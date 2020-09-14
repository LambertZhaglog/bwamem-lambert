/* The MIT License

   Copyright (c) 2008 Genome Research Ltd (GRL).

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/* Contact: Heng Li <lh3@sanger.ac.uk> */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include "utils.h"
#include "bwt.h"
#include "kvec.h"
#include <nmmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>

#ifdef USE_MALLOC_WRAPPERS
#  include "malloc_wrap.h"
#endif

void bwt_gen_cnt_table(bwt_t *bwt)
{
  int i, j;
  for (i = 0; i != 256; ++i) {
    uint32_t x = 0;
    for (j = 0; j != 4; ++j)
      x |= (((i&3) == j) + ((i>>2&3) == j) + ((i>>4&3) == j) + (i>>6 == j)) << (j<<3);
    bwt->cnt_table[i] = x;
  }
}

static inline bwtint_t bwt_invPsi(const bwt_t *bwt, bwtint_t k) // compute inverse CSA
{
  bwtint_t x = k - (k > bwt->primary);
  x = bwt_B0(bwt, x);
  x = bwt->L2[x] + bwt_occ(bwt, k, x);
  return k == bwt->primary? 0 : x;
}

// bwt->bwt and bwt->occ must be precalculated
void bwt_cal_sa(bwt_t *bwt, int intv)
{
  bwtint_t isa, sa, i; // S(isa) = sa
  int intv_round = intv;

  kv_roundup32(intv_round);
  xassert(intv_round == intv, "SA sample interval is not a power of 2.");
  xassert(bwt->bwt, "bwt_t::bwt is not initialized.");

  if (bwt->sa) free(bwt->sa);
  bwt->sa_intv = intv;
  bwt->n_sa = (bwt->seq_len + intv) / intv;
  bwt->sa = (bwtint_t*)calloc(bwt->n_sa, sizeof(bwtint_t));
  if(bwt->sa==NULL){
    printf("bwt_cal_sa error:: cannot allocate enough memory\n");
  }
  // calculate SA value
  isa = 0; sa = bwt->seq_len;
  for (i = 0; i < bwt->seq_len; ++i) {
    if (isa % intv == 0) bwt->sa[isa/intv] = sa;
    --sa;
    isa = bwt_invPsi(bwt, isa);
  }
  if (isa % intv == 0) bwt->sa[isa/intv] = sa;
  bwt->sa[0] = (bwtint_t)-1; // before this line, bwt->sa[0] = bwt->seq_len
}

#define _set_sahigh2(sahigh2,l,c) ((sahigh2)[(l)>>4]|=(c)<<(((~(l))&15)<<1))
#define _get_sahigh2(sahigh2,l) (((sahigh2)[(l)>>4]>>(((~(l))&15)<<1))&3)
#define get_sahigh2(l) (((l)>>32)&3)
void bwt_cal_sa_and_sample(bwt_t *bwt){
  xassert(bwt->bwt,"bwt_t::bwt is not initialized.");

  if(bwt->sa) free(bwt->sa);
  uint32_t *bwtsa=(uint32_t*)calloc(bwt->seq_len+1+(bwt->seq_len+1+15)/16,sizeof(uint32_t));
  if(bwtsa==NULL){
    printf("bwt_cal_sa_and_sample error:: cannot allocate enough memory\n");
  }
  bwtint_t isa,sa,i;
  uint32_t *salow32=bwtsa,*sahigh2=bwtsa+bwt->seq_len+1;
  isa=0;sa=bwt->seq_len;
  for(i=0;i<bwt->seq_len/2;i++){
    --sa;
    isa=bwt_invPsi(bwt,isa);
  }
  for(;i<bwt->seq_len;i++){
    --sa;
    isa=bwt_invPsi(bwt,isa);
    salow32[isa]=sa&0xffffffff;
    _set_sahigh2(sahigh2,isa,(sa>>32)&3);
  }
  //now isa is isaZero, sa=0
  i=0;
  uint32_t *sahigh2NEW=(uint32_t*)calloc((bwt->seq_len/2+15)/16,sizeof(uint32_t));
  if(sahigh2NEW==NULL){
    printf("bwt_cal_sa_and_sample error:: cannot allocate sahigh2NEW memory\n");
  }
  for(uint64_t j=0;i<bwt->seq_len+1;i++){
    if(i==isa){
      salow32[j]=0;
      _set_sahigh2(sahigh2NEW,j,0);
      j++;
    }
    if(_get_sahigh2(sahigh2,i)!=0 || salow32[i]!=0){
      salow32[j]=salow32[i];
      _set_sahigh2(sahigh2NEW,j,_get_sahigh2(sahigh2,i));
      j++;
    }
  }
  memmove(bwtsa+bwt->seq_len/2,sahigh2NEW,(bwt->seq_len/2+15)/16*sizeof(uint32_t));
  bwtsa=(uint32_t *)realloc(bwtsa,(bwt->seq_len/2+(bwt->seq_len/2+15)/16)*sizeof(uint32_t));
  if(bwtsa==NULL){
    printf("bwt_cal_sa_and_sample error:: cannot reallocate memory error\n");
  }
  free(sahigh2NEW);
  bwt->sa=(bwtint_t*)bwtsa;
}

static int __occ_aux(uint64_t y, int c)
{
  // reduce nucleotide counting to bits counting
  y = ((c&2)? y : ~y) >> 1 & ((c&1)? y : ~y) & 0x5555555555555555ull;
  return _mm_popcnt_u64(y);
}

bwtint_t bwt_occ(const bwt_t *bwt, bwtint_t k, ubyte_t c)
{
  bwtint_t n;
  uint32_t *p, *end;

  if (k == bwt->seq_len) return bwt->L2[c+1] - bwt->L2[c];
  if (k == (bwtint_t)(-1)) return 0;
  k -= (k >= bwt->primary); // because $ is not in bwt

  // retrieve Occ at k/OCC_INTERVAL
  n = ((bwtint_t*)(p = bwt_occ_intv(bwt, k)))[c];
  p += sizeof(bwtint_t); // jump to the start of the first BWT cell

  // calculate Occ up to the last k/32
  end = p + (((k>>5) - ((k&~OCC_INTV_MASK)>>5))<<1);
  for (; p < end; p += 2) n += __occ_aux((uint64_t)p[0]<<32 | p[1], c);

  // calculate Occ
  n += __occ_aux(((uint64_t)p[0]<<32 | p[1]) & ~((1ull<<((~k&31)<<1)) - 1), c);
  if (c == 0) n -= ~k&31; // corrected for the masked bits

  return n;
}

/*********************
 * Bidirectional BWT *
 *********************/

static void bwt_reverse_intvs(bwtintv_v *p)
{
  if (p->n > 1) {
    int j;
    for (j = 0; j < p->n>>1; ++j) {
      bwtintv_t tmp = p->a[p->n - 1 - j];
      p->a[p->n - 1 - j] = p->a[j];
      p->a[j] = tmp;
    }
  }
}
static void remove_duplicate_intvs(bwtintv_v *p){
  if(p->n>1){
    int num=p->n;
    int j=0;
    for(int i=1;i<num;i++){
      if(p->a[j].len==p->a[i].len){
	p->a[j]=p->a[i];
	p->n--;
      }else{
	j++;
	p->a[j]=p->a[i];
      }
    }
  }
}

int forwardExtensionTwoStepFsRs(const lbwt_t *bwt,char c[2],bwtintv_t *current,bwtintv_t *next){
    //return value: the step of function, if c[0]>4,return 0, if c[1]>4, return 1, else return 2;
  uint64_t lineS=current->rs/128;
  uint64_t lineE=(current->rs+current->len)/128;
  char cccc[2];
  cccc[0]=(bwt->occArray[lineS].base[0]==0)&(bwt->occArray[lineS].offset[0]==0);
  cccc[1]=(bwt->occArray[lineE].base[0]==0)&(bwt->occArray[lineE].offset[0]==0);
    int rshiftS=63-current->rs%64;//
    int rshiftE=63-(current->rs+current->len)%64;
    int greatS=(current->rs)%128>=64;
    int greatE=(current->rs+current->len)%128>=64;
    uint64_t occ1S[4], occ1E[4];
    uint32_t *tmp=(bwt->occArray[lineS].base);
    occ1S[0]=tmp[0]+tmp[1]+tmp[2]+tmp[3];
    occ1S[1]=tmp[4]+tmp[5]+tmp[6]+tmp[7];
    occ1S[2]=tmp[8]+tmp[9]+tmp[10]+tmp[11];
    occ1S[3]=tmp[12]+tmp[13]+tmp[14]+tmp[15];
    tmp=(bwt->occArray[lineE].base);
    occ1E[0]=tmp[0]+tmp[1]+tmp[2]+tmp[3];
    occ1E[1]=tmp[4]+tmp[5]+tmp[6]+tmp[7];
    occ1E[2]=tmp[8]+tmp[9]+tmp[10]+tmp[11];
    occ1E[3]=tmp[12]+tmp[13]+tmp[14]+tmp[15];
    uint64_t nextSelectS[2],nextUnselectS[2];
    uint64_t nextSelectE[2],nextUnselectE[2];
    uint64_t next1SelectS[2],next1UnselectS[2];
    uint64_t next1SelectE[2],next1UnselectE[2];
    uint64_t baseUnselectS,baseUnselectE;//select for RS value, unselect for FS value
    uint64_t cntSelectS,cntSelectE;//next for get the original char and can be used to caculate next1,cnt for change next to number, base for accumulate the Oline.base
    uint64_t cntUnselectS,cntUnselectE;
    switch(c[0]){
        case 0://a
            nextSelectS[0]=bwt->occArray[lineS].offset[4] & bwt->occArray[lineS].offset[6];
            nextSelectS[1]=bwt->occArray[lineS].offset[5] & bwt->occArray[lineS].offset[7];
            nextUnselectS[0]=0;
            nextUnselectS[1]=0;
            baseUnselectS=0;
            nextSelectE[0]=bwt->occArray[lineE].offset[4] & bwt->occArray[lineE].offset[6];
            nextSelectE[1]=bwt->occArray[lineE].offset[5] & bwt->occArray[lineE].offset[7];
            nextUnselectE[0]=0;
            nextUnselectE[1]=0;
            baseUnselectE=0;
            break;
        case 1://c
            nextSelectS[0]=bwt->occArray[lineS].offset[4] &(~ bwt->occArray[lineS].offset[6]);
            nextSelectS[1]=bwt->occArray[lineS].offset[5] &(~ bwt->occArray[lineS].offset[7]);
            nextUnselectS[0]=bwt->occArray[lineS].offset[4] & bwt->occArray[lineS].offset[6];
            nextUnselectS[1]=bwt->occArray[lineS].offset[5] & bwt->occArray[lineS].offset[7];
            baseUnselectS=occ1S[3];
            nextSelectE[0]=bwt->occArray[lineE].offset[4] &(~ bwt->occArray[lineE].offset[6]);
            nextSelectE[1]=bwt->occArray[lineE].offset[5] &(~ bwt->occArray[lineE].offset[7]);
            nextUnselectE[0]=bwt->occArray[lineE].offset[4] & bwt->occArray[lineE].offset[6];
            nextUnselectE[1]=bwt->occArray[lineE].offset[5] & bwt->occArray[lineE].offset[7];
            baseUnselectE=occ1E[3];
            break;
        case 2://g
            nextSelectS[0]=(~bwt->occArray[lineS].offset[4]) & bwt->occArray[lineS].offset[6];
            nextSelectS[1]=(~bwt->occArray[lineS].offset[5]) & bwt->occArray[lineS].offset[7];
            nextUnselectS[0]=bwt->occArray[lineS].offset[4];
            nextUnselectS[1]=bwt->occArray[lineS].offset[5];
            baseUnselectS=occ1S[3]+occ1S[2];
            nextSelectE[0]=(~bwt->occArray[lineE].offset[4]) & bwt->occArray[lineE].offset[6];
            nextSelectE[1]=(~bwt->occArray[lineE].offset[5]) & bwt->occArray[lineE].offset[7];
            nextUnselectE[0]=bwt->occArray[lineE].offset[4];
            nextUnselectE[1]=bwt->occArray[lineE].offset[5];
            baseUnselectE=occ1E[3]+occ1E[2];
            break;
        case 3://t
            nextSelectS[0]=(~bwt->occArray[lineS].offset[4]) &(~ bwt->occArray[lineS].offset[6]);
            nextSelectS[1]=(~bwt->occArray[lineS].offset[5]) &(~ bwt->occArray[lineS].offset[7]);
            nextUnselectS[0]=(bwt->occArray[lineS].offset[4]|bwt->occArray[lineS].offset[6]);
            nextUnselectS[1]=(bwt->occArray[lineS].offset[5]|bwt->occArray[lineS].offset[7]);
            baseUnselectS=occ1S[3]+occ1S[1]+occ1S[2];
            nextSelectE[0]=(~bwt->occArray[lineE].offset[4]) &(~ bwt->occArray[lineE].offset[6]);
            nextSelectE[1]=(~bwt->occArray[lineE].offset[5]) &(~ bwt->occArray[lineE].offset[7]);
            nextUnselectE[0]=(bwt->occArray[lineE].offset[4]|bwt->occArray[lineE].offset[6]);
            nextUnselectE[1]=(bwt->occArray[lineE].offset[5]|bwt->occArray[lineE].offset[7]);
            baseUnselectE=occ1E[3]+occ1E[1]+occ1E[2];
            break;
        default:return 0;
    }
    if(greatS){
      cntSelectS=_mm_popcnt_u64(nextSelectS[0])+_mm_popcnt_u64((nextSelectS[1]>>rshiftS)>>1);
      cntUnselectS=_mm_popcnt_u64(nextUnselectS[0])+_mm_popcnt_u64((nextUnselectS[1]>>rshiftS)>>1);
    }else{
      cntSelectS=_mm_popcnt_u64((nextSelectS[0]>>rshiftS)>>1);
      cntUnselectS=_mm_popcnt_u64((nextUnselectS[0]>>rshiftS)>>1);
    }
    if(greatE){
      cntSelectE=_mm_popcnt_u64(nextSelectE[0])+_mm_popcnt_u64((nextSelectE[1]>>rshiftE)>>1);
      cntUnselectE=_mm_popcnt_u64(nextUnselectE[0])+_mm_popcnt_u64((nextUnselectE[1]>>rshiftE)>>1);
    }else{
      cntSelectE=_mm_popcnt_u64((nextSelectE[0]>>rshiftE)>>1);
      cntUnselectE=_mm_popcnt_u64((nextUnselectE[0]>>rshiftE)>>1);
    }
    next[0].rs=bwt->c1Array[3-c[0]]+occ1S[3-c[0]]+cntSelectS;
    next[0].len=occ1E[3-c[0]]-occ1S[3-c[0]]+cntSelectE-cntSelectS;
    next[0].fs=current->fs+cntUnselectE-cntUnselectS+baseUnselectE-baseUnselectS;
    //printf("cntS=%lu,cntE=%lu,occ1S=%lu,occ1E=%lu\n",cntSelectS,cntSelectE,occ1S[3-c[0]],occ1E[3-c[0]]);
    next[0].readBegin=current->readBegin;
    next[0].readEnd=current->readEnd+1;
    switch(c[1]){
        case 0://a
            next1SelectS[0]=bwt->occArray[lineS].offset[0] & bwt->occArray[lineS].offset[2];
            next1SelectS[1]=bwt->occArray[lineS].offset[1] & bwt->occArray[lineS].offset[3];
            next1UnselectS[0]=0;
            next1UnselectS[1]=0;
            baseUnselectS=0;
            next1SelectE[0]=bwt->occArray[lineE].offset[0] & bwt->occArray[lineE].offset[2];
            next1SelectE[1]=bwt->occArray[lineE].offset[1] & bwt->occArray[lineE].offset[3];
            next1UnselectE[0]=0;
            next1UnselectE[1]=0;
            baseUnselectE=0;
            break;
        case 1://c
            next1SelectS[0]=bwt->occArray[lineS].offset[0] &(~ bwt->occArray[lineS].offset[2]);
            next1SelectS[1]=bwt->occArray[lineS].offset[1] &(~ bwt->occArray[lineS].offset[3]);
            next1UnselectS[0]=bwt->occArray[lineS].offset[0] & bwt->occArray[lineS].offset[2];
            next1UnselectS[1]=bwt->occArray[lineS].offset[1] & bwt->occArray[lineS].offset[3];
            baseUnselectS=bwt->occArray[lineS].base[(3-c[0])*4+3];
            next1SelectE[0]=bwt->occArray[lineE].offset[0] &(~ bwt->occArray[lineE].offset[2]);
            next1SelectE[1]=bwt->occArray[lineE].offset[1] &(~ bwt->occArray[lineE].offset[3]);
            next1UnselectE[0]=bwt->occArray[lineE].offset[0] & bwt->occArray[lineE].offset[2];
            next1UnselectE[1]=bwt->occArray[lineE].offset[1] & bwt->occArray[lineE].offset[3];
            baseUnselectE=bwt->occArray[lineE].base[(3-c[0])*4+3];
            break;
        case 2://g
            next1SelectS[0]=(~bwt->occArray[lineS].offset[0]) & bwt->occArray[lineS].offset[2];
            next1SelectS[1]=(~bwt->occArray[lineS].offset[1]) & bwt->occArray[lineS].offset[3];
            next1UnselectS[0]=bwt->occArray[lineS].offset[0];
            next1UnselectS[1]=bwt->occArray[lineS].offset[1];
            baseUnselectS=bwt->occArray[lineS].base[(3-c[0])*4+3]+bwt->occArray[lineS].base[(3-c[0])*4+2];
            next1SelectE[0]=(~bwt->occArray[lineE].offset[0]) & bwt->occArray[lineE].offset[2];
            next1SelectE[1]=(~bwt->occArray[lineE].offset[1]) & bwt->occArray[lineE].offset[3];
            next1UnselectE[0]=bwt->occArray[lineE].offset[0];
            next1UnselectE[1]=bwt->occArray[lineE].offset[1];
            baseUnselectE=bwt->occArray[lineE].base[(3-c[0])*4+3]+bwt->occArray[lineE].base[(3-c[0])*4+2];
            break;
        case 3://t
            next1SelectS[0]=(~bwt->occArray[lineS].offset[0]) &(~ bwt->occArray[lineS].offset[2]);
            next1SelectS[1]=(~bwt->occArray[lineS].offset[1]) &(~ bwt->occArray[lineS].offset[3]);
            next1UnselectS[0]=(bwt->occArray[lineS].offset[0]|bwt->occArray[lineS].offset[2]);
            next1UnselectS[1]=(bwt->occArray[lineS].offset[1]|bwt->occArray[lineS].offset[3]);
            baseUnselectS=bwt->occArray[lineS].base[(3-c[0])*4+3]+bwt->occArray[lineS].base[(3-c[0])*4+2]+bwt->occArray[lineS].base[(3-c[0])*4+1];
            next1SelectE[0]=(~bwt->occArray[lineE].offset[0]) &(~ bwt->occArray[lineE].offset[2]);
            next1SelectE[1]=(~bwt->occArray[lineE].offset[1]) &(~ bwt->occArray[lineE].offset[3]);
            next1UnselectE[0]=(bwt->occArray[lineE].offset[0]|bwt->occArray[lineE].offset[2]);
            next1UnselectE[1]=(bwt->occArray[lineE].offset[1]|bwt->occArray[lineE].offset[3]);
            baseUnselectE=bwt->occArray[lineE].base[(3-c[0])*4+3]+bwt->occArray[lineE].base[(3-c[0])*4+2]+bwt->occArray[lineE].base[(3-c[0])*4+1];
            break;
        default:
            return 1;
    }
    if(greatS){
      cntSelectS=_mm_popcnt_u64(nextSelectS[0]&next1SelectS[0])+_mm_popcnt_u64(((nextSelectS[1]&next1SelectS[1])>>rshiftS)>>1);
      cntUnselectS=_mm_popcnt_u64(next1UnselectS[0]&nextSelectS[0])+_mm_popcnt_u64(((next1UnselectS[1]&nextSelectS[1])>>rshiftS)>>1);
    }else{
      cntSelectS=_mm_popcnt_u64(((nextSelectS[0]&next1SelectS[0])>>rshiftS)>>1);
      cntUnselectS=_mm_popcnt_u64(((nextSelectS[0]&next1UnselectS[0])>>rshiftS)>>1);
    }
    if(greatE){
      cntSelectE=_mm_popcnt_u64(nextSelectE[0]&next1SelectE[0])+_mm_popcnt_u64(((nextSelectE[1]&next1SelectE[1])>>rshiftE)>>1);
      cntUnselectE=_mm_popcnt_u64(nextSelectE[0]&next1UnselectE[0])+_mm_popcnt_u64(((nextSelectE[1]&next1UnselectE[1])>>rshiftE)>>1);
    }else{
      cntSelectE=_mm_popcnt_u64(((nextSelectE[0]&next1SelectE[0])>>rshiftE)>>1);
      cntUnselectE=_mm_popcnt_u64(((nextSelectE[0]&next1UnselectE[0])>>rshiftE)>>1);
    }
    next[1].rs=bwt->c2Array[(3-c[1])*4+(3-c[0])]+bwt->occArray[lineS].base[(3-c[0])*4+(3-c[1])]+cntSelectS;
    next[1].len=bwt->occArray[lineE].base[(3-c[0])*4+(3-c[1])]-bwt->occArray[lineS].base[(3-c[0])*4+(3-c[1])]+cntSelectE-cntSelectS;
    next[1].fs=next[0].fs+cntUnselectE-cntUnselectS+baseUnselectE-baseUnselectS;
    next[1].readBegin=current->readBegin;
    next[1].readEnd=current->readEnd+2;
    c[0]=cccc[0];
    c[1]=cccc[1];
    return 2;
}

int backwardExtensionTwoStepFs(const lbwt_t *bwt, char cc[2], bwtintv_t *current, bwtintv_t *next){
  //c[0]=q[current->readBegin-2], c[1]=q[current->readEnd-1]
  int c[2];
  c[1]=cc[0];c[0]=cc[1];
    uint64_t lineS=current->fs/128;
    uint64_t lineE=(current->fs+current->len)/128;
    int rshiftS=63-current->fs%64;//
    int rshiftE=63-(current->fs+current->len)%64;
    int greatS=(current->fs)%128>=64;
    int greatE=(current->fs+current->len)%128>=64;
    uint64_t occ1S[4], occ1E[4];
    uint32_t *tmp=(bwt->occArray[lineS].base);
    occ1S[0]=tmp[0]+tmp[1]+tmp[2]+tmp[3];
    occ1S[1]=tmp[4]+tmp[5]+tmp[6]+tmp[7];
    occ1S[2]=tmp[8]+tmp[9]+tmp[10]+tmp[11];
    occ1S[3]=tmp[12]+tmp[13]+tmp[14]+tmp[15];
    tmp=(bwt->occArray[lineE].base);
    occ1E[0]=tmp[0]+tmp[1]+tmp[2]+tmp[3];
    occ1E[1]=tmp[4]+tmp[5]+tmp[6]+tmp[7];
    occ1E[2]=tmp[8]+tmp[9]+tmp[10]+tmp[11];
    occ1E[3]=tmp[12]+tmp[13]+tmp[14]+tmp[15];
    uint64_t nextSelectS[2];
    uint64_t nextSelectE[2];
    uint64_t next1SelectS[2];
    uint64_t next1SelectE[2];//select for FS value
    uint64_t cntSelectS,cntSelectE;//next for get the original char and can be used to caculate next1,cnt for change next to number
    switch(c[0]){
        case 0://a
	  nextSelectS[0]=(~bwt->occArray[lineS].offset[4]) & (~bwt->occArray[lineS].offset[6]);
	  nextSelectS[1]=(~bwt->occArray[lineS].offset[5]) & (~bwt->occArray[lineS].offset[7]);
	  nextSelectE[0]=(~bwt->occArray[lineE].offset[4]) & (~bwt->occArray[lineE].offset[6]);
	  nextSelectE[1]=(~bwt->occArray[lineE].offset[5]) & (~bwt->occArray[lineE].offset[7]);
	  break;
        case 1://c
	  nextSelectS[0]=(~bwt->occArray[lineS].offset[4]) & bwt->occArray[lineS].offset[6];
	  nextSelectS[1]=(~bwt->occArray[lineS].offset[5]) & bwt->occArray[lineS].offset[7];
	  nextSelectE[0]=(~bwt->occArray[lineE].offset[4]) & bwt->occArray[lineE].offset[6];
	  nextSelectE[1]=(~bwt->occArray[lineE].offset[5]) & bwt->occArray[lineE].offset[7];
	  break;
        case 2://g
	  nextSelectS[0]=bwt->occArray[lineS].offset[4] & (~bwt->occArray[lineS].offset[6]);
	  nextSelectS[1]=bwt->occArray[lineS].offset[5] & (~bwt->occArray[lineS].offset[7]);
	  nextSelectE[0]=bwt->occArray[lineE].offset[4] & (~bwt->occArray[lineE].offset[6]);
	  nextSelectE[1]=bwt->occArray[lineE].offset[5] & (~bwt->occArray[lineE].offset[7]);
	  break;
        case 3://t
	  nextSelectS[0]=bwt->occArray[lineS].offset[4] & bwt->occArray[lineS].offset[6];
	  nextSelectS[1]=bwt->occArray[lineS].offset[5] & bwt->occArray[lineS].offset[7];
	  nextSelectE[0]=bwt->occArray[lineE].offset[4] & bwt->occArray[lineE].offset[6];
	  nextSelectE[1]=bwt->occArray[lineE].offset[5] & bwt->occArray[lineE].offset[7];
	  break;
        default:return 0;
    }
    if(greatS){
      cntSelectS=_mm_popcnt_u64(nextSelectS[0])+_mm_popcnt_u64((nextSelectS[1]>>rshiftS)>>1);
    }else{
      cntSelectS=_mm_popcnt_u64((nextSelectS[0]>>rshiftS)>>1);
    }
    if(greatE){
      cntSelectE=_mm_popcnt_u64(nextSelectE[0])+_mm_popcnt_u64((nextSelectE[1]>>rshiftE)>>1);
    }else{
      cntSelectE=_mm_popcnt_u64((nextSelectE[0]>>rshiftE)>>1);
    }
    next[0].fs=bwt->c1Array[c[0]]+occ1S[c[0]]+cntSelectS;
    next[0].len=occ1E[c[0]]-occ1S[c[0]]+cntSelectE-cntSelectS;
    next[0].readBegin=current->readBegin-1;
    next[0].readEnd=current->readEnd;
    switch(c[1]){
        case 0://a
	  next1SelectS[0]=(~bwt->occArray[lineS].offset[0]) & (~bwt->occArray[lineS].offset[2]);
	  next1SelectS[1]=(~bwt->occArray[lineS].offset[1]) & (~bwt->occArray[lineS].offset[3]);
	  next1SelectE[0]=(~bwt->occArray[lineE].offset[0]) & (~bwt->occArray[lineE].offset[2]);
	  next1SelectE[1]=(~bwt->occArray[lineE].offset[1]) & (~bwt->occArray[lineE].offset[3]);
	  break;
        case 1://c
	  next1SelectS[0]=(~bwt->occArray[lineS].offset[0]) & bwt->occArray[lineS].offset[2];
	  next1SelectS[1]=(~bwt->occArray[lineS].offset[1]) & bwt->occArray[lineS].offset[3];
	  next1SelectE[0]=(~bwt->occArray[lineE].offset[0]) & bwt->occArray[lineE].offset[2];
	  next1SelectE[1]=(~bwt->occArray[lineE].offset[1]) & bwt->occArray[lineE].offset[3];
	  break;
        case 2://g
	  next1SelectS[0]=bwt->occArray[lineS].offset[0] & (~bwt->occArray[lineS].offset[2]);
	  next1SelectS[1]=bwt->occArray[lineS].offset[1] & (~bwt->occArray[lineS].offset[3]);
	  next1SelectE[0]=bwt->occArray[lineE].offset[0] & (~bwt->occArray[lineE].offset[2]);
	  next1SelectE[1]=bwt->occArray[lineE].offset[1] & (~bwt->occArray[lineE].offset[3]);
	  break;
        case 3://t
	  next1SelectS[0]=bwt->occArray[lineS].offset[0] & bwt->occArray[lineS].offset[2];
	  next1SelectS[1]=bwt->occArray[lineS].offset[1] & bwt->occArray[lineS].offset[3];
	  next1SelectE[0]=bwt->occArray[lineE].offset[0] & bwt->occArray[lineE].offset[2];
	  next1SelectE[1]=bwt->occArray[lineE].offset[1] & bwt->occArray[lineE].offset[3];
	  break;
        default:
	  return 1;
    }
    if(greatS){
      cntSelectS=_mm_popcnt_u64(nextSelectS[0]&next1SelectS[0])+_mm_popcnt_u64(((nextSelectS[1]&next1SelectS[1])>>rshiftS)>>1);
    }else{
      cntSelectS=_mm_popcnt_u64(((nextSelectS[0]&next1SelectS[0])>>rshiftS)>>1);
    }
    if(greatE){
      cntSelectE=_mm_popcnt_u64(nextSelectE[0]&next1SelectE[0])+_mm_popcnt_u64(((nextSelectE[1]&next1SelectE[1])>>rshiftE)>>1);
    }else{
      cntSelectE=_mm_popcnt_u64(((nextSelectE[0]&next1SelectE[0])>>rshiftE)>>1);
    }
    next[1].fs=bwt->c2Array[c[1]*4+c[0]]+bwt->occArray[lineS].base[c[0]*4+c[1]]+cntSelectS;
    next[1].len=bwt->occArray[lineE].base[c[0]*4+c[1]]-bwt->occArray[lineS].base[c[0]*4+c[1]]+cntSelectE-cntSelectS;
    next[1].readBegin=current->readBegin-2;
    next[1].readEnd=current->readEnd;
    return 2;
}

void printInterval(bwtintv_t *ik){
  printf("%lu,%lu,%lu,%d,%d\n",ik->fs,ik->rs,ik->len,ik->readBegin,ik->readEnd);
}
  
  
int bwt_smem1a(const lbwt_t *lbwt, int len, const uint8_t *q, int x, int min_intv, bwtintv_v *mem, bwtintv_v *tmpvec[2])
{
  int i, ret;
  bwtintv_v a[2], *curr;
  bwtintv_t *ik,*jk,*swapij;
  ik=(bwtintv_t *)calloc(2,sizeof(bwtintv_t));
  jk=(bwtintv_t *)calloc(2,sizeof(bwtintv_t));
	
  mem->n = 0;
  if (q[x] > 3) return x + 1;
  if (min_intv < 1) min_intv = 1; // the interval size should be at least 1
  kv_init(a[0]); kv_init(a[1]);
  curr = tmpvec && tmpvec[1]? tmpvec[1] : &a[1];
  //bwt_set_intv(bwt, q[x], ik); // the initial interval of a single base
  ik[1].fs=lbwt->c1Array[q[x]];ik[1].rs=lbwt->c1Array[3-q[x]];
  ik[1].len=lbwt->c1Array[q[x]+1]-lbwt->c1Array[q[x]];
  ik[1].readBegin=x;ik[1].readEnd=x+1;
  kv_push(bwtintv_t,*curr,ik[1]);
  //printInterval(ik+1);
	
  char cc[2];
  for (i = x + 1, curr->n = 0; i<len; i=i+2) { // forward search
    cc[0]=q[i];
    cc[1]=(i==len-1?4:q[i+1]);
    int retf=forwardExtensionTwoStepFsRs(lbwt,cc,ik+1,jk);
    //printInterval(jk); printInterval(jk+1);
    if(retf==0){
      break;
    }else if(retf==1){
      if(jk[0].len<min_intv) break;
      kv_push(bwtintv_t,*curr,jk[0]);
    }else{
      if(jk[0].len<min_intv) break;
      kv_push(bwtintv_t,*curr,jk[0]);
      if(jk[1].len<min_intv) break;
      kv_push(bwtintv_t,*curr,jk[1]);
      swapij=ik; ik=jk;jk=swapij;
    }
  }
  remove_duplicate_intvs(curr);
  bwt_reverse_intvs(curr); // s.t. smaller intervals (i.e. longer matches) visited first
  ret = curr->a[0].readEnd; // this will be the returned value
	
  int lastBegin=x+1;
  for(int ii=0;ii<curr->n;ii++){
    ik[0]=curr->a[ii];
    for(int jj=x-1;jj>-1;jj=jj-2){
      cc[1]=q[jj];cc[0]=jj==0?4:q[jj-1];
      int retf=backwardExtensionTwoStepFs(lbwt,cc,ik,jk);
      /* printf("backward"); */
      /* printInterval(jk);printInterval(jk+1); */
      if(retf==0){
	break;
      }else if(retf==1){
	if(jk[0].len>=min_intv){
	  ik[0]=jk[0];
	}
	break;
      }else{
	if(jk[1].len>=min_intv){
	  ik[0]=jk[1];
	}else if(jk[0].len>=min_intv){
	  ik[0]=jk[0];
	  break;
	}else{
	  break;
	}
      }
    }
    if(ik[0].readBegin<lastBegin){
      lastBegin=ik[0].readBegin;
      kv_push(bwtintv_t,*mem,ik[0]);
    }
  }
  bwt_reverse_intvs(mem); // s.t. sorted by the start coordinate

  free(ik);free(jk);
  if (tmpvec == 0 || tmpvec[0] == 0) free(a[0].a);
  if (tmpvec == 0 || tmpvec[1] == 0) free(a[1].a);
  return ret;
}

int bwt_smem1(const lbwt_t *lbwt, int len, const uint8_t *q, int x, int min_intv, bwtintv_v *mem, bwtintv_v *tmpvec[2])
{
  return bwt_smem1a(lbwt, len, q, x, min_intv, mem, tmpvec);
}

int bwt_seed_strategy1(const lbwt_t *lbwt, int len, const uint8_t *q, int x, int min_len, int max_intv, bwtintv_t *mem)
{
  int i;
  char cc[2];
  bwtintv_t *ik,*jk,*swapij;
  bwtintv_t ika[2],jka[2];
  ik=ika;jk=jka;
  memset(mem,0,sizeof(bwtintv_t));
  if(q[x]>3) return x+1;
  ik[1].fs=lbwt->c1Array[q[x]];ik[1].rs=lbwt->c1Array[3-q[x]];
  ik[1].len=lbwt->c1Array[q[x]+1]-lbwt->c1Array[q[x]];
  ik[1].readBegin=x;ik[1].readEnd=x+1;
  for(i=x+1;i<len;i=i+2){
    cc[0]=q[i];
    cc[1]=(i==len-1?4:q[i+1]);
    int retf=forwardExtensionTwoStepFsRs(lbwt,cc,ik+1,jk);
    if(retf==0){
      return i+1;
    }else if(retf==1){
      if(jk[0].len<max_intv && jk[0].readEnd-jk[0].readBegin>=min_len){
	*mem=jk[0];
	return i+1;
      }
      return i+2;
    }else {
      if(jk[0].len<max_intv && jk[0].readEnd-jk[0].readBegin>=min_len){
	*mem=jk[0];
	return i+1;
      }else if(jk[1].len<max_intv && jk[1].readEnd-jk[1].readBegin>=min_len){
	*mem=jk[1];
	return i+2;
      }else{
	swapij=ik;ik=jk;jk=swapij;
      }
    }
  }
  return len;
}

/*************************
 * Read/write BWT and SA *
 *************************/

void bwt_dump_bwt(const char *fn, const bwt_t *bwt)
{
  FILE *fp;
  fp = xopen(fn, "wb");
  err_fwrite(&bwt->primary, sizeof(bwtint_t), 1, fp);
  err_fwrite(bwt->L2+1, sizeof(bwtint_t), 4, fp);
  err_fwrite(bwt->bwt, 4, bwt->bwt_size, fp);
  err_fflush(fp);
  err_fclose(fp);
}

void lbwt_dump_lbwt(const char *fn,const lbwt_t *lbwt){
  FILE *fp;
  fp=xopen(fn,"wb");
  err_fwrite(&lbwt->refLen,sizeof(uint64_t),1,fp);
  err_fwrite(lbwt->c1Array,sizeof(uint64_t),5,fp);
  err_fwrite(lbwt->c2Array,sizeof(uint64_t),16,fp);
  err_fwrite(lbwt->occArray,sizeof(Occline),(lbwt->refLen*2+127)/128,fp);
  err_fflush(fp);
  err_fclose(fp);
}
void bwt_dump_sa(const char *fn, const bwt_t *bwt)
{
  FILE *fp;
  fp = xopen(fn, "wb");
  err_fwrite(&bwt->primary, sizeof(bwtint_t), 1, fp);
  err_fwrite(bwt->L2+1, sizeof(bwtint_t), 4, fp);
  err_fwrite(&bwt->sa_intv, sizeof(bwtint_t), 1, fp);
  err_fwrite(&bwt->seq_len, sizeof(bwtint_t), 1, fp);
  err_fwrite(bwt->sa + 1, sizeof(bwtint_t), bwt->n_sa - 1, fp);
  err_fflush(fp);
  err_fclose(fp);
}

void bwt_dump_sa_lambert(const char *fn, const bwt_t *bwt){
  FILE *fp;
  fp=xopen(fn,"wb");
  uint64_t seq_len=bwt->seq_len/2;
  err_fwrite(&seq_len,sizeof(uint64_t),1,fp);//the number of entries of suffix array
  err_fwrite(bwt->sa,sizeof(uint32_t),seq_len+(seq_len+15)/16,fp);//the encoded suffix array, low32 at begin, then the high2 at end.
  err_fflush(fp);
  err_fclose(fp);
}

static bwtint_t fread_fix(FILE *fp, bwtint_t size, void *a)
{ // Mac/Darwin has a bug when reading data longer than 2GB. This function fixes this issue by reading data in small chunks
  const int bufsize = 0x1000000; // 16M block
  bwtint_t offset = 0;
  while (size) {
    int x = bufsize < size? bufsize : size;
    if ((x = err_fread_noeof(a + offset, 1, x, fp)) == 0) break;
    size -= x; offset += x;
  }
  return offset;
}

void bwt_restore_sa(const char *fn, bwt_t *bwt)
{
  char skipped[256];
  FILE *fp;
  bwtint_t primary;

  fp = xopen(fn, "rb");
  err_fread_noeof(&primary, sizeof(bwtint_t), 1, fp);
  xassert(primary == bwt->primary, "SA-BWT inconsistency: primary is not the same.");
  err_fread_noeof(skipped, sizeof(bwtint_t), 4, fp); // skip
  err_fread_noeof(&bwt->sa_intv, sizeof(bwtint_t), 1, fp);
  err_fread_noeof(&primary, sizeof(bwtint_t), 1, fp);
  xassert(primary == bwt->seq_len, "SA-BWT inconsistency: seq_len is not the same.");

  bwt->n_sa = (bwt->seq_len + bwt->sa_intv) / bwt->sa_intv;
  bwt->sa = (bwtint_t*)calloc(bwt->n_sa, sizeof(bwtint_t));
  if(bwt->sa==NULL){
    printf("bwt_restore_sa error:: cannot allocate enough memory\n");
  }
  bwt->sa[0] = -1;

  fread_fix(fp, sizeof(bwtint_t) * (bwt->n_sa - 1), bwt->sa + 1);
  err_fclose(fp);
}

void bwt_restore_sa_lambert(const char *fn,lbwt_t *lbwt){
  FILE *fp;
  fp=xopen(fn,"rb");
  uint64_t seq_len;
  err_fread_noeof(&seq_len,sizeof(uint64_t),1,fp);
  if(lbwt->refLen*2!=seq_len){
    printf("bwt_restore_sa_lambert error:: refLen*2!=seq_len\n");
  }
  lbwt->sa_low32=(uint32_t*)calloc(seq_len,sizeof(uint32_t));
  lbwt->sa_high2=(uint32_t*)calloc((seq_len+15)/16,sizeof(uint32_t));
  if(lbwt->sa_low32==NULL){
    printf("bwt_restore_sa_lambert error:: cannot allocate enough memory\n");
  }
  if(lbwt->sa_high2==NULL){
    printf("bwt_restore_sa_lambert error:: cannot allocate enough memory\n");
  }
  err_fread_noeof(lbwt->sa_low32,sizeof(uint32_t),seq_len,fp);
  err_fread_noeof(lbwt->sa_high2,sizeof(uint32_t),(seq_len+15)/16,fp);
  err_fclose(fp);
}
  
bwt_t *bwt_restore_bwt(const char *fn)
{
  bwt_t *bwt;
  FILE *fp;

  bwt = (bwt_t*)calloc(1, sizeof(bwt_t));
  fp = xopen(fn, "rb");
  err_fseek(fp, 0, SEEK_END);
  bwt->bwt_size = (err_ftell(fp) - sizeof(bwtint_t) * 5) >> 2;
  bwt->bwt = (uint32_t*)calloc(bwt->bwt_size, 4);
  err_fseek(fp, 0, SEEK_SET);
  err_fread_noeof(&bwt->primary, sizeof(bwtint_t), 1, fp);
  err_fread_noeof(bwt->L2+1, sizeof(bwtint_t), 4, fp);
  fread_fix(fp, bwt->bwt_size<<2, bwt->bwt);
  bwt->seq_len = bwt->L2[4];
  err_fclose(fp);
  bwt_gen_cnt_table(bwt);

  return bwt;
}

lbwt_t *lbwt_restore_lbwt(const char *fn){
  lbwt_t *lbwt;
  FILE *fp;
  lbwt=(lbwt_t*)calloc(1,sizeof(lbwt_t));
  fp=xopen(fn,"rb");
  err_fread_noeof(&lbwt->refLen,sizeof(uint64_t),1,fp);
  err_fread_noeof(lbwt->c1Array,sizeof(uint64_t),5,fp);
  err_fread_noeof(lbwt->c2Array,sizeof(uint64_t),16,fp);
  lbwt->occArray=(Occline*)valloc((lbwt->refLen*2+127)/128*sizeof(Occline));
  err_fread_noeof(lbwt->occArray,sizeof(Occline),(lbwt->refLen*2+127)/128,fp);
  if(lbwt->occArray==NULL){
    printf("lbwt_restore_lbwt error:: cannot allocate enough memory\n");
  }
  err_fflush(fp);
  err_fclose(fp);
  return lbwt;
}

void bwt_destroy(bwt_t *bwt)
{
  if (bwt == 0) return;
  free(bwt->sa); free(bwt->bwt);
  free(bwt);
}

void lbwt_destroy(lbwt_t *lbwt){
  if(lbwt==0) return;
  free(lbwt->occArray);free(lbwt->sa_low32);free(lbwt->sa_high2);
  free(lbwt);
}

#define _get_sahigh2(sahigh2,l) (((sahigh2)[(l)>>4]>>(((~(l))&15)<<1))&3)
#define _get_pac(pac, l) ((pac)[(l)>>2]>>((~(l)&3)<<1)&3)
void constructOccArray(lbwt_t *lbwt, char *pac, bwt_t *bwt){
  uint64_t seq_len=bwt->seq_len/2;
  lbwt->occArray=(Occline*)calloc((seq_len+127)/128,sizeof(Occline));
  if(lbwt->occArray==NULL){
    printf("constructOccArray error:: cannot allocate enough memory\n");
  }
  uint32_t *sahigh2=(uint32_t*)bwt->sa+seq_len;
  uint32_t *bwtsa=(uint32_t*)bwt->sa;
  //c1Array, c2Array set initial value
  for(int i=0;i<4;i++){
    lbwt->c1Array[i]=0;
  }
  for(int i=0;i<16;i++){
    lbwt->c2Array[i]=0;
  }
  //fill occArray
  for(uint64_t i=0;i<seq_len;i++){
    uint64_t offsetOcc=i%128;
    uint64_t lineOcc=i/128;
    //set Occline.base and initialize Occline.offset
    if(i%128==0){
      for(int j=0;j<16;j++){
	lbwt->occArray[lineOcc].base[j]=lbwt->c2Array[(j&3)*4+(j>>2)];
      }
      for(int j=0;j<8;j++){
	lbwt->occArray[lineOcc].offset[j]=0ull;
      }
    }
    //get B string and the second last string
    uint64_t sa=bwtsa[i]+((_get_sahigh2(sahigh2,i)&3ull)<<32);
    char b[2];
    b[0]=_get_pac(pac,sa+seq_len-2);
    b[1]=_get_pac(pac,sa+seq_len-1);
    /* temporal code used to generate human readable suffix array and pac information for debug */
    /* char dict[4];dict[0]='A';dict[1]='C';dict[2]='G';dict[3]='T'; */
    /* printf("i=%lu,sa=%lu,prefix=",i,sa); */
    /* for(int jj=0;jj<20;jj++) */
    /*   printf("%c",dict[(int)_get_pac(pac,sa+jj)]); */
    /* printf("\n"); */
    /* temporal code end */
    
    //count
    lbwt->c1Array[(int)b[1]]++;
    lbwt->c2Array[(int)b[0]*4+b[1]]++;
    //set Occline.offset
    uint64_t c[4];
    c[0]=(b[0]&2)>>1; c[1]=b[0]&1; c[2]=(b[1]&2)>>1; c[3]=b[1]&1;
    for(int j=0;j<4;j++){
      if(offsetOcc<64){
	lbwt->occArray[lineOcc].offset[2*j]|=c[j]<<(63-offsetOcc);
      }else{
	lbwt->occArray[lineOcc].offset[2*j+1]|=c[j]<<(127-offsetOcc);
      }
    }
  }
  //correct the c2Array
  uint64_t tmp=0;
  for(int i=0;i<16;i++){
    uint64_t tmp2=lbwt->c2Array[i];
    lbwt->c2Array[i]=tmp;
    tmp+=tmp2;
  }
  if(tmp!=seq_len){
    printf("build FDM-index error::the occArray error!\n");
  }
  //correct the c1Array
  tmp=0;
  for(int i=0;i<4;i++){
    uint64_t tmp2=lbwt->c1Array[i];
    lbwt->c1Array[i]=tmp;
    tmp+=tmp2;
  }
  lbwt->c1Array[4]=tmp;
}
