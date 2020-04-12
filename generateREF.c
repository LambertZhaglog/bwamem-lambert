#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
int main(){
  printf(">gi|46575915|ref|NM_008261.2| Mus musculus hepatic nuclear factor 4, alpha (Hnf4a), mRNA\n");
  char table[]={'A','T','C','G'};
  char buf[82];
  buf[80]='\n';
  buf[81]='\0';
  time_t t;
  srand((unsigned) time(&t));
  for(int line=1000000;line>0;line--){
    for(int i=0;i<80;i++){
      buf[i]=table[rand()%4];
    }
    printf("%s",buf);
  }
  return 1;
}
