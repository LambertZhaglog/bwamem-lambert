#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
int main(){
  char *pp=">gi|1828828282|read"; 
  char table[]={'A','T','C','G'};
  char buf[322];
  buf[320]='\n';
  buf[321]='\0';
  time_t t;
  srand((unsigned) time(&t));
  for(int read=10;read>0;read--){
    for(int i=0;i<320;i++){
      buf[i]=table[rand()%4];
    }
    printf("%s%d\n%s",pp,read,buf);
  }
  return 1;
}
