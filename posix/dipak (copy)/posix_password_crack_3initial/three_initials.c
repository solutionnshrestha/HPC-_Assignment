#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <time.h>
int n_passwords = 4;

char *encrypted_passwords[] = {
"$6$KB$XIt7PXVcgoQg9dthQcv.k4j3U9EwPy5G8QR9ELGoZ5NhsBQJQdYa4q.TFxD5b3tiB.sMFqUd5pOAjf8HatYMQ/",

"$6$KB$xz84XVy2vJZsXw7rnKqTBRwf9Gfl16ihNyyYMvZw46A9RFQmOM3jvt0CaDuIdT02DYmOHnv7xAtob1Z0.GP6f0",

"$6$KB$NjcXqugXU7QgthD3M1.RxPbFZCWWYHBWdqcQSqoa6f4HlFoTkiYkmVRroMqh.rqRYUjOWadwzXZOqgLv1jdpq1",

"$6$KB$2IiEWJWJbJv29Dh.V5.fpKfkrEeq6CJhx4HFtJiTkb1.Bn0Mj/B2sLJMtlF1A3Jul4vm3m0Bj35ib2/s0D3r71"


};

/**
 Required by lack of standard function in C.   
*/

void substr(char *dest, char *src, int start, int length){
  memcpy(dest, src + start, length);
  *(dest + length) = '\0';
}

/**
 This function can crack the kind of password explained above. All
combinations
 that are tried are displayed and when the password is found, #, is put
at the 
 start of the line. Note that one of the most time consuming operations
that 
 it performs is the output of intermediate results, so performance
experiments 
 for this kind of program should not include this. i.e. comment out the
printfs.
*/

void crack(char *salt_and_encrypted){
  int a,x, y, z;     // Loop counters
  char salt[7];    // String used in hashing the password. Need space for \0
  char plain[7];   // The combination of letters currently being checked
  char *enc;       // Pointer to the encrypted password
  int count = 0;   // The number of combinations explored so far

  substr(salt, salt_and_encrypted, 0, 6);
for(a='A'; a<='Z';a++){
  for(x='A'; x<='Z'; x++){
    for(y='A'; y<='Z'; y++){
      for(z=0; z<=99; z++){
        sprintf(plain, "%c%c%c%02d", a,x, y, z); 
        enc = (char *) crypt(plain, salt);
        count++;
        if(strcmp(salt_and_encrypted, enc) == 0){
          printf("#%-8d%s %s\n", count, plain, enc);
        } else {
          printf(" %-8d%s %s\n", count, plain, enc);
        }
      }
    }
  }
}
  printf("%d solutions explored\n", count);
}


int time_difference(struct timespec *start, struct timespec *finish, 
                    long long int *difference) {
  long long int ds =  finish->tv_sec - start->tv_sec; 
  long long int dn =  finish->tv_nsec - start->tv_nsec; 

  if(dn < 0 ) {
    ds--;
    dn += 1000000000; 
  } 
  *difference = ds * 1000000000 + dn;
  return !(*difference > 0);
}


int main(int argc, char *argv[]){
  int i;
struct timespec start, finish;   
  long long int time_elapsed;
  clock_gettime(CLOCK_MONOTONIC, &start);


  for(i=0;i<n_passwords;i<i++)
 {
    crack(encrypted_passwords[i]);
  }

clock_gettime(CLOCK_MONOTONIC, &finish);
time_difference(&start, &finish, &time_elapsed);
  printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed, 
         (time_elapsed/1.0e9)); 


  return 0;
}







