#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <sys/stat.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include <crypt.h>



int n_passwords = 4;
pthread_t t1, t2;
char *encrypted_passwords[] = {

"$6$KB$yPvAP5BWyF7oqOITVloN9mCAVdel.65miUZrEel72LcJy2KQDuYE6xccHS2ycoxqXDzW.lvbtDU5HuZ733K0X0",

"$6$KB$H8RIsbCyr2r7L1lklCPKY0tLK9k5WudNWpxkNbx2bCBRHCsI3qyVRY.0nrovdkDLDJRsogQE9mA3OqcIafVsV0",

"$6$KB$3rRxD6TE6OA8TgmpyLHqDg3Rf1njR3q3klxFb0vBpKtUmAub40Q9rxRvW6xwrWyZEOq4HMzQ0DZt94CEdDwGI/",

"$6$KB$tzNVqzdv42qd7V72nEGgMTpox6VZLTvDxJBU6y7FRGIWDfdn2EBdm4RPSt/A/4rwVZprY8X2HAKYxai0EpQ781"
};

void substr(char *dest, char *src, int start, int length){
  memcpy(dest, src + start, length);
  *(dest + length) = '\0';
}
  
void *kernel_function_1(void *salt_and_encrypted){

int x, y, z;     // Loop counters
  char salt[7];    // String used in hashing the password. Need space

  char plain[7];   // The combination of letters currently being checked
  char *enc;       // Pointer to the encrypted password
  int count = 0;   // The number of combinations explored so far

  substr(salt, salt_and_encrypted, 0, 6);
	for(x='A'; x<='M'; x++){
	    for(y='A'; y<='Z'; y++){
	      for(z=0; z<=99; z++){
		printf("Thread1");
		sprintf(plain, "%c%c%02d",x, y, z);
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
  printf("%d solutions explored\n", count);

}
void *kernel_function_2(void *salt_and_encrypted){

int x, y, z;     // Loop counters
  char salt[7];    // String used in hashing the password. Need space

  char plain[7];   // The combination of letters currently being checked
  char *enc;       // Pointer to the encrypted password
  int count = 0;   // The number of combinations explored so far

  substr(salt, salt_and_encrypted, 0, 6);
	for(x='N'; x<='Z'; x++){
	    for(y='A'; y<='Z'; y++){
	      for(z=0; z<=99; z++){
		printf("Thread2");
		sprintf(plain, "%c%c%02d", x, y, z);
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
  //int account = 0;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for(i=0;i<n_passwords;i<i++) {
   pthread_create(&t1, NULL, kernel_function_1, encrypted_passwords[i]);
  pthread_create(&t2, NULL, kernel_function_2, encrypted_passwords[i]);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);		

  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  time_difference(&start, &finish, &time_elapsed);
  printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed, 
         (time_elapsed/1.0e9)); 


  return 0;
}
