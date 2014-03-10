#ifndef MIC_DEV 
#define MIC_DEV 0 
#endif 
 
#include <stdio.h> 
#include <stdlib.h> 
#include <omp.h> 
#include <mkl.h> /* needed for the dsecnd() timing function. */ 
#include <math.h> 
 
/* Program test1.c: multiply two matrices using explicit looping of elements. */ 
 
/*---------------------------------------------------------------------*/ 
/* Simple "naive" method to multiply two square matrices A and B 
 to generate matrix C. */ 
 
void myMult(int size, 
 float (* restrict A)[size], 
 float (* restrict B)[size], 
 float (* restrict C)[size]) 
{ 
 #pragma offload target(mic:MIC_DEV) in(A:length(size*size)) \ 
 in( B:length(size*size)) \ 
out(C:length(size*size)) 
 { 
 /* Initialize the C matrix with zeroes. */ 
 
 #pragma omp parallel for default(none) shared(C,size) 
 for(int i = 0; i < size; ++i) 
 for(int j = 0; j < size; ++j) 
 C[i][j] = 0.f; 
 
 /* Compute matrix multiplication. */ 
 
 #pragma omp parallel for default(none) shared(A,B,C,size) 
 for(int i = 0; i < size; ++i) 
 for(int k = 0; k < size; ++k) 
 for(int j = 0; j < size; ++j) 
 C[i][j] += A[i][k] * B[k][j]; 
 } 
} 
 
/*---------------------------------------------------------------------*/ 
/* Read input parameters; set-up dummy input data; multiply matrices using 
 the myMult() function above; average repeated runs. */ 
 
int main(int argc, char *argv[]) 
{ 
 if(argc != 4) { 
 fprintf(stderr,"Use: %s size nThreads nIter\n",argv[0]); 
 return -1; 
 } 
 
 int i,j,nt; 
 int size=atoi(argv[1]); 
 int nThreads=atoi(argv[2]); 
 int nIter=atoi(argv[3]); 
 
 omp_set_num_threads(nThreads); 
 
 /* when compiled in "mic-offload" mode, this memory gets allocated on host, 
 when compiled in "mic-native" mode, it gets allocated on mic. */ 
 
 float (*restrict A)[size] = malloc(sizeof(float)*size*size); 
 float (*restrict B)[size] = malloc(sizeof(float)*size*size); 
 float (*restrict C)[size] = malloc(sizeof(float)*size*size); 
 
 /* this first pragma is just to get the actual #threads used  11
 (sanity check). */ 
 
 #pragma omp parallel 
 { 
 nt = omp_get_num_threads(); 
 
 /* Fill the A and B arrays with dummy test data. */ 
 #pragma omp parallel for default(none) shared(A,B,size) private(i,j) 
 for(i = 0; i < size; ++i) { 
 for(j = 0; j < size; ++j) { 
 A[i][j] = (float)i + j; 
 B[i][j] = (float)i - j; 
 } 
 } 
 } 
 
 /* warm up run to overcome setup overhead in benchmark runs below. */ 
 
 myMult(size, A,B,C); 
 
 double aveTime,minTime=1e6,maxTime=0.; 
 
 /* run the matrix multiplication function nIter times and compute 
 average runtime. */ 
 
 for(i=0; i < nIter; i++) { 
 double startTime = dsecnd(); 
 myMult(size, A,B,C); 
 double endTime = dsecnd(); 
 double runtime = endTime-startTime; 
 maxTime=(maxTime > runtime)?maxTime:runtime; 
 minTime=(minTime < runtime)?minTime:runtime; 
 aveTime += runtime; 
 } 
 aveTime /= nIter; 
 
 printf("%s nThreads %d matrix %d maxRT %g minRT %g aveRT %g GFlop/s %g\n", 
 argv[0],nt,size,maxTime,minTime,aveTime, 2e-9*size*size*size/aveTime); 
 
 free(A); 
 free(B); 
 free(C); 
 
 return 0; 
} 
