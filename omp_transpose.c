/******************************************************************************
   Example 14   :        Omp_MatrixTranspose.c

   Objective    : Write an OpenMP Program for Transpose of a Matrix
                  and measure the performance
                  This example demonstrates the use of
                  PARALLEL Directive and Private clause

   Input        : Size of matrices (numofrows and noofcols )

   Output       : Each thread transposes assaigned Row and finally
                  master prints the final resultant matrix

******************************************************************************/

#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

/* Main Program */

main()
{
	int             NoofRows, NoofCols, i, j;
	float         **Matrix, **Trans, **Checkoutput, flops;

	struct timeval  TimeValue_Start;
	struct timezone TimeZone_Start;

	struct timeval  TimeValue_Final;
	struct timezone TimeZone_Final;
	long            time_start, time_end, time_overhead;

	printf("Read The Matrix Size Noofrows And Colums Of Matrix \n");
	scanf("%d%d", &NoofRows, &NoofCols);

	if (NoofRows <= 0 || NoofCols <= 0) {
		printf("The NoofRows And NoofCols Should Be Of Positive Sign\n");
		exit(1);
	}
	/* Matrix Elements */

	Matrix = (float **) malloc(sizeof(float) * NoofRows);
	for (i = 0; i < NoofRows; i++) {
		Matrix[i] = (float *) malloc(sizeof(float) * NoofCols);
		for (j = 0; j < NoofCols; j++)
			Matrix[i][j] = (i * j) * 5 + i;
	}

	/* Dynamic Memory Allocation */

	Trans = (float **) malloc(sizeof(float) * NoofCols);
	Checkoutput = (float **) malloc(sizeof(float) * NoofCols);

	/* Initializing The Output Matrices Elements As Zero */

	for (i = 0; i < NoofCols; i++) {
		Checkoutput[i] = (float *) malloc(sizeof(float) * NoofRows);
		Trans[i] = (float *) malloc(sizeof(float) * NoofRows);
		for (j = 0; j < NoofRows; j++) {
			Checkoutput[i][j] = 0.0;
			Trans[i][j] = 0.0;
		}
	}

	gettimeofday(&TimeValue_Start, &TimeZone_Start);

	/* OpenMP Parallel For Directive */

#pragma omp parallel for private(j)
	for (i = 0; i < NoofRows; i = i + 1)
		for (j = 0; j < NoofCols; j = j + 1)
			Trans[j][i] = Matrix[i][j];

	gettimeofday(&TimeValue_Final, &TimeZone_Final);

	time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
	time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
	time_overhead = time_end - time_start;


	/* Serial Computation */

	for (i = 0; i < NoofRows; i = i + 1)
		for (j = 0; j < NoofCols; j = j + 1)
			Checkoutput[j][i] = Matrix[i][j];

	for (i = 0; i < NoofCols; i = i + 1)
		for (j = 0; j < NoofRows; j = j + 1)
			if (Checkoutput[i][j] == Trans[i][j])
				continue;
			else {
				printf("There Is A Difference From Serial And Parallel Calculation \n");
				exit(1);
			}

	printf("\nTime Overhead = %ld\n", time_overhead);

	printf("The Input Matrix Is \n");

	for (i = 0; i < NoofRows; i++) {
		for (j = 0; j < NoofCols; j++)
			printf("%f \t", Matrix[i][j]);
		printf("\n");
	}

	printf("\nThe Transpose Matrix Is \n");
	for (i = 0; i < NoofCols; i = i + 1) {
		for (j = 0; j < NoofRows; j = j + 1)
			printf("%f \t", Trans[i][j]);
		printf("\n");
	}

	/* Calculation Of Flops */

	flops = (float) 2 *NoofRows * NoofCols / (float) time_overhead;
	printf("\nNoofRows=%d\t NoofCols=%d \t Flops=%fMFlops\n", NoofRows, NoofCols, flops);

	/* Freeing Allocated Memory */

	free(Matrix);
	free(Checkoutput);
	free(Trans);

}
