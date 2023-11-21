
#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <math.h>
#include<ctime>

#define DataType double
#define TPBD 16

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numArows,
                      int numAcolumns, int numBrows, int numBcolumns){
  //@@ Insert code to implement matrix multiplication here
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  if(x>=numArows || y>= numBcolumns){
    return;
  }
  int numCcolumns = numBcolumns;

  DataType sum = 0;
  for (int k = 0; k < numAcolumns; ++k)
  {
    sum += A[k + x * numAcolumns] * B[y + k*numBcolumns];
  }
  C[y + x*numCcolumns] = sum;
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numArows;    // number of rows in the matrix A
  int numAcolumns; // number of columns in the matrix A
  int numBrows;    // number of rows in the matrix B
  int numBcolumns; // number of columns in the matrix B
  int numCrows;
  int numCcolumns;

  //@@ Insert code below to read in numArows, numAcolumns, numBcolumns from args
  if(argc != 5){
    printf("Please specify numArows, numAcolumns, numBcolumns from args!\n");
    return 0;
  }
  numArows = atoi(argv[1]);
  numAcolumns = atoi(argv[2]);
  numBrows = atoi(argv[3]);
  numBcolumns = atoi(argv[4]);
  if(numArows == 0 || numAcolumns == 0 || numBrows == 0 || numBcolumns == 0){
    printf("Please have all arguments be positive integers!\n");
    return 0;
  }
  if(numAcolumns != numBrows){
    printf("Dimension mismatch: numAcolumns != numBrows! (%d != %d)\n",numAcolumns,numBrows);
    return 0;
  }

  numCrows = numArows;
  numCcolumns = numBcolumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numArows, numAcolumns, numBrows, numBcolumns, numCrows, numCcolumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType*)malloc(numArows*numAcolumns*sizeof(DataType));
  hostB = (DataType*)malloc(numBrows*numBcolumns*sizeof(DataType));
  hostC = (DataType*)malloc(numCrows*numCcolumns*sizeof(DataType));
  resultRef = (DataType*)malloc(numCrows*numCcolumns*sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU

  DataType lower_bound = 0;
  DataType upper_bound = 10;
  std::random_device r;
  std::uniform_real_distribution<DataType> unif(lower_bound,upper_bound);
  std::default_random_engine re(1);

  for (int i = 0; i < numArows*numAcolumns; ++i)
  {
    hostA[i] = unif(re);
  }
  for (int i = 0; i < numBrows*numBcolumns; ++i)
  {
    hostB[i] = unif(re);
  }


  std::clock_t start;
  start = std::clock();/*
  for (int i = 0; i < numCcolumns; ++i)
  {
    for (int j = 0; j < numCrows; ++j)
    {
      for (int k = 0; k < numAcolumns; ++k)
      {
        resultRef[i + j*numCcolumns] += hostA[k+j*numAcolumns]*hostB[i+k*numBcolumns];
      }
    }
  }*/
  printf("CPU matmul: %f ms\n",(std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000));
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numArows*numAcolumns*sizeof(DataType));
  cudaMalloc(&deviceB, numBrows*numBcolumns*sizeof(DataType));
  cudaMalloc(&deviceC, numCrows*numCcolumns*sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  start = std::clock();
  cudaMemcpy(deviceA, hostA, numArows*numAcolumns*sizeof(DataType),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBrows*numBcolumns*sizeof(DataType),cudaMemcpyHostToDevice);
  printf("data copy host to device: %f ms\n",(std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000));

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(TPBD,TPBD);
  dim3 dimGrid((numCcolumns+TPBD-1)/TPBD,(numCrows+TPBD-1)/TPBD);

  //@@ Launch the GPU Kernel here
  start = std::clock();
  gemm<<<dimGrid, dimBlock >>>(deviceA,deviceB,deviceC,numArows,numAcolumns,numBrows,numBcolumns);
  cudaDeviceSynchronize();
  printf("kernel time: %f ms\n",(std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000));

  //@@ Copy the GPU memory back to the CPU here
  start = std::clock();
  cudaMemcpy(hostC, deviceC, numCrows*numCcolumns*sizeof(DataType), cudaMemcpyDeviceToHost);
  printf("data copy device to host: %f ms\n",(std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000));
  

  //@@ Insert code below to compare the output with the reference
  bool diff = false;
  for (int i = 0; i < numCrows*numCcolumns; ++i)
  {
    if(fabs(hostC[i]-resultRef[i])>0.1){
      //printf("%d ",i);
      diff = true;
    }
  }
  if(!diff){
    printf("results equal!\n");
  }else{
    printf("\nresults not equal!\n");
  }

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);
  
  return 0;
}
