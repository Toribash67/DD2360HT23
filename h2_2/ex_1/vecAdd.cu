//%%writefile vecAdd.cu
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>

#define DataType double
#define N 64
#define TPB 64

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < len){
    out[tid] = in1[tid] + in2[tid];
  }
}

//@@ Insert code to implement timer start
DataType cpuSecond(){
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

//@@ Insert code to implement timer stop

int main(int argc, char **argv) {
  cudaProfilerStart();
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *)malloc(sizeof(DataType)*inputLength);
  hostInput2 = (DataType *)malloc(sizeof(DataType)*inputLength);
  hostOutput = (DataType *)malloc(sizeof(DataType)*inputLength);
  resultRef = (DataType *)malloc(sizeof(DataType)*inputLength);
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  int i;
  for(i=0; i<inputLength; i++){
    hostInput1[i] = rand()%100;
    hostInput2[i] = rand()%100;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, sizeof(DataType)*inputLength);
  cudaMalloc(&deviceInput2, sizeof(DataType)*inputLength); 
  cudaMalloc(&deviceOutput, sizeof(DataType)*inputLength); 

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, sizeof(DataType)*inputLength, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, sizeof(DataType)*inputLength, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutput, hostOutput, sizeof(DataType)*inputLength, cudaMemcpyHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here
  //@@ Launch the GPU Kernel here
  DataType iStart = cpuSecond();
  vecAdd<<<(inputLength+TPB-1)/TPB, TPB>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  DataType iElaps = cpuSecond() - iStart;

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, sizeof(DataType)*inputLength, cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  bool equal = true;
  for(i=0;i<inputLength;i++){
    if(hostOutput[i] != resultRef[i]){
      equal = false;
      break;
    }
  }
  if(equal){
    printf("check passed! runtime:%f\n", iElaps);
  }
  else{
    printf("check error!\n");
  }

  //@@ Free the GPU memory hereclear
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);
  cudaDeviceReset();
  cudaProfilerStop();
  return 0;
}
