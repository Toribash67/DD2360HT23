%%writefile vecAdd_2.cu
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double
#define TPB 32
#define nstreams 4

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int t_n = gridDim.x * blockDim.x;
  while (tid < len)
  {
    out[tid] = in1[tid] + in2[tid];
    tid += t_n;
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
  cudaHostAlloc((void **)&hostInput1, inputLength*sizeof(DataType), cudaHostAllocDefault);
  cudaHostAlloc((void **)&hostInput2, inputLength*sizeof(DataType), cudaHostAllocDefault);
  cudaHostAlloc((void **)&hostOutput, inputLength*sizeof(DataType), cudaHostAllocDefault);
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
  int S_seg = atoi(argv[2]);
  printf("the segment length is %d\n", S_seg);
  int circle1 = inputLength / (S_seg * nstreams);
  int circle2 = (inputLength - circle1 * S_seg * nstreams) / S_seg;
  int circle3 = inputLength - circle1 * S_seg * nstreams - circle2 * S_seg;
  int j;
  int c;
  const int streamSize = S_seg;
  const int streamBytes = streamSize * sizeof(double);
  cudaStream_t stream[nstreams];
  for (i=0; i<nstreams; i++){
    cudaStreamCreate(&stream[i]);
  }

  //@@ Initialize the 1D grid and block dimensions here
  //@@ Launch the GPU Kernel here
  DataType iStart = cpuSecond();
  for(i=0; i<circle1; i++){
    c = i * S_seg * nstreams;
    for(j=0; j<nstreams; j++){
      int offset = j * streamSize + c;
      cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], streamBytes, cudaMemcpyHostToDevice, stream[j]);
      cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], streamBytes, cudaMemcpyHostToDevice, stream[j]);
      vecAdd<<<((streamSize+TPB-1)/TPB), TPB, 0, stream[j]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], streamSize);
      cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], streamBytes, cudaMemcpyDeviceToHost, stream[j]);
      }
    cudaDeviceSynchronize();
  }
  for(i=0; i<circle2; i++){
      int offset = i * streamSize + circle1 * S_seg * nstreams;
      cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
      cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
      vecAdd<<<((streamSize+TPB-1)/TPB), TPB, 0, stream[i]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], streamSize);
      cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
      }
  int remainSize = circle3;
  int remainBytes = remainSize * sizeof(DataType);
  cudaMemcpy(&deviceInput1[inputLength-remainSize], &hostInput1[inputLength-remainSize], remainBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(&deviceInput2[inputLength-remainSize], &hostInput2[inputLength-remainSize], remainBytes, cudaMemcpyHostToDevice);
  vecAdd<<<(remainSize+TPB-1)/TPB, TPB>>>(&deviceInput1[inputLength-remainSize], &deviceInput2[inputLength-remainSize], &deviceOutput[inputLength-remainSize], remainSize);
  cudaMemcpy(&hostOutput[inputLength-remainSize], &deviceOutput[inputLength-remainSize], remainBytes, cudaMemcpyDeviceToHost);

  DataType iElaps = cpuSecond() - iStart;

  for (i=0; i<nstreams; i++){
    cudaStreamDestroy(stream[i]);
  }

  //@@ Copy the GPU memory back to the CPU here

  //@@ Insert code below to compare the output with the reference
  bool equal = false;
  int err;
  for (i=0; i<inputLength; i++){
    if(hostOutput[i] == resultRef[i]){
    equal = true;
  }
  else{
    equal = false;
    err = i;
    break;
  }
  }

  if(equal == true){
    printf("check passed! runtime: %f", iElaps);
  }
  else{
    printf("check error! error: %d", err);
  }


  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);

  return 0;
}
