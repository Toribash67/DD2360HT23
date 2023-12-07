%%writefile histogram.cu

#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define TPB 512
#define nstreams 20

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 //unsigned int num_elements,
                                 unsigned int num_bins) {   //size 4096

//@@ Insert code below to compute histogram of input using shared memory and atomics
__shared__ unsigned int histogram[TPB];
int tid = threadIdx.x + blockIdx.x * blockDim.x;
const int s_id = threadIdx.x;

if (threadIdx.x < TPB){
  histogram[s_id] = 0;
}
__syncthreads();

while (tid < num_bins){
  histogram[s_id] = input[tid];
  atomicAdd(&bins[histogram[s_id]], 1);
  tid += blockDim.x * gridDim.x;
}
__syncthreads();

}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
int saturate = 127;
int i;
for (i=0; i<num_bins; i++){
  if (bins[i] > saturate){
    bins[i] = 0;
  }
}

}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  cudaHostAlloc((void **)&hostInput, inputLength*sizeof(unsigned int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&hostBins, NUM_BINS*sizeof(unsigned int), cudaHostAllocDefault);
  resultRef = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  int i;
  for (i=0; i<inputLength; i++){
    hostInput[i] = rand()%(NUM_BINS);
  }

  //@@ Insert code below to create reference result in CPU
  int j;
  for(j=0; j<inputLength; j++){
    resultRef[hostInput[j]]++;
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength*sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS*sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here, use multiple streams to optimize
  //@@ Insert code to initialize GPU results
  //@@ Initialize the grid and block dimensions here
  //@@ Launch the GPU Kernel here
  int k;
  const int streamSize = inputLength/nstreams;
  const int streamBytes = streamSize * sizeof(int);
  cudaStream_t stream[nstreams];
  for (k=0; k<nstreams; k++){
    cudaStreamCreate(&stream[k]);
  }

  for (k=0; k<nstreams; k++){
    int offset = k * streamSize;
    cudaMemcpyAsync(&deviceInput[offset], &hostInput[offset], streamBytes, cudaMemcpyHostToDevice, stream[k]);
    histogram_kernel<<<(streamSize+TPB-1)/TPB, TPB, 0, stream[k]>>>(&deviceInput[offset], deviceBins, streamSize);
   // cudaMemcpyAsync(&hostInput[offset], &deviceInput[offset], streamBytes, cudaMemcpyDeviceToHost, stream[k]);
  }

  for (k=0; k<nstreams; k++){
    cudaStreamDestroy(stream[k]);
  }

  //@@ Remaining part runned by default kernel
  const int remainSize = inputLength - streamSize * nstreams;
  const int remainBytes = remainSize * sizeof(int);
  cudaMemcpy(&deviceInput[inputLength-remainSize], &hostInput[inputLength-remainSize], remainBytes, cudaMemcpyHostToDevice);
  histogram_kernel<<<(remainSize+TPB-1)/TPB, TPB>>>(&deviceInput[inputLength-remainSize], deviceBins, remainSize); 

  //@@ Initialize the second grid and block dimensions here
  //@@ Launch the second GPU Kernel here
  convert_kernel<<<(NUM_BINS+TPB-1)/TPB, TPB>>>(deviceBins, NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  bool equal = true;
  for(i=0; i<NUM_BINS; i++){
    if(hostBins[i] != resultRef[i]){
      equal = false;
      break;
    }
  }
  if(equal == true){
    printf("check passed!/n");
    //print the histogram's content
    for(i=0; i<64; i++){
      for(j=0; j<64; j++){
        printf("%d ",hostBins[64*i+j]);
      }
      printf("\n");
    }
  }
  else {
    printf("check error!/n");
  }

  //@@ Free the GPU memory here
  cudaFree(deviceBins);
  cudaFree(deviceInput);

  //@@ Free the CPU memory here
  cudaFreeHost(hostBins);
  cudaFreeHost(hostInput);

  return 0;
}


