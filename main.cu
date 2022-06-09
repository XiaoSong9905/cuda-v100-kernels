#include <cstdio>
#include "cub/cub.cuh"

#define cudaCheckError() {                                          \
  cudaError_t e=cudaGetLastError();                                  \
  if(e!=cudaSuccess) {                                               \
  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
  exit(0); \
  }                                                                  \
}

void RunTestCub(char* label, int N, int REPEAT, int* src, int checksum) {
  int *in, *out;
  cudaEvent_t start,stop;
  
  cudaMalloc(&in,sizeof(int)*N);
  cudaMalloc(&out,sizeof(int)*1024);  //only stable version needs multiple elements, all others only need 1
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaCheckError();

  cudaMemcpy(in,src,N*sizeof(int),cudaMemcpyHostToDevice);

  size_t temp_storage_bytes;
  void* temp_storage=NULL;
  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, in, out, N, cub::Sum(), 0);
  cudaMalloc(&temp_storage,temp_storage_bytes);

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaEventRecord(start);

  for(int i=0;i<REPEAT;i++) {
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, in, out, N, cub::Sum(), 0);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  cudaCheckError();

  float time_ms;
  cudaEventElapsedTime(&time_ms,start,stop);
  float time_s=time_ms/(float)1e3;

  float GB=(float)N*sizeof(int)*REPEAT;
  float GBs=GB/time_s/(float)1e9;

  int sum;
  cudaMemcpy(&sum,out,sizeof(int),cudaMemcpyDeviceToHost);
  cudaCheckError();

  char *valid;
  if(sum==checksum) 
    valid="CORRECT";
  else
    valid="INCORRECT";

  printf("%s: %s, Time: %f s, GB/s: %f\n", label, valid, time_s, GBs); 
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(in);
  cudaFree(out);
  cudaFree(temp_storage);
  cudaCheckError();
}


int main(int argc, char** argv)
{
  if(argc!=3) {
    printf("Usage: ./reduce num_elems repeat\n");
    exit(0);
  }
  int NUM_ELEMS=atoi(argv[1]);
  int REPEAT=atoi(argv[2]);

  printf("NUM_ELEMS: %d, REPEAT: %d\n", NUM_ELEMS, REPEAT);

  int* vals=(int*)malloc(NUM_ELEMS*sizeof(int));
  int checksum =0;
  for(int i=0;i<NUM_ELEMS;i++) {
    vals[i]=rand()%4;
    checksum+=vals[i];
  }

  RunTestCub("device_reduce_cub",NUM_ELEMS,REPEAT,vals,checksum);
  
  free(vals);

}
