extern "C"
__global__ void reduce(int *g_idata, int *g_odata, int n) {
  extern __shared__ int sdata[];

  int blockSize = min(blockDim.x, n - blockDim.x * blockIdx.x);
  
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * n;
  sdata[tid] = g_idata[i];
  __syncthreads();
  
  for (unsigned int s = (blockSize + 1) / 2; s > 0; s -= 1) {
    if (tid < s) {
      sdata[tid] = (tid + s) < blockSize ? min(sdata[tid], sdata[tid + s]) : sdata[tid];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x + blockIdx.y * gridDim.x] = sdata[0]; //sdata[0];
}
