extern "C"
__global__ void reduce(int *g_idata, int *g_odata, int n) {
  extern __shared__ int sdata[];

  unsigned int blockSize = min(blockDim.x, n - blockDim.x * blockIdx.x);
  
  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * n;
  sdata[tid * 2]     = g_idata[i * 2];
  sdata[tid * 2 + 1] = g_idata[i * 2 + 1];
  __syncthreads();
  
  for (unsigned int s = (blockSize + 1) / 2; s > 0; s -= 1) {
    if (tid < s && (tid + s) < blockSize) {
      sdata[tid * 2]     = min(sdata[tid * 2], sdata[(tid + s) * 2]);
      sdata[tid * 2 + 1] = (sdata[tid * 2] < sdata[(tid + s) * 2]) ? sdata[tid * 2 + 1] : sdata[(tid + s) * 2 + 1];
    }
    __syncthreads();
  }

  int offset = blockIdx.x + blockIdx.y * gridDim.x;
  if (tid == 0) {
    g_odata[offset * 2]     = sdata[0]; 
    g_odata[offset * 2 + 1] = sdata[1]; 
  }
}
