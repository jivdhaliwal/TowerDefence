extern "C"

__global__ void calcDistance(int* globalInputData, int n, int range, int* globalQueryData, int* globalOutputData) {
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = blockIdx.y;
  unsigned int offset = x + y * n;

  if (x < n) {
  
  
    int dX = globalInputData[x * 2] - globalQueryData[y * 2];
    int dY = globalInputData[x * 2 + 1] - globalQueryData[y * 2 + 1];
    int distance = dX * dX + dY * dY;
    globalOutputData[offset * 2] = distance;
    globalOutputData[offset * 2 + 1] = x;
  }
  
}