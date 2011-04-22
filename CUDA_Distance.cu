extern "C"

__global__ void calcDistance(float* globalInputData, int n, float* globalQueryData, float* globalOutputData) {
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = blockIdx.y;
  unsigned int offset = x + y * n;

  if (x < n) {
    // int distance = 0;
    // 
    // distance = 
    // for (int i = 0; i < size; i++) {
    //   int d = globalInputData[x][i] - globalQueryData[y][i];
    //   distance += d * d; 
    // }
  
    float dX = globalInputData[x * 2] - globalQueryData[y * 2];
    float dY = globalInputData[x * 2 + 1] - globalQueryData[y * 2 + 1];
    globalOutputData[offset * 2] = dX * dX + dY * dY;
    globalOutputData[offset * 2 + 1] = x;
  }
  
}