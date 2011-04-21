extern "C"

__global__ void calcDistance(int** globalInputData, int size, int n, int** globalQueryData, int* globalOutputData) {  
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = blockIdx.y;
  int offset = x + y * n;

  if (x < n) {
    int distance = 0;
    for (int i = 0; i < size; i++) {
      int d = globalInputData[x][i] - globalQueryData[y][i];
      distance += d * d; 
    }
  
    globalOutputData[offset * 2] = distance;
    globalOutputData[offset * 2 + 1] = x;
  }
  
}