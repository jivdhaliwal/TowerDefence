package cuda;

import jcuda.*;
import jcuda.driver.*;

public class Reduction extends cudaKernel {
  public Reduction(String filename) throws Exception {
    super(filename);
  }
  
  public int[] reduction(CUdeviceptr deviceInput, int size, int towers, int range) {    
    int threadNum = Math.min(size, 512);
    int blockNum  = (int) Math.ceil((double)size / 512); 
    
    CUdeviceptr deviceOutput = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(deviceOutput, towers * blockNum * 2 * Sizeof.INT);
    
    launchKernel(deviceInput, deviceOutput, size, threadNum, blockNum, towers);
    
    ////////////
    while (blockNum > 1) {
      threadNum = Math.min(blockNum, 512);
      
      size = blockNum;
      blockNum = (int) Math.ceil((double)size / 512);
      
      JCudaDriver.cuMemAlloc(deviceInput, towers * size * 2 * Sizeof.INT);
      JCudaDriver.cuMemcpyDtoD(deviceInput, deviceOutput, towers * size * 2 * Sizeof.INT);
      
      
      JCudaDriver.cuMemFree(deviceOutput);
      JCudaDriver.cuMemAlloc(deviceOutput, towers * blockNum * 2 * Sizeof.INT);
      
      launchKernel(deviceInput, deviceOutput, size, threadNum, blockNum, towers);
      
    }
    ////////////
    
    int hostOutput[] = new int[towers * blockNum * 2];
    JCudaDriver.cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, towers * blockNum * 2 * Sizeof.INT);
    JCudaDriver.cuMemFree(deviceOutput);
    
    int result[] = new int[towers];
    for (int i = 0; i < towers; i ++) {
      if (hostOutput[i * 2] <= range)
        result[i] = hostOutput[i * 2 + 1];
      else
        result[i] = -1;
    }
    return result;
    
  }
  
  private void launchKernel(CUdeviceptr deviceInput, CUdeviceptr deviceOutput, int n, int threadNum, int blockNumX, int blockNumY) {
    CUfunction function = new CUfunction();
    JCudaDriver.cuModuleGetFunction(function, getModule(), "reduce");
    
    Pointer dIn  = Pointer.to(deviceInput);
    Pointer dOut = Pointer.to(deviceOutput);
    Pointer pN   = Pointer.to(new int[]{n});
    
    int offset = 0;
    offset = JCudaDriver.align(offset, Sizeof.POINTER);
    JCudaDriver.cuParamSetv(function, offset, dIn, Sizeof.POINTER);
    offset += Sizeof.POINTER;
    
    offset = JCudaDriver.align(offset, Sizeof.POINTER);
    JCudaDriver.cuParamSetv(function, offset, dOut, Sizeof.POINTER);
    offset += Sizeof.POINTER;

    offset = JCudaDriver.align(offset, Sizeof.INT);
    JCudaDriver.cuParamSetv(function, offset, pN, Sizeof.INT);
    offset += Sizeof.INT;
   
    JCudaDriver.cuParamSetSize(function, offset);
    JCudaDriver.cuFuncSetBlockShape(function, threadNum, 1, 1);
    JCudaDriver.cuFuncSetSharedSize(function, threadNum * 2 * Sizeof.INT);
 
    
    JCudaDriver.cuLaunchGrid(function, blockNumX, blockNumY);
    JCudaDriver.cuCtxSynchronize();
    
    JCudaDriver.cuMemFreeHost(dIn);
    JCudaDriver.cuMemFreeHost(dOut);
    JCudaDriver.cuMemFreeHost(pN);
    
    JCudaDriver.cuMemFree(deviceInput);
  }
  
  
}
