package cuda;

import jcuda.*;
import jcuda.driver.*;

public class DistanceCalc extends cudaKernel {
  public DistanceCalc(String filename) throws Exception {
    super(filename);
  }
  
  public void calcDist(int critters[][], int towers[][], int threadPerBlock, CUdeviceptr deviceOutput) {
    CUfunction function = new CUfunction();
    JCudaDriver.cuModuleGetFunction(function, getModule(), "calcDistance");

    int numThreads = critters.length;
    
    int numBlocks = (int) Math.ceil((double)numThreads / threadPerBlock);

    int numTowers = towers.length;
    int size = critters[0].length;

    CUdeviceptr hostDevicePointers[] = new CUdeviceptr[numThreads];
    for(int i = 0; i < numThreads; i++) {
        hostDevicePointers[i] = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(hostDevicePointers[i], size * Sizeof.INT);
        
        JCudaDriver.cuMemcpyHtoD(hostDevicePointers[i],
            Pointer.to(critters[i]), size * Sizeof.INT);
    }
    
    CUdeviceptr queryHostDevicePointers[] = new CUdeviceptr[numTowers];
    for (int i = 0; i < numTowers; i++) {
      queryHostDevicePointers[i] = new CUdeviceptr();
      JCudaDriver.cuMemAlloc(queryHostDevicePointers[i], size * Sizeof.INT);
      
      JCudaDriver.cuMemcpyHtoD(queryHostDevicePointers[i],
          Pointer.to(towers[i]), size * Sizeof.INT);
    }
    
    CUdeviceptr deviceInput = new CUdeviceptr();
    CUdeviceptr deviceQuery = new CUdeviceptr();
    
    JCudaDriver.cuMemAlloc(deviceInput, numThreads * Sizeof.POINTER);
    JCudaDriver.cuMemcpyHtoD(deviceInput, Pointer.to(hostDevicePointers),
        numThreads * Sizeof.POINTER);

    
    JCudaDriver.cuMemAlloc(deviceQuery, numTowers * Sizeof.POINTER);
    JCudaDriver.cuMemcpyHtoD(deviceQuery, Pointer.to(queryHostDevicePointers), numTowers * Sizeof.POINTER);
    
    //CUdeviceptr deviceOutput = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(deviceOutput, numTowers * numThreads * 2 * Sizeof.INT);
    
    JCudaDriver.cuFuncSetBlockShape(function, Math.min(threadPerBlock, numThreads), 1, 1);
    
    Pointer dIn = Pointer.to(deviceInput);
    Pointer dQu = Pointer.to(deviceQuery);
    Pointer dOut = Pointer.to(deviceOutput);
    
    Pointer pSize = Pointer.to(new int[]{size});
    Pointer pN    = Pointer.to(new int[]{numThreads});

    int offset = 0;
    offset = JCudaDriver.align(offset, Sizeof.POINTER);
    JCudaDriver.cuParamSetv(function, offset, dIn, Sizeof.POINTER);
    offset += Sizeof.POINTER;

    offset = JCudaDriver.align(offset, Sizeof.INT);
    JCudaDriver.cuParamSetv(function, offset, pSize, Sizeof.INT);
    offset += Sizeof.INT;
    
    offset = JCudaDriver.align(offset, Sizeof.INT);
    JCudaDriver.cuParamSetv(function, offset, pN, Sizeof.INT);
    offset += Sizeof.INT;
    
    offset = JCudaDriver.align(offset, Sizeof.POINTER);
    JCudaDriver.cuParamSetv(function, offset, dQu, Sizeof.POINTER);
    offset += Sizeof.POINTER;
    
    offset = JCudaDriver.align(offset, Sizeof.POINTER);
    JCudaDriver.cuParamSetv(function, offset, dOut, Sizeof.POINTER);
    offset += Sizeof.POINTER;

    JCudaDriver.cuParamSetSize(function, offset);
    
    JCudaDriver.cuLaunchGrid(function, numBlocks, numTowers);
    JCudaDriver.cuCtxSynchronize();
    
    for(int i = 0; i < numThreads; i++) {
        JCudaDriver.cuMemFree(hostDevicePointers[i]);
    }
    for (int i = 0; i < numTowers; i++) {
      JCudaDriver.cuMemFree(queryHostDevicePointers[i]);
    }
    JCudaDriver.cuMemFreeHost(dIn);
    JCudaDriver.cuMemFreeHost(dOut);
    JCudaDriver.cuMemFreeHost(pSize);
    JCudaDriver.cuMemFreeHost(pN);
    JCudaDriver.cuMemFreeHost(dQu);
   
    JCudaDriver.cuMemFree(deviceInput);
    JCudaDriver.cuMemFree(deviceQuery);

    
  }    
  
}
