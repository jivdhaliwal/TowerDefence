package cuda;

import jcuda.*;
import jcuda.driver.*;

public class DistanceCalc extends cudaKernel {
  public DistanceCalc(String filename) throws Exception {
    super(filename);
  }
  
  public void calcDist(float critters[], float towers[], CUdeviceptr deviceOutput) {
   
    CUfunction function = new CUfunction();
    JCudaDriver.cuModuleGetFunction(function, getModule(), "calcDistance");

    int numThreads = critters.length / 2;
    int numTowers = towers.length / 2;
    
    int numBlocks = (int) Math.ceil((double)numThreads / 512);

    CUdeviceptr deviceInput = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(deviceInput, critters.length * Sizeof.FLOAT);
    JCudaDriver.cuMemcpyHtoD(deviceInput, Pointer.to(critters), critters.length * Sizeof.FLOAT);
    
    CUdeviceptr deviceQuery = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(deviceQuery, towers.length * Sizeof.FLOAT);
    JCudaDriver.cuMemcpyHtoD(deviceQuery, Pointer.to(towers), towers.length * Sizeof.FLOAT);
    
    JCudaDriver.cuMemAlloc(deviceOutput, numTowers * numThreads * 2 * Sizeof.FLOAT);
    
    JCudaDriver.cuFuncSetBlockShape(function, Math.min(512, numThreads), 1, 1);
    
    
    Pointer dIn = Pointer.to(deviceInput);
    Pointer dQu = Pointer.to(deviceQuery);
    Pointer dOut = Pointer.to(deviceOutput);
    
    Pointer pN    = Pointer.to(new int[]{numThreads});

    int offset = 0;
    offset = JCudaDriver.align(offset, Sizeof.POINTER);
    JCudaDriver.cuParamSetv(function, offset, dIn, Sizeof.POINTER);
    offset += Sizeof.POINTER;

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
    
    JCudaDriver.cuMemFreeHost(dIn);
    JCudaDriver.cuMemFreeHost(dOut);

    JCudaDriver.cuMemFreeHost(pN);
    JCudaDriver.cuMemFreeHost(dQu);
   
    JCudaDriver.cuMemFree(deviceInput);
    JCudaDriver.cuMemFree(deviceQuery);
    
  }    
//  public static void main(String[] args) {
//    JCudaDriver.cuInit(0);
//    DistanceCalc dc;
//    CUcontext pctx = new CUcontext();
//    CUdevice dev = new CUdevice();
//    JCudaDriver.cuDeviceGet(dev, 0);
//    JCudaDriver.cuCtxCreate(pctx, 0, dev);
//    try {
//      dc = new DistanceCalc("CUDA_Distance.cu");
//      CUdeviceptr deviceOutput = new CUdeviceptr();
//      
//      int hostInput[] = new int[6];
//      for(int i = 0; i < 3; i++)
//      {
//         hostInput[i * 2] = i;// (int)(Math.random() * 100);
//         hostInput[i * 2 + 1] = i;// (int)(Math.random() * 100);
//      }
//      for (int i : hostInput)
//        System.out.println(i);
//      int towerNo = 2;
//      
//      int hostQuery[] = new int[towerNo * 2];
//      for (int i = 0; i < towerNo; i++) {
//        for (int j = 0; j < 2; j++) {
//          hostQuery[i * 2] = i; //(int)(Math.random() * 100);
//          hostQuery[i * 2 + 1] = i; //(int)(Math.random() * 100);
//        }
//      }
//      
//      dc.calcDist(hostInput, hostQuery, 512, deviceOutput);
//      JCudaDriver.cuMemFree(deviceOutput);
//      
//    } catch (Exception ex) {
//      ex.printStackTrace();
//    }
//  }
  
}
