package cuda;

import jcuda.driver.*;

public class CudaCritterSelector {
  private DistanceCalc dc;
   Reduction rd;
  
  public CudaCritterSelector() {
    JCudaDriver.cuInit(0);
    CUcontext pctx = new CUcontext();
    CUdevice dev = new CUdevice();
    JCudaDriver.cuDeviceGet(dev, 0);
    JCudaDriver.cuCtxCreate(pctx, 0, dev);
    try {
      rd = new Reduction("reduction.cu");
      dc = new DistanceCalc("CUDA_Distance.cu");
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
  
  public int[] selectCritters(int critters[], int towers[], int range) {
    CUdeviceptr deviceOutput = new CUdeviceptr();
    dc.calcDist(critters, towers, range, deviceOutput);
    return rd.reduction(deviceOutput, critters.length / 2, towers.length / 2, range);
  }
  
  
  public static void main(String[] args) throws Exception {
    int critterNo = 100000;
    int hostInput[] = new int[critterNo * 2];
    for(int i = 0; i < critterNo; i++)
    {
       hostInput[i * 2] = (int)Math.floor(Math.random() * 200);
       hostInput[i * 2 + 1] = (int)Math.floor(Math.random() * 200);
    }
    
    int towerNo = 6;
    int hostQuery[] = new int[towerNo * 2];
    for (int i = 0; i < towerNo; i++) {
      for (int j = 0; j < 2; j++) {
        hostQuery[i * 2] = (int)Math.floor(Math.random() * 200);
        hostQuery[i * 2 + 1] = (int)Math.floor(Math.random() * 200);
      }
    }
    long start, end;
    
    start = System.currentTimeMillis();
    System.out.print("CPU: ");
    for (int k = 0; k < 60; k ++ ) {
      int min[] = new int[towerNo];
      for (int i = 0; i < critterNo; i ++){
        for (int j = 0; j < towerNo; j ++) {
          int d = (int)(Math.pow((hostInput[i*2] - hostQuery[j*2]), 2)) + 
          (int)(Math.pow((hostInput[i*2 +1] - hostQuery[j*2 +1]), 2));
          if (i == 0)
            min[j] = d;
          else
            min[j] = Math.min(min[j], d);
        }
      }
    }
    end   = System.currentTimeMillis();
    System.out.println("finished in " + (end - start) + "ms");
    
    System.out.print("GPU: ");
    CudaCritterSelector c = new CudaCritterSelector();
    start = System.currentTimeMillis();
    //int result[] =
    for (int k = 0; k < 60; k ++ )
      c.selectCritters(hostInput, hostQuery, 2);
    end   = System.currentTimeMillis();
    System.out.println("finished in " + (end - start) + "ms");
    
//    for (int i:result)
//      System.out.println(i);

    
  }
}
