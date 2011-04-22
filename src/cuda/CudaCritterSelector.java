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
  
  public int[] selectCritters(float critters[], float towers[], int range) {
    CUdeviceptr deviceOutput = new CUdeviceptr();
    dc.calcDist(critters, towers, deviceOutput);
    return rd.reduction(deviceOutput, critters.length / 2, towers.length / 2, range);
  }
  
  
  public static void main(String[] args) throws Exception {
    int critterNo = 2000;
    float hostInput[] = new float[critterNo * 2];
    for(int i = 0; i < critterNo; i++)
    {
       hostInput[i * 2] = (float)Math.floor(Math.random() * 200);
       hostInput[i * 2 + 1] = (float)Math.floor(Math.random() * 200);
    }
    
    int towerNo = 5;
    float hostQuery[] = new float[towerNo * 2];
    for (int i = 0; i < towerNo; i++) {
      for (int j = 0; j < 2; j++) {
        hostQuery[i * 2] = (float)Math.floor(Math.random() * 200);
        hostQuery[i * 2 + 1] = (float)Math.floor(Math.random() * 200);
      }
    }
 
    
    CudaCritterSelector c = new CudaCritterSelector();
    long start, end;
    start = System.currentTimeMillis();
    int result[] = c.selectCritters(hostInput, hostQuery, 1000);
    end   = System.currentTimeMillis();
    System.out.println("finished in " + (end - start) + "ms");
    
    for (int i:result)
      System.out.println(i);

    
  }
}
