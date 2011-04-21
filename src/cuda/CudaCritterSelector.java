package cuda;

import jcuda.driver.*;

public class cudaCritterSelector {
  private DistanceCalc dc;
   Reduction rd;
  
  public cudaCritterSelector() {
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
  
  public int[] selectCritters(int critters[][], int towers[][], int range) {
    CUdeviceptr deviceOutput = new CUdeviceptr();
    dc.calcDist(critters, towers, 512, deviceOutput);
    return rd.reduction(deviceOutput, critters.length, towers.length, range);
  }
  
  
  public static void main(String[] args) throws Exception {
    
    int hostInput[][] = new int[2000][2];
    for(int i = 0; i < 2000; i++)
    {
        for (int j=0; j<2; j++)
        {
            hostInput[i][j] = (int)(Math.random() * 100);
        }
    }
    
    int towerNo = 20;
    
    int hostQuery[][] = new int[towerNo][2];
    for (int i = 0; i < towerNo; i++) {
      for (int j = 0; j < 2; j++) {
        hostQuery[i][j] = (i + j * j + 1);
      }
    }
    cudaCritterSelector c = new cudaCritterSelector();
    for (int i : c.selectCritters(hostInput, hostQuery, 10))
      System.out.println(i);

  }
}
