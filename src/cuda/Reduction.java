package cuda;
import java.io.*;

import jcuda.*;
import jcuda.driver.*;

public class Reduction {
  private CUmodule module;
  public Reduction(String filename) throws Exception {
    String cubinFileName = prepareCubinFile(filename);
    JCudaDriver.cuInit(0);
    CUcontext pctx = new CUcontext();
    CUdevice dev = new CUdevice();
    JCudaDriver.cuDeviceGet(dev, 0);
    JCudaDriver.cuCtxCreate(pctx, 0, dev);


    // Load the CUBIN file.
    module = new CUmodule();
    JCudaDriver.cuModuleLoad(module, cubinFileName);
  }
  private static String prepareCubinFile(String cuFileName) throws IOException
  {
      int endIndex = cuFileName.lastIndexOf('.');
      if (endIndex == -1)
      {
          endIndex = cuFileName.length()-1;
      }
      String cubinFileName = cuFileName.substring(0, endIndex+1)+"cubin";
      File cubinFile = new File(cubinFileName);
      if (cubinFile.exists())
      {
          //return cubinFileName;
      }

      File cuFile = new File(cuFileName);
      if (!cuFile.exists())
      {
          throw new IOException("Input file not found: "+cuFileName);
      }
      String modelString = "-m"+System.getProperty("sun.arch.data.model");
      String command =
          "/usr/local/cuda/bin/nvcc " + modelString + " -arch sm_11 -cubin "+
          cuFile.getPath()+" -o "+cubinFileName;

      System.out.println("Executing\n"+command);
      Process process = Runtime.getRuntime().exec(command);

      String errorMessage = new String(toByteArray(process.getErrorStream()));
      String outputMessage = new String(toByteArray(process.getInputStream()));
      int exitValue = 0;
      try
      {
          exitValue = process.waitFor();
      }
      catch (InterruptedException e)
      {
          Thread.currentThread().interrupt();
          throw new IOException("Interrupted while waiting for nvcc output", e);
      }

      System.out.println("nvcc process exitValue "+exitValue);
      if (exitValue != 0)
      {
          System.out.println("errorMessage:\n"+errorMessage);
          System.out.println("outputMessage:\n"+outputMessage);
          throw new IOException("Could not create .cubin file: "+errorMessage);
      }
      return cubinFileName;
  }

  /**
   * Fully reads the given InputStream and returns it as a byte array.
   *
   * @param inputStream The input stream to read
   * @return The byte array containing the data from the input stream
   * @throws IOException If an I/O error occurs
   */
  private static byte[] toByteArray(InputStream inputStream) throws IOException
  {
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      byte buffer[] = new byte[8192];
      while (true)
      {
          int read = inputStream.read(buffer);
          if (read == -1)
          {
              break;
          }
          baos.write(buffer, 0, read);
      }
      return baos.toByteArray();
  }
  
  
  
  public void reduction(int[] input, int towers) {

    
    int size = input.length / 2 / towers;
    
    int threadNum = Math.min(size, 512);
    int blockNum  = (int) Math.ceil((double)size / 512); 
    
    CUdeviceptr deviceInput = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(deviceInput, size * towers * 2 * Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(deviceInput,
      Pointer.to(input), size * towers * 2 * Sizeof.INT);
    
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
    
    for (int result : hostOutput)
      System.out.println(result);
  }
  
  private void launchKernel(CUdeviceptr deviceInput, CUdeviceptr deviceOutput, int n, int threadNum, int blockNumX, int blockNumY) {
    CUfunction function = new CUfunction();
    JCudaDriver.cuModuleGetFunction(function, module, "reduce");
    
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
  
  public static void main(String[] args) throws Exception {
//    int threads = 66;
//    int inputs[] = new int[threads * 2];
//    for (int i = 0; i < threads; i ++)
//      inputs[i * 2] = threads - i; //(int)(Math.random() * 100);
//   
//    for (int i = 0; i < threads; i ++)
//      inputs[i * 2 + 1] = i % (threads / 2);
//   
    int inputs[] = new int[]{0, 0, 2, 1, 8, 2, 18, 3, 32, 4, 50, 5, 72, 6, 98, 7, 128, 8, 162, 9, 200, 10, 2, 0, 0, 1, 2, 2, 8, 3, 18, 4, 32, 5, 50, 6, 72, 7, 98, 8, 128, 9, 162, 10, 8, 0, 2, 1, 0, 2, 2, 3, 8, 4, 18, 5, 32, 6, 50, 7, 72, 8, 98, 9, 128, 10};
    Reduction r = new Reduction("reduction.cu");
    r.reduction(inputs, 3);
  }
  
}
