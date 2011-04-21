package cuda;

import java.io.*;

import jcuda.*;
import jcuda.driver.*;

public class CudaCritterSelector {
  private CUmodule module;
  public CudaCritterSelector(String filename) throws Exception {
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
  
  public void calcDist(int critters[][], int towers[][], int threadPerBlock) {

    CUfunction function = new CUfunction();
    JCudaDriver.cuModuleGetFunction(function, module, "calcDistance");



    int numThreads = critters.length;
    
    int numBlocks = (int) Math.ceil((double)numThreads / threadPerBlock);

    int numTowers = towers.length;
    int size = critters[0].length;

    // Allocate and fill host input memory: A 2D int array with
    // 'numThreads' rows and 'size' columns, each row filled with
    // the values from 0 to size-1.
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
    
    
    // Allocate device memory for the array pointers, and copy
    // the array pointers from the host to the device.
    CUdeviceptr deviceInput = new CUdeviceptr();
    CUdeviceptr deviceQuery = new CUdeviceptr();
    
    JCudaDriver.cuMemAlloc(deviceInput, numThreads * Sizeof.POINTER);
    JCudaDriver.cuMemcpyHtoD(deviceInput, Pointer.to(hostDevicePointers),
        numThreads * Sizeof.POINTER);

    
    JCudaDriver.cuMemAlloc(deviceQuery, numTowers * Sizeof.POINTER);
    JCudaDriver.cuMemcpyHtoD(deviceQuery, Pointer.to(queryHostDevicePointers), numTowers * Sizeof.POINTER);
    
    CUdeviceptr deviceOutput = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(deviceOutput, numTowers * numThreads * Sizeof.INT);
    
    JCudaDriver.cuFuncSetBlockShape(function, Math.min(threadPerBlock, numThreads), 1, 1);
    
    // Set up the parameters for the function call: One pointer (to
    // pointers) for the input, one int for the size, and one pointer
    // for the output array. Note that for 'cuParamSetv' you have
    // to pass a pointer to a pointer, in order to set the value
    // of the pointer as the parameter.
    Pointer dIn = Pointer.to(deviceInput);
    
    Pointer dOut = Pointer.to(deviceOutput);
    Pointer pSize = Pointer.to(new int[]{size});
    Pointer pN    = Pointer.to(new int[]{numThreads});
    
    Pointer dQu = Pointer.to(deviceQuery);
    

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
    
    
    int hostOutput[] = new int[numTowers * numThreads];
    JCudaDriver.cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
        numTowers * numThreads * Sizeof.INT);
 
    
    for (int output : hostOutput)
      System.out.println(output);
    
    for(int i = 0; i < numThreads; i++)
    {
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
    JCudaDriver.cuMemFree(deviceOutput);
    
  }
  
  
  /**
   * The extension of the given file name is replaced with "cubin".
   * If the file with the resulting name does not exist, it is
   * compiled from the given file using NVCC. The name of the
   * cubin file is returned.
   *
   * @param cuFileName The name of the .CU file
   * @return The name of the CUBIN file
   * @throws IOException If an I/O error occurs
   */
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
  
  
  public static void main(String[] args) throws Exception {
    
    int hostInput[][] = new int[11][2];
    for(int i = 0; i < 11; i++)
    {
        for (int j=0; j<2; j++)
        {
            hostInput[i][j] = j + i + 5;
        }
    }
    
    int towerNo = 3;
    
    int hostQuery[][] = new int[towerNo][2];
    for (int i = 0; i < towerNo; i++) {
      for (int j = 0; j < 2; j++) {
        hostQuery[i][j] = (i + j * j);
      }
    }
    

    CudaCritterSelector c = new CudaCritterSelector("CUDA_Distance.cu");
    c.calcDist(hostInput, hostQuery, 5);
  }
  
  
  
  
  
}
