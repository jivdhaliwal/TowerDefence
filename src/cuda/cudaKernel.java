package cuda;

import java.io.*;

import jcuda.driver.*;

public class cudaKernel {
  private CUmodule module;
  protected CUmodule getModule() {
    return module;
  }
  
  public cudaKernel(String filename) throws Exception {
    String cubinFileName = prepareCubinFile(filename);
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
          return cubinFileName;
      }

      File cuFile = new File(cuFileName);
      if (!cuFile.exists())
      {
          throw new IOException("Input file not found: "+cuFileName);
      }
      String modelString = "-m"+System.getProperty("sun.arch.data.model");
//      String command =
//        "nvcc " + modelString + " -arch sm_11 -cubin "+
//        cuFile.getPath()+" -o "+cubinFileName;

      
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
}
