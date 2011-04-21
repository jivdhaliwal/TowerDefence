package cuda;
/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2009 Marco Hutter - http://www.jcuda.org
 */

import java.io.*;

import jcuda.*;
import jcuda.driver.*;

/**
 * This is a sample class demonstrating how to use the JCuda driver
 * bindings to load a CUBIN file and execute a function. The sample
 * reads a CUDA file, compiles it to a CUBIN file using NVCC, and
 * loads the CUBIN file as a module. <br />
 * <br />
 * The the sample creates a 2D float array and passes it to a kernel
 * that sums up the elements of each row of the array (each in its
 * own thread) and writes the sums into an 1D output array.
 */
public class JCudaDriverCubinSample
{
    public static void main(String args[]) throws IOException
    {
        String cubinFileName = prepareCubinFile("JCudaCubinSample_kernel.cu");

        // Initialize the driver and create a context for the first device.
        JCudaDriver.cuInit(0);
        CUcontext pctx = new CUcontext();
        CUdevice dev = new CUdevice();
        JCudaDriver.cuDeviceGet(dev, 0);
        JCudaDriver.cuCtxCreate(pctx, 0, dev);


        // Load the CUBIN file.
        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, cubinFileName);


        // Obtain a function pointer to the "sampleKernel" function.
        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "sampleKernel");


        int numThreads = 8;
        int size = 128;

        // Allocate and fill host input memory: A 2D float array with
        // 'numThreads' rows and 'size' columns, each row filled with
        // the values from 0 to size-1.
        float hostInput[][] = new float[numThreads][size];
        for(int i = 0; i < numThreads; i++)
        {
            for (int j=0; j<size; j++)
            {
                hostInput[i][j] = (float)j;
            }
        }


        // Allocate arrays on the device, one for each row. The pointers
        // to these array are stored in host memory.
        CUdeviceptr hostDevicePointers[] = new CUdeviceptr[numThreads];
        for(int i = 0; i < numThreads; i++)
        {
            hostDevicePointers[i] = new CUdeviceptr();
            JCudaDriver.cuMemAlloc(hostDevicePointers[i], size * Sizeof.FLOAT);
        }


        // Copy the contents of the rows from the host input data to
        // the device arrays that have just been allocated.
        for(int i = 0; i < numThreads; i++)
        {
            JCudaDriver.cuMemcpyHtoD(hostDevicePointers[i],
                Pointer.to(hostInput[i]), size * Sizeof.FLOAT);
        }


        // Allocate device memory for the array pointers, and copy
        // the array pointers from the host to the device.
        CUdeviceptr deviceInput = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(deviceInput, numThreads * Sizeof.POINTER);
        JCudaDriver.cuMemcpyHtoD(deviceInput, Pointer.to(hostDevicePointers),
            numThreads * Sizeof.POINTER);


        // Allocate device output memory: A single column with
        // height 'numThreads'.
        CUdeviceptr deviceOutput = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(deviceOutput, numThreads * Sizeof.FLOAT);


        // Set up the execution parameters.
        JCudaDriver.cuFuncSetBlockShape(function, numThreads, 1, 1);


        // Set up the parameters for the function call: One pointer (to
        // pointers) for the input, one int for the size, and one pointer
        // for the output array. Note that for 'cuParamSetv' you have
        // to pass a pointer to a pointer, in order to set the value
        // of the pointer as the parameter.
        Pointer dIn = Pointer.to(deviceInput);
        Pointer dOut = Pointer.to(deviceOutput);
        Pointer pSize = Pointer.to(new int[]{size});

        int offset = 0;
        offset = JCudaDriver.align(offset, Sizeof.POINTER);
        JCudaDriver.cuParamSetv(function, offset, dIn, Sizeof.POINTER);
        offset += Sizeof.POINTER;

        offset = JCudaDriver.align(offset, Sizeof.INT);
        JCudaDriver.cuParamSetv(function, offset, pSize, Sizeof.INT);
        offset += Sizeof.INT;

        offset = JCudaDriver.align(offset, Sizeof.POINTER);
        JCudaDriver.cuParamSetv(function, offset, dOut, Sizeof.POINTER);
        offset += Sizeof.POINTER;

        JCudaDriver.cuParamSetSize(function, offset);

        // Call the function.
        JCudaDriver.cuLaunch(function);
        JCudaDriver.cuCtxSynchronize();


        // Allocate host output memory and copy the device output
        // to the host.
        float hostOutput[] = new float[numThreads];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
            numThreads * Sizeof.FLOAT);


        // Verify the result
        boolean passed = true;
        for(int i = 0; i < numThreads; i++)
        {
            float expected = 0;
            for(int j = 0; j < size; j++)
            {
                expected += hostInput[i][j];
            }
            if (Math.abs(hostOutput[i] - expected) > 1e-5)
            {
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));


        // Clean up.
        for(int i = 0; i < numThreads; i++)
        {
            JCudaDriver.cuMemFree(hostDevicePointers[i]);
        }
        JCudaDriver.cuMemFree(deviceInput);
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


}
