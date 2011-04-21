package cuda;

import jcuda.*;
import jcuda.runtime.*;
import jcuda.jcudpp.*;

class JCudppSample
{
    // Program main
    public static void main(String args[])
    {
        int numElements = 20;
        int memSize = Sizeof.FLOAT * numElements;

        // allocate host memory
        float h_idata[] = new float[numElements];

        // initialize the memory
        for (int i = 0; i < numElements; i++)
        {
            h_idata[i] = (float)i; // Math.random();
        }
        
        for (float h: h_idata)
          System.out.println(h);

        // allocate device memory
        Pointer d_idata = new Pointer();
        JCuda.cudaMalloc(d_idata, memSize);

        // copy host memory to device
        JCuda.cudaMemcpy(d_idata, Pointer.to(h_idata), memSize,
            cudaMemcpyKind.cudaMemcpyHostToDevice);

        // allocate device memory for result
        Pointer d_odata = new Pointer();
        JCuda.cudaMalloc(d_odata, memSize);

        CUDPPConfiguration config = new CUDPPConfiguration();
        config.op = CUDPPOperator.CUDPP_ADD;
        config.datatype = CUDPPDatatype.CUDPP_FLOAT;
        config.algorithm = CUDPPAlgorithm.CUDPP_SCAN;
        config.options = CUDPPOption.CUDPP_OPTION_FORWARD |
                         CUDPPOption.CUDPP_OPTION_EXCLUSIVE;

        CUDPPHandle scanplan = new CUDPPHandle();
        JCudpp.cudppPlan(scanplan, config, numElements, 1, 0);

        // Run the scan
        JCudpp.cudppScan(scanplan, d_odata, d_idata, numElements);

        // allocate mem for the result on host side
        float h_odata[] = new float[numElements];

        // copy result from device to host
        JCuda.cudaMemcpy(Pointer.to(h_odata), d_odata, memSize,
            cudaMemcpyKind.cudaMemcpyDeviceToHost);
        
        for (float result : h_odata)
          System.out.println(result);

        JCudpp.cudppDestroyPlan(scanplan);


        JCuda.cudaFree(d_idata);
        JCuda.cudaFree(d_odata);

    }
}