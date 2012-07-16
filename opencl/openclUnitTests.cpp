#include <boost/program_options.hpp>

#include "gtest/gtest.h"
#include "opencl_fft/clFFT.h"

#include <iostream>

#if defined(__APPLE__)
# include <OpenCL/opencl.h>
#else // if defined(__APPLE__)
# include <CL/cl.h>
#endif // if defined(__APPLE__)

typedef struct
{
    double *real;
    double *imag;
} clFFT_SplitComplexDouble;

TEST(FFTTest, InverseFFTTest)
{
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id   = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
                         &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue queue = clCreateCommandQueue(context,
                                                  device_id,
                                                  0,
                                                  &ret);

    clFFT_Direction dir         = clFFT_Forward;
    clFFT_Dim3 n                = { 4096, 1, 1 };
    clFFT_DataFormat dataFormat = clFFT_SplitComplexFormat;
    clFFT_Dimension dim        = clFFT_1D;
    cl_int error_code;
    cl_mem data_in_real  = NULL;
    cl_mem data_in_imag  = NULL;
    cl_mem data_out_real = NULL;
    cl_mem data_out_imag = NULL;

    unsigned int batchSize = 1;
    int numIter            = 1;

    int length = n.x * n.y * n.z * batchSize;

    clFFT_SplitComplex data_i_split    = { NULL, NULL };
    clFFT_SplitComplex data_cl_split   = { NULL, NULL };
    clFFT_Complex *data_i          = NULL;
    clFFT_Complex *data_cl         = NULL;
    clFFT_SplitComplexDouble data_iref = { NULL, NULL };
    clFFT_SplitComplexDouble data_oref = { NULL, NULL };

    data_i_split.real  = (float *)malloc(sizeof(float) * length);
    data_i_split.imag  = (float *)malloc(sizeof(float) * length);
    data_cl_split.real = (float *)malloc(sizeof(float) * length);
    data_cl_split.imag = (float *)malloc(sizeof(float) * length);

    data_iref.real = (double *)malloc(sizeof(double) * length);
    data_iref.imag = (double *)malloc(sizeof(double) * length);
    data_oref.real = (double *)malloc(sizeof(double) * length);
    data_oref.imag = (double *)malloc(sizeof(double) * length);

    // std::cout << "Input: " << std::endl;
    for (int i = 0; i < length; i++)
    {
        data_i_split.real[i]  = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
        data_i_split.imag[i]  = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
        data_cl_split.real[i] = 0.0f;
        data_cl_split.imag[i] = 0.0f;
        data_iref.real[i]     = data_i_split.real[i];
        data_iref.imag[i]     = data_i_split.imag[i];
        data_oref.real[i]     = data_iref.real[i];
        data_oref.imag[i]     = data_iref.imag[i];

        // std::cout << "data[" << i << "] = { " << data_i_split.real[i] << ", "
        // << data_i_split.imag[i] << " } " << std::endl;
    }

    clFFT_Plan plan = clFFT_CreatePlan(context, n, dim, dataFormat, &error_code);

    if (!plan || error_code)
    {
        std::cout << "clFFT_CreatePlan failed" << std::endl;
        exit(1);
    }

    data_in_real = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  length * sizeof(float),
                                  data_i_split.real,
                                  &error_code);

    if (!data_in_real || error_code)
    {
        std::cout << "clCreateBuffer failed" << std::endl;
        exit(1);
    }

    data_in_imag = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  length * sizeof(float),
                                  data_i_split.imag,
                                  &error_code);

    if (!data_in_imag || error_code)
    {
        std::cout << "clCreateBuffer failed" << std::endl;
        exit(1);
    }

    // clFFT_OUT_OF_PLACE
    data_out_real = clCreateBuffer(context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   length * sizeof(float),
                                   data_cl_split.real,
                                   &error_code);

    if (!data_out_real || error_code)
    {
        std::cout << "clCreateBuffer failed" << std::endl;
        exit(1);
    }

    data_out_imag = clCreateBuffer(context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   length * sizeof(float),
                                   data_cl_split.imag,
                                   &error_code);

    if (!data_out_imag || error_code)
    {
        std::cout << "clCreateBuffer failed" << std::endl;
        exit(1);
    }

    error_code = CL_SUCCESS;

    for (int iter = 0; iter < numIter; iter++)
    {
        error_code |= clFFT_ExecutePlannar(queue,
                                           plan,
                                           batchSize,
                                           dir,
                                           data_in_real,
                                           data_in_imag,
                                           data_out_real,
                                           data_out_imag,
                                           0,
                                           NULL,
                                           NULL);
    }

    if (error_code)
    {
        std::cout << "clFFT_Execute failed" << std::endl;
        exit(1);
    }

    error_code |= clEnqueueReadBuffer(queue,
                                      data_out_real,
                                      CL_TRUE,
                                      0,
                                      length * sizeof(float),
                                      data_cl_split.real,
                                      0,
                                      NULL,
                                      NULL);
    error_code |= clEnqueueReadBuffer(queue,
                                      data_out_imag,
                                      CL_TRUE,
                                      0,
                                      length * sizeof(float),
                                      data_cl_split.imag,
                                      0,
                                      NULL,
                                      NULL);

    if (error_code)
    {
        std::cout << "clEnqueueReadBuffer failed" << std::endl;
        exit(1);
    }

    //   std::cout << "Output: " << std::endl;
    //   for(int i = 0; i < length; i++)
    // {
    //       std::cout << "data[" << i << "] = { " << data_cl_split.real[i] <<
    // ", " << data_cl_split.imag[i] << " } " << std::endl;
    // }

    dir = clFFT_Inverse;

    error_code = CL_SUCCESS;

    for (int iter = 0; iter < numIter; iter++)
    {
        error_code |= clFFT_ExecutePlannar(queue,
                                           plan,
                                           batchSize,
                                           dir,
                                           data_out_real,
                                           data_out_imag,
                                           data_in_real,
                                           data_in_imag,
                                           0,
                                           NULL,
                                           NULL);
    }

    if (error_code)
    {
        std::cout << "clFFT_Execute failed" << std::endl;
        exit(1);
    }

    error_code |= clEnqueueReadBuffer(queue,
                                      data_in_real,
                                      CL_TRUE,
                                      0,
                                      length * sizeof(float),
                                      data_cl_split.real,
                                      0,
                                      NULL,
                                      NULL);
    error_code |= clEnqueueReadBuffer(queue,
                                      data_in_imag,
                                      CL_TRUE,
                                      0,
                                      length * sizeof(float),
                                      data_cl_split.imag,
                                      0,
                                      NULL,
                                      NULL);

    if (error_code)
    {
        std::cout << "clEnqueueReadBuffer failed" << std::endl;
        exit(1);
    }

    //   std::cout << "Inverse Output: " << std::endl;
    //   for(int i = 0; i < length; i++)
    // {
    //       std::cout << "data[" << i << "] = { " << data_cl_split.real[i] /
    // length << ", " << data_cl_split.imag[i] / length << " } " << std::endl;
    // }

    for (int i = 0; i < length; i++)
    {
        EXPECT_NEAR(data_i_split.real[i],
                    data_cl_split.real[i] / length,
                    0.000001);
        EXPECT_NEAR(data_i_split.imag[i],
                    data_cl_split.imag[i] / length,
                    0.000001);
    }

    // Clean up
    ret = clFlush(queue);
    ret = clFinish(queue);
    ret = clReleaseCommandQueue(queue);
    ret = clReleaseContext(context);
}
