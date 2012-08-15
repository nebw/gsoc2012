#include "gtest/gtest.h"
#include "opencl_fft/clFFT.h"
#include "BaseSimulator.h"
#include "CLSimulator.h"
#include "CPUSimulator.h"

#include "cpplog/cpplog.hpp"

#include <iostream>
#include <memory>

#include <boost/filesystem.hpp>

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

    data_i_split.real  = (float *)malloc(sizeof(float) * length);
    data_i_split.imag  = (float *)malloc(sizeof(float) * length);
    data_cl_split.real = (float *)malloc(sizeof(float) * length);
    data_cl_split.imag = (float *)malloc(sizeof(float) * length);

    for (int i = 0; i < length; i++)
    {
        data_i_split.real[i]  = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
        data_i_split.imag[i]  = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
        data_cl_split.real[i] = 0.0f;
        data_cl_split.imag[i] = 0.0f;
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

TEST(BasicSimTests, CPUSimInitializationTest)
{
    state state0;
    state0.V = -70.0f;
    state0.h = 1.0f;
    state0.n = 0.0f;
    state0.z = 0.0f;
    state0.s_AMPA = 0.0f;
    state0.s_NMDA = 0.0f;
    state0.x_NMDA = 0.0f;
    state0.s_GABAA = 0.0f;
    state0.I_app = 1.0f;

    EXPECT_NO_THROW(
        CPUSimulator cpuSim = CPUSimulator(1024,
                                           1,
                                           1,
                                           1000,
                                           0.1f,
                                           state0)
    );
}

TEST(BasicSimTests, CPUSimSimulationReturnsTest)
{
    state state0;
    state0.V = -70.0f;
    state0.h = 1.0f;
    state0.n = 0.0f;
    state0.z = 0.0f;
    state0.s_AMPA = 0.0f;
    state0.s_NMDA = 0.0f;
    state0.x_NMDA = 0.0f;
    state0.s_GABAA = 0.0f;
    state0.I_app = 1.0f;

    CPUSimulator cpuSim = CPUSimulator(256,
                                       1,
                                       1,
                                       1000,
                                       0.1f,
                                       state0);

    EXPECT_NO_THROW(cpuSim.simulate());
}

TEST(BasicSimTests, CLSimInitializationTest)
{
    state state0;
    state0.V = -70.0f;
    state0.h = 1.0f;
    state0.n = 0.0f;
    state0.z = 0.0f;
    state0.s_AMPA = 0.0f;
    state0.s_NMDA = 0.0f;
    state0.x_NMDA = 0.0f;
    state0.s_GABAA = 0.0f;
    state0.I_app = 1.0f;

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    EXPECT_NO_THROW(
        CLSimulator clSim = CLSimulator(1024,
                                        1,
                                        1,
                                        1000,
                                        0.1f,
                                        state0,
                                        BaseSimulator::NO_PLOT,
                                        BaseSimulator::NO_MEASURE,
                                        BaseSimulator::NO_FFTW,
                                        BaseSimulator::NO_CLFFT,
                                        path,
                                        logger)
    );
}

TEST(BasicSimTests, CLSimSimulationRungeKuttaOnlyReturnsTest)
{
    state state0;
    state0.V = -70.0f;
    state0.h = 1.0f;
    state0.n = 0.0f;
    state0.z = 0.0f;
    state0.s_AMPA = 0.0f;
    state0.s_NMDA = 0.0f;
    state0.x_NMDA = 0.0f;
    state0.s_GABAA = 0.0f;
    state0.I_app = 1.0f;

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    CLSimulator clSim = CLSimulator(256,
                                    1,
                                    1,
                                    1000,
                                    0.1f,
                                    state0,
                                    BaseSimulator::NO_PLOT,
                                    BaseSimulator::NO_MEASURE,
                                    BaseSimulator::NO_FFTW,
                                    BaseSimulator::NO_CLFFT,
                                    path,
                                    logger);

    EXPECT_NO_THROW(clSim.simulate());
}

TEST(BasicSimTests, CLSimSimulationWithFFTWReturnsTest)
{
    state state0;
    state0.V = -70.0f;
    state0.h = 1.0f;
    state0.n = 0.0f;
    state0.z = 0.0f;
    state0.s_AMPA = 0.0f;
    state0.s_NMDA = 0.0f;
    state0.x_NMDA = 0.0f;
    state0.s_GABAA = 0.0f;
    state0.I_app = 1.0f;

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    CLSimulator clSim = CLSimulator(256,
                                    1,
                                    1,
                                    1000,
                                    0.1f,
                                    state0,
                                    BaseSimulator::NO_PLOT,
                                    BaseSimulator::NO_MEASURE,
                                    BaseSimulator::FFTW,
                                    BaseSimulator::NO_CLFFT,
                                    path,
                                    logger);

    EXPECT_NO_THROW(clSim.simulate());
}

TEST(BasicSimTests, CLSimSimulationWithClFFTWReturnsTest)
{
    state state0;
    state0.V = -70.0f;
    state0.h = 1.0f;
    state0.n = 0.0f;
    state0.z = 0.0f;
    state0.s_AMPA = 0.0f;
    state0.s_NMDA = 0.0f;
    state0.x_NMDA = 0.0f;
    state0.s_GABAA = 0.0f;
    state0.I_app = 1.0f;

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    CLSimulator clSim = CLSimulator(256,
                                    1,
                                    1,
                                    1000,
                                    0.1f,
                                    state0,
                                    BaseSimulator::NO_PLOT,
                                    BaseSimulator::NO_MEASURE,
                                    BaseSimulator::NO_FFTW,
                                    BaseSimulator::CLFFT,
                                    path,
                                    logger);

    EXPECT_NO_THROW(clSim.simulate());
}

TEST(BasicSimTests, CLSimSimulationFFTWClFFTWAssertionsReturnsTest)
{
    state state0;
    state0.V = -70.0f;
    state0.h = 1.0f;
    state0.n = 0.0f;
    state0.z = 0.0f;
    state0.s_AMPA = 0.0f;
    state0.s_NMDA = 0.0f;
    state0.x_NMDA = 0.0f;
    state0.s_GABAA = 0.0f;
    state0.I_app = 1.0f;

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    CLSimulator clSim = CLSimulator(256,
                                    1,
                                    1,
                                    1000,
                                    0.1f,
                                    state0,
                                    BaseSimulator::NO_PLOT,
                                    BaseSimulator::NO_MEASURE,
                                    BaseSimulator::FFTW,
                                    BaseSimulator::CLFFT,
                                    path,
                                    logger);

    EXPECT_NO_THROW(clSim.simulate());
}