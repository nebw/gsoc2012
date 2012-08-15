#include "BaseSimulator.h"
#include "CLSimulator.h"
#include "CPUSimulator.h"
#include "gtest/gtest.h"
#include "opencl_fft/clFFT.h"

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
                                           500,
                                           0.1f,
                                           state0,
                                           CPUSimulator::CONVOLUTION)
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
                                       500,
                                       0.1f,
                                       state0,
                                       CPUSimulator::CONVOLUTION);

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
                                        500,
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
                                    500,
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
                                    500,
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
                                    500,
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
                                    500,
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

TEST(SimResultTests, RungeKuttaApproximationRelErrorTest)
{
    const unsigned int numNeurons = 2;
    const unsigned int timesteps = 1000;

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

    CPUSimulator cpuSim = CPUSimulator(numNeurons,
                                       1,
                                       1,
                                       timesteps,
                                       0.1f,
                                       state0,
                                       CPUSimulator::NO_CONVOLUTION);

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    CLSimulator clSim = CLSimulator(numNeurons,
                                    1,
                                    1,
                                    timesteps,
                                    0.1f,
                                    state0,
                                    BaseSimulator::NO_PLOT,
                                    BaseSimulator::NO_MEASURE,
                                    BaseSimulator::NO_FFTW,
                                    BaseSimulator::NO_CLFFT,
                                    path,
                                    logger,
                                    true);

    unsigned int t;

    for (t = 0; t < timesteps - 1; ++t)
    {
        if ((t + 2) % (timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        cpuSim.step();
        auto& cpuSimState = cpuSim.getCurrentStates();
        clSim.step();
        auto& clSimState = clSim.getCurrentStates();

        for (unsigned int i = 0; i < numNeurons; ++i)
        {
            EXPECT_NEAR(cpuSimState[i].V, clSimState[i].V, 0.00001);
            EXPECT_NEAR(cpuSimState[i].h, clSimState[i].h, 0.00001);
            EXPECT_NEAR(cpuSimState[i].n, clSimState[i].n, 0.00001);
            EXPECT_NEAR(cpuSimState[i].z, clSimState[i].z, 0.00001);
            EXPECT_NEAR(cpuSimState[i].s_AMPA, clSimState[i].s_AMPA, 0.00001);
            EXPECT_NEAR(cpuSimState[i].x_NMDA, clSimState[i].x_NMDA, 0.00001);
            EXPECT_NEAR(cpuSimState[i].s_NMDA, clSimState[i].s_NMDA, 0.00001);
            EXPECT_NEAR(cpuSimState[i].s_GABAA, clSimState[i].s_GABAA, 0.00001);
        }

        cpuSim.setCurrentStates(clSimState);
    }
    std::cout << std::endl;
}

TEST(SimResultTests, RungeKuttaApproximationAbsErrorTest)
{
    const unsigned int numNeurons = 2;
    // cumulative floating point inaccuracies become too large for more than 100 timesteps
    const unsigned int timesteps = 100;

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

    CPUSimulator cpuSim = CPUSimulator(numNeurons,
                                       1,
                                       1,
                                       timesteps,
                                       0.1f,
                                       state0,
                                       CPUSimulator::NO_CONVOLUTION);

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    CLSimulator clSim = CLSimulator(numNeurons,
                                    1,
                                    1,
                                    timesteps,
                                    0.1f,
                                    state0,
                                    BaseSimulator::NO_PLOT,
                                    BaseSimulator::NO_MEASURE,
                                    BaseSimulator::NO_FFTW,
                                    BaseSimulator::NO_CLFFT,
                                    path,
                                    logger,
                                    true);

    unsigned int t;

    for (t = 0; t < timesteps - 1; ++t)
    {
        if ((t + 2) % (timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        cpuSim.step();
        auto& cpuSimState = cpuSim.getCurrentStates();
        clSim.step();
        auto& clSimState = clSim.getCurrentStates();

        for (unsigned int i = 0; i < numNeurons; ++i)
        {
            EXPECT_FLOAT_EQ(cpuSimState[i].V, clSimState[i].V);
            EXPECT_FLOAT_EQ(cpuSimState[i].h, clSimState[i].h);
            EXPECT_FLOAT_EQ(cpuSimState[i].n, clSimState[i].n);
            EXPECT_FLOAT_EQ(cpuSimState[i].z, clSimState[i].z);
            EXPECT_FLOAT_EQ(cpuSimState[i].s_AMPA, clSimState[i].s_AMPA);
            EXPECT_FLOAT_EQ(cpuSimState[i].x_NMDA, clSimState[i].x_NMDA);
            EXPECT_FLOAT_EQ(cpuSimState[i].s_NMDA, clSimState[i].s_NMDA);
            EXPECT_FLOAT_EQ(cpuSimState[i].s_GABAA, clSimState[i].s_GABAA);
        }
    }
    std::cout << std::endl;
}

TEST(SimResultTests, FFTWConvolutionRelErrorTest)
{
    const unsigned int numNeurons = 256;
    const unsigned int timesteps = 1000;

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

    CPUSimulator cpuSim = CPUSimulator(numNeurons,
                                       1,
                                       1,
                                       timesteps,
                                       0.1f,
                                       state0,
                                       CPUSimulator::CONVOLUTION);

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    CLSimulator clSim = CLSimulator(numNeurons,
                                    1,
                                    1,
                                    timesteps,
                                    0.1f,
                                    state0,
                                    BaseSimulator::NO_PLOT,
                                    BaseSimulator::NO_MEASURE,
                                    BaseSimulator::FFTW,
                                    BaseSimulator::NO_CLFFT,
                                    path,
                                    logger,
                                    true);

    unsigned int t;

    for (t = 0; t < timesteps - 1; ++t)
    {
        if ((t + 2) % (timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        cpuSim.step();
        auto& cpuSimState = cpuSim.getCurrentStates();
        auto& cpuSumFootprintAMPA = cpuSim.getCurrentSumFootprintAMPA();
        auto& cpuSumFootprintNMDA = cpuSim.getCurrentSumFootprintNMDA();
        clSim.step();
        auto& clSimState = clSim.getCurrentStates();
        auto& clSumFootprintAMPA = clSim.getCurrentSumFootprintAMPA();
        auto& clSumFootprintNMDA = clSim.getCurrentSumFootprintNMDA();

        cpuSim.setCurrentStates(clSimState);

        for (unsigned int i = 0; i < numNeurons; ++i)
        {
            EXPECT_FLOAT_EQ(cpuSimState[i].V, clSimState[i].V);
            EXPECT_FLOAT_EQ(cpuSimState[i].h, clSimState[i].h);
            EXPECT_FLOAT_EQ(cpuSimState[i].n, clSimState[i].n);
            EXPECT_FLOAT_EQ(cpuSimState[i].z, clSimState[i].z);
            EXPECT_FLOAT_EQ(cpuSimState[i].s_AMPA, clSimState[i].s_AMPA);
            EXPECT_FLOAT_EQ(cpuSimState[i].x_NMDA, clSimState[i].x_NMDA);
            EXPECT_FLOAT_EQ(cpuSimState[i].s_NMDA, clSimState[i].s_NMDA);
            EXPECT_FLOAT_EQ(cpuSimState[i].s_GABAA, clSimState[i].s_GABAA);
            EXPECT_NEAR(cpuSumFootprintAMPA[i], clSumFootprintAMPA[i], 0.00001);
            EXPECT_NEAR(cpuSumFootprintNMDA[i], clSumFootprintNMDA[i], 0.00001);
        }

        cpuSim.setCurrentSumFootprintAMPA(clSumFootprintAMPA);
        cpuSim.setCurrentSumFootprintNMDA(clSumFootprintNMDA);
    }
    std::cout << std::endl;
}

TEST(SimResultTests, ClFFTConvolutionRelErrorTest)
{
    const unsigned int numNeurons = 256;
    const unsigned int timesteps = 1000;

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

    CPUSimulator cpuSim = CPUSimulator(numNeurons,
                                       1,
                                       1,
                                       timesteps,
                                       0.1f,
                                       state0,
                                       CPUSimulator::CONVOLUTION);

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    CLSimulator clSim = CLSimulator(numNeurons,
                                    1,
                                    1,
                                    timesteps,
                                    0.1f,
                                    state0,
                                    BaseSimulator::NO_PLOT,
                                    BaseSimulator::NO_MEASURE,
                                    BaseSimulator::NO_FFTW,
                                    BaseSimulator::CLFFT,
                                    path,
                                    logger,
                                    true);

    unsigned int t;

    for (t = 0; t < timesteps - 1; ++t)
    {
        if ((t + 2) % (timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        cpuSim.step();
        auto& cpuSimState = cpuSim.getCurrentStates();
        auto& cpuSumFootprintAMPA = cpuSim.getCurrentSumFootprintAMPA();
        auto& cpuSumFootprintNMDA = cpuSim.getCurrentSumFootprintNMDA();
        clSim.step();
        auto& clSimState = clSim.getCurrentStates();
        auto& clSumFootprintAMPA = clSim.getCurrentSumFootprintAMPA();
        auto& clSumFootprintNMDA = clSim.getCurrentSumFootprintNMDA();

        cpuSim.setCurrentStates(clSimState);

        for (unsigned int i = 0; i < numNeurons; ++i)
        {
            EXPECT_FLOAT_EQ(cpuSimState[i].V, clSimState[i].V);
            EXPECT_FLOAT_EQ(cpuSimState[i].h, clSimState[i].h);
            EXPECT_FLOAT_EQ(cpuSimState[i].n, clSimState[i].n);
            EXPECT_FLOAT_EQ(cpuSimState[i].z, clSimState[i].z);
            EXPECT_FLOAT_EQ(cpuSimState[i].s_AMPA, clSimState[i].s_AMPA);
            EXPECT_FLOAT_EQ(cpuSimState[i].x_NMDA, clSimState[i].x_NMDA);
            EXPECT_FLOAT_EQ(cpuSimState[i].s_NMDA, clSimState[i].s_NMDA);
            EXPECT_FLOAT_EQ(cpuSimState[i].s_GABAA, clSimState[i].s_GABAA);
            // error can be > 0.001
            EXPECT_NEAR(cpuSumFootprintAMPA[i], clSumFootprintAMPA[i], 0.01);
            EXPECT_NEAR(cpuSumFootprintNMDA[i], clSumFootprintNMDA[i], 0.01);
        }

        cpuSim.setCurrentSumFootprintAMPA(clSumFootprintAMPA);
        cpuSim.setCurrentSumFootprintNMDA(clSumFootprintNMDA);
    }
    std::cout << std::endl;
}
