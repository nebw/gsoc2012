/**
 * Copyright (C) 2012 Benjamin Wild
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "BaseSimulator.h"
#include "CLSimulator.h"
#include "CPUSimulator.h"

#include "gtest/gtest.h"

#include <clFFT.h>
#include <cpplog.hpp>

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

    size_t batchSize = 1;
    int numIter            = 1;

    int length = n.x * n.y * n.z * batchSize;

    clFFT_SplitComplex data_i_split = { (float *)malloc(sizeof(float) * length),
                                        (float *)malloc(sizeof(float) * length) };
    clFFT_SplitComplex data_cl_split = { (float *)malloc(sizeof(float) * length),
                                         (float *)malloc(sizeof(float) * length) };

    for (int i = 0; i < length; i++)
    {
        data_i_split.real[i]  = 2.0f * float(rand()) / float(RAND_MAX) - 1.0f;
        data_i_split.imag[i]  = 2.0f * float(rand()) / float(RAND_MAX) - 1.0f;
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
        ASSERT_NEAR(data_i_split.real[i],
                    data_cl_split.real[i] / length,
                    0.000001);
        ASSERT_NEAR(data_i_split.imag[i],
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

    ASSERT_NO_THROW(
        CPUSimulator cpuSim(256,
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

    CPUSimulator cpuSim(64,
                        1,
                        1,
                        500,
                        0.1f,
                        state0,
                        CPUSimulator::CONVOLUTION);

    ASSERT_NO_THROW(cpuSim.simulate());
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

    ASSERT_NO_THROW(
        CLSimulator(4,
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

    CLSimulator clSim(256,
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

    ASSERT_NO_THROW(clSim.simulate());
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

    CLSimulator clSim(256,
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

    ASSERT_NO_THROW(clSim.simulate());
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

    CLSimulator clSim(4,
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

    ASSERT_NO_THROW(clSim.simulate());
}

TEST(BasicSimTests, CLSimSimulationFFTWClFFTW1DAssertionsReturnsTest)
{
    const size_t nX = 64;
    const size_t nY = 1;
    const size_t nZ = 1;

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

    CLSimulator clSim(nX,
                      nY,
                      nZ,
                      500,
                      0.1f,
                      state0,
                      BaseSimulator::NO_PLOT,
                      BaseSimulator::NO_MEASURE,
                      BaseSimulator::FFTW,
                      BaseSimulator::CLFFT,
                      path,
                      logger);

    ASSERT_NO_THROW(clSim.simulate());
}

TEST(BasicSimTests, CLSimSimulationFFTWClFFTW2DAssertionsReturnsTest)
{
    const size_t nX = 8;
    const size_t nY = 8;
    const size_t nZ = 1;

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

    CLSimulator clSim(nX,
                      nY,
                      nZ,
                      500,
                      0.1f,
                      state0,
                      BaseSimulator::NO_PLOT,
                      BaseSimulator::NO_MEASURE,
                      BaseSimulator::FFTW,
                      BaseSimulator::CLFFT,
                      path,
                      logger);

    ASSERT_NO_THROW(clSim.simulate());
}

TEST(SimResultTests, RungeKuttaApproximationRelErrorTest)
{
    const size_t numNeurons = 2;
    const size_t timesteps = 1000;

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

    CPUSimulator cpuSim(numNeurons,
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

    CLSimulator clSim(numNeurons,
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

    size_t t;

    for (t = 0; t < timesteps - 1; ++t)
    {
        if ((t + 2) % (timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        cpuSim.step();
        auto const cpuSimStateNew = cpuSim.getCurrentStatesNew();
        clSim.step();
        auto const clSimStateNew = clSim.getCurrentStatesNew();
        auto const clSimStateOld = clSim.getCurrentStatesOld();

        for (size_t i = 0; i < numNeurons; ++i)
        {
            ASSERT_NEAR(cpuSimStateNew[i].V, clSimStateNew[i].V, 0.00001);
            ASSERT_NEAR(cpuSimStateNew[i].h, clSimStateNew[i].h, 0.00001);
            ASSERT_NEAR(cpuSimStateNew[i].n, clSimStateNew[i].n, 0.00001);
            ASSERT_NEAR(cpuSimStateNew[i].z, clSimStateNew[i].z, 0.00001);
            ASSERT_NEAR(cpuSimStateNew[i].s_AMPA, clSimStateNew[i].s_AMPA, 0.00001);
            ASSERT_NEAR(cpuSimStateNew[i].x_NMDA, clSimStateNew[i].x_NMDA, 0.00001);
            ASSERT_NEAR(cpuSimStateNew[i].s_NMDA, clSimStateNew[i].s_NMDA, 0.00001);
        }

        cpuSim.setCurrentStatesOld(clSimStateOld);
        cpuSim.setCurrentStatesNew(clSimStateNew);
    }
    std::cout << std::endl;
}

TEST(SimResultTests, RungeKuttaApproximationAbsErrorTest)
{
    const size_t numNeurons = 64;
    // cumulative floating point inaccuracies become too large for more than 100 timesteps
    const size_t timesteps = 100;

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

    CPUSimulator cpuSim(numNeurons,
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

    CLSimulator clSim(numNeurons,
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

    size_t t;

    for (t = 0; t < timesteps - 1; ++t)
    {
        if ((t + 2) % (timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        cpuSim.step();
        auto const cpuSimStateNew = cpuSim.getCurrentStatesNew();
        clSim.step();
        auto const clSimStateNew = clSim.getCurrentStatesNew();

        for (size_t i = 0; i < numNeurons; ++i)
        {
            ASSERT_NEAR(cpuSimStateNew[i].V, clSimStateNew[i].V, 0.000001);
            ASSERT_NEAR(cpuSimStateNew[i].h, clSimStateNew[i].h, 0.000001);
            ASSERT_NEAR(cpuSimStateNew[i].n, clSimStateNew[i].n, 0.000001);
            ASSERT_NEAR(cpuSimStateNew[i].z, clSimStateNew[i].z, 0.000001);
            ASSERT_NEAR(cpuSimStateNew[i].s_AMPA, clSimStateNew[i].s_AMPA, 0.000001);
            ASSERT_NEAR(cpuSimStateNew[i].x_NMDA, clSimStateNew[i].x_NMDA, 0.000001);
            ASSERT_NEAR(cpuSimStateNew[i].s_NMDA, clSimStateNew[i].s_NMDA, 0.000001);
        }
    }
    std::cout << std::endl;
}

TEST(SimResultTests, FFTWConvolution1DRelErrorTest)
{
    const size_t numNeurons = 64;
    const size_t timesteps = 500;

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

    CPUSimulator cpuSim(numNeurons,
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

    CLSimulator clSim(numNeurons,
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

    size_t t;

    for (t = 0; t < timesteps - 1; ++t)
    {
        if ((t + 2) % (timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        cpuSim.step();
        auto const cpuSimStateNew = cpuSim.getCurrentStatesNew();
        auto& cpuSumFootprintAMPA = cpuSim.getCurrentSumFootprintAMPA();
        auto& cpuSumFootprintNMDA = cpuSim.getCurrentSumFootprintNMDA();
        clSim.step();
        auto const clSimStateOld = clSim.getCurrentStatesOld();
        auto const clSimStateNew = clSim.getCurrentStatesNew();
        auto& clSumFootprintAMPA = clSim.getCurrentSumFootprintAMPA();
        auto& clSumFootprintNMDA = clSim.getCurrentSumFootprintNMDA();

        cpuSim.setCurrentStatesOld(clSimStateOld);
        cpuSim.setCurrentStatesNew(clSimStateNew);

        for (size_t i = 0; i < numNeurons; ++i)
        {
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].V, clSimStateNew[i].V);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].h, clSimStateNew[i].h);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].n, clSimStateNew[i].n);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].z, clSimStateNew[i].z);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].s_AMPA, clSimStateNew[i].s_AMPA);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].x_NMDA, clSimStateNew[i].x_NMDA);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].s_NMDA, clSimStateNew[i].s_NMDA);
            ASSERT_NEAR(cpuSumFootprintAMPA[i], clSumFootprintAMPA[i], 0.00001);
            ASSERT_NEAR(cpuSumFootprintNMDA[i], clSumFootprintNMDA[i], 0.00001);
        }

        cpuSim.setCurrentSumFootprintAMPA(clSumFootprintAMPA);
        cpuSim.setCurrentSumFootprintNMDA(clSumFootprintNMDA);
    }
    std::cout << std::endl;
}

TEST(SimResultTests, FFTWConvolution2DRelErrorTest)
{
    const size_t nX = 8;
    const size_t nY = 8;
    const size_t nZ = 1;
    const size_t numNeurons = nX * nY * nZ;
    const size_t timesteps = 500;

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

    CPUSimulator cpuSim(nX,
                        nY,
                        nZ,
                        timesteps,
                        0.1f,
                        state0,
                        CPUSimulator::CONVOLUTION);

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    CLSimulator clSim(nX,
                      nY,
                      nZ,
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

    size_t t;

    for (t = 0; t < timesteps - 1; ++t)
    {
        if ((t + 2) % (timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        cpuSim.step();
        auto const cpuSimStateNew = cpuSim.getCurrentStatesNew();
        auto& cpuSumFootprintAMPA = cpuSim.getCurrentSumFootprintAMPA();
        auto& cpuSumFootprintNMDA = cpuSim.getCurrentSumFootprintNMDA();
        clSim.step();
        auto const clSimStateOld = clSim.getCurrentStatesOld();
        auto const clSimStateNew = clSim.getCurrentStatesNew();
        auto& clSumFootprintAMPA = clSim.getCurrentSumFootprintAMPA();
        auto& clSumFootprintNMDA = clSim.getCurrentSumFootprintNMDA();

        cpuSim.setCurrentStatesOld(clSimStateOld);
        cpuSim.setCurrentStatesNew(clSimStateNew);

        for (size_t i = 0; i < numNeurons; ++i)
        {
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].V, clSimStateNew[i].V);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].h, clSimStateNew[i].h);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].n, clSimStateNew[i].n);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].z, clSimStateNew[i].z);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].s_AMPA, clSimStateNew[i].s_AMPA);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].x_NMDA, clSimStateNew[i].x_NMDA);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].s_NMDA, clSimStateNew[i].s_NMDA);
            ASSERT_NEAR(cpuSumFootprintAMPA[i], clSumFootprintAMPA[i], 0.00001);
            ASSERT_NEAR(cpuSumFootprintNMDA[i], clSumFootprintNMDA[i], 0.00001);
        }

        cpuSim.setCurrentSumFootprintAMPA(clSumFootprintAMPA);
        cpuSim.setCurrentSumFootprintNMDA(clSumFootprintNMDA);
    }
    std::cout << std::endl;
}

TEST(SimResultTests, ClFFTConvolution1DRelErrorTest)
{
    const size_t nX = 64;
    const size_t nY = 1;
    const size_t nZ = 1;
    const size_t numNeurons = nX * nY * nZ;
    const size_t timesteps = 500;

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

    CPUSimulator cpuSim(nX,
                        nY,
                        nZ,
                        timesteps,
                        0.1f,
                        state0,
                        CPUSimulator::CONVOLUTION);

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    CLSimulator clSim(nX,
                      nY,
                      nZ,
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

    size_t t;

    for (t = 0; t < timesteps - 1; ++t)
    {
        if ((t + 2) % (timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        cpuSim.step();
        auto const cpuSimStateNew = cpuSim.getCurrentStatesNew();
        auto& cpuSumFootprintAMPA = cpuSim.getCurrentSumFootprintAMPA();
        auto& cpuSumFootprintNMDA = cpuSim.getCurrentSumFootprintNMDA();
        clSim.step();
        auto const clSimStateOld = clSim.getCurrentStatesOld();
        auto const clSimStateNew = clSim.getCurrentStatesNew();
        auto& clSumFootprintAMPA = clSim.getCurrentSumFootprintAMPA();
        auto& clSumFootprintNMDA = clSim.getCurrentSumFootprintNMDA();

        cpuSim.setCurrentStatesOld(clSimStateOld);
        cpuSim.setCurrentStatesNew(clSimStateNew);

        for (size_t i = 0; i < numNeurons; ++i)
        {
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].V, clSimStateNew[i].V);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].h, clSimStateNew[i].h);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].n, clSimStateNew[i].n);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].z, clSimStateNew[i].z);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].s_AMPA, clSimStateNew[i].s_AMPA);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].x_NMDA, clSimStateNew[i].x_NMDA);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].s_NMDA, clSimStateNew[i].s_NMDA);
            ASSERT_NEAR(cpuSumFootprintAMPA[i], clSumFootprintAMPA[i], 0.00001);
            ASSERT_NEAR(cpuSumFootprintNMDA[i], clSumFootprintNMDA[i], 0.00001);
        }

        cpuSim.setCurrentSumFootprintAMPA(clSumFootprintAMPA);
        cpuSim.setCurrentSumFootprintNMDA(clSumFootprintNMDA);
    }
    std::cout << std::endl;
}

TEST(SimResultTests, ClFFTConvolution2DRelErrorTest)
{
    const size_t nX = 8;
    const size_t nY = 8;
    const size_t nZ = 1;
    const size_t numNeurons = nX * nY * nZ;
    const size_t timesteps = 500;

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

    CPUSimulator cpuSim(nX,
                        nY,
                        nZ,
                        timesteps,
                        0.1f,
                        state0,
                        CPUSimulator::CONVOLUTION);

    auto path = boost::filesystem::path(CL_SOURCE_DIR);
    path /= "/kernels.cl";

    auto stdErrLogger = std::make_shared<cpplog::StdErrLogger>();
    auto logger = std::make_shared<cpplog::FilteringLogger>(LL_ERROR, stdErrLogger.get());

    CLSimulator clSim(nX,
                      nY,
                      nZ,
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

    size_t t;

    for (t = 0; t < timesteps - 1; ++t)
    {
        if ((t + 2) % (timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        cpuSim.step();
        auto const cpuSimStateNew = cpuSim.getCurrentStatesNew();
        auto& cpuSumFootprintAMPA = cpuSim.getCurrentSumFootprintAMPA();
        auto& cpuSumFootprintNMDA = cpuSim.getCurrentSumFootprintNMDA();
        clSim.step();
        auto const clSimStateOld = clSim.getCurrentStatesOld();
        auto const clSimStateNew = clSim.getCurrentStatesNew();
        auto& clSumFootprintAMPA = clSim.getCurrentSumFootprintAMPA();
        auto& clSumFootprintNMDA = clSim.getCurrentSumFootprintNMDA();

        cpuSim.setCurrentStatesOld(clSimStateOld);
        cpuSim.setCurrentStatesNew(clSimStateNew);

        for (size_t i = 0; i < numNeurons; ++i)
        {
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].V, clSimStateNew[i].V);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].h, clSimStateNew[i].h);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].n, clSimStateNew[i].n);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].z, clSimStateNew[i].z);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].s_AMPA, clSimStateNew[i].s_AMPA);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].x_NMDA, clSimStateNew[i].x_NMDA);
            ASSERT_FLOAT_EQ(cpuSimStateNew[i].s_NMDA, clSimStateNew[i].s_NMDA);
            ASSERT_NEAR(cpuSumFootprintAMPA[i], clSumFootprintAMPA[i], 0.05);
            ASSERT_NEAR(cpuSumFootprintNMDA[i], clSumFootprintNMDA[i], 0.05);
        }

        cpuSim.setCurrentSumFootprintAMPA(clSumFootprintAMPA);
        cpuSim.setCurrentSumFootprintNMDA(clSumFootprintNMDA);
    }
    std::cout << std::endl;
}
