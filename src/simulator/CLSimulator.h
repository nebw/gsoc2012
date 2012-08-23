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

#pragma once

#include "Definitions.h"

#include "BasePlotter.h"
#include "BaseSimulator.h"
#include "CLWrapper.h"

#include <clFFT.h>

#include <boost/filesystem.hpp>

#include "fftw3.h"

class CLSimulator : public BaseSimulator {
public:

    CLSimulator(const size_t nX,
                const size_t nY,
                const size_t nZ,
                const size_t timesteps,
                const float dt,
                state const& state_0,
                const Plot plot,
                const Measure measure,
                const FFT_FFTW fftw,
                const FFT_clFFT clfft,
                boost::filesystem::path const& programPath,
                Logger const& logger,
                const bool readToHostMemory = false);
    ~CLSimulator();

    void step();
    void simulate();

    state const* getCurrentStatesOld() const;
    state const* getCurrentStatesNew() const;
    std::unique_ptr<float[]> const& getCurrentSumFootprintAMPA() const;
    std::unique_ptr<float[]> const& getCurrentSumFootprintNMDA() const;
    std::unique_ptr<float[]> const& getCurrentSumFootprintGABAA() const;

    std::vector<size_t> getTimesCalculations() const;
    std::vector<size_t> getTimesFFTW() const;
    std::vector<size_t> getTimesClFFT() const;

    CLWrapper getClWrapper() const;

private:

    enum Receptor {
        AMPA = 0,
        NMDA,
        GABAA
    };

    // OpenCL
    cl_int _err;
    CLWrapper _wrapper;
    cl::Program _program;
    cl::Event _event;

    // Simulation configuration
    const size_t _nX;
    const size_t _nY;
    const size_t _nZ;
    const size_t _numNeurons;
    const size_t _timesteps;
    const float _dt;
    
    // Initial state
    const state _state_0;
    size_t _t;
    size_t _ind_old;
    size_t _ind_new;
    
    // Configuration
    const bool _plot;
    const bool _measure;
    const bool _fftw;
    const bool _clfft;
    
    std::unique_ptr<BasePlotter> _plotter;
    Logger _logger;

    // Debug settings
    const bool _readToHostMemory;

    // Measurements
    std::vector<size_t> _timesCalculations;
    std::vector<size_t> _timesFFTW;
    std::vector<size_t> _timesClFFT;

    // FFT variables
    size_t _nFFT;
    size_t _nFFTx;
    size_t _nFFTy;
    size_t _nFFTz;
    float _scaleFFT;

    // Data
    std::vector<std::unique_ptr<state[]> > _states;
    std::unique_ptr<float[]> _sumFootprintAMPA;
    std::unique_ptr<float[]> _sumFootprintNMDA;
    std::unique_ptr<float[]> _sumFootprintGABAA;

    // Data (FFTW)
    fftwf_complex *_distances_split;
    fftwf_complex *_sVals_split;
    fftwf_complex *_convolution_split;
    fftwf_complex *_distances_f_split;
    fftwf_complex *_sVals_f_split;
    fftwf_complex *_convolution_f_split;
    fftwf_plan _p_distances_fftw;
    fftwf_plan _p_sVals_fftw;
    fftwf_plan _p_inv_fftw;

    // Data (OpenCL)
    std::vector<cl::Buffer> _states_cl;
    cl::Buffer _sumFootprintAMPA_cl;
    cl::Buffer _sumFootprintNMDA_cl;
    cl::Buffer _sumFootprintGABAA_cl;

    // OpenCL_FFT
    std::unique_ptr<float[]> _distances_real;
    std::unique_ptr<float[]> _sVals_real;
    std::unique_ptr<float[]> _convolution_real;
    std::unique_ptr<float[]> _zeros;
    cl::Buffer _distances_real_cl;
    cl::Buffer _distances_imag_cl;
    cl::Buffer _sVals_real_cl;
    cl::Buffer _sVals_imag_cl;
    cl::Buffer _convolution_real_cl;
    cl::Buffer _convolution_imag_cl;
    cl::Buffer _distances_f_real_cl;
    cl::Buffer _distances_f_imag_cl;
    cl::Buffer _sVals_f_real_cl;
    cl::Buffer _sVals_f_imag_cl;
    cl::Buffer _convolution_f_real_cl;
    cl::Buffer _convolution_f_imag_cl;

    clFFT_Plan _p_cl;
    cl::Kernel _kernel_convolution;
    cl::Kernel _kernel_prepareFFT_AMPA;
    cl::Kernel _kernel_prepareFFT_NMDA;
    cl::Kernel _kernel_prepareFFT_GABAA;
    cl::Kernel _kernel_postConvolution_AMPA;
    cl::Kernel _kernel_postConvolution_NMDA;
    cl::Kernel _kernel_postConvolution_GABAA;

    // Kernels
    cl::Kernel _kernel_f_dV_dt;
    cl::Kernel _kernel_f_dn_dt;
    cl::Kernel _kernel_f_I_Na_dh_dt;
    cl::Kernel _kernel_f_dz_dt;
    cl::Kernel _kernel_f_dsAMPA_dt;
    cl::Kernel _kernel_f_dxNMDA_dt;
    cl::Kernel _kernel_f_dsNMDA_dt;

    void initializeFFTW();
    void initializeHostVariables(state const& state_0);
    void initializeClFFT();
    void initializeCLKernelsAndBuffers();

    void convolutionFFTW();
    void convolutionClFFT();

    void f_I_FFT_fftw(const Receptor rec);
    void f_I_FFT_clFFT(const Receptor rec);

    void executeKernels();

    void assertConvolutionResults();
    void assertInitializationResults();

    static inline float _f_w_EE(const float j);
};
