#pragma once

#include "Definitions.h"
#include "CLWrapper.h"
#include "Plotter.h"

#include "OpenCL_FFT\clFFT.h"
#include "fftw3.h"

#include <boost/filesystem.hpp>

class Simulator
{
public:
    Simulator(const unsigned int numNeurons,
              const unsigned int timesteps,
              const float dt,
              state const& state_0,
              const bool plot,
              const bool measure,
              boost::filesystem3::path const& programPath,
              Logger const& logger);

    void step();
    void simulate();

private:
    // OpenCL
    CLWrapper _wrapper;
    cl::Program _program;
    cl::Event _event;
    cl_int _err;

    Plotter _plotter;
    Logger _logger;

    // Simulation configuration
    unsigned int _numNeurons;
    unsigned int _timesteps;
    float _dt;

    // Initial state
    state _state_0;
    unsigned int _t;

    // Configuration
    bool _plot;
    bool _measure;

    // FFT variables
    unsigned int _nFFT;
    float _scaleFFT;

    // Data
    state *_states;
    float *_sumFootprintAMPA;
    float *_sumFootprintNMDA;
    float *_sumFootprintGABAA;

    // Data (FFTW)
    fftwf_complex *_distances;
    fftwf_complex *_sVals;
    fftwf_complex *_convolution;
    fftwf_complex *_distances_f;
    fftwf_complex *_sVals_f;
    fftwf_complex *_convolution_f;
    
    fftwf_plan _p_distances;
    fftwf_plan _p_sVals;
    fftwf_plan _p_inv;

    // Data (OpenCL)
    cl::Buffer _states_cl;
    cl::Buffer _sumFootprintAMPA_cl;
    cl::Buffer _sumFootprintNMDA_cl;
    cl::Buffer _sumFootprintGABAA_cl;

    // Kernels
    cl::Kernel _kernel_f_dV_dt;
    cl::Kernel _kernel_f_dn_dt;
    cl::Kernel _kernel_f_I_Na_dh_dt;
    cl::Kernel _kernel_f_dz_dt;
    cl::Kernel _kernel_f_dsAMPA_dt;
    cl::Kernel _kernel_f_dxNMDA_dt;
    cl::Kernel _kernel_f_dsNMDA_dt;

    static inline float _f_w_EE(const int j);

    void f_I_FFT(const unsigned int ind_old, const std::string var);
};
