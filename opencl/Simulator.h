#pragma once

#include "Definitions.h"
#include "CLWrapper.h"
#include "Plotter.h"

#include "opencl_fft/clFFT.h"
#include "fftw3.h"

#include <boost/filesystem.hpp>

class Simulator
{
public:
    enum Plot {
        NO_PLOT = 0,
        PLOT
    };

    enum Measure {
        NO_MEASURE = 0,
        MEASURE
    };

    enum FFT_FFTW {
        NO_FFTW = 0,
        FFTW
    };

    enum FFT_clFFT {
        NO_CLFFT = 0,
        CLFFT
    };

    Simulator(const unsigned int numNeurons,
              const unsigned int timesteps,
              const float dt,
              state const& state_0,
              const Plot plot,
              const Measure measure,
              const FFT_FFTW fftw,
              const FFT_clFFT clfft,
              boost::filesystem::path const& programPath,
              Logger const& logger);

    void step();
    void simulate();

    std::vector<unsigned long> getTimesCalculations() const;
    std::vector<unsigned long> getTimesFFTW() const;
    std::vector<unsigned long> getTimesClFFT() const;

private:
    enum Receptor {
        AMPA = 0,
        NMDA,
        GABAA
    };

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
    bool _fftw;
    bool _clfft;

    // Measurements
    std::vector<unsigned long> _timesCalculations;
    std::vector<unsigned long> _timesFFTW;
    std::vector<unsigned long> _timesClFFT;

    // FFT variables
    unsigned int _nFFT;
    float _scaleFFT;

    // Data
    std::unique_ptr<state[]> _states;
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
    cl::Buffer _states_cl;
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

    // Kernels
    cl::Kernel _kernel_f_dV_dt;
    cl::Kernel _kernel_f_dn_dt;
    cl::Kernel _kernel_f_I_Na_dh_dt;
    cl::Kernel _kernel_f_dz_dt;
    cl::Kernel _kernel_f_dsAMPA_dt;
    cl::Kernel _kernel_f_dxNMDA_dt;
    cl::Kernel _kernel_f_dsNMDA_dt;

    void handleClError(cl_int err);

    static inline float _f_w_EE(const int j);

    void f_I_FFT_fftw(const unsigned int ind_old, const Receptor rec);
    void f_I_FFT_clFFT(const unsigned int ind_old, const Receptor rec);
};
