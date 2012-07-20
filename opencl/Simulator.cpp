#include "stdafx.h"

#include "Simulator.h"

#include "CL/cl.hpp"

#include <boost/foreach.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
#include <Windows.h>
#endif

Simulator::Simulator(const unsigned int numNeurons,
                     const unsigned int timesteps,
                     const float dt,
                     state const& state_0,
                     const bool plot,
                     const bool measure,
                     boost::filesystem3::path const& programPath,
                     Logger const& logger)
    : _wrapper(CLWrapper()),
      _plotter(Plotter(numNeurons, 0, dt)),
      _numNeurons(numNeurons),
      _timesteps(timesteps),
      _dt(dt),
      _state_0(_state_0),
      _t(0),
      _plot(plot),
      _measure(measure),
      _logger(logger),
      _nFFT(2 * numNeurons - 1),
      _scaleFFT(1.f / _nFFT)
{
    _program = _wrapper.loadProgram(programPath.string());

    LOG_INFO(*logger) << "Configuration: ";
    LOG_INFO(*logger) << "numNeurons: " << _numNeurons;
    LOG_INFO(*logger) << "timesteps: " << _timesteps;
    LOG_INFO(*logger) << "dt: " << _dt;
    LOG_INFO(*logger) << "V_0: " << state_0.V;
    LOG_INFO(*logger) << "h_0: " << state_0.h;
    LOG_INFO(*logger) << "n_0: " << state_0.n;
    LOG_INFO(*logger) << "z_0: " << state_0.z;
    LOG_INFO(*logger) << "sAMPA_0: " << state_0.s_AMPA;
    LOG_INFO(*logger) << "xNMDA_0: " << state_0.x_NMDA;
    LOG_INFO(*logger) << "sNMDA_0: " << state_0.s_NMDA;
    LOG_INFO(*logger) << "sGABAA_0: " << state_0.s_GABAA;
    LOG_INFO(*logger) << "IApp: " << state_0.I_app;

    // 2 states (old and new) per neuron per timestep
    _states = (state *)malloc(2 * numNeurons * sizeof(state));

    _sumFootprintAMPA = (float *)malloc(numNeurons * sizeof(float));
    _sumFootprintNMDA = (float *)malloc(numNeurons * sizeof(float));
    _sumFootprintGABAA = (float *)malloc(numNeurons * sizeof(float));

    _distances = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
    _sVals = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
    _convolution = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
    _distances_f = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
    _sVals_f = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
    _convolution_f = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
    _p_distances = fftwf_plan_dft_1d(_nFFT, _distances, _distances_f, FFTW_FORWARD, FFTW_ESTIMATE);
    _p_sVals = fftwf_plan_dft_1d(_nFFT, _sVals, _sVals_f, FFTW_FORWARD, FFTW_ESTIMATE);
    _p_inv = fftwf_plan_dft_1d(_nFFT, _convolution_f, _convolution, FFTW_BACKWARD, FFTW_ESTIMATE);

    // initialize initial states
    for (unsigned int i = 0; i < numNeurons; ++i)
    {
        _states[i] = state_0;
        _sumFootprintAMPA[i] = 0;
        _sumFootprintNMDA[i] = 0;
        _sumFootprintGABAA[i] = 0;
    }

    for (unsigned int i = numNeurons; i < 2 * numNeurons; ++i)
    {
        _states[i] = state_0;
    }

    // initialize distances
    unsigned int j = 0;

    for (unsigned int i = numNeurons - 1; i > 0; --i)
    {
        _distances[j][0] = _f_w_EE(i);
        _distances[j][1] = 0;
        ++j;
    }

    for (unsigned int i = 0; i < numNeurons; ++i)
    {
        _distances[j][0] = _f_w_EE(i);
        _distances[j][1] = 0;
        ++j;
    }

    // opencl initialization
    _states_cl = cl::Buffer(_wrapper.getContext(),
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            2 * numNeurons * sizeof(state),
                            _states,
                            &_err);
    _sumFootprintAMPA_cl = cl::Buffer(_wrapper.getContext(),
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      numNeurons * sizeof(float),
                                      _sumFootprintAMPA,
                                      &_err);
    _sumFootprintNMDA_cl = cl::Buffer(_wrapper.getContext(),
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      numNeurons * sizeof(float),
                                      _sumFootprintNMDA,
                                      &_err);
    _sumFootprintGABAA_cl = cl::Buffer(_wrapper.getContext(),
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      numNeurons * sizeof(float),
                                      _sumFootprintGABAA,
                                      &_err);

    _kernel_f_dV_dt = cl::Kernel(_program, "f_dV_dt", &_err);

    _kernel_f_dn_dt = cl::Kernel(_program, "f_dn_dt", &_err);
    _kernel_f_I_Na_dh_dt = cl::Kernel(_program, "f_I_Na_dh_dt", &_err);
    _kernel_f_dz_dt = cl::Kernel(_program, "f_dz_dt", &_err);
    _kernel_f_dsAMPA_dt = cl::Kernel(_program, "f_dsAMPA_dt", &_err);
    _kernel_f_dxNMDA_dt = cl::Kernel(_program, "f_dxNMDA_dt", &_err);
    _kernel_f_dsNMDA_dt = cl::Kernel(_program, "f_dsNMDA_dt", &_err);

    cl::Kernel kernels[6] = { 
        _kernel_f_dn_dt, 
        _kernel_f_I_Na_dh_dt,
        _kernel_f_dz_dt, 
        _kernel_f_dsAMPA_dt, 
        _kernel_f_dxNMDA_dt, 
        _kernel_f_dsNMDA_dt
    };

    // set constant kernel arguments
    _err = _kernel_f_dV_dt.setArg(0, _states_cl);
    _err = _kernel_f_dV_dt.setArg(1, _sumFootprintAMPA_cl);
    _err = _kernel_f_dV_dt.setArg(2, _sumFootprintNMDA_cl);
    _err = _kernel_f_dV_dt.setArg(3, _sumFootprintGABAA_cl);
    _err = _kernel_f_dV_dt.setArg(4, numNeurons);
    _err = _kernel_f_dV_dt.setArg(6, dt);

    BOOST_FOREACH(cl::Kernel kernel, kernels)
    {
        _err = kernel.setArg(0, _states_cl);
        _err = kernel.setArg(1, _numNeurons);
        _err = kernel.setArg(3, _dt);
    }

    if(_plot)
    {
        _plotter.step(&_states[0], _numNeurons, _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA);
    }
}

void Simulator::step() 
{
    unsigned int ind_old = _t % 2;
    unsigned int ind_new = 1 - ind_old;

    _err = _kernel_f_dV_dt.setArg(5, ind_old);
    _err = _kernel_f_dn_dt.setArg(2, ind_old);
    _err = _kernel_f_I_Na_dh_dt.setArg(2, ind_old);
    _err = _kernel_f_dz_dt.setArg(2, ind_old); 
    _err = _kernel_f_dsAMPA_dt.setArg(2, ind_old);
    _err = _kernel_f_dxNMDA_dt.setArg(2, ind_old);
    _err = _kernel_f_dsNMDA_dt.setArg(2, ind_old);
    
//    unsigned long startTime;
//    if(_measure)
//    {
//#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
//        timeBeginPeriod(1);
//        startTime = timeGetTime();
//#endif
//    }

    f_I_FFT(ind_old, "AMPA");
    _err = _wrapper.getQueue().enqueueWriteBuffer(_sumFootprintAMPA_cl, CL_FALSE, 0, _numNeurons * sizeof(float), _sumFootprintAMPA, NULL, &_event);
    f_I_FFT(ind_old, "NMDA");
    _err = _wrapper.getQueue().enqueueWriteBuffer(_sumFootprintNMDA_cl, CL_TRUE, 0, _numNeurons * sizeof(float), _sumFootprintNMDA, NULL, &_event);
    //f_I_FFT(ind_old, "GABAA");

    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dV_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dn_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_I_Na_dh_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dz_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dsAMPA_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dxNMDA_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dsNMDA_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);

    _wrapper.getQueue().finish();

    _err = _wrapper.getQueue().enqueueReadBuffer(_states_cl, CL_TRUE, ind_new * _numNeurons * sizeof(state), _numNeurons * sizeof(state), &_states[ind_new * _numNeurons], NULL, &_event);

//    if(_measure)
//    {
//#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
//        unsigned long elapsedTime = timeGetTime() - startTime;
//        LOG_INFO(*_logger) << "Execution time: " << elapsedTime / 1000.0 << "s";
//        timeEndPeriod(1);
//#endif
//    }
}

void Simulator::simulate()
{
    unsigned long startTime;
    if(_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        timeBeginPeriod(1);
        startTime = timeGetTime();
#endif
    }

    if(_t == 0)
    {
        LOG_INFO(*_logger) << "Timestep 1/" << _timesteps;
    }

    for(_t; _t < _timesteps - 1; ++_t)
    {
        if((_t + 2) % (_timesteps / 100) == 0)
        {
            LOG_INFO(*_logger) << "Timestep " << _t + 2 << "/" << _timesteps;
        }

        step();

        if(_plot)
        {
            unsigned int ind_old = _t % 2;
            unsigned int ind_new = 1 - ind_old;

            _plotter.step(&_states[ind_new * _numNeurons], _t, _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA);
        }
    }

    if(_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        unsigned long elapsedTime = timeGetTime() - startTime;
        LOG_INFO(*_logger) << "Execution time: " << elapsedTime / 1000.0 << "s";
        timeEndPeriod(1);
#endif
    }

    if(_plot)
    {
        _plotter.plot();
    }    
}

inline float Simulator::_f_w_EE(const int j)
{
    static const float sigma = 1;
    static const float p     = 32;

    // TODO: p varies between 8 to 64
    //
    return tanh(1 / (2 * sigma * p))
            * exp(-abs(j) / (sigma * p));
}

void Simulator::f_I_FFT(const unsigned int ind_old, const std::string var)
{
    for (unsigned int i = 0; i < _numNeurons; ++i)   {
        if(var == "AMPA")
        {
            _sVals[i][0] = _states[ind_old*_numNeurons+i].s_AMPA;
        } else if(var == "NMDA")
        {
            _sVals[i][0] = _states[ind_old*_numNeurons+i].s_NMDA;
        } else if(var == "GABAA")
        {
            _sVals[i][0] = _states[ind_old*_numNeurons+i].s_GABAA;
        }
        _sVals[i][1] = 0;
    }

    for (unsigned int i = _numNeurons; i < _nFFT; ++i)
    {
        _sVals[i][0] = 0;
        _sVals[i][1] = 0;
    }

    fftwf_execute(_p_distances);

    fftwf_execute(_p_sVals);

    // convolution in frequency domain
    for (unsigned int i = 0; i < _nFFT; ++i)
    {
        _convolution_f[i][0] = (_distances_f[i][0] * _sVals_f[i][0]
                                - _distances_f[i][1] * _sVals_f[i][1]) * _scaleFFT;
        _convolution_f[i][1] = (_distances_f[i][0] * _sVals_f[i][1]
                                + _distances_f[i][1] * _sVals_f[i][0]) * _scaleFFT;
    }

    fftwf_execute(_p_inv);

    for (unsigned int indexOfNeuron = 0; indexOfNeuron < _numNeurons; ++indexOfNeuron)
    {
        if(var == "AMPA")
        {
            _sumFootprintAMPA[indexOfNeuron] = _convolution[indexOfNeuron+_numNeurons-1][0];
        } else if(var == "NMDA")
        {
            _sumFootprintNMDA[indexOfNeuron] = _convolution[indexOfNeuron+_numNeurons-1][0];
        } else if(var == "GABAA")
        {
            _sumFootprintGABAA[indexOfNeuron] = _convolution[indexOfNeuron+_numNeurons-1][0];
        }
    }
}