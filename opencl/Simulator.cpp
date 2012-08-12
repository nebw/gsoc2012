#include "stdafx.h"

#include "Simulator.h"
#include "util.h"

#include "CL/cl.hpp"

#include <cassert>
#include <numeric>

#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
#include <Windows.h>
#endif

Simulator::Simulator(const unsigned int numNeurons,
                     const unsigned int timesteps,
                     const float dt,
                     state const& state_0,
                     const Plot plot,
                     const Measure measure,
                     const FFT_FFTW fftw,
                     const FFT_clFFT clfft,
                     boost::filesystem3::path const& programPath,
                     Logger const& logger)
    : _wrapper(CLWrapper()),
      _plotter(Plotter(numNeurons, 0, dt)),
      _numNeurons(numNeurons),
      _timesteps(timesteps),
      _dt(dt),
      _state_0(_state_0),
      _t(0),
      _plot(plot == PLOT),
      _measure(measure == MEASURE),
      _fftw(fftw == FFTW),
      _clfft(clfft == CLFFT),
      _logger(logger),
      //TODO: _nFFT(2 * numNeurons - 1),
      _nFFT(2 * numNeurons),
      _scaleFFT(1.f / _nFFT),
      _err(CL_SUCCESS)
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
    _states = std::unique_ptr<state[]>(new state[2 * numNeurons]);

    _sumFootprintAMPA = std::unique_ptr<float[]>(new float[numNeurons]);
    _sumFootprintNMDA = std::unique_ptr<float[]>(new float[numNeurons]);
    _sumFootprintGABAA = std::unique_ptr<float[]>(new float[numNeurons]);

    if(_fftw)
    {
        _distances_split = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
        _sVals_split = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
        _convolution_split = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
        _distances_f_split = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
        _sVals_f_split = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
        _convolution_f_split = (fftwf_complex *)fftwf_malloc(_nFFT * sizeof(fftwf_complex));
        _p_distances_fftw = fftwf_plan_dft_1d(_nFFT, _distances_split, _distances_f_split, FFTW_FORWARD, FFTW_ESTIMATE);
        _p_sVals_fftw = fftwf_plan_dft_1d(_nFFT, _sVals_split, _sVals_f_split, FFTW_FORWARD, FFTW_ESTIMATE);
        _p_inv_fftw = fftwf_plan_dft_1d(_nFFT, _convolution_f_split, _convolution_split, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    if(_clfft)
    {
        assert(isPowerOfTwo(_nFFT));
        clFFT_Dim3 n = { _nFFT, 1, 1 };
        clFFT_DataFormat dataFormat = clFFT_SplitComplexFormat;
        clFFT_Dimension dim = clFFT_1D;
        _p_cl = clFFT_CreatePlan(_wrapper.getContextC(), n, dim, dataFormat, &_err);
        handleClError(_err);

        _distances_real = std::unique_ptr<float[]>(new float[_nFFT]);
        _sVals_real = std::unique_ptr<float[]>(new float[_nFFT]);
        _convolution_real = std::unique_ptr<float[]>(new float[_nFFT]);
        _zeros = std::unique_ptr<float[]>(new float[_nFFT]);
    }

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

    if(_fftw)
    {
        // initialize distances
        unsigned int j = 0;

        for (unsigned int i = numNeurons - 1; i > 0; --i)
        {
            _distances_split[j][0] = _f_w_EE(i);
            _distances_split[j][1] = 0;
            ++j;
        }

        for (unsigned int i = 0; i < numNeurons; ++i)
        {
            _distances_split[j][0] = _f_w_EE(i);
            _distances_split[j][1] = 0;
            ++j;
        }

        _distances_split[j][0] = 0;
        _distances_split[j][1] = 0;

        fftwf_execute(_p_distances_fftw);
    }

    if(_clfft)
    {
        // initialize distances
        unsigned int j = 0;

        for(unsigned int i = numNeurons - 1; i > 0; --i)
        {
            _distances_real[j] = _f_w_EE(i);
            ++j;
        }

        for(unsigned int i = 0; i < numNeurons; ++i)
        {
            _distances_real[j] = _f_w_EE(i);
            ++j;
        }

        _distances_real[j] = 0;

        for(unsigned int i = 0; i < _nFFT; ++i)
        {
            _zeros[i] = 0;
        }

        _distances_real_cl = cl::Buffer(_wrapper.getContext(),
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        _nFFT * sizeof(float),
                                        _distances_real.get(),
                                        &_err);
        _distances_imag_cl = cl::Buffer(_wrapper.getContext(),
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        _nFFT * sizeof(float),
                                        _zeros.get(),
                                        &_err);
        _sVals_real_cl = cl::Buffer(_wrapper.getContext(),
                                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    _nFFT * sizeof(float),
                                    _zeros.get(),
                                    &_err);
        _sVals_imag_cl = cl::Buffer(_wrapper.getContext(),
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    _nFFT * sizeof(float),
                                    _zeros.get(),
                                    &_err);
        _convolution_real_cl = cl::Buffer(_wrapper.getContext(),
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          _nFFT * sizeof(float),
                                          _zeros.get(),
                                          &_err);
        _convolution_imag_cl = cl::Buffer(_wrapper.getContext(),
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          _nFFT * sizeof(float),
                                          _zeros.get(),
                                          &_err);
        _distances_f_real_cl = cl::Buffer(_wrapper.getContext(),
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          _nFFT * sizeof(float),
                                          _zeros.get(),
                                          &_err);
        _distances_f_imag_cl = cl::Buffer(_wrapper.getContext(),
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          _nFFT * sizeof(float),
                                          _zeros.get(),
                                          &_err);
        _sVals_f_real_cl = cl::Buffer(_wrapper.getContext(),
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      _nFFT * sizeof(float),
                                      _zeros.get(),
                                      &_err);
        _sVals_f_imag_cl = cl::Buffer(_wrapper.getContext(),
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      _nFFT * sizeof(float),
                                      _zeros.get(),
                                      &_err);
        _convolution_f_real_cl = cl::Buffer(_wrapper.getContext(),
                                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                            _nFFT * sizeof(float),
                                            _zeros.get(),
                                            &_err);
        _convolution_f_imag_cl = cl::Buffer(_wrapper.getContext(),
                                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                            _nFFT * sizeof(float),
                                            _zeros.get(),
                                            &_err);

        _kernel_convolution = cl::Kernel(_program, "convolution", &_err);
        handleClError(_kernel_convolution.setArg(0, _convolution_f_real_cl));
        handleClError(_kernel_convolution.setArg(1, _convolution_f_imag_cl));
        handleClError(_kernel_convolution.setArg(2, _distances_f_real_cl));
        handleClError(_kernel_convolution.setArg(3, _distances_f_imag_cl));
        handleClError(_kernel_convolution.setArg(4, _sVals_f_real_cl));
        handleClError(_kernel_convolution.setArg(5, _sVals_f_imag_cl));
        handleClError(_kernel_convolution.setArg(6, _scaleFFT));

        handleClError(clFFT_ExecutePlannar(_wrapper.getQueueC(),
            _p_cl,
            1,
            clFFT_Forward,
            _distances_real_cl(),
            _distances_imag_cl(),
            _distances_f_real_cl(),
            _distances_f_imag_cl(),
            0,
            NULL,
            NULL));

        _wrapper.getQueue().finish();
    }

    if(_fftw && _clfft)
    {
        boost::scoped_array<float> distances_real(new float[_nFFT]);
        boost::scoped_array<float> distances_imag(new float[_nFFT]);
        boost::scoped_array<float> distances_f_real(new float[_nFFT]);
        boost::scoped_array<float> distances_f_imag(new float[_nFFT]);

        _err = _wrapper.getQueue().enqueueReadBuffer(_distances_real_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_real.get(), NULL, &_event);
        _err = _wrapper.getQueue().enqueueReadBuffer(_distances_imag_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_imag.get(), NULL, &_event);
        _err = _wrapper.getQueue().enqueueReadBuffer(_distances_f_real_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_f_real.get(), NULL, &_event);
        _err = _wrapper.getQueue().enqueueReadBuffer(_distances_f_imag_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_f_imag.get(), NULL, &_event);

        for(unsigned int i = 0; i < _nFFT; ++i)
        {
            assertAlmostEquals(_distances_split[i][0], distances_real[i]);
            assertAlmostEquals(_distances_split[i][1], distances_imag[i]);
            assertNear(_distances_f_split[i][0], distances_f_real[i], 0.000001);
            assertNear(_distances_f_split[i][1], distances_f_imag[i], 0.000001);
        }
    }

    // opencl initialization
    _states_cl = cl::Buffer(_wrapper.getContext(),
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            2 * numNeurons * sizeof(state),
                            _states.get(),
                            &_err);
    _sumFootprintAMPA_cl = cl::Buffer(_wrapper.getContext(),
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      numNeurons * sizeof(float),
                                      _sumFootprintAMPA.get(),
                                      &_err);
    _sumFootprintNMDA_cl = cl::Buffer(_wrapper.getContext(),
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      numNeurons * sizeof(float),
                                      _sumFootprintNMDA.get(),
                                      &_err);
    _sumFootprintGABAA_cl = cl::Buffer(_wrapper.getContext(),
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      numNeurons * sizeof(float),
                                      _sumFootprintGABAA.get(),
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
    _kernel_prepareFFT_AMPA = cl::Kernel(_program, "prepareFFT_AMPA", &_err);
    handleClError(_kernel_prepareFFT_AMPA.setArg(0, _states_cl));
    handleClError(_kernel_prepareFFT_AMPA.setArg(1, _sVals_real_cl));
    handleClError(_kernel_prepareFFT_AMPA.setArg(2, _numNeurons));

    _kernel_prepareFFT_NMDA = cl::Kernel(_program, "prepareFFT_NMDA", &_err);
    handleClError(_kernel_prepareFFT_NMDA.setArg(0, _states_cl));
    handleClError(_kernel_prepareFFT_NMDA.setArg(1, _sVals_real_cl));
    handleClError(_kernel_prepareFFT_NMDA.setArg(2, _numNeurons));

    _kernel_prepareFFT_GABAA = cl::Kernel(_program, "prepareFFT_GABAA", &_err);
    handleClError(_kernel_prepareFFT_GABAA.setArg(0, _states_cl));
    handleClError(_kernel_prepareFFT_GABAA.setArg(1, _sVals_real_cl));
    handleClError(_kernel_prepareFFT_GABAA.setArg(2, _numNeurons));

    handleClError(_kernel_f_dV_dt.setArg(0, _states_cl));
    handleClError(_kernel_f_dV_dt.setArg(1, _sumFootprintAMPA_cl));
    handleClError(_kernel_f_dV_dt.setArg(2, _sumFootprintNMDA_cl));
    handleClError(_kernel_f_dV_dt.setArg(3, _sumFootprintGABAA_cl));
    handleClError(_kernel_f_dV_dt.setArg(4, numNeurons));
    handleClError(_kernel_f_dV_dt.setArg(6, dt));

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
    
    std::unique_ptr<float[]> sumFootPrintAMPA_tmp;
    std::unique_ptr<float[]> sumFootPrintNMDA_tmp;

    if(_fftw)
    {
        unsigned long startTime;
        if(_measure)
        {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
            timeBeginPeriod(1);
            startTime = timeGetTime();
#endif
        }

        f_I_FFT_fftw(ind_old, AMPA);
        _err = _wrapper.getQueue().enqueueWriteBuffer(_sumFootprintAMPA_cl, CL_FALSE, 0, _numNeurons * sizeof(float), _sumFootprintAMPA.get(), NULL, &_event);
        f_I_FFT_fftw(ind_old, NMDA);
        _err = _wrapper.getQueue().enqueueWriteBuffer(_sumFootprintNMDA_cl, CL_TRUE, 0, _numNeurons * sizeof(float), _sumFootprintNMDA.get(), NULL, &_event);
        //f_I_FFT(ind_old, "GABAA");

        if(_measure)
        {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
            unsigned long elapsedTime = timeGetTime() - startTime;
            _timesFFTW.push_back(elapsedTime);
            timeEndPeriod(1);
#endif
        }
    }

    if(_fftw && _clfft) {
        sumFootPrintAMPA_tmp = std::unique_ptr<float[]>(new float[_numNeurons]);
        sumFootPrintNMDA_tmp = std::unique_ptr<float[]>(new float[_numNeurons]);
        
        for(unsigned int i = 0; i < _numNeurons; ++i) {
            sumFootPrintAMPA_tmp[i] = _sumFootprintAMPA[i];
            sumFootPrintNMDA_tmp[i] = _sumFootprintNMDA[i];
        }
    }

    if(_clfft)
    {
        unsigned long startTime;
        if(_measure)
        {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
            timeBeginPeriod(1);
            startTime = timeGetTime();
#endif
        }

        f_I_FFT_clFFT(ind_old, AMPA);
        _err = _wrapper.getQueue().enqueueWriteBuffer(_sumFootprintAMPA_cl, CL_FALSE, 0, _numNeurons * sizeof(float), _sumFootprintAMPA.get(), NULL, &_event);
        f_I_FFT_clFFT(ind_old, NMDA);
        _err = _wrapper.getQueue().enqueueWriteBuffer(_sumFootprintNMDA_cl, CL_TRUE, 0, _numNeurons * sizeof(float), _sumFootprintNMDA.get(), NULL, &_event);

        if(_measure)
        {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
            unsigned long elapsedTime = timeGetTime() - startTime;
            _timesClFFT.push_back(elapsedTime);
            timeEndPeriod(1);
#endif
        }
    }

    if(_fftw && _clfft)
    {
        for(unsigned int i = 0; i < _numNeurons; ++i)
        {
            assertNear(_sumFootprintAMPA[i], sumFootPrintAMPA_tmp[i], 0.05);
            assertNear(_sumFootprintNMDA[i], sumFootPrintNMDA_tmp[i], 0.05);
        }
    }
    
    unsigned long startTime;
    if(_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        timeBeginPeriod(1);
        startTime = timeGetTime();
#endif
    }
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dV_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dn_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_I_Na_dh_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dz_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dsAMPA_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dxNMDA_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dsNMDA_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);

    _wrapper.getQueue().finish();
    if(_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        unsigned long elapsedTime = timeGetTime() - startTime;
        _timesCalculations.push_back(elapsedTime);
        timeEndPeriod(1);
#endif
    }

    _err = _wrapper.getQueue().enqueueReadBuffer(_states_cl, CL_TRUE, ind_new * _numNeurons * sizeof(state), _numNeurons * sizeof(state), &_states[ind_new * _numNeurons], NULL, &_event);
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

    //if(_t == 0)
    //{
    //    LOG_INFO(*_logger) << "Timestep 1/" << _timesteps;
    //}

    for(_t; _t < _timesteps - 1; ++_t)
    {
        if((_t + 2) % (_timesteps / 100) == 0)
        {
            std::cout << ".";
            //LOG_INFO(*_logger) << "Timestep " << _t + 2 << "/" << _timesteps;
        }

        step();

        if(_plot)
        {
            unsigned int ind_old = _t % 2;
            unsigned int ind_new = 1 - ind_old;

            _plotter.step(&_states[ind_new * _numNeurons], _t, _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA);
        }
    }

    std::cout << std::endl;;

    if(_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        unsigned long elapsedTime = timeGetTime() - startTime;
        LOG_INFO(*_logger) << "Execution time: " << elapsedTime / 1000.0 << "s";
        timeEndPeriod(1);
#endif

        if(_fftw)
        {
            double avgTimeFFTW = std::accumulate(_timesFFTW.begin(), _timesFFTW.end(), 0.0) / _timesFFTW.size();
            LOG_INFO(*_logger) << "Average execution time FFTW: " << avgTimeFFTW << "ms";
        }
        if(_clfft)
        {
            double avgTimeClFFT = std::accumulate(_timesClFFT.begin(), _timesClFFT.end(), 0.0) / _timesClFFT.size();
            LOG_INFO(*_logger) << "Average execution time clFFT: " << avgTimeClFFT << "ms";
        }
        double avgTimeCalculations = std::accumulate(_timesCalculations.begin(), _timesCalculations.end(), 0.0) / _timesCalculations.size();
        LOG_INFO(*_logger) << "Average execution time calculations: " << avgTimeCalculations << "ms";
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

void Simulator::f_I_FFT_fftw(const unsigned int ind_old, const Receptor rec)
{
    for (unsigned int i = 0; i < _numNeurons; ++i)   {
        if(rec == AMPA)
        {
            _sVals_split[i][0] = _states[ind_old*_numNeurons+i].s_AMPA;
        } else if(rec == NMDA)
        {
            _sVals_split[i][0] = _states[ind_old*_numNeurons+i].s_NMDA;
        } else if(rec == GABAA)
        {
            _sVals_split[i][0] = _states[ind_old*_numNeurons+i].s_GABAA;
        }
        _sVals_split[i][1] = 0;
    }

    for (unsigned int i = _numNeurons; i < _nFFT; ++i)
    {
        _sVals_split[i][0] = 0;
        _sVals_split[i][1] = 0;
    }

    fftwf_execute(_p_sVals_fftw);

    // convolution in frequency domain
    for (unsigned int i = 0; i < _nFFT; ++i)
    {
        _convolution_f_split[i][0] = (_distances_f_split[i][0] * _sVals_f_split[i][0]
                                - _distances_f_split[i][1] * _sVals_f_split[i][1]) * _scaleFFT;
        _convolution_f_split[i][1] = (_distances_f_split[i][0] * _sVals_f_split[i][1]
                                + _distances_f_split[i][1] * _sVals_f_split[i][0]) * _scaleFFT;
    }

    fftwf_execute(_p_inv_fftw);

    for (unsigned int indexOfNeuron = 0; indexOfNeuron < _numNeurons; ++indexOfNeuron)
    {
        if(rec == AMPA)
        {
            _sumFootprintAMPA[indexOfNeuron] = _convolution_split[indexOfNeuron+_numNeurons-1][0];
        } else if(rec == NMDA)
        {
            _sumFootprintNMDA[indexOfNeuron] = _convolution_split[indexOfNeuron+_numNeurons-1][0];
        } else if(rec == GABAA)
        {
            _sumFootprintGABAA[indexOfNeuron] = _convolution_split[indexOfNeuron+_numNeurons-1][0];
        }
    }
}

void Simulator::f_I_FFT_clFFT(const unsigned int ind_old, const Receptor rec)
{
    try 
    {
        switch (rec)
        {
        case AMPA:
            handleClError(_kernel_prepareFFT_AMPA.setArg(3, ind_old));
            _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_prepareFFT_AMPA, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
    	    break;
        case NMDA:
            handleClError(_kernel_prepareFFT_NMDA.setArg(3, ind_old));
            _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_prepareFFT_NMDA, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
            break;
        case GABAA:
            handleClError(_kernel_prepareFFT_GABAA.setArg(3, ind_old));
            _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_prepareFFT_GABAA, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, &_event);
            break;
        }

        _wrapper.getQueue().finish();

        handleClError(clFFT_ExecutePlannar(_wrapper.getQueueC(),
            _p_cl,
            1,
            clFFT_Forward,
            _sVals_real_cl(),
            _sVals_imag_cl(),
            _sVals_f_real_cl(),
            _sVals_f_imag_cl(),
            0,
            NULL,
            NULL));

        _wrapper.getQueue().finish();

        _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_convolution, cl::NullRange, cl::NDRange(_nFFT), cl::NullRange, NULL, &_event);

        _wrapper.getQueue().finish();

        handleClError(clFFT_ExecutePlannar(_wrapper.getQueueC(),
            _p_cl,
            1,
            clFFT_Inverse,
            _convolution_f_real_cl(),
            _convolution_f_imag_cl(),
            _convolution_real_cl(),
            _convolution_imag_cl(),
            0,
            NULL,
            NULL));

        _wrapper.getQueue().finish();

        _err = _wrapper.getQueue().enqueueReadBuffer(_convolution_real_cl, CL_TRUE, 0, _nFFT * sizeof(float), _convolution_real.get(), NULL, &_event);

        for(unsigned int indexOfNeuron = 0; indexOfNeuron < _numNeurons; ++indexOfNeuron)
        {
            if(rec == AMPA)
            {
                _sumFootprintAMPA[indexOfNeuron] = _convolution_real[indexOfNeuron+_numNeurons-1];
            } else if(rec == NMDA)
            {
                _sumFootprintNMDA[indexOfNeuron] = _convolution_real[indexOfNeuron+_numNeurons-1];
            } else if(rec == GABAA)
            {
                _sumFootprintGABAA[indexOfNeuron] = _convolution_real[indexOfNeuron+_numNeurons-1];
            }
        }
    }
    catch (cl::Error err) 
    {
        handleClError(err);
    }

    //if(_clfft && _fftw)
    //{
    //    boost::scoped_array<float> distances_real(new float[_nFFT]);
    //    boost::scoped_array<float> distances_imag(new float[_nFFT]);
    //    boost::scoped_array<float> distances_f_real(new float[_nFFT]);
    //    boost::scoped_array<float> distances_f_imag(new float[_nFFT]);
    //    boost::scoped_array<float> sVals_real(new float[_nFFT]);
    //    boost::scoped_array<float> sVals_imag(new float[_nFFT]);
    //    boost::scoped_array<float> sVals_f_real(new float[_nFFT]);
    //    boost::scoped_array<float> sVals_f_imag(new float[_nFFT]);
    //    boost::scoped_array<float> convolution_real(new float[_nFFT]);
    //    boost::scoped_array<float> convolution_imag(new float[_nFFT]);
    //    boost::scoped_array<float> convolution_f_real(new float[_nFFT]);
    //    boost::scoped_array<float> convolution_f_imag(new float[_nFFT]);

    //    _err = _wrapper.getQueue().enqueueReadBuffer(_distances_real_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_real.get(), NULL, &_event);
    //    _err = _wrapper.getQueue().enqueueReadBuffer(_distances_imag_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_imag.get(), NULL, &_event);
    //    _err = _wrapper.getQueue().enqueueReadBuffer(_distances_f_real_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_f_real.get(), NULL, &_event);
    //    _err = _wrapper.getQueue().enqueueReadBuffer(_distances_f_imag_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_f_imag.get(), NULL, &_event);
    //    _err = _wrapper.getQueue().enqueueReadBuffer(_sVals_real_cl, CL_TRUE, 0, _nFFT * sizeof(float), sVals_real.get(), NULL, &_event);
    //    _err = _wrapper.getQueue().enqueueReadBuffer(_sVals_imag_cl, CL_TRUE, 0, _nFFT * sizeof(float), sVals_imag.get(), NULL, &_event);
    //    _err = _wrapper.getQueue().enqueueReadBuffer(_sVals_f_real_cl, CL_TRUE, 0, _nFFT * sizeof(float), sVals_f_real.get(), NULL, &_event);
    //    _err = _wrapper.getQueue().enqueueReadBuffer(_sVals_f_imag_cl, CL_TRUE, 0, _nFFT * sizeof(float), sVals_f_imag.get(), NULL, &_event);
    //    _err = _wrapper.getQueue().enqueueReadBuffer(_convolution_real_cl, CL_TRUE, 0, _nFFT * sizeof(float), convolution_real.get(), NULL, &_event);
    //    _err = _wrapper.getQueue().enqueueReadBuffer(_convolution_imag_cl, CL_TRUE, 0, _nFFT * sizeof(float), convolution_imag.get(), NULL, &_event);
    //    _err = _wrapper.getQueue().enqueueReadBuffer(_convolution_f_real_cl, CL_TRUE, 0, _nFFT * sizeof(float), convolution_f_real.get(), NULL, &_event);
    //    _err = _wrapper.getQueue().enqueueReadBuffer(_convolution_f_imag_cl, CL_TRUE, 0, _nFFT * sizeof(float), convolution_f_imag.get(), NULL, &_event);

    //    for(unsigned int i = 0; i < _nFFT; ++i) {
    //        /*assertNear(_distances_split[i][0], distances_real[i], 0.000001);
    //        assertNear(_distances_split[i][1], distances_imag[i], 0.000001);
    //        assertNear(_distances_f_split[i][0], distances_f_real[i], 0.000001);
    //        assertNear(_distances_f_split[i][1], distances_f_imag[i], 0.000001);
    //        assertNear(_sVals_split[i][0], sVals_real[i], 0.000001);
    //        assertNear(_sVals_split[i][1], sVals_imag[i], 0.000001);
    //        assertNear(_sVals_f_split[i][0], sVals_f_real[i], 0.0001);
    //        assertNear(_sVals_f_split[i][1], sVals_f_imag[i], 0.0001);
    //        assertNear(_convolution_split[i][0], convolution_real[i], 0.000001);
    //        assertNear(_convolution_split[i][1], convolution_imag[i], 0.000001);
    //        assertNear(_convolution_f_split[i][0], convolution_f_real[i], 0.000001);
    //        assertNear(_convolution_f_split[i][1], convolution_f_imag[i], 0.000001);*/
    //    }
    //}
}

void Simulator::handleClError(cl_int err)
{
    if (err)
    {
        handleClError(cl::Error(err));
    }
}

void Simulator::handleClError(cl::Error err)
{
    std::cout << "OpenCL Error: " << err.what() << " " << oclErrorString(err.err()) << std::endl;
    getchar();
    throw err;
}

std::vector<unsigned long> Simulator::getTimesCalculations() const
{
    assert(_measure);
    return _timesCalculations;
}

std::vector<unsigned long> Simulator::getTimesFFTW() const
{
    assert(_measure && _fftw);
    return _timesFFTW;
}

std::vector<unsigned long> Simulator::getTimesClFFT() const
{
    assert(_measure && _clfft);
    return _timesClFFT;
}