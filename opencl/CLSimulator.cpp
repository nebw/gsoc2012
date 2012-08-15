#include "stdafx.h"

#include "CLSimulator.h"
#include "GnuPlotPlotter.h"
#include "OpenGLPlotter.h"
#include "util.h"

#include "CL/cl.hpp"

#include <cassert>
#include <ctime>
#include <numeric>

#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
# include <Windows.h>
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)

CLSimulator::CLSimulator(const unsigned int nX,
                         const unsigned int nY,
                         const unsigned int nZ,
                         const unsigned int timesteps,
                         const float dt,
                         state const& state_0,
                         const Plot plot,
                         const Measure measure,
                         const FFT_FFTW fftw,
                         const FFT_clFFT clfft,
                         boost::filesystem::path const& programPath,
                         Logger const& logger,
                         const bool readToHostMemory /*= false*/)
    : _wrapper(CLWrapper(logger)),
      _nX(nX),
      _nY(nY),
      _nZ(nZ),
      _numNeurons(nX * nY * nZ),
      _timesteps(timesteps),
      _dt(dt),
      _state_0(state_0),
      _t(0),
      _plot(plot != NO_PLOT),
      _measure(measure == MEASURE),
      _fftw(fftw == FFTW),
      _clfft(clfft == CLFFT),
      _logger(logger),
      _readToHostMemory(readToHostMemory),
      // TODO: _nFFT(2 * numNeurons - 1),
      _nFFT(2 * nX * nY * nZ),
      _scaleFFT(1.f / _nFFT),
      _err(CL_SUCCESS)
{
    switch (plot)
    {
    case NO_PLOT:
        break;

    case PLOT_GNUPLOT:
        _plotter = std::unique_ptr<BasePlotter>(new GnuPlotPlotter(_nX, _nY, _nZ, 0, dt));
        break;

    case PLOT_OPENGL:
        _plotter = std::unique_ptr<BasePlotter>(new OpenGLPlotter(_numNeurons, 0, dt));
        break;
    }

    _program = _wrapper.loadProgram(programPath.string());
    LOG_INFO(*logger) << "Configuration: ";
    LOG_INFO(*logger) << "numNeurons: " << _numNeurons;
    LOG_INFO(*logger) << "nX: " << _nX;
    LOG_INFO(*logger) << "nY: " << _nY;
    LOG_INFO(*logger) << "nZ: " << _nZ;
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

    initializeHostVariables(state_0);

    if (_fftw)
    {
        initializeFFTW();
    }

    if (_clfft)
    {
        try
        {
            initializeClFFT();
        }
        catch (cl::Error err)
        {
            handleClError(err);
        }
    }

    if (_fftw && _clfft)
    {
        assertInitializationResults();
    }

    initializeCLKernelsAndBuffers();

    if (_plot)
    {
        _plotter->step(&_states[0], _numNeurons, _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA);
    }
}

void CLSimulator::step()
{
    unsigned int ind_old = _t % 2;
    unsigned int ind_new = 1 - ind_old;

    try
    {
        // make sure that enqueueReadBuffer from last timestep has finished if gnuplot is enabled
        if (_plot)
        {
            _err = _wrapper.getQueue().finish();
        }

        // set dynamic kernel args
        _err = _kernel_f_dV_dt.setArg(5, ind_old);
        _err = _kernel_f_dn_dt.setArg(2, ind_old);
        _err = _kernel_f_I_Na_dh_dt.setArg(2, ind_old);
        _err = _kernel_f_dz_dt.setArg(2, ind_old);
        _err = _kernel_f_dsAMPA_dt.setArg(2, ind_old);
        _err = _kernel_f_dxNMDA_dt.setArg(2, ind_old);
        _err = _kernel_f_dsNMDA_dt.setArg(2, ind_old);

        // compute convolution
        if (_fftw)
        {
            convolutionFFTW(ind_old);
        }

        if (_clfft)
        {
            convolutionClFFT(ind_old);
        }

        // compare results of fftw with clfft if both are enabled
        if (_fftw && _clfft)
        {
            assertConvolutionResults();
        }

        // execute opencl kernels for runge-kutta approximations
        executeKernels();

        // read states from GPU memory for gnuplot plotting or unit tests
        if (_plot || _readToHostMemory)
        {
            _err = _wrapper.getQueue().enqueueReadBuffer(_states_cl, CL_FALSE, ind_new * _numNeurons * sizeof(state), _numNeurons * sizeof(state), &_states[ind_new * _numNeurons], NULL, NULL);
            _err = _wrapper.getQueue().enqueueReadBuffer(_sumFootprintAMPA_cl, CL_FALSE, 0, _numNeurons * sizeof(float), _sumFootprintAMPA.get(), NULL, NULL);
            _err = _wrapper.getQueue().enqueueReadBuffer(_sumFootprintNMDA_cl, CL_FALSE, 0, _numNeurons * sizeof(float), _sumFootprintNMDA.get(), NULL, NULL);
            _err = _wrapper.getQueue().enqueueReadBuffer(_sumFootprintGABAA_cl, CL_FALSE, 0, _numNeurons * sizeof(float), _sumFootprintGABAA.get(), NULL, NULL);

            if (_readToHostMemory)
            {
                _wrapper.getQueue().finish();
            }
        }

        ++_t;
    }
    catch (cl::Error err)
    {
        handleClError(err);
    }
}

void CLSimulator::simulate()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    unsigned long startTime;
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
    clock_t tStart;
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)

    if (_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        timeBeginPeriod(1);
        startTime = timeGetTime();
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
        tStart = clock();
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    }

    // if(_t == 0)
    // {
    //    LOG_INFO(*_logger) << "Timestep 1/" << _timesteps;
    // }

    for (; _t < _timesteps - 1;)
    {
        if ((_t + 2) % (_timesteps / 100) == 0)
        {
            std::cout << ".";
            // LOG_INFO(*_logger) << "Timestep " << _t + 2 << "/" << _timesteps;
        }

        step();

        if (_plot)
        {
            unsigned int ind_old = _t % 2;
            unsigned int ind_new = 1 - ind_old;

            _plotter->step(&_states[ind_new * _numNeurons], _t, _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA);
        }
    }

    std::cout << std::endl;

    if (_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        unsigned long elapsedTime = timeGetTime() - startTime;
        LOG_INFO(*_logger) << "Execution time: " << elapsedTime / 1000.0 << "s";
        timeEndPeriod(1);
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
        unsigned long elapsedTime = (double)(clock() - tStart) / CLOCKS_PER_SEC;
        LOG_INFO(*_logger) << "Execution time: " << elapsedTime / 1000.0 << "s";
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)

        if (_fftw)
        {
            double avgTimeFFTW = std::accumulate(_timesFFTW.begin(), _timesFFTW.end(), 0.0) / _timesFFTW.size();
            LOG_INFO(*_logger) << "Average execution time FFTW: " << avgTimeFFTW << "ms";
        }

        if (_clfft)
        {
            double avgTimeClFFT = std::accumulate(_timesClFFT.begin(), _timesClFFT.end(), 0.0) / _timesClFFT.size();
            LOG_INFO(*_logger) << "Average execution time clFFT: " << avgTimeClFFT << "ms";
        }
        double avgTimeCalculations = std::accumulate(_timesCalculations.begin(), _timesCalculations.end(), 0.0) / _timesCalculations.size();
        LOG_INFO(*_logger) << "Average execution time calculations: " << avgTimeCalculations << "ms";
    }

    if (_plot)
    {
        _plotter->plot();
    }
}

inline float CLSimulator::_f_w_EE(const int j)
{
    static const float sigma = 1;
    static const float p     = 32;

    // TODO: p varies between 8 to 64
    //
    return tanh(1 / (2 * sigma * p))
           * exp(-abs(j) / (sigma * p));
}

void CLSimulator::f_I_FFT_fftw(const unsigned int ind_old, const Receptor rec)
{
    for (unsigned int i = 0; i < _numNeurons; ++i)   {
        if (rec == AMPA)
        {
            _sVals_split[i][0] = _states[ind_old * _numNeurons + i].s_AMPA;
        } else if (rec == NMDA)
        {
            _sVals_split[i][0] = _states[ind_old * _numNeurons + i].s_NMDA;
        } else if (rec == GABAA)
        {
            _sVals_split[i][0] = _states[ind_old * _numNeurons + i].s_GABAA;
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
        if (rec == AMPA)
        {
            _sumFootprintAMPA[indexOfNeuron] = _convolution_split[indexOfNeuron + _numNeurons - 1][0];
        } else if (rec == NMDA)
        {
            _sumFootprintNMDA[indexOfNeuron] = _convolution_split[indexOfNeuron + _numNeurons - 1][0];
        } else if (rec == GABAA)
        {
            _sumFootprintGABAA[indexOfNeuron] = _convolution_split[indexOfNeuron + _numNeurons - 1][0];
        }
    }
}

void CLSimulator::f_I_FFT_clFFT(const unsigned int ind_old, const Receptor rec)
{
    // initialize sVals_real for FFT
    switch (rec)
    {
    case AMPA:
        handleClError(_kernel_prepareFFT_AMPA.setArg(3, ind_old));
        _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_prepareFFT_AMPA, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
        break;

    case NMDA:
        handleClError(_kernel_prepareFFT_NMDA.setArg(3, ind_old));
        _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_prepareFFT_NMDA, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
        break;

    case GABAA:
        handleClError(_kernel_prepareFFT_GABAA.setArg(3, ind_old));
        _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_prepareFFT_GABAA, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
        break;
    }

    _wrapper.getQueue().finish();

    // transform sVals into frequency domain using FFT
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

    // execute convolution in frequency domain
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_convolution, cl::NullRange, cl::NDRange(_nFFT), cl::NullRange, NULL, NULL);

    _wrapper.getQueue().finish();

    // inverse transform convolution_f using FFT
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

    // update sumFootprint array for current receptor
    switch (rec)
    {
    case AMPA:
        _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_postConvolution_AMPA, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
        break;

    case NMDA:
        _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_postConvolution_NMDA, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
        break;

    case GABAA:
        _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_postConvolution_GABAA, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
        break;
    }
}

std::vector<unsigned long> CLSimulator::getTimesCalculations() const
{
    assert(_measure);
    return _timesCalculations;
}

std::vector<unsigned long> CLSimulator::getTimesFFTW() const
{
    assert(_measure && _fftw);
    return _timesFFTW;
}

std::vector<unsigned long> CLSimulator::getTimesClFFT() const
{
    assert(_measure && _clfft);
    return _timesClFFT;
}

void CLSimulator::convolutionFFTW(const unsigned int ind_old)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    unsigned long startTime;
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
    clock_t tStart;
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)

    if (_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        timeBeginPeriod(1);
        startTime = timeGetTime();
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
        tStart = clock();
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    }

    if (!_plot)
    {
        _err = _wrapper.getQueue().enqueueReadBuffer(_states_cl, CL_TRUE, ind_old * _numNeurons * sizeof(state), _numNeurons * sizeof(state), &_states[ind_old * _numNeurons], NULL, NULL);
    }

    f_I_FFT_fftw(ind_old, AMPA);
    _err = _wrapper.getQueue().enqueueWriteBuffer(_sumFootprintAMPA_cl, CL_FALSE, 0, _numNeurons * sizeof(float), _sumFootprintAMPA.get(), NULL, NULL);
    f_I_FFT_fftw(ind_old, NMDA);
    _err = _wrapper.getQueue().enqueueWriteBuffer(_sumFootprintNMDA_cl, CL_TRUE, 0, _numNeurons * sizeof(float), _sumFootprintNMDA.get(), NULL, NULL);
    // f_I_FFT(ind_old, "GABAA");

    if (_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        unsigned long elapsedTime = timeGetTime() - startTime;
        _timesFFTW.push_back(elapsedTime);
        timeEndPeriod(1);
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
        _timesFFTW.push_back((double)(clock() - tStart) / CLOCKS_PER_SEC);
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    }
}

void CLSimulator::convolutionClFFT(const unsigned int ind_old)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    unsigned long startTime;
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
    clock_t tStart;
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)

    if (_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        timeBeginPeriod(1);
        startTime = timeGetTime();
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
        tStart = clock();
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    }

    f_I_FFT_clFFT(ind_old, AMPA);
    f_I_FFT_clFFT(ind_old, NMDA);

    if (_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        unsigned long elapsedTime = timeGetTime() - startTime;
        _timesClFFT.push_back(elapsedTime);
        timeEndPeriod(1);
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
        _timesClFFT.push_back((double)(clock() - tStart) / CLOCKS_PER_SEC);
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    }
}

void CLSimulator::executeKernels()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    unsigned long startTime;
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
    clock_t tStart;
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)

    if (_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        timeBeginPeriod(1);
        startTime = timeGetTime();
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
        tStart = clock();
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    }
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dV_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dn_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_I_Na_dh_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dz_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dsAMPA_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dxNMDA_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
    _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_f_dsNMDA_dt, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);

    _wrapper.getQueue().finish();

    if (_measure)
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
        unsigned long elapsedTime = timeGetTime() - startTime;
        _timesCalculations.push_back(elapsedTime);
        timeEndPeriod(1);
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
        _timesCalculations.push_back((double)(clock() - tStart) / CLOCKS_PER_SEC);
#endif // if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
    }
}

void CLSimulator::assertConvolutionResults()
{
    std::unique_ptr<float[]> sumFootprintAMPA_tmp(std::unique_ptr<float[]>(new float[_numNeurons]));
    _err = _wrapper.getQueue().enqueueReadBuffer(_sumFootprintAMPA_cl, CL_FALSE, 0, _numNeurons * sizeof(float), sumFootprintAMPA_tmp.get(), NULL, NULL);
    std::unique_ptr<float[]> sumFootprintNMDA_tmp(std::unique_ptr<float[]>(new float[_numNeurons]));
    _err = _wrapper.getQueue().enqueueReadBuffer(_sumFootprintNMDA_cl, CL_TRUE, 0, _numNeurons * sizeof(float), sumFootprintNMDA_tmp.get(), NULL, NULL);

    for (unsigned int i = 0; i < _numNeurons; ++i)
    {
        assertNear(_sumFootprintAMPA[i], sumFootprintAMPA_tmp[i], 0.05);
        assertNear(_sumFootprintNMDA[i], sumFootprintNMDA_tmp[i], 0.05);
    }
}

void CLSimulator::initializeFFTW()
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

    if (_fftw)
    {
        // initialize distances
        unsigned int j = 0;

        for (unsigned int i = _numNeurons - 1; i > 0; --i)
        {
            _distances_split[j][0] = _f_w_EE(i);
            _distances_split[j][1] = 0;
            ++j;
        }

        for (unsigned int i = 0; i < _numNeurons; ++i)
        {
            _distances_split[j][0] = _f_w_EE(i);
            _distances_split[j][1] = 0;
            ++j;
        }

        _distances_split[j][0] = 0;
        _distances_split[j][1] = 0;

        fftwf_execute(_p_distances_fftw);
    }
}

void CLSimulator::initializeHostVariables(state const& state_0)
{
    // 2 states (old and new) per neuron per timestep
    _states = std::unique_ptr<state[]>(new state[2 * _numNeurons]);

    _sumFootprintAMPA = std::unique_ptr<float[]>(new float[_numNeurons]);
    _sumFootprintNMDA = std::unique_ptr<float[]>(new float[_numNeurons]);
    _sumFootprintGABAA = std::unique_ptr<float[]>(new float[_numNeurons]);

    _distances_real = std::unique_ptr<float[]>(new float[_nFFT]);
    _sVals_real = std::unique_ptr<float[]>(new float[_nFFT]);
    _convolution_real = std::unique_ptr<float[]>(new float[_nFFT]);
    _zeros = std::unique_ptr<float[]>(new float[_nFFT]);

    // initialize initial states
    for (unsigned int i = 0; i < _numNeurons; ++i)
    {
        _states[i] = state_0;
        _sumFootprintAMPA[i] = 0;
        _sumFootprintNMDA[i] = 0;
        _sumFootprintGABAA[i] = 0;
    }

    for (unsigned int i = _numNeurons; i < 2 * _numNeurons; ++i)
    {
        _states[i] = state_0;
    }
}

void CLSimulator::initializeClFFT()
{
    // initialize distances
    unsigned int j = 0;

    for (unsigned int i = _numNeurons - 1; i > 0; --i)
    {
        _distances_real[j] = _f_w_EE(i);
        ++j;
    }

    for (unsigned int i = 0; i < _numNeurons; ++i)
    {
        _distances_real[j] = _f_w_EE(i);
        ++j;
    }

    _distances_real[j] = 0;

    for (unsigned int i = 0; i < _nFFT; ++i)
    {
        _zeros[i] = 0;
    }

    assert(isPowerOfTwo(_nFFT));
    clFFT_Dim3 n = { _nFFT, 1, 1 };
    clFFT_DataFormat dataFormat = clFFT_SplitComplexFormat;
    clFFT_Dimension dim = clFFT_1D;
    _p_cl = clFFT_CreatePlan(_wrapper.getContextC(), n, dim, dataFormat, &_err);
    handleClError(_err);

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

void CLSimulator::assertInitializationResults()
{
    boost::scoped_array<float> distances_real(new float[_nFFT]);
    boost::scoped_array<float> distances_imag(new float[_nFFT]);
    boost::scoped_array<float> distances_f_real(new float[_nFFT]);
    boost::scoped_array<float> distances_f_imag(new float[_nFFT]);

    _err = _wrapper.getQueue().enqueueReadBuffer(_distances_real_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_real.get(), NULL, NULL);
    _err = _wrapper.getQueue().enqueueReadBuffer(_distances_imag_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_imag.get(), NULL, NULL);
    _err = _wrapper.getQueue().enqueueReadBuffer(_distances_f_real_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_f_real.get(), NULL, NULL);
    _err = _wrapper.getQueue().enqueueReadBuffer(_distances_f_imag_cl, CL_TRUE, 0, _nFFT * sizeof(float), distances_f_imag.get(), NULL, NULL);

    for (unsigned int i = 0; i < _nFFT; ++i)
    {
        assertAlmostEquals(_distances_split[i][0], distances_real[i]);
        assertAlmostEquals(_distances_split[i][1], distances_imag[i]);
        assertNear(_distances_f_split[i][0], distances_f_real[i], 0.000001);
        assertNear(_distances_f_split[i][1], distances_f_imag[i], 0.000001);
    }
}

void CLSimulator::initializeCLKernelsAndBuffers()
{
    _states_cl = cl::Buffer(_wrapper.getContext(),
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            2 * _numNeurons * sizeof(state),
                            _states.get(),
                            &_err);
    _sumFootprintAMPA_cl = cl::Buffer(_wrapper.getContext(),
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      _numNeurons * sizeof(float),
                                      _sumFootprintAMPA.get(),
                                      &_err);
    _sumFootprintNMDA_cl = cl::Buffer(_wrapper.getContext(),
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      _numNeurons * sizeof(float),
                                      _sumFootprintNMDA.get(),
                                      &_err);
    _sumFootprintGABAA_cl = cl::Buffer(_wrapper.getContext(),
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       _numNeurons * sizeof(float),
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

    _kernel_postConvolution_AMPA = cl::Kernel(_program, "postConvolution_AMPA", &_err);
    handleClError(_kernel_postConvolution_AMPA.setArg(0, _convolution_real_cl));
    handleClError(_kernel_postConvolution_AMPA.setArg(1, _sumFootprintAMPA_cl));
    handleClError(_kernel_postConvolution_AMPA.setArg(2, _numNeurons));

    _kernel_postConvolution_NMDA = cl::Kernel(_program, "postConvolution_NMDA", &_err);
    handleClError(_kernel_postConvolution_NMDA.setArg(0, _convolution_real_cl));
    handleClError(_kernel_postConvolution_NMDA.setArg(1, _sumFootprintNMDA_cl));
    handleClError(_kernel_postConvolution_NMDA.setArg(2, _numNeurons));

    _kernel_postConvolution_GABAA = cl::Kernel(_program, "postConvolution_GABAA", &_err);
    handleClError(_kernel_postConvolution_GABAA.setArg(0, _convolution_real_cl));
    handleClError(_kernel_postConvolution_GABAA.setArg(1, _sumFootprintGABAA_cl));
    handleClError(_kernel_postConvolution_GABAA.setArg(2, _numNeurons));

    handleClError(_kernel_f_dV_dt.setArg(0, _states_cl));
    handleClError(_kernel_f_dV_dt.setArg(1, _sumFootprintAMPA_cl));
    handleClError(_kernel_f_dV_dt.setArg(2, _sumFootprintNMDA_cl));
    handleClError(_kernel_f_dV_dt.setArg(3, _sumFootprintGABAA_cl));
    handleClError(_kernel_f_dV_dt.setArg(4, _numNeurons));
    handleClError(_kernel_f_dV_dt.setArg(6, _dt));

    BOOST_FOREACH(cl::Kernel kernel, kernels)
    {
        _err = kernel.setArg(0, _states_cl);
        _err = kernel.setArg(1, _numNeurons);
        _err = kernel.setArg(3, _dt);
    }
}

CLSimulator::~CLSimulator()
{
    if (_fftw)
    {
        fftwf_free(_distances_split);
        fftwf_free(_convolution_split);
        fftwf_free(_sVals_split);
        fftwf_free(_distances_f_split);
        fftwf_free(_convolution_f_split);
        fftwf_free(_sVals_f_split);
        fftwf_destroy_plan(_p_distances_fftw);
        fftwf_destroy_plan(_p_sVals_fftw);
        fftwf_destroy_plan(_p_inv_fftw);
    }

    if (_clfft)
    {
        clFFT_DestroyPlan(_p_cl);
    }
}

std::unique_ptr<state[]> const& CLSimulator::getCurrentStates() const
{
    return _states;
}

std::unique_ptr<float[]> const& CLSimulator::getCurrentSumFootprintAMPA() const
{
    return _sumFootprintAMPA;
}

std::unique_ptr<float[]> const& CLSimulator::getCurrentSumFootprintNMDA() const
{
    return _sumFootprintNMDA;
}

std::unique_ptr<float[]> const& CLSimulator::getCurrentSumFootprintGABAA() const
{
    return _sumFootprintGABAA;
}
