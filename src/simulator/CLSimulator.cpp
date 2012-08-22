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

#include "stdafx.h"

#include "CLSimulator.h"
#include "GnuPlotPlotter.h"
#include "OpenGLPlotter.h"
#include "util.h"

#include <cassert>
#include <ctime>
#include <numeric>

#include <boost/chrono.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>

#ifdef _MSVC_VER
    #pragma warning(push, 0)        
#elif __GCC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wall"
#endif

#include <Cl/cl.hpp>

#ifdef _MSVC_VER
    #pragma warning(pop)
#elif __GCC__
    #pragma GCC diagnostic pop
#endif

CLSimulator::CLSimulator(const size_t nX,
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
      _nFFTx(2 * nX),
      _nFFTy(nY > 1 ? 2 * nY : 1),
      _nFFTz(nZ > 1 ? 2 * nZ : 1),
      _err(CL_SUCCESS)
{
    _nFFT = (_nFFTx * _nFFTy * _nFFTz);
    _scaleFFT = (1.f / _nFFT);

    switch (plot)
    {
    case NO_PLOT :
        break;

    case PLOT_GNUPLOT :
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
        _plotter->step(&_states[0][0], _numNeurons, _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA);
    }
}

void CLSimulator::step()
{
    _ind_old = _t % 2;
    _ind_new = 1 - _ind_old;

    try
    {
        // make sure that enqueueReadBuffer from last timestep has finished if gnuplot is enabled
        if (_plot)
        {
            _err = _wrapper.getQueue().finish();
        }

        // set dynamic kernel args
        _err = _kernel_f_dV_dt.setArg(0, _states_cl[_ind_old]);
        _err = _kernel_f_dV_dt.setArg(1, _states_cl[_ind_new]);
        _err = _kernel_f_dn_dt.setArg(0, _states_cl[_ind_old]);
        _err = _kernel_f_dn_dt.setArg(1, _states_cl[_ind_new]);
        _err = _kernel_f_I_Na_dh_dt.setArg(0, _states_cl[_ind_old]);
        _err = _kernel_f_I_Na_dh_dt.setArg(1, _states_cl[_ind_new]);
        _err = _kernel_f_dz_dt.setArg(0, _states_cl[_ind_old]);
        _err = _kernel_f_dz_dt.setArg(1, _states_cl[_ind_new]);
        _err = _kernel_f_dsAMPA_dt.setArg(0, _states_cl[_ind_old]);
        _err = _kernel_f_dsAMPA_dt.setArg(1, _states_cl[_ind_new]);
        _err = _kernel_f_dxNMDA_dt.setArg(0, _states_cl[_ind_old]);
        _err = _kernel_f_dxNMDA_dt.setArg(1, _states_cl[_ind_new]);
        _err = _kernel_f_dsNMDA_dt.setArg(0, _states_cl[_ind_old]);
        _err = _kernel_f_dsNMDA_dt.setArg(1, _states_cl[_ind_new]);

        // compute convolution
        if (_fftw)
        {
            convolutionFFTW();
        }

        if (_clfft)
        {
            convolutionClFFT();
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
            _err = _wrapper.getQueue().enqueueReadBuffer(_states_cl[_ind_new], CL_FALSE, 0, _numNeurons * sizeof(state), _states[_ind_new].get(), NULL, NULL);
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
    boost::chrono::high_resolution_clock::time_point startTime;

    if (_measure)
    {
        startTime = boost::chrono::high_resolution_clock::now();
    }

    for (; _t < _timesteps - 1;)
    {
        if ((_t + 2) % (_timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        step();

        if (_plot)
        {
            size_t ind_old = _t % 2;
            size_t ind_new = 1 - ind_old;

            _plotter->step(_states[ind_new].get(), _t, _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA);
        }
    }

    std::cout << std::endl;

    if (_measure)
    {
        auto const& endTime =
            boost::chrono::duration_cast<boost::chrono::microseconds>(
                boost::chrono::high_resolution_clock::now() - startTime);
        LOG_INFO(*_logger) << "Execution time: " << endTime.count() / 1000000.0 << "s";

        if (_fftw)
        {
            auto const& avgTimeFFTW = std::accumulate(_timesFFTW.begin(), _timesFFTW.end(), 0.0) / _timesFFTW.size();
            LOG_INFO(*_logger) << "Average execution time FFTW: " << avgTimeFFTW << "us";
        }

        if (_clfft)
        {
            auto const& avgTimeClFFT = std::accumulate(_timesClFFT.begin(), _timesClFFT.end(), 0.0) / _timesClFFT.size();
            LOG_INFO(*_logger) << "Average execution time clFFT: " << avgTimeClFFT << "us";
        }
        double avgTimeCalculations = std::accumulate(_timesCalculations.begin(), _timesCalculations.end(), 0.0) / _timesCalculations.size();
        LOG_INFO(*_logger) << "Average execution time calculations: " << avgTimeCalculations << "us";
    }

    if (_plot)
    {
        _plotter->plot();
    }
}

inline float CLSimulator::_f_w_EE(const float j)
{
    static const float sigma = 1;
    static const float p     = 32;

    // p varies between 8 to 64
    return tanh(1 / (2 * sigma * p))
           * exp(-abs(j) / (sigma * p));
}

void CLSimulator::f_I_FFT_fftw(const Receptor rec)
{
    for (size_t x = 0; x < _nX; ++x) {
        for (size_t y = 0; y < _nY; ++y) {
            size_t index_states = x + y * _nY;
            size_t index_sVals = x + y * _nFFTy;

            if (rec == AMPA)
            {
                _sVals_split[index_sVals][0] = _states[_ind_old][index_states].s_AMPA;
            } else if (rec == NMDA)
            {
                _sVals_split[index_sVals][0] = _states[_ind_old][index_states].s_NMDA;
            } else if (rec == GABAA)
            {
                _sVals_split[index_sVals][0] = _states[_ind_old][index_states].s_GABAA;
            }
            _sVals_split[index_sVals][1] = 0;
        }
    }

    for (size_t x = _nX; x < _nFFTx; ++x)
    {
        for (size_t y = _nY; y < _nFFTy; ++y)
        {
            size_t index = x + y * _nFFTx;
            _sVals_split[index][0] = 0;
            _sVals_split[index][1] = 0;
        }
    }

    fftwf_execute(_p_sVals_fftw);

    // convolution in frequency domain
    for (size_t i = 0; i < _nFFT; ++i)
    {
        _convolution_f_split[i][0] = (_distances_f_split[i][0] * _sVals_f_split[i][0]
                                      - _distances_f_split[i][1] * _sVals_f_split[i][1]) * _scaleFFT;
        _convolution_f_split[i][1] = (_distances_f_split[i][0] * _sVals_f_split[i][1]
                                      + _distances_f_split[i][1] * _sVals_f_split[i][0]) * _scaleFFT;
    }

    fftwf_execute(_p_inv_fftw);

    for (size_t x_conv = _nX - 1, x_fp = 0; x_conv < _nFFTx - 1; ++x_conv, ++x_fp)
    {
        size_t t_nFFTy = _nFFTy > 1 ? _nFFTy - 1 : 1;

        for (size_t y_conv = _nY - 1, y_fp = 0; y_conv < t_nFFTy; ++y_conv, ++y_fp)
        {
            size_t index_conv = x_conv + y_conv * _nFFTx;
            size_t index_fp = x_fp + y_fp * _nX;

            if (rec == AMPA)
            {
                _sumFootprintAMPA[index_fp] = _convolution_split[index_conv][0];
            } else if (rec == NMDA)
            {
                _sumFootprintNMDA[index_fp] = _convolution_split[index_conv][0];
            } else if (rec == GABAA)
            {
                _sumFootprintGABAA[index_fp] = _convolution_split[index_conv][0];
            }
        }
    }
}

void CLSimulator::f_I_FFT_clFFT(const Receptor rec)
{
    // initialize sVals_real for FFT
    switch (rec)
    {
    case AMPA:
        handleClError(_kernel_prepareFFT_AMPA.setArg(0, _states_cl[_ind_old]));
        _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_prepareFFT_AMPA, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
        break;

    case NMDA:
        handleClError(_kernel_prepareFFT_NMDA.setArg(0, _states_cl[_ind_old]));
        _err = _wrapper.getQueue().enqueueNDRangeKernel(_kernel_prepareFFT_NMDA, cl::NullRange, cl::NDRange(_numNeurons), cl::NullRange, NULL, NULL);
        break;

    case GABAA:
        handleClError(_kernel_prepareFFT_GABAA.setArg(0, _states_cl[_ind_old]));
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

std::vector<size_t> CLSimulator::getTimesCalculations() const
{
    assert(_measure);
    return _timesCalculations;
}

std::vector<size_t> CLSimulator::getTimesFFTW() const
{
    assert(_measure && _fftw);
    return _timesFFTW;
}

std::vector<size_t> CLSimulator::getTimesClFFT() const
{
    assert(_measure && _clfft);
    return _timesClFFT;
}

void CLSimulator::convolutionFFTW()
{
    boost::chrono::high_resolution_clock::time_point startTime;

    if (_measure)
    {
        startTime = boost::chrono::high_resolution_clock::now();
    }

    if (!_plot)
    {
        _err = _wrapper.getQueue().enqueueReadBuffer(_states_cl[_ind_old], CL_FALSE, 0, _numNeurons * sizeof(state), _states[_ind_old].get(), NULL, NULL);
    }

    _wrapper.getQueue().finish();

    f_I_FFT_fftw(AMPA);
    _err = _wrapper.getQueue().enqueueWriteBuffer(_sumFootprintAMPA_cl, CL_FALSE, 0, _numNeurons * sizeof(float), _sumFootprintAMPA.get(), NULL, NULL);
    f_I_FFT_fftw(NMDA);
    _err = _wrapper.getQueue().enqueueWriteBuffer(_sumFootprintNMDA_cl, CL_TRUE, 0, _numNeurons * sizeof(float), _sumFootprintNMDA.get(), NULL, NULL);
    // f_I_FFT(ind_old, "GABAA");

    if (_measure)
    {
        auto const& endTime =
            boost::chrono::duration_cast<boost::chrono::microseconds>(
                boost::chrono::high_resolution_clock::now() - startTime);
        _timesFFTW.push_back(size_t(endTime.count()));
    }
}

void CLSimulator::convolutionClFFT()
{
    boost::chrono::high_resolution_clock::time_point startTime;

    if (_measure)
    {
        startTime = boost::chrono::high_resolution_clock::now();
    }

    f_I_FFT_clFFT(AMPA);
    f_I_FFT_clFFT(NMDA);

    if (_measure)
    {
        auto const& endTime =
            boost::chrono::duration_cast<boost::chrono::microseconds>(
                boost::chrono::high_resolution_clock::now() - startTime);
        _timesClFFT.push_back(size_t(endTime.count()));
    }
}

void CLSimulator::executeKernels()
{
    boost::chrono::high_resolution_clock::time_point startTime;

    if (_measure)
    {
        startTime = boost::chrono::high_resolution_clock::now();
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
        auto const& endTime =
            boost::chrono::duration_cast<boost::chrono::microseconds>(
                boost::chrono::high_resolution_clock::now() - startTime);
        _timesCalculations.push_back(size_t(endTime.count()));
    }
}

void CLSimulator::assertConvolutionResults()
{
    std::unique_ptr<float[]> sumFootprintAMPA_tmp(new float[_numNeurons]);
    _err = _wrapper.getQueue().enqueueReadBuffer(_sumFootprintAMPA_cl, CL_FALSE, 0, _numNeurons * sizeof(float), sumFootprintAMPA_tmp.get(), NULL, NULL);
    std::unique_ptr<float[]> sumFootprintNMDA_tmp(new float[_numNeurons]);
    _err = _wrapper.getQueue().enqueueReadBuffer(_sumFootprintNMDA_cl, CL_FALSE, 0, _numNeurons * sizeof(float), sumFootprintNMDA_tmp.get(), NULL, NULL);

    for (size_t i = 0; i < _numNeurons; ++i)
    {
        assertNear(_sumFootprintAMPA[i], sumFootprintAMPA_tmp[i], 0.00001);
        assertNear(_sumFootprintNMDA[i], sumFootprintNMDA_tmp[i], 0.00001);
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

    assert(_nX >= 1 && _nY >= 1 && _nZ >= 1);
    assert((_nX >= _nY) && (_nY >= _nZ));

    if (_nY == 1)
    {
        _p_distances_fftw = fftwf_plan_dft_1d(_nFFT, _distances_split, _distances_f_split, FFTW_FORWARD, FFTW_ESTIMATE);
        _p_sVals_fftw = fftwf_plan_dft_1d(_nFFT, _sVals_split, _sVals_f_split, FFTW_FORWARD, FFTW_ESTIMATE);
        _p_inv_fftw = fftwf_plan_dft_1d(_nFFT, _convolution_f_split, _convolution_split, FFTW_BACKWARD, FFTW_ESTIMATE);
    } else if (_nZ == 1)
    {
        _p_distances_fftw = fftwf_plan_dft_2d(_nFFTx, _nFFTy, _distances_split, _distances_f_split, FFTW_FORWARD, FFTW_ESTIMATE);
        _p_sVals_fftw = fftwf_plan_dft_2d(_nFFTx, _nFFTy, _sVals_split, _sVals_f_split, FFTW_FORWARD, FFTW_ESTIMATE);
        _p_inv_fftw = fftwf_plan_dft_2d(_nFFTx, _nFFTy, _convolution_f_split, _convolution_split, FFTW_BACKWARD, FFTW_ESTIMATE);
    } else
    {
        _p_distances_fftw = fftwf_plan_dft_3d(_nFFTx, _nFFTy, _nFFTz, _distances_split, _distances_f_split, FFTW_FORWARD, FFTW_ESTIMATE);
        _p_sVals_fftw = fftwf_plan_dft_3d(_nFFTx, _nFFTy, _nFFTz, _sVals_split, _sVals_f_split, FFTW_FORWARD, FFTW_ESTIMATE);
        _p_inv_fftw = fftwf_plan_dft_3d(_nFFTx, _nFFTy, _nFFTz, _convolution_f_split, _convolution_split, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    for (size_t i = 0; i < _nFFT; ++i)
    {
        _sVals_split[i][0] = 0;
        _sVals_split[i][1] = 0;
    }

    for (size_t x_idx = 0, x_val = _nX - 1; x_idx < _nX; ++x_idx, --x_val) {
        for (size_t y_idx = 0, y_val = _nY - 1; y_idx < _nY; ++y_idx, --y_val) {
            float distance = sqrt(pow(float(x_val), 2.0f) + pow(float(y_val), 2.0f));
            _distances_split[x_idx + y_idx * _nFFTx][0] = _f_w_EE((float(distance)));
            _distances_split[x_idx + y_idx * _nFFTx][1] = 0;
        }
    }

    for (size_t x_idx = 0, x_val = _nX - 1; x_idx < _nX; ++x_idx, --x_val) {
        for (size_t y_idx = _nY, y_val = 1; y_idx < _nFFTy - 1; ++y_idx, ++y_val) {
            float distance = sqrt(pow(float(x_val), 2.0f) + pow(float(y_val), 2.0f));
            _distances_split[x_idx + y_idx * _nFFTx][0] = _f_w_EE((float(distance)));
            _distances_split[x_idx + y_idx * _nFFTx][1] = 0;
        }
    }

    if (_nY > 1)
    {
        for (size_t x_idx = 0; x_idx < _nFFTx; ++x_idx) {
            _distances_split[x_idx + (_nFFTy - 1) * _nFFTx][0] = 0;
            _distances_split[x_idx + (_nFFTy - 1) * _nFFTx][1] = 0;
        }
    }

    for (size_t x_idx = _nX, x_val = 1; x_idx < _nFFTx - 1; ++x_idx, ++x_val) {
        for (size_t y_idx = 0, y_val = _nY - 1; y_idx < _nY; ++y_idx, --y_val) {
            float distance = sqrt(pow(float(x_val), 2.0f) + pow(float(y_val), 2.0f));
            _distances_split[x_idx + y_idx * _nFFTx][0] = _f_w_EE((float(distance)));
            _distances_split[x_idx + y_idx * _nFFTx][1] = 0;
        }
    }

    for (size_t y_idx = 0; y_idx < _nFFTy; ++y_idx) {
        _distances_split[(_nFFTx - 1) + y_idx * _nFFTx][0] = 0;
        _distances_split[(_nFFTx - 1) + y_idx * _nFFTx][1] = 0;
    }

    for (size_t x_idx = _nX, x_val = 1; x_idx < _nFFTx - 1; ++x_idx, ++x_val) {
        for (size_t y_idx = _nY, y_val = 1; y_idx < _nFFTy - 1; ++y_idx, ++y_val) {
            float distance = sqrt(pow(float(x_val), 2.0f) + pow(float(y_val), 2.0f));
            _distances_split[x_idx + y_idx * _nFFTx][0] = _f_w_EE((float(distance)));
            _distances_split[x_idx + y_idx * _nFFTx][1] = 0;
        }
    }

    fftwf_execute(_p_distances_fftw);
}

void CLSimulator::initializeHostVariables(state const& state_0)
{
    // 2 states (old and new) per neuron per timestep
    _states.emplace_back(std::unique_ptr<state[]>(new state[_numNeurons]));
    _states.emplace_back(std::unique_ptr<state[]>(new state[_numNeurons]));

    _sumFootprintAMPA = std::unique_ptr<float[]>(new float[_numNeurons]);
    _sumFootprintNMDA = std::unique_ptr<float[]>(new float[_numNeurons]);
    _sumFootprintGABAA = std::unique_ptr<float[]>(new float[_numNeurons]);

    _distances_real = std::unique_ptr<float[]>(new float[_nFFT]);
    _sVals_real = std::unique_ptr<float[]>(new float[_nFFT]);
    _convolution_real = std::unique_ptr<float[]>(new float[_nFFT]);
    _zeros = std::unique_ptr<float[]>(new float[_nFFT]);

    // initialize initial states
    for (size_t i = 0; i < _numNeurons; ++i)
    {
        _states[0][i] = state_0;
        _states[1][i] = state_0;
        _sumFootprintAMPA[i] = 0;
        _sumFootprintNMDA[i] = 0;
        _sumFootprintGABAA[i] = 0;
    }

    for (size_t i = 0; i < _nFFT; ++i)
    {
        _zeros[i] = 0;
    }
}

void CLSimulator::initializeClFFT()
{
    /* x x x x
     * x x x x
     * x x x x
     * x x x x
     */

    for (size_t x_idx = 0, x_val = _nX - 1; x_idx < _nX; ++x_idx, --x_val) {
        for (size_t y_idx = 0, y_val = _nY - 1; y_idx < _nY; ++y_idx, --y_val) {
            float distance = sqrt(pow(float(x_val), 2.0f) + pow(float(y_val), 2.0f));
            _distances_real[x_idx + y_idx * _nFFTx] = _f_w_EE((float(distance)));
        }
    }

    /* v v x x
     * v v x x
     * x x x x
     * x x x x
     */

    for (size_t x_idx = 0, x_val = _nX - 1; x_idx < _nX; ++x_idx, --x_val) {
        for (size_t y_idx = _nY, y_val = 1; y_idx < _nFFTy - 1; ++y_idx, ++y_val) {
            float distance = sqrt(pow(float(x_val), 2.0f) + pow(float(y_val), 2.0f));
            _distances_real[x_idx + y_idx * _nFFTx] = _f_w_EE((float(distance)));
        }
    }

    /* v v v x
     * v v v x
     * x x x x
     * x x x x
     */

    if (_nY > 1)
    {
        for (size_t x_idx = 0; x_idx < _nFFTx; ++x_idx) {
            _distances_real[x_idx + (_nFFTy - 1) * _nFFTx] = 0;
        }
    }

    /* v v v 0
     * v v v 0
     * x x x 0
     * x x x 0
     */

    for (size_t x_idx = _nX, x_val = 1; x_idx < _nFFTx - 1; ++x_idx, ++x_val) {
        for (size_t y_idx = 0, y_val = _nY - 1; y_idx < _nY; ++y_idx, --y_val) {
            float distance = sqrt(pow(float(x_val), 2.0f) + pow(float(y_val), 2.0f));
            _distances_real[x_idx + y_idx * _nFFTx] = _f_w_EE((float(distance)));
        }
    }

    /* v v v 0
     * v v v 0
     * v v x 0
     * x x x 0
     */

    for (size_t y_idx = 0; y_idx < _nFFTy; ++y_idx) {
        _distances_real[(_nFFTx - 1) + y_idx * _nFFTx] = 0;
    }

    /* v v v 0
     * v v v 0
     * v v x 0
     * 0 0 0 0
     */

    for (size_t x_idx = _nX, x_val = 1; x_idx < _nFFTx - 1; ++x_idx, ++x_val) {
        for (size_t y_idx = _nY, y_val = 1; y_idx < _nFFTy - 1; ++y_idx, ++y_val) {
            float distance = sqrt(pow(float(x_val), 2.0f) + pow(float(y_val), 2.0f));
            _distances_real[x_idx + y_idx * _nFFTx] = _f_w_EE((float(distance)));
        }
    }

    /* v v v 0
     * v v v 0
     * v v v 0
     * 0 0 0 0
     */

    assert(isPowerOfTwo(_nFFT));
    assert(_nX >= 1 && _nY >= 1 && _nZ >= 1);
    assert((_nX >= _nY) && (_nY >= _nZ));
    clFFT_Dim3 n = { static_cast<unsigned int>(_nFFTx),
                     static_cast<unsigned int>(_nFFTy),
                     static_cast<unsigned int>(_nFFTz) };
    clFFT_DataFormat dataFormat = clFFT_SplitComplexFormat;
    clFFT_Dimension dim;

    if (_nY == 1)
    {
        dim = clFFT_1D;
    } else if (_nZ == 1)
    {
        dim = clFFT_2D;
    } else
    {
        dim = clFFT_3D;
    }
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

    for (size_t i = 0; i < _nFFT; ++i)
    {
        assertAlmostEquals(_distances_split[i][0], distances_real[i]);
        assertAlmostEquals(_distances_split[i][1], distances_imag[i]);
        assertNear(_distances_f_split[i][0], distances_f_real[i], 0.000001);
        assertNear(_distances_f_split[i][1], distances_f_imag[i], 0.000001);
    }
}

void CLSimulator::initializeCLKernelsAndBuffers()
{
    _states_cl.emplace_back(cl::Buffer(_wrapper.getContext(),
                                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       _numNeurons * sizeof(state),
                                       _states[0].get(),
                                       &_err));
    _states_cl.emplace_back(cl::Buffer(_wrapper.getContext(),
                                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       _numNeurons * sizeof(state),
                                       _states[1].get(),
                                       &_err));
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
    handleClError(_kernel_prepareFFT_AMPA.setArg(1, _sVals_real_cl));
    handleClError(_kernel_prepareFFT_AMPA.setArg(2, _nX));
    handleClError(_kernel_prepareFFT_AMPA.setArg(3, _nY));
    handleClError(_kernel_prepareFFT_AMPA.setArg(4, _nZ));

    _kernel_prepareFFT_NMDA = cl::Kernel(_program, "prepareFFT_NMDA", &_err);
    handleClError(_kernel_prepareFFT_NMDA.setArg(1, _sVals_real_cl));
    handleClError(_kernel_prepareFFT_NMDA.setArg(2, _nX));
    handleClError(_kernel_prepareFFT_NMDA.setArg(3, _nY));
    handleClError(_kernel_prepareFFT_NMDA.setArg(4, _nZ));

    _kernel_prepareFFT_GABAA = cl::Kernel(_program, "prepareFFT_GABAA", &_err);
    handleClError(_kernel_prepareFFT_GABAA.setArg(1, _sVals_real_cl));
    handleClError(_kernel_prepareFFT_GABAA.setArg(2, _nX));
    handleClError(_kernel_prepareFFT_GABAA.setArg(3, _nY));
    handleClError(_kernel_prepareFFT_GABAA.setArg(4, _nZ));

    _kernel_postConvolution_AMPA = cl::Kernel(_program, "postConvolution_AMPA", &_err);
    handleClError(_kernel_postConvolution_AMPA.setArg(0, _convolution_real_cl));
    handleClError(_kernel_postConvolution_AMPA.setArg(1, _sumFootprintAMPA_cl));
    handleClError(_kernel_postConvolution_AMPA.setArg(2, _nX));
    handleClError(_kernel_postConvolution_AMPA.setArg(3, _nY));
    handleClError(_kernel_postConvolution_AMPA.setArg(4, _nZ));

    _kernel_postConvolution_NMDA = cl::Kernel(_program, "postConvolution_NMDA", &_err);
    handleClError(_kernel_postConvolution_NMDA.setArg(0, _convolution_real_cl));
    handleClError(_kernel_postConvolution_NMDA.setArg(1, _sumFootprintNMDA_cl));
    handleClError(_kernel_postConvolution_NMDA.setArg(2, _nX));
    handleClError(_kernel_postConvolution_NMDA.setArg(3, _nY));
    handleClError(_kernel_postConvolution_NMDA.setArg(4, _nZ));

    _kernel_postConvolution_GABAA = cl::Kernel(_program, "postConvolution_GABAA", &_err);
    handleClError(_kernel_postConvolution_GABAA.setArg(0, _convolution_real_cl));
    handleClError(_kernel_postConvolution_GABAA.setArg(1, _sumFootprintGABAA_cl));
    handleClError(_kernel_postConvolution_GABAA.setArg(2, _nX));
    handleClError(_kernel_postConvolution_GABAA.setArg(3, _nY));
    handleClError(_kernel_postConvolution_GABAA.setArg(4, _nZ));

    handleClError(_kernel_f_dV_dt.setArg(2, _sumFootprintAMPA_cl));
    handleClError(_kernel_f_dV_dt.setArg(3, _sumFootprintNMDA_cl));
    handleClError(_kernel_f_dV_dt.setArg(4, _sumFootprintGABAA_cl));
    handleClError(_kernel_f_dV_dt.setArg(5, _dt));

    BOOST_FOREACH(cl::Kernel kernel, kernels)
    {
        _err = kernel.setArg(2, _dt);
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

state const * CLSimulator::getCurrentStatesOld() const
{
    return &_states[_ind_old][0];
}

state const * CLSimulator::getCurrentStatesNew() const
{
    return &_states[_ind_new][0];
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

CLWrapper CLSimulator::getClWrapper() const
{
    return _wrapper;
}

