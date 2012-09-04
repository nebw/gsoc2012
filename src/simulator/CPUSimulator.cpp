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

#include "CPUSimulator.h"
#include "util.h"

#include <algorithm>

#include <boost/chrono.hpp>

CPUSimulator::CPUSimulator(const size_t nX,
                           const size_t nY,
                           const size_t nZ,
                           const size_t timesteps,
                           const float dt,
                           state const& state_0,
                           const Convolution convolution)
    : _dt(dt),
      _nX(nX),
      _nY(nY),
      _nZ(nZ),
      _numNeurons(nX * nY * nZ),
      _timesteps(timesteps),
      _t(0),
      _convolution(convolution)
{
    // 2 states (old and new) per neuron per timestep
    _states = std::unique_ptr<state[]>(new state[2 * _numNeurons]);

    _sumFootprintAMPA = std::unique_ptr<float[]>(new float[_numNeurons]);
    _sumFootprintNMDA = std::unique_ptr<float[]>(new float[_numNeurons]);
    _sumFootprintGABAA = std::unique_ptr<float[]>(new float[_numNeurons]);

    _distances = std::unique_ptr<float[]>(new float[_numNeurons]);
    _sVals = std::unique_ptr<float[]>(new float[_numNeurons]);

    // initialize initial states
    for (size_t i = 0; i < _numNeurons; ++i)
    {
        _states[i] = state_0;
        _sumFootprintAMPA[i] = 0;
        _sumFootprintNMDA[i] = 0;
        _sumFootprintGABAA[i] = 0;
    }

    for (size_t i = _numNeurons; i < 2 * _numNeurons; ++i)
    {
        _states[i] = state_0;
    }

    for (size_t i = _numNeurons; i < 2 * _numNeurons; ++i)
    {
        _states[i] = state_0;
    }

    for (size_t x = 0; x < _nX; ++x) {
        for (size_t y = 0; y < _nY; ++y) {
            for (size_t z = 0; z < _nZ; ++z) {
                size_t distance = x * x + y * y + z * z;
                _distances[x + y * _nX + z * _nY] = _f_w_EE(sqrt(float(distance)));
            }
        }
    }
}

void CPUSimulator::step()
{
    _ind_old = _t % 2;
    _ind_new = 1 - _ind_old;

    if (_convolution)
    {
        computeConvolutions();
    }

    computeRungeKuttaApproximations();

    ++_t;
}

void CPUSimulator::simulate()
{
    for (; _t < _timesteps - 1;)
    {
        if ((_t + 2) % (_timesteps / 100) == 0)
        {
            std::cout << ".";
        }

        step();
    }
    std::cout << std::endl;
}

void CPUSimulator::computeConvolutions()
{
    auto const& startTime =
        boost::chrono::high_resolution_clock::now();

    convolutionAMPA();

    convolutionNMDA();

    auto const& endTime =
        boost::chrono::duration_cast<boost::chrono::microseconds>(
            boost::chrono::high_resolution_clock::now() - startTime);
    _timesConvolutions.push_back(size_t(endTime.count()));
}

void CPUSimulator::computeRungeKuttaApproximations()
{
    auto const& startTime =
        boost::chrono::high_resolution_clock::now();

    for (size_t idx = 0; idx < _numNeurons; ++idx)
    {
        runge4_f_dV_dt(idx);
        runge4_f_I_Na_dh_dt(idx);
        runge4_f_dsAMPA_dt(idx);
        runge4_f_dn_dt(idx);
        runge4_f_dz_dt(idx);
        runge4_f_dxNMDA_dt(idx);
        runge4_f_dsNMDA_dt(idx);
    }

    auto const& endTime =
        boost::chrono::duration_cast<boost::chrono::microseconds>(
            boost::chrono::high_resolution_clock::now() - startTime);
    _timesCalculations.push_back(size_t(endTime.count()));
}

state const * CPUSimulator::getCurrentStatesOld() const
{
    return &_states.get()[_ind_old * _numNeurons];
}

state const * CPUSimulator::getCurrentStatesNew() const
{
    return &_states.get()[_ind_new * _numNeurons];
}

std::unique_ptr<float[]> const& CPUSimulator::getCurrentSumFootprintAMPA() const
{
    return _sumFootprintAMPA;
}

std::unique_ptr<float[]> const& CPUSimulator::getCurrentSumFootprintNMDA() const
{
    return _sumFootprintNMDA;
}

std::unique_ptr<float[]> const& CPUSimulator::getCurrentSumFootprintGABAA() const
{
    return _sumFootprintGABAA;
}

void CPUSimulator::setCurrentStatesOld(state const *states)
{
#ifdef _MSC_VER
    std::copy(states, states + _numNeurons,
              stdext::checked_array_iterator<state *>(
                _states.get() + _ind_old * _numNeurons,
                _numNeurons));
#else // ifdef _MSC_VER
    std::copy(states, states + _numNeurons,
              _states.get() + _ind_old * _numNeurons);
#endif // ifdef _MSC_VER
}

void CPUSimulator::setCurrentStatesNew(state const *states)
{
#ifdef _MSC_VER
    std::copy(states, states + _numNeurons,
              stdext::checked_array_iterator<state *>(
                _states.get() + _ind_new * _numNeurons,
                _numNeurons));
#else // ifdef _MSC_VER
    std::copy(states, states + _numNeurons,
              _states.get() + _ind_new * _numNeurons);
#endif // ifdef _MSC_VER
}

void CPUSimulator::setCurrentSumFootprintAMPA(std::unique_ptr<float[]> const& sumFootprintAMPA)
{
#ifdef _MSC_VER
    std::copy(stdext::checked_array_iterator<float *>(sumFootprintAMPA.get(), 0),
              stdext::checked_array_iterator<float *>(sumFootprintAMPA.get(), _numNeurons),
              stdext::checked_array_iterator<float *>(_sumFootprintAMPA.get(), 0));
#else // ifdef _MSC_VER
    std::copy(sumFootprintAMPA.get(),
              sumFootprintAMPA.get() + _numNeurons,
              _sumFootprintAMPA.get());
#endif // ifdef _MSC_VER
}

void CPUSimulator::setCurrentSumFootprintNMDA(std::unique_ptr<float[]> const& sumFootprintNMDA)
{
#ifdef _MSC_VER
    std::copy(stdext::checked_array_iterator<float *>(sumFootprintNMDA.get(), 0),
              stdext::checked_array_iterator<float *>(sumFootprintNMDA.get(), _numNeurons),
              stdext::checked_array_iterator<float *>(_sumFootprintNMDA.get(), 0));
#else // ifdef _MSC_VER
    std::copy(sumFootprintNMDA.get(),
              sumFootprintNMDA.get() + _numNeurons,
              _sumFootprintNMDA.get());
#endif // ifdef _MSC_VER
}

void CPUSimulator::setCurrentSumFootprintGABAA(std::unique_ptr<float[]> const& sumFootprintGABAA)
{
#ifdef _MSC_VER
    std::copy(stdext::checked_array_iterator<float *>(sumFootprintGABAA.get(), 0),
              stdext::checked_array_iterator<float *>(sumFootprintGABAA.get(), _numNeurons),
              stdext::checked_array_iterator<float *>(_sumFootprintGABAA.get(), 0));
#else // ifdef _MSC_VER
    std::copy(sumFootprintGABAA.get(),
              sumFootprintGABAA.get() + _numNeurons,
              _sumFootprintGABAA.get());
#endif // ifdef _MSC_VER
}

std::vector<size_t> CPUSimulator::getTimesCalculations() const
{
    return _timesCalculations;
}

std::vector<size_t> CPUSimulator::getTimesConvolutions() const
{
    return _timesConvolutions;
}

float CPUSimulator::_f_w_EE(const float d)
{
    static const float sigma = 1;
    static const float p     = 32;

    // p varies between 8 to 64
    return tanh(1 / (2 * sigma * p))
           * exp(-abs(d) / (sigma * p));
}

float CPUSimulator::f_I_Na_m_inf(const float V)
{
    static const float theta_m = -30;
    static const float sigma_m = 9.5f;

    return pow((1 + exp(-(V - theta_m) / sigma_m)), -1);
}

float CPUSimulator::f_I_Na(const float V, const float h)
{
    static const float g_Na = 35;
    static const float V_Na = 55;

    return g_Na * pow(f_I_Na_m_inf(V), 3) * h * (V - V_Na);
}

float CPUSimulator::f_p_inf(const float V)
{
    static const float theta_p = -47;
    static const float sigma_p = 3;

    return pow((1 + exp(-(V - theta_p) / sigma_p)), -1);
}

float CPUSimulator::f_I_NaP(const float V)
{
    static const float g_NaP = 0.2f;
    static const float V_Na  = 55;

    return g_NaP * f_p_inf(V) * (V - V_Na);
}

float CPUSimulator::f_I_Kdr(const float V, const float n)
{
    static const float g_Kdr = 3;
    static const float V_K   = -90;

    return g_Kdr * pow(n, 4) * (V - V_K);
}

float CPUSimulator::f_I_Leak(const float V)
{
    static const float g_L = 0.05f;
    static const float V_L = -70;

    return g_L * (V - V_L);
}

float CPUSimulator::f_I_Kslow(const float V, const float z)
{
    static const float g_Kslow = 1.8f;
    static const float V_K     = -90;

    return g_Kslow * z * (V - V_K);
}

float CPUSimulator::f_I_AMPA(const float V, const float sumFootprintAMPA)
{
    static const float g_AMPA = 0.08f;
    static const float V_Glu  = 0;

    return g_AMPA * (V - V_Glu) * sumFootprintAMPA;
}

float CPUSimulator::f_f_NMDA(const float V)
{
    static const float theta_NMDA = 0;

    // theta_NMDA = -inf for [Mg2+]_0 = 0
    // and increases logarithmically with [Mg2+]_0
    static const float sigma_NMDA = 10;

    return pow(1 + exp(-(V - theta_NMDA) / sigma_NMDA), -1);
}

float CPUSimulator::f_I_NMDA(const float V, const float sumFootprintNMDA)
{
    static const float g_NMDA = 0.07f;
    static const float V_Glu  = 0;

    return g_NMDA * f_f_NMDA(V) * (V - V_Glu) * sumFootprintNMDA;
}

float CPUSimulator::f_I_GABAA(const float V, const float sumFootprintGABAA)
{
    static const float g_GABAA = 0.05f;
    static const float V_GABAA = -70;

    return g_GABAA * (V - V_GABAA) * sumFootprintGABAA;
}

float CPUSimulator::f_dV_dt(const float V, const float h, const float n, const float z, const float I_app, const float sumFootprintAMPA, const float sumFootprintNMDA, const float sumFootprintGABAA)
{
    return -f_I_Na(V, h)
           - f_I_NaP(V)
           - f_I_Kdr(V, n)
           - f_I_Kslow(V, z)
           - f_I_Leak(V)
           - f_I_AMPA(V, sumFootprintAMPA)
           - f_I_NMDA(V, sumFootprintNMDA)
           - f_I_GABAA(V, sumFootprintGABAA)
           + I_app;
}

float CPUSimulator::_f_I_Na_h_inf(const float V)
{
    static const float theta_h = -45;
    static const float sigma_h = -7;

    return pow((1 + exp(-(V - theta_h) / sigma_h)), -1);
}

float CPUSimulator::_f_I_Na_tau_h(const float V)
{
    static const float theta_th = -40.5f;
    static const float sigma_th = -6;

    return 0.1f + 0.75f * pow((1 + exp(-(V - theta_th) / sigma_th)), -1);
}

float CPUSimulator::_f_I_Na_dh_dt(const float h, const float V)
{
    return (_f_I_Na_h_inf(V) - h) / _f_I_Na_tau_h(V);
}

float CPUSimulator::_f_n_inf(const float V)
{
    static const float theta_n = -33;
    static const float sigma_n = 10;

    return pow(1 + exp(-(V - theta_n) / sigma_n), -1);
}

float CPUSimulator::_f_tau_n(const float V)
{
    static const float theta_tn = -33;
    static const float sigma_tn = -15;

    return 0.1f + 0.5f * pow(1 + exp(-(V - theta_tn) / sigma_tn), -1);
}

float CPUSimulator::_f_dn_dt(const float n, const float V)
{
    return (_f_n_inf(V) - n) / _f_tau_n(V);
}

float CPUSimulator::_f_z_inf(const float V)
{
    static const float theta_z = -39;
    static const float sigma_z = 5;

    return pow(1 + exp(-(V - theta_z) / sigma_z), -1);
}

float CPUSimulator::_f_dz_dt(const float z, const float V)
{
    static const float tau_z = 75;

    return (_f_z_inf(V) - z) / tau_z;
}

float CPUSimulator::_f_s_inf(const float V)
{
    static const float theta_s = -20;
    static const float sigma_s = 2;

    return pow(1 + exp(-(V - theta_s) / sigma_s), -1);
}

float CPUSimulator::_f_dsAMPA_dt(const float s_AMPA, const float V)
{
    static const float k_fP     = 1;
    static const float tau_AMPA = 5;

    return k_fP * _f_s_inf(V) * (1 - s_AMPA)
           - (s_AMPA / tau_AMPA);
}

float CPUSimulator::_f_dxNMDA_dt(const float x_NMDA, const float V)
{
    static const float k_xN      = 1;
    static const float tau2_NMDA = 14.3f;

    return k_xN * _f_s_inf(V) * (1 - x_NMDA)
           - (1 - _f_s_inf(V)) * x_NMDA / tau2_NMDA;
}

float CPUSimulator::_f_dsNMDA_dt(const float s_NMDA, const float x_NMDA)
{
    static const float k_fN     = 1;
    static const float tau_NMDA = 14.3f;

    return k_fN * x_NMDA * (1 - s_NMDA)
           - s_NMDA / tau_NMDA;
}

void CPUSimulator::runge4_f_dV_dt(const size_t idx)
{
    const state state_0 = _states[_ind_old * _numNeurons + idx];
    const float sumFootprintAMPA_loc = _sumFootprintAMPA[idx];
    const float sumFootprintNMDA_loc = _sumFootprintNMDA[idx];
    const float sumFootprintGABAA_loc = _sumFootprintGABAA[idx];

    float f1, f2, f3, f4;

    f1 = f_dV_dt(state_0.V, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);
    f2 = f_dV_dt(state_0.V + _dt * f1 / 2.0f, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);
    f3 = f_dV_dt(state_0.V + _dt * f2 / 2.0f, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);
    f4 = f_dV_dt(state_0.V + _dt * f3, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);

    _states[_ind_new * _numNeurons + idx].V = state_0.V + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_I_Na_dh_dt(const size_t idx)
{
    state state_0 = _states[_ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_I_Na_dh_dt(state_0.h, state_0.V);
    f2 = _f_I_Na_dh_dt(state_0.h + _dt * f1 / 2.0f, state_0.V);
    f3 = _f_I_Na_dh_dt(state_0.h + _dt * f2 / 2.0f, state_0.V);
    f4 = _f_I_Na_dh_dt(state_0.h + _dt * f3, state_0.V);

    _states[_ind_new * _numNeurons + idx].h = state_0.h + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_dn_dt(const size_t idx)
{
    state state_0 = _states[_ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_dn_dt(state_0.n, state_0.V);
    f2 = _f_dn_dt(state_0.n + _dt * f1 / 2.0f, state_0.V);
    f3 = _f_dn_dt(state_0.n + _dt * f2 / 2.0f, state_0.V);
    f4 = _f_dn_dt(state_0.n + _dt * f3, state_0.V);

    _states[_ind_new * _numNeurons + idx].n = state_0.n + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_dz_dt(const size_t idx)
{
    state state_0 = _states[_ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_dz_dt(state_0.z, state_0.V);
    f2 = _f_dz_dt(state_0.z + _dt * f1 / 2.0f, state_0.V);
    f3 = _f_dz_dt(state_0.z + _dt * f2 / 2.0f, state_0.V);
    f4 = _f_dz_dt(state_0.z + _dt * f3, state_0.V);

    _states[_ind_new * _numNeurons + idx].z = state_0.z + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_dsAMPA_dt(const size_t idx)
{
    state state_0 = _states[_ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_dsAMPA_dt(state_0.s_AMPA, state_0.V);
    f2 = _f_dsAMPA_dt(state_0.s_AMPA + _dt * f1 / 2.0f, state_0.V);
    f3 = _f_dsAMPA_dt(state_0.s_AMPA + _dt * f2 / 2.0f, state_0.V);
    f4 = _f_dsAMPA_dt(state_0.s_AMPA + _dt * f3, state_0.V);

    _states[_ind_new * _numNeurons + idx].s_AMPA = state_0.s_AMPA + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_dxNMDA_dt(const size_t idx)
{
    state state_0 = _states[_ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_dxNMDA_dt(state_0.x_NMDA, state_0.V);
    f2 = _f_dxNMDA_dt(state_0.x_NMDA + _dt * f1 / 2.0f, state_0.V);
    f3 = _f_dxNMDA_dt(state_0.x_NMDA + _dt * f2 / 2.0f, state_0.V);
    f4 = _f_dxNMDA_dt(state_0.x_NMDA + _dt * f3, state_0.V);

    _states[_ind_new * _numNeurons + idx].x_NMDA = state_0.x_NMDA + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_dsNMDA_dt(const size_t idx)
{
    state state_0 = _states[_ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_dsNMDA_dt(state_0.s_NMDA, state_0.x_NMDA);
    f2 = _f_dsNMDA_dt(state_0.s_NMDA + _dt * f1 / 2.0f, state_0.x_NMDA);
    f3 = _f_dsNMDA_dt(state_0.s_NMDA + _dt * f2 / 2.0f, state_0.x_NMDA);
    f4 = _f_dsNMDA_dt(state_0.s_NMDA + _dt * f3, state_0.x_NMDA);

    _states[_ind_new * _numNeurons + idx].s_NMDA = state_0.s_NMDA + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::convolutionAMPA()
{
    long nX = long(_nX);
    long nY = long(_nY);
    long nZ = long(_nZ);

    for (long x1 = 0; x1 < nX; ++x1) {
        for (long y1 = 0; y1 < nY; ++y1) {
            for (long z1 = 0; z1 < nZ; ++z1) {
                float sumFootprint = 0;
                size_t index1 = x1 + y1 * nX + z1 * nY;

                for (long x2 = 0; x2 < nX; ++x2) {
                    for (long y2 = 0; y2 < nY; ++y2) {
                        for (long z2 = 0; z2 < nZ; ++z2) {
                            size_t index2 = x2 + y2 * nX + z2 * nY;
                            size_t distanceIdx = abs(x2 - x1) + abs(y2 - y1) * nX + abs(z2 - z1) * nY;
                            sumFootprint += _distances[distanceIdx] * _states[_ind_old * _numNeurons + index2].s_AMPA;
                        }
                    }
                }

                _sumFootprintAMPA[index1] = sumFootprint;
            }
        }
    }
}

void CPUSimulator::convolutionNMDA()
{
    long nX = long(_nX);
    long nY = long(_nY);
    long nZ = long(_nZ);

    for (long x1 = 0; x1 < nX; ++x1) {
        for (long y1 = 0; y1 < nY; ++y1) {
            for (long z1 = 0; z1 < nZ; ++z1) {
                float sumFootprint = 0;
                size_t index1 = x1 + y1 * nX + z1 * nY;

                for (long x2 = 0; x2 < nX; ++x2) {
                    for (long y2 = 0; y2 < nY; ++y2) {
                        for (long z2 = 0; z2 < nZ; ++z2) {
                            size_t index2 = x2 + y2 * nX + z2 * nY;
                            size_t distanceIdx = abs(x2 - x1) + abs(y2 - y1) * nX + abs(z2 - z1) * nY;
                            sumFootprint += _distances[distanceIdx] * _states[_ind_old * _numNeurons + index2].s_NMDA;
                        }
                    }
                }

                _sumFootprintNMDA[index1] = sumFootprint;
            }
        }
    }
}

void CPUSimulator::convolutionGABAA()
{
    long nX = long(_nX);
    long nY = long(_nY);
    long nZ = long(_nZ);

    for (long x1 = 0; x1 < nX; ++x1) {
        for (long y1 = 0; y1 < nY; ++y1) {
            for (long z1 = 0; z1 < nZ; ++z1) {
                float sumFootprint = 0;
                size_t index1 = x1 + y1 * nX + z1 * nY;

                for (long x2 = 0; x2 < nX; ++x2) {
                    for (long y2 = 0; y2 < nY; ++y2) {
                        for (long z2 = 0; z2 < nZ; ++z2) {
                            size_t index2 = x2 + y2 * nX + z2 * nY;
                            size_t distanceIdx = abs(x2 - x1) + abs(y2 - y1) * nX + abs(z2 - z1) * nY;
                            sumFootprint += _distances[distanceIdx] * _states[_ind_old * _numNeurons + index2].s_GABAA;
                        }
                    }
                }

                _sumFootprintGABAA[index1] = sumFootprint;
            }
        }
    }
}

