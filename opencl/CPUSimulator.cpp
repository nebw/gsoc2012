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

#include <algorithm>

CPUSimulator::CPUSimulator(const unsigned int nX,
                           const unsigned int nY,
                           const unsigned int nZ,
                           const unsigned int timesteps,
                           const float dt,
                           state const& state_0,
                           const Convolution convolution)
    : _nX(nX),
      _nY(nY),
      _nZ(nZ),
      _numNeurons(nX * nY * nZ),
      _timesteps(timesteps),
      _dt(dt),
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

    for (unsigned int i = _numNeurons; i < 2 * _numNeurons; ++i)
    {
        _states[i] = state_0;
    }

    for (unsigned int i = 0; i < _numNeurons; ++i)
    {
        _distances[i] = _f_w_EE(i);
    }
}

void CPUSimulator::step()
{
    unsigned int ind_old = _t % 2;

    if (_convolution)
    {
        computeConvolutions(ind_old);
    }

    computeRungeKuttaApproximations(ind_old);

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

void CPUSimulator::computeConvolutions(unsigned int ind_old)
{
    convolutionAMPA(ind_old);

    convolutionNMDA(ind_old);

    // convolutionGABAA();
}

void CPUSimulator::computeRungeKuttaApproximations(unsigned int ind_old)
{
    for (unsigned int idx = 0; idx < _numNeurons; ++idx)
    {
        runge4_f_dV_dt(idx, ind_old);
        runge4_f_I_Na_dh_dt(idx, ind_old);
        runge4_f_dsAMPA_dt(idx, ind_old);
        runge4_f_dn_dt(idx, ind_old);
        runge4_f_dz_dt(idx, ind_old);
        runge4_f_dxNMDA_dt(idx, ind_old);
        runge4_f_dsNMDA_dt(idx, ind_old);
    }
}

std::unique_ptr<state[]> const& CPUSimulator::getCurrentStates() const
{
    return _states;
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

void CPUSimulator::setCurrentStates(std::unique_ptr<state[]> const& states)
{
    std::copy(states.get(), states.get() + (2 * _numNeurons), _states.get());
}

void CPUSimulator::setCurrentSumFootprintAMPA(std::unique_ptr<float[]> const& sumFootprintAMPA)
{
    std::copy(sumFootprintAMPA.get(), sumFootprintAMPA.get() + _numNeurons, _sumFootprintAMPA.get());
}

void CPUSimulator::setCurrentSumFootprintNMDA(std::unique_ptr<float[]> const& sumFootprintNMDA)
{
    std::copy(sumFootprintNMDA.get(), sumFootprintNMDA.get() + _numNeurons, _sumFootprintNMDA.get());
}

void CPUSimulator::setCurrentSumFootprintGABAA(std::unique_ptr<float[]> const& sumFootprintGABAA)
{
    std::copy(sumFootprintGABAA.get(), sumFootprintGABAA.get() + _numNeurons, _sumFootprintGABAA.get());
}

std::vector<unsigned long> CPUSimulator::getTimesCalculations() const
{
    throw std::exception("The method or operation is not implemented.");
}

std::vector<unsigned long> CPUSimulator::getTimesFFTW() const
{
    throw std::exception("The method or operation is not implemented.");
}

std::vector<unsigned long> CPUSimulator::getTimesClFFT() const
{
    throw std::exception("The method or operation is not implemented.");
}

float CPUSimulator::_f_w_EE(const int d)
{
    static const float sigma = 1;
    static const float p     = 32;

    // TODO: p varies between 8 to 64
    //
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

    // TODO: theta_NMDA = -inf for [Mg2+]_0 = 0
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

float CPUSimulator::_f_dsNMDA_dt(const float s_NMDA, const float x_NMDA, const float V)
{
    static const float k_fN     = 1;
    static const float tau_NMDA = 14.3f;

    return k_fN * x_NMDA * (1 - s_NMDA)
           - s_NMDA / tau_NMDA;
}

void CPUSimulator::runge4_f_dV_dt(const unsigned int idx, const unsigned int ind_old)
{
    const unsigned int ind_new = 1 - ind_old;

    const state state_0 = _states[ind_old * _numNeurons + idx];
    const float sumFootprintAMPA_loc = _sumFootprintAMPA[idx];
    const float sumFootprintNMDA_loc = _sumFootprintNMDA[idx];
    const float sumFootprintGABAA_loc = _sumFootprintGABAA[idx];

    float f1, f2, f3, f4;

    f1 = f_dV_dt(state_0.V, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);
    f2 = f_dV_dt(state_0.V + _dt * f1 / 2.0f, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);
    f3 = f_dV_dt(state_0.V + _dt * f2 / 2.0f, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);
    f4 = f_dV_dt(state_0.V + _dt * f3, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);

    _states[ind_new * _numNeurons + idx].V = state_0.V + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_I_Na_dh_dt(const unsigned int idx, const unsigned int ind_old)
{
    const unsigned int ind_new = 1 - ind_old;

    state state_0 = _states[ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_I_Na_dh_dt(state_0.h, state_0.V);
    f2 = _f_I_Na_dh_dt(state_0.h + _dt * f1 / 2.0f, state_0.V);
    f3 = _f_I_Na_dh_dt(state_0.h + _dt * f2 / 2.0f, state_0.V);
    f4 = _f_I_Na_dh_dt(state_0.h + _dt * f3, state_0.V);

    _states[ind_new * _numNeurons + idx].h = state_0.h + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_dn_dt(const unsigned int idx, const unsigned int ind_old)
{
    const unsigned int ind_new = 1 - ind_old;

    state state_0 = _states[ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_dn_dt(state_0.n, state_0.V);
    f2 = _f_dn_dt(state_0.n + _dt * f1 / 2.0f, state_0.V);
    f3 = _f_dn_dt(state_0.n + _dt * f2 / 2.0f, state_0.V);
    f4 = _f_dn_dt(state_0.n + _dt * f3, state_0.V);

    _states[ind_new * _numNeurons + idx].n = state_0.n + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_dz_dt(const unsigned int idx, const unsigned int ind_old)
{
    const unsigned int ind_new = 1 - ind_old;

    state state_0 = _states[ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_dz_dt(state_0.z, state_0.V);
    f2 = _f_dz_dt(state_0.z + _dt * f1 / 2.0f, state_0.V);
    f3 = _f_dz_dt(state_0.z + _dt * f2 / 2.0f, state_0.V);
    f4 = _f_dz_dt(state_0.z + _dt * f3, state_0.V);

    _states[ind_new * _numNeurons + idx].z = state_0.z + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_dsAMPA_dt(const unsigned int idx, const unsigned int ind_old)
{
    const unsigned int ind_new = 1 - ind_old;

    state state_0 = _states[ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_dsAMPA_dt(state_0.s_AMPA, state_0.V);
    f2 = _f_dsAMPA_dt(state_0.s_AMPA + _dt * f1 / 2.0f, state_0.V);
    f3 = _f_dsAMPA_dt(state_0.s_AMPA + _dt * f2 / 2.0f, state_0.V);
    f4 = _f_dsAMPA_dt(state_0.s_AMPA + _dt * f3, state_0.V);

    _states[ind_new * _numNeurons + idx].s_AMPA = state_0.s_AMPA + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_dxNMDA_dt(const unsigned int idx, const unsigned int ind_old)
{
    const unsigned int ind_new = 1 - ind_old;

    state state_0 = _states[ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_dxNMDA_dt(state_0.x_NMDA, state_0.V);
    f2 = _f_dxNMDA_dt(state_0.x_NMDA + _dt * f1 / 2.0f, state_0.V);
    f3 = _f_dxNMDA_dt(state_0.x_NMDA + _dt * f2 / 2.0f, state_0.V);
    f4 = _f_dxNMDA_dt(state_0.x_NMDA + _dt * f3, state_0.V);

    _states[ind_new * _numNeurons + idx].x_NMDA = state_0.x_NMDA + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::runge4_f_dsNMDA_dt(const unsigned int idx, const unsigned int ind_old)
{
    const unsigned int ind_new = 1 - ind_old;

    state state_0 = _states[ind_old * _numNeurons + idx];

    float f1, f2, f3, f4;

    f1 = _f_dsNMDA_dt(state_0.s_NMDA, state_0.x_NMDA, state_0.V);
    f2 = _f_dsNMDA_dt(state_0.s_NMDA + _dt * f1 / 2.0f, state_0.x_NMDA, state_0.V);
    f3 = _f_dsNMDA_dt(state_0.s_NMDA + _dt * f2 / 2.0f, state_0.x_NMDA, state_0.V);
    f4 = _f_dsNMDA_dt(state_0.s_NMDA + _dt * f3, state_0.x_NMDA, state_0.V);

    _states[ind_new * _numNeurons + idx].s_NMDA = state_0.s_NMDA + _dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

void CPUSimulator::convolutionAMPA(const unsigned int ind_old)
{
    for (int i = 0; i < static_cast<int>(_numNeurons); ++i)
    {
        float sumFootprint = 0;

        for (int j = 0; j < static_cast<int>(_numNeurons); ++j)
        {
            sumFootprint += _distances[abs(j - i)] * _states[ind_old * _numNeurons + j].s_AMPA;
        }

        _sumFootprintAMPA[i] = sumFootprint;
    }
}

void CPUSimulator::convolutionNMDA(const unsigned int ind_old)
{
    for (int i = 0; i < static_cast<int>(_numNeurons); ++i)
    {
        float sumFootprint = 0;

        for (int j = 0; j < static_cast<int>(_numNeurons); ++j)
        {
            sumFootprint += _distances[abs(j - i)] * _states[ind_old * _numNeurons + j].s_NMDA;
        }

        _sumFootprintNMDA[i] = sumFootprint;
    }
}

void CPUSimulator::convolutionGABAA()
{
    throw std::exception("The method or operation is not implemented.");
}
