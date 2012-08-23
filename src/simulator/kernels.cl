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

struct state
{
    float V;
    float h;
    float n;
    float z;
    float s_AMPA;
    float x_NMDA;
    float s_NMDA;
    float s_GABAA;
    float I_app;
};

inline float _f_I_Na_m_inf(const float V)
{
    const float theta_m = -30;
    const float sigma_m = 9.5f;

    return pow((1 + exp(-(V - theta_m) / sigma_m)), -1);
}

inline float _f_I_Na(const float V, const float h)
{
    const float g_Na = 35;
    const float V_Na = 55;

    return g_Na * pow(_f_I_Na_m_inf(V), 3) * h * (V - V_Na);
}

inline float _f_p_inf(const float V)
{
    const float theta_p = -47;
    const float sigma_p = 3;

    return pow((1 + exp(-(V - theta_p) / sigma_p)), -1);
}

inline float _f_I_NaP(const float V)
{
    const float g_NaP = 0.2f;
    const float V_Na  = 55;

    return g_NaP * _f_p_inf(V) * (V - V_Na);
}

inline float _f_I_Kdr(const float V, const float n)
{
    const float g_Kdr = 3;
    const float V_K   = -90;

    return g_Kdr * pow(n, 4) * (V - V_K);
}

inline float _f_I_Leak(const float V)
{
    const float g_L = 0.05f;
    const float V_L = -70;

    return g_L * (V - V_L);
}

inline float _f_I_Kslow(const float V, const float z)
{
    const float g_Kslow = 1.8f;
    const float V_K     = -90;

    return g_Kslow * z * (V - V_K);
}

inline float _f_I_AMPA(const float V, const float sumFootprintAMPA)
{
    const float g_AMPA = 0.08f;
    const float V_Glu  = 0;

    return g_AMPA * (V - V_Glu) * sumFootprintAMPA;
}

inline float _f_f_NMDA(const float V)
{
    const float theta_NMDA = 0;

    // theta_NMDA = -inf for [Mg2+]_0 = 0
    // and increases logarithmically with [Mg2+]_0
    const float sigma_NMDA = 10;

    return pow(1 + exp(-(V - theta_NMDA) / sigma_NMDA), -1);
}

inline float _f_I_NMDA(const float V, const float sumFootprintNMDA)
{
    const float g_NMDA = 0.07f;
    const float V_Glu  = 0;

    return g_NMDA * _f_f_NMDA(V) * (V - V_Glu) * sumFootprintNMDA;
}

inline float _f_I_GABAA(const float V, const float sumFootprintGABAA)
{
    const float g_GABAA = 0.05f;
    const float V_GABAA = -70;

    return g_GABAA * (V - V_GABAA) * sumFootprintGABAA;
}

inline float _f_dV_dt(const float V,
                      const float h,
                      const float n,
                      const float z,
                      const float I_app,
                      const float sumFootprintAMPA,
                      const float sumFootprintNMDA,
                      const float sumFootprintGABAA)
{
    return -_f_I_Na(V, h)
           - _f_I_NaP(V)
           - _f_I_Kdr(V, n)
           - _f_I_Kslow(V, z)
           - _f_I_Leak(V)
           - _f_I_AMPA(V, sumFootprintAMPA)
           - _f_I_NMDA(V, sumFootprintNMDA)
           - _f_I_GABAA(V, sumFootprintGABAA)
           + I_app;
}

__kernel void f_dV_dt(__global struct state *states_old,
                      __global struct state *states_new,
                      __global const float *sumFootprintAMPA,
                      __global const float *sumFootprintNMDA,
                      __global const float *sumFootprintGABAA,
                      const float dt)
{
    const unsigned int idx = get_global_id(0);
    // const unsigned int nX = 2;
    // const unsigned int nY = 2;
    // const unsigned int x = idx % nX;
    // const unsigned int y = idx / nX;
    // const unsigned int index = x + y * nY;

    const struct state state_0 = states_old[idx];
    const float sumFootprintAMPA_loc = sumFootprintAMPA[idx];
    const float sumFootprintNMDA_loc = sumFootprintNMDA[idx];
    const float sumFootprintGABAA_loc = sumFootprintGABAA[idx];

    float f1, f2, f3, f4;

    f1 = _f_dV_dt(state_0.V, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);
    f2 = _f_dV_dt(state_0.V + dt * f1 / 2.0f, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);
    f3 = _f_dV_dt(state_0.V + dt * f2 / 2.0f, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);
    f4 = _f_dV_dt(state_0.V + dt * f3, state_0.h, state_0.n, state_0.z, state_0.I_app, sumFootprintAMPA_loc, sumFootprintNMDA_loc, sumFootprintGABAA_loc);

    states_new[idx].V = state_0.V + dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

inline float _f_I_Na_h_inf(const float V)
{
    const float theta_h = -45;
    const float sigma_h = -7;

    return pow((1 + exp(-(V - theta_h) / sigma_h)), -1);
}

inline float _f_I_Na_tau_h(const float V)
{
    const float theta_th = -40.5f;
    const float sigma_th = -6;

    return 0.1f + 0.75f * pow((1 + exp(-(V - theta_th) / sigma_th)), -1);
}

inline float _f_I_Na_dh_dt(const float h, const float V)
{
    return (_f_I_Na_h_inf(V) - h) / _f_I_Na_tau_h(V);
}

__kernel void f_I_Na_dh_dt(__global struct state *states_old,
                           __global struct state *states_new,
                           const float dt)
{
    const unsigned int idx = get_global_id(0);

    struct state state_0 = states_old[idx];

    float f1, f2, f3, f4;

    f1 = _f_I_Na_dh_dt(state_0.h, state_0.V);
    f2 = _f_I_Na_dh_dt(state_0.h + dt * f1 / 2.0f, state_0.V);
    f3 = _f_I_Na_dh_dt(state_0.h + dt * f2 / 2.0f, state_0.V);
    f4 = _f_I_Na_dh_dt(state_0.h + dt * f3, state_0.V);

    states_new[idx].h = state_0.h + dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

inline float _f_n_inf(const float V)
{
    const float theta_n = -33;
    const float sigma_n = 10;

    return pow(1 + exp(-(V - theta_n) / sigma_n), -1);
}

inline float _f_tau_n(const float V)
{
    const float theta_tn = -33;
    const float sigma_tn = -15;

    return 0.1f + 0.5f * pow(1 + exp(-(V - theta_tn) / sigma_tn), -1);
}

inline float _f_dn_dt(const float n, const float V)
{
    return (_f_n_inf(V) - n) / _f_tau_n(V);
}

__kernel void f_dn_dt(__global struct state *states_old,
                      __global struct state *states_new,
                      const float dt)
{
    const unsigned int idx = get_global_id(0);

    struct state state_0 = states_old[idx];

    float f1, f2, f3, f4;

    f1 = _f_dn_dt(state_0.n, state_0.V);
    f2 = _f_dn_dt(state_0.n + dt * f1 / 2.0f, state_0.V);
    f3 = _f_dn_dt(state_0.n + dt * f2 / 2.0f, state_0.V);
    f4 = _f_dn_dt(state_0.n + dt * f3, state_0.V);

    states_new[idx].n = state_0.n + dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

inline float _f_z_inf(const float V)
{
    const float theta_z = -39;
    const float sigma_z = 5;

    return pow(1 + exp(-(V - theta_z) / sigma_z), -1);
}

inline float _f_dz_dt(const float z, const float V)
{
    const float tau_z = 75;

    return (_f_z_inf(V) - z) / tau_z;
}

__kernel void f_dz_dt(__global struct state *states_old,
                      __global struct state *states_new,
                      const float dt)
{
    const unsigned int idx = get_global_id(0);

    struct state state_0 = states_old[idx];

    float f1, f2, f3, f4;

    f1 = _f_dz_dt(state_0.z, state_0.V);
    f2 = _f_dz_dt(state_0.z + dt * f1 / 2.0f, state_0.V);
    f3 = _f_dz_dt(state_0.z + dt * f2 / 2.0f, state_0.V);
    f4 = _f_dz_dt(state_0.z + dt * f3, state_0.V);

    states_new[idx].z = state_0.z + dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

inline float _f_s_inf(const float V)
{
    const float theta_s = -20;
    const float sigma_s = 2;

    return pow(1 + exp(-(V - theta_s) / sigma_s), -1);
}

inline float _f_dsAMPA_dt(const float s_AMPA, const float V)
{
    const float k_fP     = 1;
    const float tau_AMPA = 5;

    return k_fP * _f_s_inf(V) * (1 - s_AMPA)
           - (s_AMPA / tau_AMPA);
}

__kernel void f_dsAMPA_dt(__global struct state *states_old,
                          __global struct state *states_new,
                          const float dt)
{
    const unsigned int idx = get_global_id(0);

    struct state state_0 = states_old[idx];

    float f1, f2, f3, f4;

    f1 = _f_dsAMPA_dt(state_0.s_AMPA, state_0.V);
    f2 = _f_dsAMPA_dt(state_0.s_AMPA + dt * f1 / 2.0f, state_0.V);
    f3 = _f_dsAMPA_dt(state_0.s_AMPA + dt * f2 / 2.0f, state_0.V);
    f4 = _f_dsAMPA_dt(state_0.s_AMPA + dt * f3, state_0.V);

    states_new[idx].s_AMPA = state_0.s_AMPA + dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

inline float _f_dxNMDA_dt(const float x_NMDA, const float V)
{
    const float k_xN      = 1;
    const float tau2_NMDA = 14.3f;

    return k_xN * _f_s_inf(V) * (1 - x_NMDA)
           - (1 - _f_s_inf(V)) * x_NMDA / tau2_NMDA;
}

__kernel void f_dxNMDA_dt(__global struct state *states_old,
                          __global struct state *states_new,
                          const float dt)
{
    const unsigned int idx = get_global_id(0);

    struct state state_0 = states_old[idx];

    float f1, f2, f3, f4;

    f1 = _f_dxNMDA_dt(state_0.x_NMDA, state_0.V);
    f2 = _f_dxNMDA_dt(state_0.x_NMDA + dt * f1 / 2.0f, state_0.V);
    f3 = _f_dxNMDA_dt(state_0.x_NMDA + dt * f2 / 2.0f, state_0.V);
    f4 = _f_dxNMDA_dt(state_0.x_NMDA + dt * f3, state_0.V);

    states_new[idx].x_NMDA = state_0.x_NMDA + dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

inline float _f_dsNMDA_dt(const float s_NMDA, const float x_NMDA)
{
    const float k_fN     = 1;
    const float tau_NMDA = 14.3f;

    return k_fN * x_NMDA * (1 - s_NMDA)
           - s_NMDA / tau_NMDA;
}

__kernel void f_dsNMDA_dt(__global struct state *states_old,
                          __global struct state *states_new,
                          const float dt)
{
    const unsigned int idx = get_global_id(0);

    struct state state_0 = states_old[idx];

    float f1, f2, f3, f4;

    f1 = _f_dsNMDA_dt(state_0.s_NMDA, state_0.x_NMDA);
    f2 = _f_dsNMDA_dt(state_0.s_NMDA + dt * f1 / 2.0f, state_0.x_NMDA);
    f3 = _f_dsNMDA_dt(state_0.s_NMDA + dt * f2 / 2.0f, state_0.x_NMDA);
    f4 = _f_dsNMDA_dt(state_0.s_NMDA + dt * f3, state_0.x_NMDA);

    states_new[idx].s_NMDA = state_0.s_NMDA + dt * (f1 + 2.0f * f2 + 2.0f * f3 + f4) / 6.0f;
}

__kernel void convolution(__global float *convolution_f_real, __global float *convolution_f_imag,
                          __global const float *distances_f_real, __global const float *distances_f_imag,
                          __global const float *sVals_f_real, __global const float *sVals_f_imag,
                          const float scaleFFT)
{
    const unsigned int idx = get_global_id(0);

    convolution_f_real[idx] = (distances_f_real[idx] * sVals_f_real[idx]
                               - distances_f_imag[idx] * sVals_f_imag[idx])
                              * scaleFFT;
    convolution_f_imag[idx] = (distances_f_real[idx] * sVals_f_imag[idx]
                               + distances_f_imag[idx] * sVals_f_real[idx])
                              * scaleFFT;
}

__kernel void prepareFFT_AMPA(__global const struct state *states_old,
                              __global float *sVals_real,
                              const unsigned int nX,
                              const unsigned int nY,
                              const unsigned int nZ)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int nFFTx = 2 * nX;
    const unsigned int x_sVals = idx % nX;
    const unsigned int y_sVals = nY > 1 ? idx / nY : 0;
    const unsigned int index_sVals = x_sVals + y_sVals * nFFTx;
    const unsigned int x_states = idx % nX;
    const unsigned int y_states = idx / nX;
    const unsigned int index_states = x_states + y_states * nX;

    sVals_real[index_sVals] = states_old[index_states].s_AMPA;
}

__kernel void prepareFFT_NMDA(__global const struct state *states_old,
                              __global float *sVals_real,
                              const unsigned int nX,
                              const unsigned int nY,
                              const unsigned int nZ)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int nFFTx = 2 * nX;
    const unsigned int x_sVals = idx % nX;
    const unsigned int y_sVals = nY > 1 ? idx / nY : 0;
    const unsigned int index_sVals = x_sVals + y_sVals * nFFTx;
    const unsigned int x_states = idx % nX;
    const unsigned int y_states = idx / nX;
    const unsigned int index_states = x_states + y_states * nX;

    sVals_real[index_sVals] = states_old[index_states].s_NMDA;
}

__kernel void prepareFFT_GABAA(__global const struct state *states_old,
                               __global float *sVals_real,
                               const unsigned int nX,
                               const unsigned int nY,
                               const unsigned int nZ)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int nFFTx = 2 * nX;
    const unsigned int x_sVals = idx % nX;
    const unsigned int y_sVals = nY > 1 ? idx / nY : 0;
    const unsigned int index_sVals = x_sVals + y_sVals * nFFTx;
    const unsigned int x_states = idx % nX;
    const unsigned int y_states = idx / nX;
    const unsigned int index_states = x_states + y_states * nX;

    sVals_real[index_sVals] = states_old[index_states].s_GABAA;
}

__kernel void postConvolution_AMPA(__global const float *convolution_real,
                                   __global float *sumFootprintAMPA,
                                   const unsigned int nX,
                                   const unsigned int nY,
                                   const unsigned int nZ)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int x_fp = idx % nX;
    const unsigned int y_fp = idx / nX;
    const unsigned int index_fp = x_fp + y_fp * nX;
    const unsigned int nFFTx = 2 * nX;
    const unsigned int x_conv = idx % nX + nX - 1;
    const unsigned int y_conv = nY > 1 ? idx / nY + nY - 1 : 0;
    const unsigned int index_conv = x_conv + y_conv * nFFTx;

    sumFootprintAMPA[index_fp] = convolution_real[index_conv];
}

__kernel void postConvolution_NMDA(__global const float *convolution_real,
                                   __global float *sumFootprintNMDA,
                                   const unsigned int nX,
                                   const unsigned int nY,
                                   const unsigned int nZ)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int x_fp = idx % nX;
    const unsigned int y_fp = idx / nX;
    const unsigned int index_fp = x_fp + y_fp * nX;
    const unsigned int nFFTx = 2 * nX;
    const unsigned int x_conv = idx % nX + nX - 1;
    const unsigned int y_conv = nY > 1 ? idx / nY + nY - 1 : 0;
    const unsigned int index_conv = x_conv + y_conv * nFFTx;

    sumFootprintNMDA[index_fp] = convolution_real[index_conv];
}

__kernel void postConvolution_GABAA(__global const float *convolution_real,
                                    __global float *sumFootprintGABAA,
                                    const unsigned int nX,
                                    const unsigned int nY,
                                    const unsigned int nZ)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int x_fp = idx % nX;
    const unsigned int y_fp = idx / nX;
    const unsigned int index_fp = x_fp + y_fp * nX;
    const unsigned int nFFTx = 2 * nX;
    const unsigned int x_conv = idx % nX + nX - 1;
    const unsigned int y_conv = nY > 1 ? idx / nY + nY - 1 : 0;
    const unsigned int index_conv = x_conv + y_conv * nFFTx;

    sumFootprintGABAA[index_fp] = convolution_real[index_conv];
}
