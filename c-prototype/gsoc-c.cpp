#if defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)

// # define USE_STD_THREADS
#endif // if defined(unix) || defined(__unix) || defined(__unix__) ||
       // defined(__APPLE__)
// #define PLOT

#ifdef USE_STD_THREADS
# ifndef _VARIADIC_MAX
#  define _VARIADIC_MAX 10
# endif // ifndef _VARIADIC_MAX
# include <thread>
#endif  // ifdef USE_STD_THREADS

#include "gnuplot_i.hpp"

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#if defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
# include <fftw3.h>
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || \
    defined(__TOS_WIN__)
# include "fftw/fftw3.h"
#endif // if defined(unix) || defined(__unix) || defined(__unix__) ||
       // defined(__APPLE__)


#include <ctime>
#include <unordered_set>

#ifdef USE_STD_THREADS
std::mutex m;
#endif // ifdef USE_STD_THREADS

template<class T1, class T2>
std::vector<T2>linSpaceVec(T1 n, T2 start, T2 stop) {
    std::vector<T2> vec;
    T2 step = (stop - start) / n;

    for (T2 i = start; i < stop; i += step) {
        vec.push_back(i);
    }
    return vec;
}

inline void arrAdd(const double *a, const double *b, double *c, int len)
{
    for (int i = 0; i < len; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

inline void arrMul(const double *a, double *b, double fac, int len)
{
    for (int i = 0; i < len; ++i)
    {
        b[i] = a[i] * fac;
    }
}

// state[timesteps][numNeurons][stateSize];
int runge_kutta_generic(
    const double     **states,
    double            *state_K1,
    double            *state_K2,
    double            *state_K3,
    double            *state_K4,
    double            *state_temp_1,
    double            *state_temp_2,
    double            *state_temp_3,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    const unsigned int returnIndex,
    const double       dt,
    void               (*f)(
        const          double **,
        const unsigned int,
        const unsigned int,
        const unsigned int,
        double *),
    double            *resultState)
{
    //    for (unsigned int i = 0; i < stateSize; ++i)
    //    {
    //        state_K1[i]     = states[indexOfNeuron][i];
    //        state_K2[i]     = states[indexOfNeuron][i];
    //        state_K3[i]     = states[indexOfNeuron][i];
    //        state_K4[i]     = states[indexOfNeuron][i];
    //        state_temp_1[i] = states[indexOfNeuron][i];
    //        state_temp_2[i] = states[indexOfNeuron][i];
    //        state_temp_3[i] = states[indexOfNeuron][i];
    //    }

    // f(y) -> state_temp_1
    (*f)(states, numNeurons, stateSize, indexOfNeuron, state_temp_1);

    // dt * f(y) -> state_K1
    arrMul(state_temp_1, state_K1, dt, stateSize);

    // // 1/2 * k1 -> state_temp_1
    // arrMul( state_K1, state_temp_1, 0.5, stateSize );
    // // y + 1/2 * k1 -> state_temp2
    // arrAdd( state_temp_1, states[indexOfNeuron], state_temp_2, stateSize );
    // // f(y + 1/2 * k1) -> state_temp_1
    // (*f)( (const double**)&state_temp_2, numNeurons, stateSize, 0,
    // state_temp_1 );
    // arrMul( state_temp_1, state_temp_2, 1, stateSize );
    // // dt * f(y +  1/2 * k1) -> state_K2
    // arrMul( state_temp_2, state_K2, dt, stateSize );

    // printf( "%f\n", state_K2[1] );

    // // 1/2 * k2 -> state_temp_1
    // arrMul( state_K2, state_temp_1, 0.5, stateSize );
    // // y + 1/2 * k2 -> state_temp2
    // arrAdd( state_temp_1, states[indexOfNeuron], state_temp_2, stateSize );
    // // f(y + 1/2 * k2) -> state_temp_1
    // (*f)( (const double**)&state_temp_2, numNeurons, stateSize, 0,
    // state_temp_1 );
    // arrMul( state_temp_1, state_temp_2, 1 , stateSize );
    // // dt * f(y + 1/2 * k2) -> state_K3
    // arrMul( state_temp_2, state_K3, dt, stateSize );

    // printf( "%f\n", state_K3[1] );

    // // y + k3 -> state_temp2
    // arrAdd( state_K3, states[indexOfNeuron], state_temp_2, stateSize );
    // // f(y + k3) -> state_temp_1
    // (*f)( (const double**)&state_temp_2, numNeurons, stateSize, 0,
    // state_temp_1 );
    // arrMul( state_temp_1, state_temp_2, 1 , stateSize );
    // // dt * f(y + k3) -> state_K4
    // arrMul( state_temp_2, state_K4, dt, stateSize );

    // printf( "%f\n", state_K4[1] );

    // // 2 * k2 -> state_temp_1
    // arrMul( state_K2, state_temp_1, dt, stateSize );
    // // 2 * k3 -> state_temp_2
    // arrMul( state_K3, state_temp_2, dt, stateSize );
    // // 2 * k2 + 2 * k3 -> state_temp_3
    // arrAdd( state_temp_1, state_temp_2, state_temp_3, stateSize );
    // // k1 + 2 * k2 + 2 * k3 -> state_temp_1
    // arrAdd( state_K1, state_temp_3, state_temp_1, stateSize );
    // // k4 + k1 + 2 * k2 + 2 * k3 -> state_temp_2
    // arrAdd( state_K4, state_temp_1, state_temp_2, stateSize );
    // // 1/6 * (k4 + k1 + 2 * k2 + 2 * k3) -> state_temp_1
    // arrMul( state_temp_2, state_temp_1, 1./6., stateSize );
    // // y + 1/6 * (k4 + k1 + 2 * k2 + 2 * k3) -> state_temp_3
    arrAdd(states[indexOfNeuron], state_K1, state_temp_3, stateSize);

    // printf( "%f\n", state_temp_3[1] );
    // getchar();

    resultState[returnIndex] = state_temp_3[returnIndex];

    return 0;
}

namespace golomb {
inline double _f_I_Na_m_inf(const double V)
{
    static const double theta_m = -30;
    static const double sigma_m = 9.5;

    return pow((1 + exp(-(V - theta_m) / sigma_m)), -1);
}

inline double _f_I_Na_h_inf(const double V)
{
    static const double theta_h = -45;
    static const double sigma_h = -7;

    return pow((1 + exp(-(V - theta_h) / sigma_h)), -1);
}

inline double _f_I_Na_tau_h(const double V)
{
    static const double theta_th = -40.5;
    static const double sigma_th = -6;

    return 0.1 + 0.75 * pow((1 + exp(-(V - theta_th) / sigma_th)), -1);
}

inline double _f_p_inf(const double V)
{
    static const double theta_p = -47;
    static const double sigma_p = 3;

    return pow((1 + exp(-(V - theta_p) / sigma_p)), -1);
}

inline double _f_I_NaP(const double V)
{
    static const double g_NaP = 0.2;
    static const double V_Na  = 55;

    return g_NaP * _f_p_inf(V) * (V - V_Na);
}

inline double _f_I_Kdr(const double V, const double n)
{
    static const double g_Kdr = 3;
    static const double V_K   = -90;

    return g_Kdr * pow(n, 4) * (V - V_K);
}

inline double _f_tau_n(const double V)
{
    static const double theta_tn = -33;
    static const double sigma_tn = -15;

    return 0.1 + 0.5 * pow(1 + exp(-(V - theta_tn) / sigma_tn), -1);
}

inline double _f_s_inf(const double V)
{
    static const double theta_s = -20;
    static const double sigma_s = 2;

    return pow(1 + exp(-(V - theta_s) / sigma_s), -1);
}

inline double _f_n_inf(const double V)
{
    static const double theta_n = -33;
    static const double sigma_n = 10;

    return pow(1 + exp(-(V - theta_n) / sigma_n), -1);
}

inline double _f_z_inf(const double V)
{
    static const double theta_z = -39;
    static const double sigma_z = 5;

    return pow(1 + exp(-(V - theta_z) / sigma_z), -1);
}

inline double _f_I_Kslow(const double V, const double z)
{
    static const double g_Kslow = 1.8;
    static const double V_K     = -90;

    return g_Kslow * z * (V - V_K);
}

inline double _f_I_Na(const double V, const double h)
{
    static const double g_Na = 35;
    static const double V_Na = 55;

    return g_Na * pow(_f_I_Na_m_inf(V), 3) * h * (V - V_Na);
}

inline double _f_I_Leak(const double V)
{
    static const double g_L = 0.05;
    static const double V_L = -70;

    return g_L * (V - V_L);
}

inline double _f_w_EE(const int j)
{
    static const double sigma = 1;
    static const double p     = 32;

    // TODO: p varies between 8 to 64
    //
    return tanh(1 / (2 * sigma * p))
           * exp(-abs(j) / (sigma * p));
}

double _f_I_AMPA(
    const double **states,
    const int      indexOfNeuron,
    const int      numNeurons)
{
    static const double g_AMPA = 0.08;
    static const double V_Glu  = 0;

    double sumFootprint = 0;

    for (int i = 0; i < numNeurons; ++i)
    {
        sumFootprint += _f_w_EE(indexOfNeuron - i) * states[i][5];
    }

    return g_AMPA * (states[indexOfNeuron][1] - V_Glu) * sumFootprint;
}

inline double _f_f_NMDA(const double V)
{
    static const double theta_NMDA = 0;

    // TODO: theta_NMDA = -inf for [Mg2+]_0 = 0
    // and increases logarithmically with [Mg2+]_0
    static const double sigma_NMDA = 10;

    return pow(1 + exp(-(V - theta_NMDA) / sigma_NMDA), -1);
}

double _f_I_NMDA(
    const double **states,
    const int      indexOfNeuron,
    const int      numNeurons)
{
    static const double g_NMDA = 0.07;
    static const double V_Glu  = 0;

    double sumFootprint = 0;

    for (int i = 0; i < numNeurons; ++i)
    {
        sumFootprint += _f_w_EE(indexOfNeuron - i) * states[i][6];
    }

    return g_NMDA * _f_f_NMDA(states[indexOfNeuron][1]) *
           (states[indexOfNeuron][1] - V_Glu) * sumFootprint;
}

inline double _f_w_IE(const int j)
{
    static const double sigma = 0.5;
    static const double p     = 32;

    // TODO: p varies between 8 to 64

    return tanh(1 / (2 * sigma * p))
           * exp(-abs(j) / (sigma * p));
}

double _f_I_GABAA(
    const double **states,
    const int      indexOfNeuron,
    const int      numNeurons)
{
    static const double g_GABAA = 0.05;
    static const double V_GABAA = -70;

    double sumFootprint = 0;

    for (int i = 0; i < numNeurons; ++i)
    {
        sumFootprint += _f_w_EE(indexOfNeuron - i) * states[i][8];
    }

    return g_GABAA * (states[indexOfNeuron][1] - V_GABAA) * sumFootprint;
}

inline void f_dsAMPA_dt(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    static const double k_fP     = 1;
    static const double tau_AMPA = 5;

    resultState[5] = k_fP * _f_s_inf(states[indexOfNeuron][1])
                     * (1 - states[indexOfNeuron][5])
                     - (states[indexOfNeuron][5] / tau_AMPA);
}

inline void f_dxNMDA_dt(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    static const double k_xN      = 1;
    static const double tau2_NMDA = 14.3;

    resultState[6] = k_xN * _f_s_inf(states[indexOfNeuron][1])
                     * (1 - states[indexOfNeuron][6])
                     - (1 - _f_s_inf(states[indexOfNeuron][1]))
                     * states[indexOfNeuron][6] / tau2_NMDA;
}

inline void f_dsNMDA_dt(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    static const double k_fN     = 1;
    static const double tau_NMDA = 14.3;

    resultState[7] = k_fN * states[indexOfNeuron][6]
                     * (1 - states[indexOfNeuron][7])
                     - states[indexOfNeuron][7] / tau_NMDA;
}

inline void f_I_Na_dh_dt(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    resultState[2] =
        (_f_I_Na_h_inf(states[indexOfNeuron][1])
         - states[indexOfNeuron][2])
        / _f_I_Na_tau_h(states[indexOfNeuron][1]);
}

inline void f_dn_dt(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    resultState[3] =
        (_f_n_inf(states[indexOfNeuron][1])
         - states[indexOfNeuron][3])
        / _f_tau_n(states[indexOfNeuron][1]);
}

inline void f_dz_dt(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    static const double tau_z = 75;

    resultState[4] =
        (_f_z_inf(states[indexOfNeuron][1])
         - states[indexOfNeuron][4]) / tau_z;
}

inline void f_dV_dt(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    resultState[1] =
        -_f_I_Na(states[indexOfNeuron][1], states[indexOfNeuron][2])
        - _f_I_NaP(states[indexOfNeuron][1])
        - _f_I_Kdr(states[indexOfNeuron][1], states[indexOfNeuron][3])
        - _f_I_Kslow(states[indexOfNeuron][1], states[indexOfNeuron][4])
        - _f_I_Leak(states[indexOfNeuron][1])
        - _f_I_AMPA(states, indexOfNeuron, numNeurons)
        - _f_I_NMDA(states, indexOfNeuron, numNeurons)
        - _f_I_GABAA(states, indexOfNeuron, numNeurons)
        + states[indexOfNeuron][9];
}

void f_I_NMDA_FFT(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    double           **resultStates)
{
    fftw_complex *distances, *sNMDAs, *convolution;
    fftw_complex *distances_f, *sNMDAs_f, *convolution_f;
    fftw_plan     p, pinv;
    const double scale = 1./numNeurons;

#ifdef USE_STD_THREADS
    {
        std::lock_guard<std::mutex> lk(m);
        distances = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
        sNMDAs    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
        convolution = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
        distances_f = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
        sNMDAs_f    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
        convolution_f    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
    }
#else // ifdef USE_STD_THREADS
        distances = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
        sNMDAs    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
        convolution = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
        distances_f = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
        sNMDAs_f    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
        convolution_f    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * numNeurons);
#endif // ifdef USE_STD_THREADS

    for (unsigned int i = 0; i < numNeurons; ++i)
    {
        distances[i][0] = _f_w_EE(i);
        distances[i][1] = 0;
    }

    for (unsigned int i = 0; i < numNeurons; ++i)
    {
        sNMDAs[i][0] = states[i][6];
        sNMDAs[i][1] = 0;
    }

    p = fftw_plan_dft_1d(numNeurons,
                         distances,
                         distances_f,
                         FFTW_FORWARD,
                         FFTW_ESTIMATE);
    fftw_execute(p);

    p = fftw_plan_dft_1d(numNeurons,
                         sNMDAs,
                         sNMDAs_f,
                         FFTW_FORWARD,
                         FFTW_ESTIMATE);
    fftw_execute(p);

    // convolution in frequency domain
    for(unsigned int i = 0; i < numNeurons; ++i)
    {
        convolution_f[i][0] = (distances_f[i][0] * sNMDAs_f[i][0]
            - distances_f[i][1] * sNMDAs_f[i][1]) * scale;
        convolution_f[i][1] = (distances_f[i][0] * sNMDAs_f[i][1]
            + distances_f[i][1] * sNMDAs_f[i][0]) * scale;
    }

    pinv = fftw_plan_dft_1d(numNeurons,
                            convolution_f,
                            convolution,
                            FFTW_BACKWARD,
                            FFTW_ESTIMATE);
    fftw_execute(pinv);

    if (states[1][6] > 0.1)
    {
        for (unsigned int i = 0; i < numNeurons; ++i)
        {
            printf("Input i=%f\n", states[i][6]);
        }
        // compare with expected result
        for (unsigned int i = 0; i < numNeurons; ++i)
        {
            double sumFootprint = 0;

            for (int j = 0; j < numNeurons; ++j)
            {
                sumFootprint += _f_w_EE(i - j) * states[j][6];
            }
            printf("i=%d, FFT: r%f, i%f : Expected: %f\n", i, convolution[i][0], convolution[i][1], sumFootprint);
        }
        getchar();
    }

#ifdef USE_STD_THREADS
    {
        std::lock_guard<std::mutex> lk(m);
        fftw_destroy_plan(p);
        fftw_destroy_plan(pinv);

        fftw_free(distances), fftw_free(sNMDAs), fftw_free(convolution);
        fftw_free(distances_f), fftw_free(sNMDAs_f), fftw_free(convolution_f);
    }
#else // ifdef USE_STD_THREADS
    fftw_destroy_plan(p);
    fftw_destroy_plan(pinv);

    fftw_free(distances), fftw_free(sNMDAs), fftw_free(convolution);
    fftw_free(distances_f), fftw_free(sNMDAs_f), fftw_free(convolution_f);
#endif // ifdef USE_STD_THREADS
}
}

namespace wang_buzsaki {
inline double _f_I_Na_m_alpha(const double V)
{
    return 0.5 * (V + 35.0) /
           (1 - exp(-(V + 35.0) / 10.0));
}

inline double _f_I_Na_m_beta(const double V)
{
    return 20.0 * exp(-(V + 60.0) / 18.0);
}

inline double _f_I_Na_m_inf(const double V)
{
    return _f_I_Na_m_alpha(V) /
           (_f_I_Na_m_alpha(V) + _f_I_Na_m_beta(V));
}

inline double _f_I_Na(const double V, const double h)
{
    static const double g_Na = 35;
    static const double V_Na = 55;

    return g_Na * pow(_f_I_Na_m_inf(V), 3) * h * (V - V_Na);
}

inline double _f_I_Na_h_alpha(const double V)
{
    return 0.35 * exp(-(V + 58.0) / 20.0);
}

inline double _f_I_Na_h_beta(const double V)
{
    return 5.0 / (1 + exp(-(V + 28.0) / 10.0));
}

inline void f_I_Na_dh_dt(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    resultState[2] =
        _f_I_Na_h_alpha(states[indexOfNeuron][1])
        * (1 - states[indexOfNeuron][2])
        - _f_I_Na_h_beta(states[indexOfNeuron][1])
        * states[indexOfNeuron][2];
}

inline double _f_I_Kdr(const double V, const double n)
{
    static const double g_Kdr = 9;
    static const double V_K   = -90;

    return g_Kdr * pow(n, 4) * (V - V_K);
}

inline double _f_I_Kdr_n_alpha(const double V)
{
    return 0.05 * (V + 34.0)
           / (1 - exp(-(V + 34.0) / 10.0));
}

inline double _f_I_Kdr_n_beta(const double V)
{
    return 0.625 * exp(-(V + 44.0) / 80.0);
}

inline void f_I_Kdr_dn_dt(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    resultState[3] =
        _f_I_Kdr_n_alpha(states[indexOfNeuron][1])
        * (1 - states[indexOfNeuron][3])
        - _f_I_Kdr_n_beta(states[indexOfNeuron][1])
        * states[indexOfNeuron][3];
}

inline double _f_I_Leak(const double V)
{
    static const double g_L = 0.1;
    static const double V_L = -65;

    return g_L * (V - V_L);
}

inline double _f_w_EI(const int j)
{
    static const double sigma = 1;
    static const double p     = 32;

    // TODO: p varies between 8 to 64

    return tanh(1 / (2 * sigma * p))
           * exp(-abs(j) / (sigma * p));
}

double _f_I_AMPA(
    const double **states,
    const int      indexOfNeuron,
    const int      numNeurons)
{
    static const double g_EI_AMPA = 0.2;
    static const double V_Glu     = 0;

    double sumFootprint = 0;

    for (int i = 0; i < numNeurons; ++i)
    {
        sumFootprint += _f_w_EI(indexOfNeuron - i) * states[i][5];
    }

    return g_EI_AMPA * (states[indexOfNeuron][1] - V_Glu) * sumFootprint;
}

inline double _f_f_NMDA(const double V)
{
    static const double theta_NMDA = 0;

    // TODO: theta_NMDA = -inf for [Mg2+]_0 = 0
    // and increases logarithmically with [Mg2+]_0
    static const double sigma_NMDA = 10;

    return pow(1 + exp(-(V - theta_NMDA) / sigma_NMDA), -1);
}

double _f_I_NMDA(
    const double **states,
    const int      indexOfNeuron,
    const int      numNeurons)
{
    // 0.0 or 0.05
    static const double g_EI_NMDA = 0.05;
    static const double V_Glu     = 0;

    double sumFootprint = 0;

    for (int i = 0; i < numNeurons; ++i)
    {
        sumFootprint += _f_w_EI(indexOfNeuron - i) * states[i][6];
    }

    return g_EI_NMDA * _f_f_NMDA(states[indexOfNeuron][1]) *
           (states[indexOfNeuron][1] - V_Glu) * sumFootprint;
}

inline double _f_s_inf(const double V)
{
    static const double theta_s = -20;
    static const double sigma_s = 2;

    return pow(1 + exp(-(V - theta_s) / sigma_s), -1);
}

inline void f_dsGABAA_dt(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    static const double k_fA      = 1;
    static const double tau_GABAA = 10;

    resultState[8] = k_fA * _f_s_inf(states[indexOfNeuron][1])
                     * (1 - states[indexOfNeuron][8])
                     - states[indexOfNeuron][8] / tau_GABAA;
}

inline void f_dV_dt(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    resultState[1] =
        -_f_I_Na(states[indexOfNeuron][1], states[indexOfNeuron][2])
        - _f_I_Kdr(states[indexOfNeuron][1], states[indexOfNeuron][3])
        - _f_I_Leak(states[indexOfNeuron][1])
        - _f_I_AMPA(states, indexOfNeuron, numNeurons)
        - _f_I_NMDA(states, indexOfNeuron, numNeurons)
        + states[indexOfNeuron][9];
}
}

#define runge_kutta(f, i)                \
    runge_kutta_generic(                 \
        (const double **)state[ind_old], \
        state_K1,                        \
        state_K2,                        \
        state_K3,                        \
        state_K4,                        \
        state_temp_1,                    \
        state_temp_2,                    \
        state_temp_3,                    \
        numNeurons, stateSize,           \
        indexOfNeuron,                   \
        i,                               \
        dt,                              \
        f,                               \
        state[ind_new][indexOfNeuron])

void stepFunction(
    const std::unordered_set<unsigned int>& excitatory_neurons,
    double                               ***state,
    const unsigned int                      ind_old,
    const unsigned int                      ind_new,
    double                                 *state_K1,
    double                                 *state_K2,
    double                                 *state_K3,
    double                                 *state_K4,
    double                                 *state_temp_1,
    double                                 *state_temp_2,
    double                                 *state_temp_3,
    const unsigned int                      numNeurons,
    const unsigned int                      stateSize,
    const double                            dt,
    std::vector<double>                    *spikeTimes_e,
    std::vector<double>                    *spikeNeuronIndices_e,
    std::vector<double>                    *spikeTimes_i,
    std::vector<double>                    *spikeNeuronIndices_i,
    const unsigned int                      firstNeuron,
    const unsigned int                      lastNeuron
    )
{
    for (unsigned int indexOfNeuron = firstNeuron; indexOfNeuron < lastNeuron;
         indexOfNeuron += 1)
    {
        if (excitatory_neurons.count(indexOfNeuron))
        {
            state[ind_new][indexOfNeuron][0] =
                state[ind_old][indexOfNeuron][0] + 1;
            runge_kutta((*golomb::f_dV_dt), 1);
            runge_kutta((*golomb::f_I_Na_dh_dt), 2);
            runge_kutta((*golomb::f_dn_dt), 3);
            runge_kutta((*golomb::f_dz_dt), 4);
            runge_kutta((*golomb::f_dsAMPA_dt), 5);
            runge_kutta((*golomb::f_dxNMDA_dt), 6);
            runge_kutta((*golomb::f_dsNMDA_dt), 7);
            state[ind_new][indexOfNeuron][8] =
                state[ind_old][indexOfNeuron][8];
            state[ind_new][indexOfNeuron][9] =
                state[ind_old][indexOfNeuron][9];

            if (((int)state[ind_new][indexOfNeuron][1]) >= 20)
            {
#ifdef USE_STD_THREADS
                std::lock_guard<std::mutex> lk(m);
#endif // ifdef USE_STD_THREADS
                spikeTimes_e->push_back(
                    (state[ind_new][indexOfNeuron][0]) * dt);
                spikeNeuronIndices_e->push_back(indexOfNeuron);
            }
        } else {
            state[ind_new][indexOfNeuron][0] =
                state[ind_old][indexOfNeuron][0] + 1;
            runge_kutta((*wang_buzsaki::f_dV_dt), 1);
            runge_kutta((*wang_buzsaki::f_I_Na_dh_dt), 2);
            runge_kutta((*wang_buzsaki::f_I_Kdr_dn_dt), 3);
            state[ind_new][indexOfNeuron][4] =
                state[ind_old][indexOfNeuron][4];
            state[ind_new][indexOfNeuron][5] =
                state[ind_old][indexOfNeuron][5];
            state[ind_new][indexOfNeuron][6] =
                state[ind_old][indexOfNeuron][6];
            state[ind_new][indexOfNeuron][7] =
                state[ind_old][indexOfNeuron][7];
            runge_kutta((*wang_buzsaki::f_dsGABAA_dt), 8);
            state[ind_new][indexOfNeuron][9] =
                state[ind_old][indexOfNeuron][9];

            if (((int)state[ind_new][indexOfNeuron][1]) >= 20)
            {
#ifdef USE_STD_THREADS
                std::lock_guard<std::mutex> lk(m);
#endif // ifdef USE_STD_THREADS
                spikeTimes_i->push_back(
                    (state[ind_new][indexOfNeuron][0]) * dt);
                spikeNeuronIndices_i->push_back(indexOfNeuron);
            }
        }
    }
}

// state[t, V, h, n, z, sAMPA, xNMDA, sNMDA, sGABAA, I_app]
int simulate()
{
    srand((unsigned int)time(NULL));

    const double t_0                    = 0;
    const double V_0                    = -70;
    const double h_0                    = 1.0;
    const double n_0                    = 0;
    const double z_0                    = 0;
    const double sAMPA_0                = 0;
    const double xNMDA_0                = 0;
    const double sNMDA_0                = 0;
    const double s_GABAA_0              = 0;
    const double I_app_0                = 1;
    const double dt                     = 0.1;
    const unsigned int timesteps        = 6000;
    const unsigned int numNeurons       = 10;
    const unsigned int stateSize        = 10;
    const unsigned int chanceInhibitory = 10;

#ifdef USE_STD_THREADS
    static const unsigned int numThreads = 8;
    std::thread threads[numThreads];
#endif // ifdef USE_STD_THREADS

    // state[2][numNeurons][stateSize];

    std::unordered_set<unsigned int> excitatory_neurons;
    std::unordered_set<unsigned int> inhibitory_neurons;

    for (unsigned int j = 0; j < numNeurons; ++j)
    {
        if ((rand() % 100) >= chanceInhibitory)
        {
            excitatory_neurons.insert(j);
        } else {
            inhibitory_neurons.insert(j);
        }
    }

    // allocate memory for state array and assign excitatory neurons
    double ***state = (double ***)malloc(2 * sizeof(double **));

    for (unsigned int i = 0; i < 2; ++i)
    {
        state[i] = (double **)malloc(numNeurons * sizeof(double *));

        for (unsigned int j = 0; j < numNeurons; ++j)
        {
            state[i][j] = (double *)malloc(stateSize * sizeof(double));
        }
    }

    // allocate memory for runge kutta temporary arrays
#ifdef USE_STD_THREADS
    double **state_K1 = (double **)malloc(numThreads * sizeof(double *));

    for (unsigned int i = 0; i < numThreads; ++i)
    {
        state_K1[i] = (double *)malloc(stateSize * sizeof(double));
    }
    double **state_temp_1 = (double **)malloc(numThreads * sizeof(double *));

    for (unsigned int i = 0; i < numThreads; ++i)
    {
        state_temp_1[i] = (double *)malloc(stateSize * sizeof(double));
    }
    double **state_temp_3 = (double **)malloc(numThreads * sizeof(double *));

    for (unsigned int i = 0; i < numThreads; ++i)
    {
        state_temp_3[i] = (double *)malloc(stateSize * sizeof(double));
    }
#else // ifdef USE_STD_THREADS
    double *state_K1     = (double *)malloc(stateSize * sizeof(double));
    double *state_temp_1 = (double *)malloc(stateSize * sizeof(double));
    double *state_temp_3 = (double *)malloc(stateSize * sizeof(double));
#endif // ifdef USE_STD_THREADS
    double *state_K2     = (double *)malloc(stateSize * sizeof(double));
    double *state_K3     = (double *)malloc(stateSize * sizeof(double));
    double *state_K4     = (double *)malloc(stateSize * sizeof(double));
    double *state_temp_2 = (double *)malloc(stateSize * sizeof(double));

    for (int i = 0; i < numNeurons; ++i)
    {
        state[0][i][0] = t_0;
        state[0][i][1] = V_0;
        state[0][i][2] = h_0;
        state[0][i][3] = n_0;
        state[0][i][4] = z_0;
        state[0][i][5] = sAMPA_0;
        state[0][i][6] = xNMDA_0;
        state[0][i][7] = sNMDA_0;
        state[0][i][8] = s_GABAA_0;

        if (excitatory_neurons.count(i))
        {
            state[0][i][9] = I_app_0;
        } else
        {
            state[0][i][9] = 0;
        }
    }

    std::vector<double> V_t_e, I_app_t_e, h_t_e, n_t_e, z_t_e, sAMPA_t_e,
                        xNMDA_t_e, sNMDA_t_e,
                        sGABAA_t_e;
    const unsigned int neuronToPlot_e = *(excitatory_neurons.begin());
    V_t_e.push_back(state[0][neuronToPlot_e][1]);
    I_app_t_e.push_back(state[0][neuronToPlot_e][9]);
    h_t_e.push_back(state[0][neuronToPlot_e][2]);
    n_t_e.push_back(state[0][neuronToPlot_e][3]);
    z_t_e.push_back(state[0][neuronToPlot_e][4]);
    sAMPA_t_e.push_back(state[0][neuronToPlot_e][5]);
    xNMDA_t_e.push_back(state[0][neuronToPlot_e][6]);
    sNMDA_t_e.push_back(state[0][neuronToPlot_e][7]);

    std::vector<double> V_t_i, I_app_t_i, h_t_i, n_t_i, z_t_i, sAMPA_t_i,
                        xNMDA_t_i, sNMDA_t_i,
                        sGABAA_t_i;
    const unsigned int neuronToPlot_i = *(inhibitory_neurons.begin());
    V_t_i.push_back(state[0][neuronToPlot_i][1]);
    I_app_t_i.push_back(state[0][neuronToPlot_i][9]);
    h_t_i.push_back(state[0][neuronToPlot_i][2]);
    n_t_i.push_back(state[0][neuronToPlot_i][3]);
    z_t_i.push_back(state[0][neuronToPlot_i][4]);
    sGABAA_t_i.push_back(state[0][neuronToPlot_i][8]);

    std::vector<double> spikeTimes_e, spikeNeuronIndices_e;
    std::vector<double> spikeTimes_i, spikeNeuronIndices_i;

#ifdef USE_STD_THREADS
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
#else // ifdef USE_STD_THREADS
    clock_t tStart = clock();
#endif // ifdef USE_STD_THREADS

    printf("Timestep %d/%d\n", 1, timesteps);

    for (unsigned int t = 0; t < timesteps - 1; ++t)
    {
        unsigned int ind_old = t % 2;
        unsigned int ind_new = 1 - ind_old;

        printf("Timestep %d/%d\n", t + 2, timesteps);

#ifdef USE_STD_THREADS
# define calculate(first, last) \
    stepFunction,               \
    excitatory_neurons,         \
    state,                      \
    ind_old,                    \
    ind_new,                    \
    state_K1[thread],           \
    state_K2,                   \
    state_K3,                   \
    state_K4,                   \
    state_temp_1[thread],       \
    state_temp_2,               \
    state_temp_3[thread],       \
    numNeurons,                 \
    stateSize,                  \
    dt,                         \
    &spikeTimes_e,              \
    &spikeNeuronIndices_e,      \
    &spikeTimes_i,              \
    &spikeNeuronIndices_i,      \
    first,                      \
    last

        assert((numNeurons % numThreads) == 0);
        unsigned int thread = 0;
        unsigned int first  = 0;
        unsigned int last   = numNeurons / numThreads;

        for (unsigned int i = last; i <= numNeurons;
             i += (numNeurons / numThreads))
        {
            threads[thread] = std::thread(calculate(first, i));
            ++thread;
            first = i;
        }

        for (unsigned int i = 0; i < numThreads; ++i)
        {
            threads[i].join();
        }

#else // ifdef USE_STD_THREADS

        for (unsigned int indexOfNeuron = 0; indexOfNeuron < numNeurons;
             ++indexOfNeuron)
        {
            stepFunction(
                excitatory_neurons,
                state,
                ind_old,
                ind_new,
                state_K1,
                state_K2,
                state_K3,
                state_K4,
                state_temp_1,
                state_temp_2,
                state_temp_3,
                numNeurons,
                stateSize,
                dt,
                &spikeTimes_e,
                &spikeNeuronIndices_e,
                &spikeTimes_i,
                &spikeNeuronIndices_i,
                indexOfNeuron,
                indexOfNeuron + 1
                );

            golomb::f_I_NMDA_FFT(
                (const double **)state[ind_old],
                numNeurons,
                stateSize,
                state[ind_new]
                );
        }
#endif // ifdef USE_STD_THREADS

        V_t_e.push_back(state[ind_new][neuronToPlot_e][1]);
        I_app_t_e.push_back(state[ind_new][neuronToPlot_e][9]);
        h_t_e.push_back(state[ind_new][neuronToPlot_e][2]);
        n_t_e.push_back(state[ind_new][neuronToPlot_e][3]);
        z_t_e.push_back(state[ind_new][neuronToPlot_e][4]);
        sAMPA_t_e.push_back(state[ind_new][neuronToPlot_e][5]);
        xNMDA_t_e.push_back(state[ind_new][neuronToPlot_e][6]);
        sNMDA_t_e.push_back(state[ind_new][neuronToPlot_e][7]);

        V_t_i.push_back(state[ind_new][neuronToPlot_i][1]);
        I_app_t_i.push_back(state[ind_new][neuronToPlot_i][9]);
        h_t_i.push_back(state[ind_new][neuronToPlot_i][2]);
        n_t_i.push_back(state[ind_new][neuronToPlot_i][3]);
        z_t_i.push_back(state[ind_new][neuronToPlot_i][4]);
        sGABAA_t_i.push_back(state[ind_new][neuronToPlot_i][8]);
    }

#ifdef USE_STD_THREADS
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed  = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Execution time: %f\n", elapsed);
#else // ifdef USE_STD_THREADS
    printf("Execution time: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
#endif // ifdef USE_STD_THREADS

#ifdef PLOT
    Gnuplot plot_V_Iapp_e;
    plot_V_Iapp_e.set_style("lines");
    plot_V_Iapp_e.set_title("Excitatory neuron");
    plot_V_Iapp_e.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        V_t_e, "V");
    plot_V_Iapp_e.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        I_app_t_e, "I_app");
    Gnuplot plot_hnz_e;
    plot_hnz_e.set_style("lines");
    plot_hnz_e.set_title("Excitatory neuron");
    plot_hnz_e.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        h_t_e, "h");
    plot_hnz_e.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        n_t_e, "n");
    plot_hnz_e.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        z_t_e, "z");
    Gnuplot plot_Syn_e;
    plot_Syn_e.set_style("lines");
    plot_Syn_e.set_title("Excitatory neuron");
    plot_Syn_e.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        sAMPA_t_e, "s_AMPA");
    plot_Syn_e.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        xNMDA_t_e, "x_NMDA");
    plot_Syn_e.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        sNMDA_t_e, "s_NMDA");

    Gnuplot plot_V_Iapp_i;
    plot_V_Iapp_i.set_style("lines");
    plot_V_Iapp_i.set_title("Inhibitory neuron");
    plot_V_Iapp_i.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        V_t_i, "V");
    plot_V_Iapp_i.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        I_app_t_i, "I_app");
    Gnuplot plot_hnz_i;
    plot_hnz_i.set_style("lines");
    plot_hnz_i.set_title("Inhibitory neuron");
    plot_hnz_i.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        h_t_i, "h");
    plot_hnz_i.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        n_t_i, "n");
    plot_hnz_i.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        z_t_i, "z");
    Gnuplot plot_Syn_i;
    plot_Syn_i.set_style("lines");
    plot_Syn_i.set_title("Inhibitory neuron");
    plot_Syn_i.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        sGABAA_t_i, "s_GABAA");

    Gnuplot plot_Spikes;
    plot_Spikes.set_title("Spikes");
    plot_Spikes.set_style("points");
    plot_Spikes.set_xrange(t_0, timesteps * dt);
    plot_Spikes.set_yrange(0, numNeurons - 1);

    if (!spikeTimes_e.empty())
    {
        plot_Spikes.plot_xy(
            spikeTimes_e, spikeNeuronIndices_e, "Excitatory Spikes");
    }

    if (!spikeTimes_i.empty())
    {
        plot_Spikes.plot_xy(
            spikeTimes_i, spikeNeuronIndices_i, " Inhibitory Spikes");
    }
    getchar();
#endif // ifdef PLOT

    // free memory
    free(state_K1);
    free(state_K2);
    free(state_K3);
    free(state_K4);
    free(state_temp_1);
    free(state_temp_2);
    free(state_temp_3);

    for (unsigned i = 0; i < 2; ++i)
    {
        for (unsigned j = 0; j < numNeurons; ++j)
        {
            free(state[i][j]);
        }
        free(state[i]);
    }
    free(state);

    return 0;
}

inline void f_rc(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    resultState[1] = (((states[indexOfNeuron][2] - states[indexOfNeuron][1]) /
                       states[indexOfNeuron][3]) +
                      (states[indexOfNeuron][4] / states[indexOfNeuron][5])) /
                     states[indexOfNeuron][6];
}

double rc_circuit_membrane_potential_current_pulse(
    double E_m, double R_m, double I_e, double a,
    double t, double C_m)
{
    return E_m + ((R_m * I_e) / a) * (1 - exp(-t /
                                              (R_m * C_m)));
}

double rc_circuit_membrane_potential_no_current_pulse(
    double E_m, double V_0, double t, double R_m,
    double C_m)
{
    return E_m + (V_0 - E_m) * exp(-t / (R_m * C_m));
}

// state[t, V, E_m, R_m, I_e, a, C_m]
int rc_circuit_generic()
{
    const double t_0       = 0;
    const double E_m_0     = -70;
    const double V_0       = E_m_0;
    const double R_m_0     = 10;
    const double I_e_0     = 50;
    const double a_0       = 10;
    const double C_m_0     = 1;
    const double dt        = 1;
    const int    timesteps = 200;
    const int    stateSize = 7;

    // state[timesteps][1][stateSize];

    // allocate memory for state array
    double ***state = (double ***)malloc(timesteps * sizeof(double **));

    for (unsigned int i = 0; i < timesteps; ++i)
    {
        state[i] = (double **)malloc(1 * sizeof(double *));

        for (unsigned int j = 0; j < 1; ++j)
        {
            state[i][j] = (double *)malloc(stateSize * sizeof(double));
        }
    }

    // allocate memory for runge kutta temporary arrays
    double *state_K1     = (double *)malloc(stateSize * sizeof(double));
    double *state_K2     = (double *)malloc(stateSize * sizeof(double));
    double *state_K3     = (double *)malloc(stateSize * sizeof(double));
    double *state_K4     = (double *)malloc(stateSize * sizeof(double));
    double *state_temp_1 = (double *)malloc(stateSize * sizeof(double));
    double *state_temp_2 = (double *)malloc(stateSize * sizeof(double));
    double *state_temp_3 = (double *)malloc(stateSize * sizeof(double));

    state[0][0][0] = t_0;
    state[0][0][1] = V_0;
    state[0][0][2] = E_m_0;
    state[0][0][3] = R_m_0;
    state[0][0][4] = I_e_0;
    state[0][0][5] = a_0;
    state[0][0][6] = C_m_0;

    std::vector<double> V_t;
    V_t.push_back(state[0][0][1]);

    for (unsigned int t = (unsigned int)t_0; t < timesteps - 1; t += 1)
    {
        for (unsigned int i = 0; i < stateSize; ++i)
        {
            state[t + 1][0][i] = state[t][0][i];
        }

        runge_kutta_generic(
            (const double **)state[t],
            state_K1,
            state_K2,
            state_K3,
            state_K4,
            state_temp_1,
            state_temp_2,
            state_temp_3,
            1,
            stateSize,
            0,
            1,
            dt,
            (*f_rc),
            state[t + 1][0]
            );

        state[t + 1][0][0] += 1;

        V_t.push_back(state[t + 1][0][1]);

        if (t == 99)
        {
            state[t + 1][0][4] = 0;
        }
    }

    double t   = 0;
    double R_m = 10;
    double C_m = 1;
    double a   = 10;
    double E_m = -70;
    double I_e = 50;

    std::vector<double> V_t_analytical;
    double step = 1;
    V_t_analytical.push_back(E_m);
    t += 1;

    while (t <= 100)
    {
        V_t_analytical.push_back(rc_circuit_membrane_potential_current_pulse(
                                     E_m, R_m, I_e, a, t, C_m));
        t += step;
    }
    double V_100 = V_t_analytical.back();

    while (t < 200)
    {
        V_t_analytical.push_back(rc_circuit_membrane_potential_no_current_pulse(
                                     E_m, V_100, t - 100, R_m, C_m));

        t += step;
    }

    Gnuplot plot;
    plot.set_style("linespoints");
    plot.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        V_t, "Runge-kutta-approximation");
    plot.plot_xy(
        linSpaceVec<double, double>(timesteps, 0, timesteps * dt),
        V_t_analytical, "Analytical solution");

    // getchar();

    // free memory
    free(state_K1);
    free(state_K2);
    free(state_K3);
    free(state_K4);
    free(state_temp_1);
    free(state_temp_2);
    free(state_temp_3);

    for (unsigned i = 0; i < timesteps; ++i)
    {
        for (unsigned j = 0; j < 1; ++j)
        {
            free(state[i][j]);
        }
        free(state[i]);
    }
    free(state);

    return 0;
}

int main(int argc, char *argv[])
{
    // return rc_circuit_generic();
    return simulate();
}
