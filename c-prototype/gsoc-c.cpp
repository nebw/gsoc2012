#if defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
 # define USE_STD_THREADS
#endif // if defined(unix) || defined(__unix) || defined(__unix__) || //
       // defined(__APPLE__)

#define PLOT

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
# include "fftw3.h"
#endif // if defined(unix) || defined(__unix) || defined(__unix__) ||
// defined(__APPLE__)


#include <ctime>
#include <unordered_set>

#ifdef USE_STD_THREADS
std::mutex m;
#endif // ifdef USE_STD_THREADS

template <typename T>
std::vector<T> linSpaceVec(T a, T b, size_t N) {
  T h = (b - a) / static_cast<T>(N-1);
  std::vector<T> xs(N);
  typename std::vector<T>::iterator x;
  T val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
    *x = val;
  return xs;
}

void f_I_FFT(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    double           **resultStates,
    fftw_complex      *distances,
    fftw_complex      *sVars,
    fftw_complex      *convolution,
    fftw_complex      *distances_f,
    fftw_complex      *sVars_f,
    fftw_complex      *convolution_f,
    fftw_plan        & p_distances,
    fftw_plan        & p_sVars,
    fftw_plan        & p_inv,
    const unsigned int n,
    const double       scale,
    double               (*footprint)(
      int)
        )
{
      unsigned int ind = 0;

    for (unsigned int i = numNeurons - 1; i > 0; --i)
    {
        distances[ind][0] = (*footprint)(i);
        distances[ind][1] = 0;
        ++ind;
    }

    for (unsigned int i = 0; i < numNeurons; ++i)
    {
        distances[ind][0] = (*footprint)(i);
        distances[ind][1] = 0;
        ++ind;
    }

    for (unsigned int i = 0; i < numNeurons; ++i)   {
        sVars[i][0] = states[i][6];
        sVars[i][1] = 0;
    }

    for (unsigned int i = numNeurons; i < n; ++i)
    {
        sVars[i][0] = 0;
        sVars[i][1] = 0;
    }

    fftw_execute(p_distances);

    fftw_execute(p_sVars);

    // convolution in frequency domain
    for (unsigned int i = 0; i < n; ++i)
    {
        convolution_f[i][0] = (distances_f[i][0] * sVars_f[i][0]
                               - distances_f[i][1] * sVars_f[i][1]) * scale;
        convolution_f[i][1] = (distances_f[i][0] * sVars_f[i][1]
                               + distances_f[i][1] * sVars_f[i][0]) * scale;
    }

    fftw_execute(p_inv);
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
    const double   *sumFootprintAMPA,
    const double   *sumFootprintNMDA,
    const double   *sumFootprintGABAA,
    const double       dt,
    void               (*f)(
        const          double **,
    const double   *,
    const double   *,
    const double   *,
        const unsigned int,
        const unsigned int,
        const unsigned int,
        double *),
    double            *resultState)
{
    (*f)(states, sumFootprintAMPA, sumFootprintNMDA, sumFootprintGABAA, numNeurons, stateSize, indexOfNeuron, state_temp_1);

    arrMul(state_temp_1, state_K1, dt, stateSize);

    arrAdd(states[indexOfNeuron], state_K1, state_temp_3, stateSize);

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

inline double _f_f_NMDA(const double V)
{
    static const double theta_NMDA = 0;

    // TODO: theta_NMDA = -inf for [Mg2+]_0 = 0
    // and increases logarithmically with [Mg2+]_0
    static const double sigma_NMDA = 10;

    return pow(1 + exp(-(V - theta_NMDA) / sigma_NMDA), -1);
}

inline double _f_w_IE(const int j)
{
    static const double sigma = 0.5;
    static const double p     = 32;

    // TODO: p varies between 8 to 64

    return tanh(1 / (2 * sigma * p))
           * exp(-abs(j) / (sigma * p));
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

double _f_I_NMDA(
    const double **states,
    const double   *sumFootprintAMPA,
    const int      indexOfNeuron,
    const int      numNeurons)
{
    static const double g_NMDA = 0.07;
    static const double V_Glu  = 0;

    return g_NMDA * _f_f_NMDA(states[indexOfNeuron][1]) *
           (states[indexOfNeuron][1] - V_Glu) *sumFootprintAMPA[indexOfNeuron];
}

double _f_I_GABAA(
    const double **states,
    const double   *sumFootprintGABAA,
    const int      indexOfNeuron,
    const int      numNeurons)
{
    static const double g_GABAA = 0.05;
    static const double V_GABAA = -70;

    return g_GABAA * (states[indexOfNeuron][1] - V_GABAA) * sumFootprintGABAA[indexOfNeuron];
}

double _f_I_AMPA(
    const double **states,
    const double   *sumFootprintNMDA,
    const int      indexOfNeuron,
    const int      numNeurons)
{
    static const double g_AMPA = 0.08;
    static const double V_Glu  = 0;

    return g_AMPA * (states[indexOfNeuron][1] - V_Glu) * sumFootprintNMDA[indexOfNeuron];
}

inline void f_dV_dt(
    const double     **states,
    const double   *sumFootprintAMPA,
    const double   *sumFootprintNMDA,
    const double   *sumFootprintGABAA,
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
        - _f_I_AMPA(states, sumFootprintAMPA, indexOfNeuron, numNeurons)
        - _f_I_NMDA(states, sumFootprintNMDA, indexOfNeuron, numNeurons)
        - _f_I_GABAA(states, sumFootprintGABAA, indexOfNeuron, numNeurons)
        + states[indexOfNeuron][9];
}

void f_I_NMDA_FFT(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    double           **resultStates,
    fftw_complex      *distances,
    fftw_complex      *sNMDAs,
    fftw_complex      *convolution,
    fftw_complex      *distances_f,
    fftw_complex      *sNMDAs_f,
    fftw_complex      *convolution_f,
    fftw_plan        & p_distances,
    fftw_plan        & p_sNMDAs,
    fftw_plan        & p_inv,
    const unsigned int n,
    const double       scale,
    double   *sumFootprintNMDA
    )
{
    f_I_FFT(
      states,
      numNeurons,
      stateSize,
      resultStates,
      distances,
      sNMDAs,
      convolution,
      distances_f,
      sNMDAs_f,
      convolution_f,
      p_distances,
      p_sNMDAs,
      p_inv,
      n,
      scale,
      *_f_w_EE);
    
    for (unsigned int indexOfNeuron = 0; indexOfNeuron < numNeurons; ++indexOfNeuron)
    {
      sumFootprintNMDA[indexOfNeuron] = convolution[indexOfNeuron+numNeurons-1][0];
    }
}

void f_I_AMPA_FFT(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    double           **resultStates,
    fftw_complex      *distances,
    fftw_complex      *sAMPAs,
    fftw_complex      *convolution,
    fftw_complex      *distances_f,
    fftw_complex      *sAMPAs_f,
    fftw_complex      *convolution_f,
    fftw_plan        & p_distances,
    fftw_plan        & p_sAMPAs,
    fftw_plan        & p_inv,
    const unsigned int n,
    const double       scale,
    double   *sumFootprintAMPA
    )
{
  f_I_FFT(
      states,
      numNeurons,
      stateSize,
      resultStates,
      distances,
      sAMPAs,
      convolution,
      distances_f,
      sAMPAs_f,
      convolution_f,
      p_distances,
      p_sAMPAs,
      p_inv,
      n,
      scale,
      *_f_w_EE
    );
  
    for (unsigned int indexOfNeuron = 0; indexOfNeuron < numNeurons; ++indexOfNeuron)
    {
      sumFootprintAMPA[indexOfNeuron] = convolution[indexOfNeuron+numNeurons-1][0];
    }
}

void f_I_GABAA_FFT(
    const double     **states,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    double           **resultStates,
    fftw_complex      *distances,
    fftw_complex      *sGABAAs,
    fftw_complex      *convolution,
    fftw_complex      *distances_f,
    fftw_complex      *sGABAAs_f,
    fftw_complex      *convolution_f,
    fftw_plan        & p_distances,
    fftw_plan        & p_sGABAAs,
    fftw_plan        & p_inv,
    const unsigned int n,
    const double       scale,
    double   *sumFootprintGABAA
    )
{
  f_I_FFT(
      states,
      numNeurons,
      stateSize,
      resultStates,
      distances,
      sGABAAs,
      convolution,
      distances_f,
      sGABAAs_f,
      convolution_f,
      p_distances,
      p_sGABAAs,
      p_inv,
      n,
      scale,
      *_f_w_EE
    );

    for (unsigned int indexOfNeuron = 0; indexOfNeuron < numNeurons; ++indexOfNeuron)
    {
      sumFootprintGABAA[indexOfNeuron] = convolution[indexOfNeuron+numNeurons-1][0];
    }
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
    const double   *sumFootprintAMPA,
    const int      indexOfNeuron,
    const int      numNeurons)
{
    static const double g_EI_AMPA = 0.2;
    static const double V_Glu     = 0;

    return g_EI_AMPA * (states[indexOfNeuron][1] - V_Glu) * sumFootprintAMPA[indexOfNeuron];
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
    const double   *sumFootprintNMDA,
    const int      indexOfNeuron,
    const int      numNeurons)
{
    // 0.0 or 0.05
    static const double g_EI_NMDA = 0.05;
    static const double V_Glu     = 0;

    return g_EI_NMDA * _f_f_NMDA(states[indexOfNeuron][1]) *
           (states[indexOfNeuron][1] - V_Glu) * sumFootprintNMDA[indexOfNeuron];
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
    const double   *sumFootprintAMPA,
    const double   *sumFootprintNMDA,
    const double   *sumFootprintGABAA,
    const unsigned int numNeurons,
    const unsigned int stateSize,
    const unsigned int indexOfNeuron,
    double            *resultState)
{
    resultState[1] =
        -_f_I_Na(states[indexOfNeuron][1], states[indexOfNeuron][2])
        - _f_I_Kdr(states[indexOfNeuron][1], states[indexOfNeuron][3])
        - _f_I_Leak(states[indexOfNeuron][1])
        - _f_I_AMPA(states, sumFootprintAMPA, indexOfNeuron, numNeurons)
        - _f_I_NMDA(states, sumFootprintNMDA, indexOfNeuron, numNeurons)
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
        _numNeurons, stateSize,           \
        indexOfNeuron,                   \
        i,                               \
        _dt,                              \
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
    const double   *sumFootprintAMPA,
    const double   *sumFootprintNMDA,
    const double   *sumFootprintGABAA,
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
        runge_kutta_generic(
          (const double **)state[ind_old],
          state_K1,
          state_K2,
          state_K3,
          state_K4,
          state_temp_1,
          state_temp_2,
          state_temp_3,
          numNeurons,
          stateSize,
          indexOfNeuron,
          1,
          sumFootprintAMPA,
          sumFootprintNMDA,
          sumFootprintGABAA,
          dt,
          (*golomb::f_dV_dt),
          state[ind_new][indexOfNeuron]);
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
        runge_kutta_generic(
          (const double **)state[ind_old],
          state_K1,
          state_K2,
          state_K3,
          state_K4,
          state_temp_1,
          state_temp_2,
          state_temp_3,
          numNeurons,
          stateSize,
          indexOfNeuron,
          1,
          sumFootprintAMPA,
          sumFootprintNMDA,
          sumFootprintGABAA,
          dt,
          (*wang_buzsaki::f_dV_dt),
          state[ind_new][indexOfNeuron]);
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
    const unsigned int timesteps        = 5000;
    const unsigned int numNeurons       = 1000;
    const unsigned int stateSize        = 10;
    const unsigned int chanceInhibitory = 10;
    const unsigned int numFFTs          = 4;
    const unsigned int n                = 2 * numNeurons - 1;
    const double scale                  = 1. / n;

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

    // allocate memory for fftw
    fftw_complex *distances =
        (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));
    fftw_complex *sNMDAs =
        (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));
    fftw_complex *convolution =
        (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));
    fftw_complex *distances_f =
        (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));
    fftw_complex *sNMDAs_f =
        (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));
    fftw_complex *convolution_f =
        (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));
    fftw_plan p_distances = fftw_plan_dft_1d(n,
                                             distances,
                                             distances_f,
                                             FFTW_FORWARD,
                                             FFTW_ESTIMATE);
    fftw_plan p_sNMDAs = fftw_plan_dft_1d(n,
                                          sNMDAs,
                                          sNMDAs_f,
                                          FFTW_FORWARD,
                                          FFTW_ESTIMATE);
    fftw_plan p_inv = fftw_plan_dft_1d(n,
                                       convolution_f,
                                       convolution,
                                       FFTW_BACKWARD,
                                       FFTW_ESTIMATE);

    double *sumFootprintAMPA = (double *)malloc(numNeurons * sizeof(double));
    double *sumFootprintGABAA = (double *)malloc(numNeurons * sizeof(double));
    double *sumFootprintNMDA = (double *)malloc(numNeurons * sizeof(double));

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
    _sumFootprintAMPA, \
    _sumFootprintNMDA, \
    _sumFootprintGABAA, \
    _numNeurons,                 \
    stateSize,                  \
    _dt,                         \
    &spikeTimes_e,              \
    &spikeNeuronIndices_e,      \
    &spikeTimes_i,              \
    &spikeNeuronIndices_i,      \
    first,                      \
    last

      golomb::f_I_NMDA_FFT(
          (const double **)state[ind_old],
          numNeurons,
          stateSize,
          state[ind_new],
          distances,
          sNMDAs,
          convolution,
          distances_f,
          sNMDAs_f,
          convolution_f,
          p_distances,
          p_sNMDAs,
          p_inv,
          n,
          scale,
          sumFootprintNMDA
          );

      golomb::f_I_AMPA_FFT(
          (const double **)state[ind_old],
          numNeurons,
          stateSize,
          state[ind_new],
          distances,
          sNMDAs,
          convolution,
          distances_f,
          sNMDAs_f,
          convolution_f,
          p_distances,
          p_sNMDAs,
          p_inv,
          n,
          scale,
          sumFootprintAMPA
          );

      golomb::f_I_GABAA_FFT(
          (const double **)state[ind_old],
          numNeurons,
          stateSize,
          state[ind_new],
          distances,
          sNMDAs,
          convolution,
          distances_f,
          sNMDAs_f,
          convolution_f,
          p_distances,
          p_sNMDAs,
          p_inv,
          n,
          scale,
          sumFootprintGABAA
          );

      //TODO: FFT for IE

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
                sumFootprintAMPA,
                sumFootprintNMDA,
                sumFootprintGABAA,
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
        }

    golomb::f_I_NMDA_FFT(
        (const double **)state[ind_old],
        numNeurons,
        stateSize,
        state[ind_new],
        distances,
        sNMDAs,
        convolution,
        distances_f,
        sNMDAs_f,
        convolution_f,
        p_distances,
        p_sNMDAs,
        p_inv,
        n,
        scale,
        sumFootprintNMDA
        );

    golomb::f_I_AMPA_FFT(
        (const double **)state[ind_old],
        numNeurons,
        stateSize,
        state[ind_new],
        distances,
        sNMDAs,
        convolution,
        distances_f,
        sNMDAs_f,
        convolution_f,
        p_distances,
        p_sNMDAs,
        p_inv,
        n,
        scale,
        sumFootprintAMPA
        );

    golomb::f_I_GABAA_FFT(
        (const double **)state[ind_old],
        numNeurons,
        stateSize,
        state[ind_new],
        distances,
        sNMDAs,
        convolution,
        distances_f,
        sNMDAs_f,
        convolution_f,
        p_distances,
        p_sNMDAs,
        p_inv,
        n,
        scale,
        sumFootprintGABAA
        );
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
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        V_t_e, "V");
    plot_V_Iapp_e.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        I_app_t_e, "I_app");
    Gnuplot plot_hnz_e;
    plot_hnz_e.set_style("lines");
    plot_hnz_e.set_title("Excitatory neuron");
    plot_hnz_e.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        h_t_e, "h");
    plot_hnz_e.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        n_t_e, "n");
    plot_hnz_e.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        z_t_e, "z");
    Gnuplot plot_Syn_e;
    plot_Syn_e.set_style("lines");
    plot_Syn_e.set_title("Excitatory neuron");
    plot_Syn_e.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        sAMPA_t_e, "s_AMPA");
    plot_Syn_e.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        xNMDA_t_e, "x_NMDA");
    plot_Syn_e.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        sNMDA_t_e, "s_NMDA");

    Gnuplot plot_V_Iapp_i;
    plot_V_Iapp_i.set_style("lines");
    plot_V_Iapp_i.set_title("Inhibitory neuron");
    plot_V_Iapp_i.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        V_t_i, "V");
    plot_V_Iapp_i.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        I_app_t_i, "I_app");
    Gnuplot plot_hnz_i;
    plot_hnz_i.set_style("lines");
    plot_hnz_i.set_title("Inhibitory neuron");
    plot_hnz_i.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        h_t_i, "h");
    plot_hnz_i.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        n_t_i, "n");
    plot_hnz_i.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
        z_t_i, "z");
    Gnuplot plot_Syn_i;
    plot_Syn_i.set_style("lines");
    plot_Syn_i.set_title("Inhibitory neuron");
    plot_Syn_i.plot_xy(
        linSpaceVec<double>(0, timesteps * dt, timesteps),
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

    fftw_free(distances);
    fftw_free(sNMDAs);
    fftw_free(convolution);
    fftw_free(distances_f);
    fftw_free(sNMDAs_f);
    fftw_free(convolution_f);

    fftw_destroy_plan(p_distances);
    fftw_destroy_plan(p_sNMDAs);
    fftw_destroy_plan(p_inv);

    return 0;
}

int main(int argc, char *argv[])
{
    // return rc_circuit_generic();
    return simulate();
}
