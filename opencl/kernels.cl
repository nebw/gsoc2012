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

//__kernel void stepKernel(__global struct state* states, const unsigned int numNeurons)
//{
//   const int idx = get_global_id(0);
//
//   states[1*numNeurons+idx].V = states[0*numNeurons+idx].V + 1;
//}

inline float _f_I_Na_m_inf(const float V)
{
    const float theta_m = -30;
    const float sigma_m = 9.5;

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
    const float g_NaP = 0.2;
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
    const float g_L = 0.05;
    const float V_L = -70;

    return g_L * (V - V_L);
}

inline float _f_I_Kslow(const float V, const float z)
{
    const float g_Kslow = 1.8;
    const float V_K     = -90;

    return g_Kslow * z * (V - V_K);
}

inline float _f_dV_dt(const float V, const float h, const float n, const float z, const float I_app)
{
    return - _f_I_Na(V, h)
           - _f_I_NaP(V)
           - _f_I_Kdr(V, n)
           - _f_I_Kslow(V, z)
           - _f_I_Leak(V)
           + I_app;
}

__kernel void f_dV_dt(__global struct state* states, const unsigned int numNeurons, const unsigned int ind_old, const float dt)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int ind_new = 1 - ind_old;

    struct state state_0 = states[ind_old*numNeurons+idx];

    float V_0 = states[ind_old*numNeurons+idx].V;
    float h_0 = states[ind_old*numNeurons+idx].h;
    float n_0 = states[ind_old*numNeurons+idx].n;
    float z_0 = states[ind_old*numNeurons+idx].z;
    float I_app_0 = states[ind_old*numNeurons+idx].I_app;
    
    float f1, f2, f3, f4;

    f1 = _f_dV_dt(V_0, h_0, n_0, z_0, I_app_0);
    f2 = _f_dV_dt(state_0.V + dt * f1 / 2.0, state_0.h, state_0.n, state_0.z, state_0.I_app);
    f3 = _f_dV_dt(state_0.V + dt * f2 / 2.0, state_0.h, state_0.n, state_0.z, state_0.I_app);
    f4 = _f_dV_dt(state_0.V + dt * f3, state_0.h, state_0.n, state_0.z, state_0.I_app);

    states[ind_new*numNeurons+idx].V = state_0.V + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0;
}

inline float _f_I_Na_h_inf(const float V)
{
    const float theta_h = -45;
    const float sigma_h = -7;

    return pow((1 + exp(-(V - theta_h) / sigma_h)), -1);
}

inline float _f_I_Na_tau_h(const float V)
{
    const float theta_th = -40.5;
    const float sigma_th = -6;

    return 0.1 + 0.75 * pow((1 + exp(-(V - theta_th) / sigma_th)), -1);
}

inline float _f_I_Na_dh_dt(const float h, const float V)
{
    return (_f_I_Na_h_inf(V) - h) / _f_I_Na_tau_h(V);
}

__kernel void f_I_Na_dh_dt(__global struct state* states, const unsigned int numNeurons, const unsigned int ind_old, const float dt)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int ind_new = 1 - ind_old;

    struct state state_0 = states[ind_old*numNeurons+idx];

    float f1, f2, f3, f4;

    f1 = _f_I_Na_dh_dt(state_0.h, state_0.V);
    f2 = _f_I_Na_dh_dt(state_0.h + dt * f1 / 2.0, state_0.V);
    f3 = _f_I_Na_dh_dt(state_0.h + dt * f2 / 2.0, state_0.V);
    f4 = _f_I_Na_dh_dt(state_0.h + dt * f3, state_0.V);

    states[ind_new*numNeurons+idx].h = state_0.h + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0;
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

    return 0.1 + 0.5 * pow(1 + exp(-(V - theta_tn) / sigma_tn), -1);
}

inline float _f_dn_dt(const float n, const float V)
{
    return (_f_n_inf(V) - n) / _f_tau_n(V);
}

__kernel void f_dn_dt(__global struct state* states, const unsigned int numNeurons, const unsigned int ind_old, const float dt)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int ind_new = 1 - ind_old;

    struct state state_0 = states[ind_old*numNeurons+idx];

    float f1, f2, f3, f4;

    f1 = _f_dn_dt(state_0.n, state_0.V);
    f2 = _f_dn_dt(state_0.n + dt * f1 / 2.0, state_0.V);
    f3 = _f_dn_dt(state_0.n + dt * f2 / 2.0, state_0.V);
    f4 = _f_dn_dt(state_0.n + dt * f3, state_0.V);

    states[ind_new*numNeurons+idx].n = state_0.n + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0;
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

__kernel void f_dz_dt(__global struct state* states, const unsigned int numNeurons, const unsigned int ind_old, const float dt)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int ind_new = 1 - ind_old;

    struct state state_0 = states[ind_old*numNeurons+idx];

    float f1, f2, f3, f4;

    f1 = _f_dz_dt(state_0.z, state_0.V);
    f2 = _f_dz_dt(state_0.z + dt * f1 / 2.0, state_0.V);
    f3 = _f_dz_dt(state_0.z + dt * f2 / 2.0, state_0.V);
    f4 = _f_dz_dt(state_0.z + dt * f3, state_0.V);

    states[ind_new*numNeurons+idx].z = state_0.z + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0;
}
