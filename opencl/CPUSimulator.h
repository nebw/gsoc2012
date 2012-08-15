#pragma once

#include "BaseSimulator.h"
#include "Definitions.h"

class CPUSimulator : public BaseSimulator {
public:
    CPUSimulator(const unsigned int nX,
                 const unsigned int nY,
                 const unsigned int nZ,
                 const unsigned int timesteps,
                 const float dt,
                 state const& state_0);

    virtual void step() override;

    virtual void simulate() override;

    virtual std::unique_ptr<state[]> const& getCurrentStates() const override;

    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintAMPA() const override;

    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintNMDA() const override;

    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintGABAA() const override;

    virtual std::vector<unsigned long> getTimesCalculations() const override;

    virtual std::vector<unsigned long> getTimesFFTW() const override;

    virtual std::vector<unsigned long> getTimesClFFT() const override;

private:
    std::unique_ptr<state[]> _states;
    std::unique_ptr<float[]> _sumFootprintAMPA;
    std::unique_ptr<float[]> _sumFootprintNMDA;
    std::unique_ptr<float[]> _sumFootprintGABAA;

    std::unique_ptr<float[]> _distances;
    std::unique_ptr<float[]> _sVals;

    const float _dt;
    const unsigned int _nX;
    const unsigned int _nY;
    const unsigned int _nZ;
    const unsigned int _numNeurons;
    const unsigned int _timesteps;

    unsigned int _t;

    float f_I_Na_m_inf(const float V);
    float f_I_Na(const float V,
                 const float h);
    float f_p_inf(const float V);
    float f_I_NaP(const float V);
    float f_I_Kdr(const float V,
                  const float n);
    float f_I_Leak(const float V);
    float f_I_Kslow(const float V,
                    const float z);
    float f_I_AMPA(const float V,
                   const float sumFootprintAMPA);
    float f_f_NMDA(const float V);
    float f_I_NMDA(const float V,
                   const float sumFootprintNMDA);
    float f_I_GABAA(const float V,
                    const float sumFootprintGABAA);
    float _f_I_Na_h_inf(const float V);
    float _f_I_Na_tau_h(const float V);
    float _f_n_inf(const float V);
    float _f_tau_n(const float V);
    float _f_dn_dt(const float n,
                   const float V);
    float _f_z_inf(const float V);
    float _f_dz_dt(const float z,
                   const float V);
    float _f_s_inf(const float V);
    float _f_dsAMPA_dt(const float s_AMPA,
                       const float V);
    float _f_dxNMDA_dt(const float x_NMDA,
                       const float V);
    float _f_dsNMDA_dt(const float s_NMDA,
                       const float x_NMDA,
                       const float V);
    float _f_I_Na_dh_dt(const float h,
                        const float V);
    float f_dV_dt(const float V,
                  const float h,
                  const float n,
                  const float z,
                  const float I_app,
                  const float sumFootprintAMPA,
                  const float sumFootprintNMDA,
                  const float sumFootprintGABAA);

    // runge-kutta approximations
    void runge4_f_dV_dt(const unsigned int idx,
                        const unsigned int ind_old);
    void runge4_f_I_Na_dh_dt(const unsigned int idx,
                             const unsigned int ind_old);
    void runge4_f_dsAMPA_dt(const unsigned int idx,
                            const unsigned int ind_old);
    void runge4_f_dn_dt(const unsigned int idx,
                        const unsigned int ind_old);
    void runge4_f_dz_dt(const unsigned int idx,
                        const unsigned int ind_old);
    void runge4_f_dxNMDA_dt(const unsigned int idx,
                            const unsigned int ind_old);
    void runge4_f_dsNMDA_dt(const unsigned int idx,
                            const unsigned int ind_old);

    void convolutionAMPA();
    void convolutionNMDA();
    void convolutionGABAA();

    void computeRungeKuttaApproximations( unsigned int ind_old );
    void computeConvolutions( unsigned int ind_old );
};
