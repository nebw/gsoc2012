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

#pragma once

#include "BaseSimulator.h"
#include "Definitions.h"

class CPUSimulator : public BaseSimulator {
public:

    enum Convolution {
        NO_CONVOLUTION = 0,
        CONVOLUTION = 1
    };

    CPUSimulator(const size_t nX,
                 const size_t nY,
                 const size_t nZ,
                 const size_t timesteps,
                 const float dt,
                 state const& state_0,
                 const Convolution convolution);

    void step() override;

    void simulate() override;

    std::unique_ptr<state[]> const& getCurrentStates() const override;
    std::unique_ptr<float[]> const& getCurrentSumFootprintAMPA() const override;
    std::unique_ptr<float[]> const& getCurrentSumFootprintNMDA() const override;
    std::unique_ptr<float[]> const& getCurrentSumFootprintGABAA() const override;

    void setCurrentStates(std::unique_ptr<state[]> const& states);
    void setCurrentSumFootprintAMPA(std::unique_ptr<float[]> const& sumFootprintAMPA);
    void setCurrentSumFootprintNMDA(std::unique_ptr<float[]> const& sumFootprintNMDA);
    void setCurrentSumFootprintGABAA(std::unique_ptr<float[]> const& sumFootprintGABAA);

    std::vector<__int64> getTimesCalculations() const;
    std::vector<__int64> getTimesConvolutions() const;

private:
    std::unique_ptr<state[]> _states;
    std::unique_ptr<float[]> _sumFootprintAMPA;
    std::unique_ptr<float[]> _sumFootprintNMDA;
    std::unique_ptr<float[]> _sumFootprintGABAA;

    std::unique_ptr<float[]> _distances;
    std::unique_ptr<float[]> _sVals;

    std::vector<__int64> _timesConvolutions;
    std::vector<__int64> _timesCalculations;

    const float _dt;
    const size_t _nX;
    const size_t _nY;
    const size_t _nZ;
    const size_t _numNeurons;
    const size_t _timesteps;

    size_t _t;    const Convolution _convolution;    float _f_w_EE(const float d);

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
    void runge4_f_dV_dt(const size_t idx,
                        const size_t ind_old);
    void runge4_f_I_Na_dh_dt(const size_t idx,
                             const size_t ind_old);
    void runge4_f_dsAMPA_dt(const size_t idx,
                            const size_t ind_old);
    void runge4_f_dn_dt(const size_t idx,
                        const size_t ind_old);
    void runge4_f_dz_dt(const size_t idx,
                        const size_t ind_old);
    void runge4_f_dxNMDA_dt(const size_t idx,
                            const size_t ind_old);
    void runge4_f_dsNMDA_dt(const size_t idx,
                            const size_t ind_old);

    void convolutionAMPA(const size_t ind_old);
    void convolutionNMDA(const size_t ind_old);
    void convolutionGABAA(const size_t ind_old);

    void computeRungeKuttaApproximations(size_t ind_old);
    void computeConvolutions(size_t ind_old);
};
