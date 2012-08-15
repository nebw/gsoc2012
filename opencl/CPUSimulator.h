#pragma once

#include "Definitions.h"
#include "BaseSimulator.h"

class CPUSimulator : public BaseSimulator
{
    virtual void step() override;

    virtual void simulate() override;

    virtual std::unique_ptr<state[]> const& getCurrentStates() const override;

    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintAMPA() const override;

    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintNMDA() const override;

    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintGABAA() const override;

    virtual std::vector<unsigned long> getTimesCalculations() const override;

    virtual std::vector<unsigned long> getTimesFFTW() const override;

    virtual std::vector<unsigned long> getTimesClFFT() const override;

};