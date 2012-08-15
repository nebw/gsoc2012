#pragma once

#include "Definitions.h"

#include <boost/filesystem.hpp>

class BaseSimulator
{
public:
    enum Plot {
        NO_PLOT = 0,
        PLOT_GNUPLOT,
        PLOT_OPENGL
    };

    enum Measure {
        NO_MEASURE = 0,
        MEASURE
    };

    enum FFT_FFTW {
        NO_FFTW = 0,
        FFTW
    };

    enum FFT_clFFT {
        NO_CLFFT = 0,
        CLFFT
    };

    virtual ~BaseSimulator() {};

    virtual void step() = 0;
    virtual void simulate() = 0;

    virtual std::unique_ptr<state[]> const& getCurrentStates() const = 0;
    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintAMPA() const = 0;
    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintNMDA() const = 0;
    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintGABAA() const = 0;

    virtual std::vector<unsigned long> getTimesCalculations() const = 0;
    virtual std::vector<unsigned long> getTimesFFTW() const = 0;
    virtual std::vector<unsigned long> getTimesClFFT() const = 0;
};