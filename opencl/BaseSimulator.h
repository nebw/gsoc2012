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

#include "Definitions.h"

class BaseSimulator {
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

    BaseSimulator() {}
    virtual ~BaseSimulator() {}

    virtual void step() = 0;
    virtual void simulate() = 0;

    virtual state const* getCurrentStatesOld() const = 0;
    virtual state const* getCurrentStatesNew() const = 0;
    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintAMPA() const = 0;
    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintNMDA() const = 0;
    virtual std::unique_ptr<float[]> const& getCurrentSumFootprintGABAA() const = 0;

private:
    BaseSimulator(const BaseSimulator&);
    BaseSimulator& operator=(BaseSimulator&);
};
