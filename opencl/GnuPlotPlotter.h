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

#include "BasePlotter.h"
#include "Definitions.h"

class GnuPlotPlotter : public BasePlotter {
public:

    GnuPlotPlotter(const size_t nX,
                   const size_t nY,
                   const size_t nZ,
                   size_t index,
                   float dt);

    void step(const state *curState,
              const size_t t,
              std::unique_ptr<float[]> const& sumFootprintAMPA,
              std::unique_ptr<float[]> const& sumFootprintNMDA,
              std::unique_ptr<float[]> const& sumFootprintGABAA) override;

    void plot() override;

private:

    const size_t _nX;
    const size_t _nY;
    const size_t _nZ;
    const size_t _numNeurons;
    size_t _index;
    float _dt;

    std::vector<float> _V, _h, _n, _z, _sAMPA, _xNMDA, _sNMDA, _IApp;
    std::vector<float> _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA;
    std::vector<float> _spikeTimes;
    std::vector<size_t> _spikeNeuronIndicesX, _spikeNeuronIndicesY;
    std::vector<bool> _spikeArr;
};
