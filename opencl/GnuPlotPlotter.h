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
#include "BasePlotter.h"

class GnuPlotPlotter : public BasePlotter
{
public:
    GnuPlotPlotter(const unsigned int nX,
                   const unsigned int nY,
                   const unsigned int nZ,
                   unsigned int index, 
                   float dt);

    void step(const state *curState, 
              const unsigned int t, 
              std::unique_ptr<float[]> const& sumFootprintAMPA, 
              std::unique_ptr<float[]> const& sumFootprintNMDA, 
              std::unique_ptr<float[]> const& sumFootprintGABAA) override;

    void plot() override;

private:
    const unsigned int _nX;
    const unsigned int _nY;
    const unsigned int _nZ;
    const unsigned int _numNeurons;
    unsigned int _index;
    float _dt;

    std::vector<float> _V, _h, _n, _z, _sAMPA, _xNMDA, _sNMDA, _IApp;
    std::vector<float> _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA;
    std::vector<float> _spikeTimes;
    std::vector<unsigned int> _spikeNeuronIndicesX, _spikeNeuronIndicesY;
    std::vector<bool> _spikeArr;
};

