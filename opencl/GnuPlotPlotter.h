#pragma once

#include "Definitions.h"
#include "BasePlotter.h"

class GnuPlotPlotter : public BasePlotter
{
public:
    GnuPlotPlotter(unsigned int numNeurons, unsigned int index, float dt);

    void step(const state *curState, 
              const unsigned int t, 
              std::unique_ptr<float[]> const& sumFootprintAMPA, 
              std::unique_ptr<float[]> const& sumFootprintNMDA, 
              std::unique_ptr<float[]> const& sumFootprintGABAA) override;

    void plot() override;

private:
    unsigned int _numNeurons;
    unsigned int _index;
    float _dt;

    std::vector<float> _V, _h, _n, _z, _sAMPA, _xNMDA, _sNMDA, _IApp;
    std::vector<float> _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA;
    std::vector<float> _spikeTimes, _spikeNeuronIndices;
    std::vector<bool> _spikeArr;
};
