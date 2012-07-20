#pragma once

#include "Definitions.h"

//#include "gnuplot_i\gnuplot_i.hpp"

class Plotter
{
public:
    Plotter(unsigned int numNeurons, unsigned int index, float dt);
    void step(const state *curState, const unsigned int t, const float* sumFootprintAMPA, const float* sumFootprintNMDA, const float* sumFootprintGABAA);
    void plot();

private:
    unsigned int _numNeurons;
    unsigned int _index;
    float _dt;

    std::vector<float> _V, _h, _n, _z, _sAMPA, _xNMDA, _sNMDA, _IApp;
    std::vector<float> _sumFootprintAMPA, _sumFootprintNMDA, _sumFootprintGABAA;
    std::vector<float> _spikeTimes, _spikeNeuronIndices;
    std::vector<bool> _spikeArr;
};

