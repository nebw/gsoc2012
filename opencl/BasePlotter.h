#pragma once

#include "Definitions.h"

class BasePlotter
{
public:
    virtual ~BasePlotter() {};

    virtual void step(const state *curState, 
                      const unsigned int t, 
                      std::unique_ptr<float[]> const& sumFootprintAMPA, 
                      std::unique_ptr<float[]> const& sumFootprintNMDA,
                      std::unique_ptr<float[]> const& sumFootprintGABAA) = 0;
    virtual void plot() = 0;
};