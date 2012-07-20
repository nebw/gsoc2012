#pragma once

#include "cpplog/cpplog.hpp"

#include <boost/shared_ptr.hpp>

typedef struct
{
    float V;
    float h;
    float n;
    float z;
    float s_AMPA;
    float x_NMDA;
    float s_NMDA;
    float s_GABAA;
    float I_app;
} state;

typedef struct
{
    float *real;
    float *imag;
} clFFT_SplitComplexFloat;

typedef boost::shared_ptr<cpplog::BaseLogger> Logger;