#pragma once

#include "cpplog/cpplog.hpp"

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

typedef float splitComplex[2];

typedef shared_ptr<cpplog::BaseLogger> Logger;