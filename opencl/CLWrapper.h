#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include "CL/cl.hpp"

#include <vector>

class CLWrapper {
public:

    CLWrapper();
    ~CLWrapper();

    cl::Program loadProgram(std::string path);

    cl::Context getContext() const;
    cl::CommandQueue getQueue() const;

    // //setup the data for the kernel
    // //these are implemented in part1.cpp (in the future we will make these
    // more general)
    // void popCorn();
    // //execute the kernel
    // void runKernel();

private:

    cl::Buffer states;

    // device variables
    unsigned int deviceUsed;
    std::vector<cl::Device> devices;

    cl::Context context;

    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel  kernel;

    // debugging variables
    cl_int err;

    // /cl_event event;
    cl::Event event;
};
