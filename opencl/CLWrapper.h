#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include "CL/cl.hpp"

#include <vector>

class CLWrapper {
public:
    CLWrapper();

    cl::Program loadProgram(std::string path);

    cl::Context getContext() const;
    cl::CommandQueue getQueue() const;

    cl_context getContextC() const;
    cl_command_queue getQueueC() const;

private:
    unsigned int deviceUsed;
    std::vector<cl::Device> devices;

    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;

    cl_int err;

    cl::Event event;
};
