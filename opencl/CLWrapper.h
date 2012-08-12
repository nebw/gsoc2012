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

    unsigned int _deviceUsed;
    std::vector<cl::Device> _devices;

    cl::Context _context;
    cl::CommandQueue _queue;
    cl::Program _program;
    cl::Kernel _kernel;

    cl_int _err;

    cl::Event _event;
};
