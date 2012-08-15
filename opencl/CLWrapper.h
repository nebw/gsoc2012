#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include "Definitions.h"

#include "CL/cl.hpp"

#include <vector>

class CLWrapper {
public:

    CLWrapper(Logger const& logger);

    cl::Program loadProgram(std::string path);

    cl::Context const& getContext() const;
    cl::CommandQueue const& getQueue() const;

    cl_context getContextC() const;
    cl_command_queue getQueueC() const;

private:

    Logger _logger;

    unsigned int _deviceUsed;
    std::vector<cl::Device> _devices;

    cl::Context _context;
    cl::CommandQueue _queue;
    cl::Kernel _kernel;

    cl_int _err;

    cl::Event _event;
};
