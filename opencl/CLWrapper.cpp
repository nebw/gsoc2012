#include <iostream>
#include <stdio.h>
#include <string>

#include "CLWrapper.h"
#include "util.h"

#include <cassert>

#include <boost/foreach.hpp>
//#include <boost/thread/thread.hpp>

CLWrapper::CLWrapper()
{
    try {
        std::cout << "Initializing OpenCL..." << std::endl;

        std::vector<cl::Platform> platforms;
        _err = cl::Platform::get(&platforms);
        std::cout << "cl::Platform::get(): " << oclErrorString(_err) << std::endl;

        assert(platforms.size() > 0);

        unsigned int numPlatform = 0;
        std::cout << std::endl;
        BOOST_FOREACH(cl::Platform const & platform, platforms)
        {
            std::cout << "Platform " << numPlatform << ":" << std::endl;
            std::cout << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
            std::cout << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
            std::cout << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
            std::cout << platform.getInfo<CL_PLATFORM_PROFILE>() << std::endl;
            std::cout << platform.getInfo<CL_PLATFORM_EXTENSIONS>() << std::endl;
            std::cout << std::endl;
            ++numPlatform;
        }

        cl_context_properties properties[] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };

        _context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

        _devices = _context.getInfo<CL_CONTEXT_DEVICES>();
        _queue = cl::CommandQueue(_context, _devices[0], 0, &_err);
    }
    catch (cl::Error er) {
        std::cout << "OpenCL Error: " << er.what() << " " << oclErrorString(er.err()) << std::endl;
        throw er;
    }
}

cl::Program CLWrapper::loadProgram(std::string path)
{
    try
    {
        cl::Program program;

        std::cout << ("Loading the program...") << std::endl;

        int pl;
        char *kernel_source;
        kernel_source = file_contents(path.c_str(), &pl);

        std::cout << "Kernel size: " << pl << std::endl;

        cl::Program::Sources source(1, std::make_pair(kernel_source, pl));
        program = cl::Program(_context, source);

        std::cout << "Building OpenCL program..." << std::endl;

        std::string includes(CL_SOURCE_DIR);
        includes = "-I" + includes;
        _err      = program.build(_devices, includes.c_str());

        std::cout << "Done building OpenCL program" << std::endl;

        std::cout << "Build Status: " << oclErrorString(program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(
                                                            _devices[0])) << std::endl;
        std::cout << "Build Options:\t" <<
        program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(_devices[0]) << std::endl;
        std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
            _devices[0]) << std::endl;

        return program;
    }
    catch (cl::Error er) {
        std::cout << "OpenCL Error: " << er.what() << " " << oclErrorString(er.err()) << std::endl;
        throw er;
    }
}

cl::Context const& CLWrapper::getContext() const
{
    return _context;
}

cl::CommandQueue const& CLWrapper::getQueue() const
{
    return _queue;
}

cl_context CLWrapper::getContextC() const
{
    return _context();
}

cl_command_queue CLWrapper::getQueueC() const
{
    return _queue();
}
