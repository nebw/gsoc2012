/**
 * Copyright (C) 2012 Benjamin Wild
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "stdafx.h"

#include "CLWrapper.h"
#include "util.h"

#include <cassert>
#include <iostream>
#include <stdio.h>
#include <string>

#include <boost/foreach.hpp>
#include <boost/format.hpp>

CLWrapper::CLWrapper(Logger const& logger)
    : _logger(logger)
{
    try {
        LOG_INFO(*_logger) << ("Initializing OpenCL...");

        std::vector<cl::Platform> platforms;
        _err = cl::Platform::get(&platforms);
        LOG_INFO(*_logger) << boost::format("cl::Platform::get(): %1%")
        % std::string(oclErrorString(_err));

        assert(platforms.size() > 0);

        size_t numPlatform = 0;
        LOG_INFO(*_logger) << std::endl;
        BOOST_FOREACH(cl::Platform const & platform, platforms)
        {
            LOG_INFO(*_logger) << boost::format("Platform %1%:") % numPlatform;
            LOG_INFO(*_logger) << (platform.getInfo<CL_PLATFORM_VENDOR>());
            LOG_INFO(*_logger) << (platform.getInfo<CL_PLATFORM_NAME>());
            LOG_INFO(*_logger) << (platform.getInfo<CL_PLATFORM_VERSION>());
            LOG_INFO(*_logger) << (platform.getInfo<CL_PLATFORM_PROFILE>());
            LOG_INFO(*_logger) << (platform.getInfo<CL_PLATFORM_EXTENSIONS>());
            LOG_INFO(*_logger) << std::endl;
            ++numPlatform;
        }

        cl_context_properties properties[] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };

        _context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

        _devices = _context.getInfo<CL_CONTEXT_DEVICES>();
        _queue = cl::CommandQueue(_context, _devices[0], 0, &_err);
    }
    catch (cl::Error er) {
        LOG_INFO(*_logger) << boost::format("OpenCl Error: %1% %2%")
        % er.what() % oclErrorString(er.err());
        throw er;
    }
}

cl::Program CLWrapper::loadProgram(std::string path)
{
    try
    {
        cl::Program program;

        LOG_INFO(*_logger) << "Loading the program...";

        int pl;
        char *kernel_source;
        kernel_source = file_contents(path.c_str(), &pl);

        LOG_INFO(*_logger) << boost::format("Kernel size: %1%") % pl;

        cl::Program::Sources source(1, std::make_pair(kernel_source, pl));
        program = cl::Program(_context, source);

        LOG_INFO(*_logger) << ("Building OpenCL program...");

        std::string includes(CL_SOURCE_DIR);
        includes = "-I" + includes;
        _err      = program.build(_devices, includes.c_str());

        LOG_INFO(*_logger) << ("Done building OpenCL program");

        LOG_INFO(*_logger) << boost::format("Build Status: %1%")
        % oclErrorString(program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(_devices[0]));
        LOG_INFO(*_logger) << boost::format("Build Options:\t%1%")
        % program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(_devices[0]);
        LOG_INFO(*_logger) << boost::format("Build Log:\t%1%")
        % program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(_devices[0]);

        return program;
    }
    catch (cl::Error er) {
        LOG_INFO(*_logger) << boost::format("OpenCL Error: %1% %2%")
        % er.what() % oclErrorString(er.err());
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

std::string CLWrapper::getDeviceVendor()
{
    std::string vendor;

    _devices[0].getInfo(CL_DEVICE_VENDOR, &vendor);
    return vendor;
}

std::string CLWrapper::getDeviceName()
{
    std::string name;

    _devices[0].getInfo(CL_DEVICE_NAME, &name);
    return name;
}

