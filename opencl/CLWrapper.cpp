#include <stdio.h>
#include <string>
#include <iostream>

#include "CLWrapper.h"
#include "util.h"

CLWrapper::CLWrapper()
{
    printf("Initialize OpenCL object and context\n");

    // setup devices and context

    // this function is defined in util.cpp
    // it comes from the NVIDIA SDK example code
    // /err = oclGetPlatformID(&platform);
    // oclErrorString is also defined in util.cpp and comes from the NVIDIA SDK
    // /printf("oclGetPlatformID: %s\n", oclErrorString(err));
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    printf("cl::Platform::get(): %s\n", oclErrorString(err));

    if (platforms.size() == 0) {
        printf("Platform size 0\n");
    }


    // Get the number of GPU devices available to the platform
    // we should probably expose the device type to the user
    // the other common option is CL_DEVICE_TYPE_CPU
    // /err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL,
    // &numDevices);
    // /printf("clGetDeviceIDs (get number of devices): %s\n",
    // oclErrorString(err));


    // Create the device list
    // /devices = new cl_device_id [numDevices];
    // /err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices,
    // NULL);
    // /printf("clGetDeviceIDs (create device list): %s\n",
    // oclErrorString(err));


    // for right now we just use the first available device
    // later you may have criteria (such as support for different extensions)
    // that you want to use to select the device
    deviceUsed = 0;

    // create the context
    // /context = clCreateContext(0, 1, &devices[deviceUsed], NULL, NULL, &err);
    // context properties will be important later, for now we go with defualts
    cl_context_properties properties[] =
    { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };

    context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
    devices = context.getInfo<CL_CONTEXT_DEVICES>();

    // create the command queue we will use to execute OpenCL commands
    // /command_queue = clCreateCommandQueue(context, devices[deviceUsed], 0,
    // &err);
    try {
        queue = cl::CommandQueue(context, devices[deviceUsed], 0, &err);
    }
    catch (cl::Error er) {
        printf("ERROR: %s(%d)\n", er.what(), er.err());
    }
}

CLWrapper::~CLWrapper()
{
    /*
       printf("Releasing OpenCL memory\n");
       if(kernel)clReleaseKernel(kernel);
       if(program)clReleaseProgram(program);
       if(command_queue)clReleaseCommandQueue(command_queue);
       //need to release any other OpenCL memory objects here
       if(cl_a)clReleaseMemObject(cl_a);
       if(cl_b)clReleaseMemObject(cl_b);
       if(cl_c)clReleaseMemObject(cl_c);

       if(context)clReleaseContext(context);

       if(devices)delete(devices);
       printf("OpenCL memory released\n");

     */
}

cl::Program CLWrapper::loadProgram(std::string path)
{
    //TODO: use boost::filesystem

    // Program Setup
    cl::Program program;

    printf("load the program\n");


    int   pl;
    char *kernel_source;
    kernel_source = file_contents(path.c_str(), &pl);

    // pl = kernel_source.size();
    printf("kernel size: %d\n", pl);

    // printf("kernel: \n %s\n", kernel_source.c_str());
    try
    {
        cl::Program::Sources source(1,
                                    std::make_pair(kernel_source, pl));
        program = cl::Program(context, source);
    }
    catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }
    printf("build program\n");
    try
    {
        std::string includes(CL_SOURCE_DIR);
        includes = "-I" + includes;
        err      = program.build(devices, includes.c_str());
    }
    catch (cl::Error er) {
        printf("program.build: %s\n", oclErrorString(er.err()));

        // if(err != CL_SUCCESS){
    }
    printf("done building program\n");
    std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(
        devices[0]) << std::endl;
    std::cout << "Build Options:\t" <<
    program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
    std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
        devices[0]) << std::endl;

    return program;
}

cl::Context CLWrapper::getContext() const
{
    return context;
}

cl::CommandQueue CLWrapper::getQueue() const
{
    return queue;
}