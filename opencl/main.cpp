#include <boost/program_options.hpp>

#include "opencl_fft/clFFT.h"

#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

int main(int argc, char** argv)
{
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
    int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }
 
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("vector_add_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, 
            &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    //clFFT_Direction dir = clFFT_Forward;
    clFFT_Dim3 n = { 1024, 1, 1 };
    clFFT_DataFormat dataFormat = clFFT_SplitComplexFormat;
    clFFT_Dimension dim = clFFT_1D;
    cl_int *error_code = (cl_int*)malloc(sizeof(cl_int));

    clFFT_Plan fftPlan = clFFT_CreatePlan( context, n, dim, dataFormat, error_code );

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);

    exit(0);
}

//#define __NO_STD_VECTOR // Use cl::vector instead of STL version
//#include "cl.hpp"
//
//#include <iostream>
//
//using namespace std;
//
//void displayPlatformInfo(cl::vector< cl::Platform > platformList,
//			 int deviceType)
//{
//  // print out some device specific information
//  cout << "Platform number is: " << platformList.size() << endl;
//    
//  string platformVendor;
//  platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, 
//			  &platformVendor);
// 
//  cout << "device Type " 
//       << ((deviceType==CL_DEVICE_TYPE_GPU)?"GPU":"CPU") << endl;
//  cout << "Platform is by: " << platformVendor << "\n";
//}
//
//int main(int argc, char** argv)
//{
//    int deviceType = CL_DEVICE_TYPE_GPU; // CL_DEVICE_TYPE_CPU;
//
//    cl::vector< cl::Platform > platformList;
//    cl::Platform::get(&platformList);
//    
//    displayPlatformInfo(platformList, deviceType);
//
//    cl_context_properties cprops[3] = 
//    {CL_CONTEXT_PLATFORM, 
//     (cl_context_properties)(platformList[0])(), 0};
// 
//    cl::Context context(deviceType, cprops);
//    
//    cl::vector<cl::Device> devices = 
//      context.getInfo<CL_CONTEXT_DEVICES>();
//    
//#ifdef PROFILING
//    cl::CommandQueue queue(context, devices[0], 
//			   CL_QUEUE_PROFILING_ENABLE);
//#else
//    cl::CommandQueue queue(context, devices[0], 0);
//#endif
//
//    clFFT_Plan clFFT_CreatePlan( context, clFFT_Dim3 n, clFFT_Dimension dim, clFFT_DataFormat dataFormat, cl_int *error_code );
//}
