project(clFFT)

include_directories(${CMAKE_SOURCE_DIR}/includes)

add_library (clFFT clFFT.h fft_base_kernels.h fft_internal.h fft_setup.cpp fft_kernelstring.cpp fft_execute)