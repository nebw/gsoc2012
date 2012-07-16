project(opencl)

add_subdirectory(gtest)
enable_testing()

include_directories(
      ${Boost_INCLUDE_DIR}
      ${OPENCL_INCLUDE_DIRS}
      ${gtest_SOURCE_DIR}/include
      ${gtest_SOURCE_DIR}
)

add_executable(openclUnitTests
      openclUnitTests.cpp
      opencl_fft/fft_execute.cpp
      opencl_fft/fft_setup.cpp
      opencl_fft/fft_kernelstring.cpp
)

target_link_libraries(openclUnitTests 
      ${Boost_LIBRARIES}
      ${OPENCL_LIBRARIES}
      gtest
      gtest_main
)

add_test(openclUnitTests openclUnitTests)

ADD_DEFINITIONS(-DCL_SOURCE_DIR="${opencl_SOURCE_DIR}")

SET(SOURCE
  main.cpp
  util.cpp
  CLWrapper.cpp
)

SET(HEADERS
  stdafx.h
  util.h
  CLWrapper.h
)

SET(OPENCL_FFT
  OpenCL_FFT/fft_execute.cpp
  OpenCL_FFT/fft_setup.cpp
  OpenCL_FFT/fft_kernelstring.cpp
  OpenCL_FFT/clFFT.h
  OpenCL_FFT/fft_internal.h
  OpenCL_FFT/fft_base_kernels.h
)

ADD_MSVC_PRECOMPILED_HEADER("stdafx.h" "stdafx.cpp" SOURCE)

add_executable(opencl
      ${SOURCE} 
      ${HEADERS}
      ${OPENCL_FFT}
)

target_link_libraries(opencl 
      ${Boost_LIBRARIES}
      ${OPENCL_LIBRARIES}
)