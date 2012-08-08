project(opencl)

add_subdirectory(gtest)
enable_testing()

include_directories(
      ${Boost_INCLUDE_DIR}
      ${OPENCL_INCLUDE_DIRS}
      ${gtest_SOURCE_DIR}/include
      ${gtest_SOURCE_DIR}
      ${opencl_SOURCE_DIR}
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
  Simulator.cpp
  Plotter.cpp
  kernels.cl
)

SET(HEADERS
  stdafx.h
  util.h
  CLWrapper.h
  Simulator.h
  Plotter.h
  Definitions.h
  gnuplot_i/gnuplot_i.hpp
)

SET(OPENCL_FFT
  opencl_fft/fft_execute.cpp
  opencl_fft/fft_setup.cpp
  opencl_fft/fft_kernelstring.cpp
  opencl_fft/clFFT.h
  opencl_fft/fft_internal.h
  opencl_fft/fft_base_kernels.h
)

ADD_MSVC_PRECOMPILED_HEADER("stdafx.h" "stdafx.cpp" SOURCE)

add_executable(opencl
      ${SOURCE} 
      ${HEADERS}
      ${OPENCL_FFT}
)

if(MSVC)
  if(CMAKE_GENERATOR MATCHES "Win64")
    set(FFTW_LIBRARIES
      "${opencl_SOURCE_DIR}/fftw_win64/libfftw3-3.lib"
      "${opencl_SOURCE_DIR}/fftw_win64/libfftw3f-3.lib"
      "${opencl_SOURCE_DIR}/fftw_win64/libfftw3l-3.lib"
    )
  else()
    set(FFTW_LIBRARIES
      "${opencl_SOURCE_DIR}/fftw_win32/libfftw3-3.lib"
      "${opencl_SOURCE_DIR}/fftw_win32/libfftw3f-3.lib"
      "${opencl_SOURCE_DIR}/fftw_win32/libfftw3l-3.lib"
    )
  endif()
  
  message("  FFTW_LIBRARIES: ${FFTW_LIBRARIES}")
  
  target_link_libraries(opencl 
        ${Boost_LIBRARIES}
        ${OPENCL_LIBRARIES}
        ${FFTW_LIBRARIES}
        Winmm
  )
else()
  target_link_libraries(opencl 
        ${Boost_LIBRARIES}
        ${OPENCL_LIBRARIES}
        ${FFTW_LIBRARIES}
  )
endif()
