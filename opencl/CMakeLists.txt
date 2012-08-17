#
# Copyright (C) 2012 Benjamin Wild
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to 
# deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#

project(opencl)

add_subdirectory(gtest)
enable_testing()

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
  
  set(GLFW_INCLUDE_DIR "${opencl_SOURCE_DIR}/glfw/")
  if(CMAKE_GENERATOR MATCHES "Win64")
      # TODO
  else()
    set(GLFW_LIBRARY
      "${opencl_SOURCE_DIR}/glfw/x86/GLFW.lib"
      "${opencl_SOURCE_DIR}/glfw/x86/GLFWDLL.lib"
    )
  endif()
  
  set(GLEW_INCLUDE_DIR "${opencl_SOURCE_DIR}/glew/")
  if(CMAKE_GENERATOR MATCHES "Win64")
    set(GLEW_LIBRARY
      "${opencl_SOURCE_DIR}/glew/x64/glew32.lib"
      "${opencl_SOURCE_DIR}/glew/x64/glew32s.lib"
    )
  else()
    set(GLEW_LIBRARY
      "${opencl_SOURCE_DIR}/glew/x86/glew32.lib"
      "${opencl_SOURCE_DIR}/glew/x86/glew32s.lib"
    )
  endif() 
  
  message("  FFTW_LIBRARIES: ${FFTW_LIBRARIES}")
endif()

SET(INCLUDEDIRS
      ${opencl_SOURCE_DIR}
      SYSTEM
      ${Boost_INCLUDE_DIR}
      ${OPENCL_INCLUDE_DIRS}
      ${OPENGL_INCLUDE_DIR}
      ${GLFW_INCLUDE_DIR}
      ${GLEW_INCLUDE_DIR}
      ${gtest_SOURCE_DIR}/include
      ${gtest_SOURCE_DIR}
)

ADD_DEFINITIONS(-DCL_SOURCE_DIR="${opencl_SOURCE_DIR}")

SET(HEADERS
  stdafx.h
  util.h
  utilShader.h
  CLWrapper.h
  BaseSimulator.h
  CLSimulator.h
  CPUSimulator.h
  Plotter.h
  BasePlotter.h
  GnuPlotPlotter.h
  OpenGLPlotter.h
  Definitions.h
  gnuplot_i/gnuplot_i.h
)

SET(SOURCE
  util.cpp
  utilShader.cpp
  CLWrapper.cpp
  CLSimulator.cpp
  CPUSimulator.cpp
  GnuPlotPlotter.cpp
  OpenGLPlotter.cpp
  kernels.cl
  vertexShader.glsl
  fragmentShader.glsl
  gnuplot_i/gnuplot_i.cpp
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

add_executable(simulate
      main.cpp
      ${SOURCE} 
      ${HEADERS}
      ${OPENCL_FFT}
)

SET(LIBRARIES
  ${Boost_LIBRARIES}
  ${OPENCL_LIBRARIES}
  ${FFTW_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${GLFW_LIBRARY}
  ${GLEW_LIBRARY}
)

include_directories(${INCLUDEDIRS})

add_executable(execute_unit_tests
      openclUnitTests.cpp
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
  
  target_link_libraries(simulate
        ${LIBRARIES}
        Winmm)
  
  target_link_libraries(execute_unit_tests
      ${LIBRARIES}
      Winmm
      gtest
      gtest_main)
else()
  target_link_libraries(simulate
        ${LIBRARIES}
  )
  
  target_link_libraries(execute_unit_tests
      ${LIBRARIES}
      gtest
      gtest_main)
endif()

add_test(openclUnitTests openclUnitTests)

if(MSVC)
#  list(REMOVE_ITEM Boost_LIBRARIES debug)
#  list(REMOVE_ITEM Boost_LIBRARIES optimized)
#  INSTALL(FILES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${FFTW_LIBRARIES} ${Boost_LIBRARIES}
  INSTALL(FILES 
    "${opencl_SOURCE_DIR}/fftw_win32/libfftw3f-3.dll"
    "${opencl_SOURCE_DIR}/glfw/x86/GLFW.dll"
    "${opencl_SOURCE_DIR}/glew/x86/glew32.dll"
    DESTINATION "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/Debug"
    CONFIGURATIONS Debug)
  INSTALL(FILES 
    "${opencl_SOURCE_DIR}/fftw_win32/libfftw3f-3.dll"
    "${opencl_SOURCE_DIR}/glfw/x86/GLFW.dll"
    "${opencl_SOURCE_DIR}/glew/x86/glew32.dll"
    DESTINATION "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/Release"
    CONFIGURATIONS Release)
endif()
