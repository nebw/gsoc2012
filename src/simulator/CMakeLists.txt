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

## Compiler flags for GCC and Clang
if(NOT MSVC)
  # demand strict ISO C++ compliance
  set(CMAKE_CXX_FLAGS "-pedantic ${CMAKE_CXX_FLAGS}")
  # enable extra compiler warnings
  set(CMAKE_CXX_FLAGS "-Wall -Wextra ${CMAKE_CXX_FLAGS}")
  # even more warnings
  set(CMAKE_CXX_FLAGS "-Wpointer-arith -Wcast-qual -Wno-long-long -Wno-variadic-macros -Wno-multichar -Woverloaded-virtual -Wsign-promo -Winit-self -Wuninitialized -Wmissing-declarations ${CMAKE_CXX_FLAGS}")
  # treat warnings as errors
  # set(CMAKE_CXX_FLAGS "-Werror ${CMAKE_CXX_FLAGS}")
  # set(CMAKE_CXX_FLAGS "-Weffc++ ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "-Wno-deprecated ${CMAKE_CXX_FLAGS}")
else()
  # enable visual studio static code analysis
  # set(CMAKE_CXX_FLAGS "-analyze ${CMAKE_CXX_FLAGS}")
  # extra warnings
  set(CMAKE_CXX_FLAGS "/W4 ${CMAKE_CXX_FLAGS}")
endif()
## Compiler flags for GCC only
if(CMAKE_COMPILER_IS_GNUCXX)
  # enforce ansi standards
  set(CMAKE_CXX_FLAGS "-ansi ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "-Wstrict-null-sentinel ${CMAKE_CXX_FLAGS}")
endif()

enable_testing()

if(MSVC)
  if(CMAKE_GENERATOR MATCHES "Win64")
    set(FFTW_LIBRARIES
      "${CMAKE_SOURCE_DIR}/libs/fftw/x64/libfftw3-3.lib"
      "${CMAKE_SOURCE_DIR}/libs/fftw/x64/libfftw3f-3.lib"
      "${CMAKE_SOURCE_DIR}/libs/fftw/x64/libfftw3l-3.lib"
    )
  else()
    set(FFTW_LIBRARIES
      "${CMAKE_SOURCE_DIR}/libs/fftw/x86/libfftw3-3.lib"
      "${CMAKE_SOURCE_DIR}/libs/fftw/x86/libfftw3f-3.lib"
      "${CMAKE_SOURCE_DIR}/libs/fftw/x86/libfftw3l-3.lib"
    )
  endif()
  
  if(CMAKE_GENERATOR MATCHES "Win64")
      # TODO
  else()
    set(GLFW_LIBRARY
      "${CMAKE_SOURCE_DIR}/libs/glfw/x86/GLFW.lib"
      "${CMAKE_SOURCE_DIR}/libs/glfw/x86/GLFWDLL.lib"
    )
  endif()
  
  if(CMAKE_GENERATOR MATCHES "Win64")
    set(GLEW_LIBRARY
      "${CMAKE_SOURCE_DIR}/libs/glew/x64/glew32.lib"
      "${CMAKE_SOURCE_DIR}/libs/glew/x64/glew32s.lib"
    )
  else()
    set(GLEW_LIBRARY
      "${CMAKE_SOURCE_DIR}/libs/glew/x86/glew32.lib"
      "${CMAKE_SOURCE_DIR}/libs/glew/x86/glew32s.lib"
    )
  endif() 
  
  message("  FFTW_LIBRARIES: ${FFTW_LIBRARIES}")
endif()

SET(INCLUDEDIRS
      ${opencl_SOURCE_DIR}
      SYSTEM
      ${CMAKE_SOURCE_DIR}/includes
      ${clFFT_SOURCE_DIR}
      ${gnuplot_i_SOURCE_DIR}
      ${cpplog_SOURCE_DIR}
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
  BasePlotter.h
  GnuPlotPlotter.h
  OpenGLPlotter.h
  Definitions.h
  main.h
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
)

ADD_MSVC_PRECOMPILED_HEADER("stdafx.h" "stdafx.cpp" SOURCE)

add_executable(simulate
      main.cpp
      ${SOURCE} 
      ${HEADERS}
)

SET(LIBRARIES
  clFFT
  gnuplot_i
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
)

target_link_libraries(simulate
    ${LIBRARIES}
)

target_link_libraries(execute_unit_tests
    ${LIBRARIES}
    gtest
    gtest_main)

add_test(openclUnitTests openclUnitTests)

if(MSVC)
  INSTALL(FILES 
    "${CMAKE_SOURCE_DIR}/libs/fftw/x86/libfftw3f-3.dll"
    "${CMAKE_SOURCE_DIR}/libs/glfw/x86/GLFW.dll"
    "${CMAKE_SOURCE_DIR}/libs/glew/x86/glew32.dll"
    DESTINATION "${CMAKE_BINARY_DIR}/src/simulator/Debug"
    CONFIGURATIONS Debug)
  INSTALL(FILES 
    "${CMAKE_SOURCE_DIR}/libs/fftw/x86/libfftw3f-3.dll"
    "${CMAKE_SOURCE_DIR}/libs/glfw/x86/GLFW.dll"
    "${CMAKE_SOURCE_DIR}/libs/glew/x86/glew32.dll"
    DESTINATION "${CMAKE_BINARY_DIR}/src/simulator/Release"
    CONFIGURATIONS Release)
endif()
