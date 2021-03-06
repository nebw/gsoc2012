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

cmake_minimum_required(VERSION 2.6)
cmake_policy(SET CMP0015 NEW)

## Source: http://stackoverflow.com/a/2956392/1474346
MACRO(ADD_MSVC_PRECOMPILED_HEADER PrecompiledHeader PrecompiledSource SourcesVar)
  IF(MSVC)
    GET_FILENAME_COMPONENT(PrecompiledBasename ${PrecompiledHeader} NAME_WE)
    SET(PrecompiledBinary "$(IntDir)/${PrecompiledBasename}.pch")
    SET(Sources ${${SourcesVar}})

    SET_SOURCE_FILES_PROPERTIES(${PrecompiledSource}
                                PROPERTIES COMPILE_FLAGS "/Yc\"${PrecompiledHeader}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_OUTPUTS "${PrecompiledBinary}")
    SET_SOURCE_FILES_PROPERTIES(${Sources}
                                PROPERTIES COMPILE_FLAGS "/Yu\"${PrecompiledHeader}\" /FI\"${PrecompiledHeader}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_DEPENDS "${PrecompiledBinary}")  
    # Add precompiled header to SourcesVar
    LIST(APPEND ${SourcesVar} ${PrecompiledSource})
  ENDIF(MSVC)
ENDMACRO(ADD_MSVC_PRECOMPILED_HEADER)

project(gsoc2012)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME OFF)
add_definitions( -DBOOST_ALL_NO_LIB ) 
#add_definitions( -DBOOST_ALL_DYN_LINK )

find_package(Boost COMPONENTS chrono program_options system filesystem REQUIRED)
message("Boost information:")
message("  Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message("  Boost_LIBRARIES: ${Boost_LIBRARIES}")
message("  Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")

find_package(OpenCL REQUIRED)
message("OpenCL information:") 
message("  OPENCL_INCLUDE_DIRS: ${OPENCL_INCLUDE_DIRS}") 
message("  OPENCL_LIBRARIES: ${OPENCL_LIBRARIES}") 
message("  OPENCL_HAS_CPP_BINDINGS: ${OPENCL_HAS_CPP_BINDINGS}") 

if(NOT MSVC)
  find_package(FFTW REQUIRED)
  message("FFTW information:") 
  message("  FFTW_INCLUDES: ${FFTW_INCLUDES}") 
  message("  FFTW_LIBRARIES: ${FFTW_LIBRARIES}") 
endif()

find_package(OpenGL REQUIRED)
if(OPENGL_FOUND)
  message("OpenGL information:") 
  message("  OPENGL_INCLUDE_DIR: ${OPENGL_INCLUDE_DIR}") 
  message("  OPENGL_LIBRARIES: ${OPENGL_LIBRARIES}") 
endif()

if(NOT MSVC)
  find_package(GLEW REQUIRED)
  if(GLEW_FOUND)
  message("GLEW information:") 
    message("  GLEW_INCLUDE_PATH: ${GLEW_INCLUDE_PATH}") 
    message("  GLEW_LIBRARY: ${GLEW_LIBRARY}") 
  endif()
endif()

if(NOT MSVC)
  find_package(GLFW REQUIRED)
  if(GLFW_FOUND)
  message("GLFW information:") 
    message("  GLFW_INCLUDE_DIR: ${GLFW_INCLUDE_DIR}") 
    message("  GLFW_LIBRARY: ${GLFW_LIBRARY}") 
  endif()
endif()

## Compiler flags for GCC and Clang
if(NOT MSVC)
  # enable support for c++11
  set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
endif()

if( MSVC ) # VS2012 doesn't support correctly the tuples yet
  add_definitions( /D _VARIADIC_MAX=10 )
endif()

add_subdirectory(src)
