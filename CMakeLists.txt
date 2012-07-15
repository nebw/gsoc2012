cmake_minimum_required(VERSION 2.6)
cmake_policy(SET CMP0015 NEW)
project(gsoc2012)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/findopencl/")

##set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON) 
##set(Boost_USE_STATIC_RUNTIME OFF)

add_definitions( -DBOOST_ALL_NO_LIB )

find_package(Boost COMPONENTS program_options REQUIRED)
message("Boost information:")
message("  Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message("  Boost_LIBRARIES: ${Boost_LIBRARIES}")
message("  Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")

find_package(OpenCL REQUIRED)
message("OpenCL information:") 
message("  OPENCL_INCLUDE_DIRS: ${OPENCL_INCLUDE_DIRS}") 
message("  OPENCL_LIBRARIES: ${OPENCL_LIBRARIES}") 
message("  OPENCL_HAS_CPP_BINDINGS: ${OPENCL_HAS_CPP_BINDINGS}") 

## Compiler flags for GCC and Clang
if(NOT MSVC)
  # enable extra compiler warnings
  set(CMAKE_CXX_FLAGS "-Wall -Wextra ${CMAKE_CXX_FLAGS}")
  # even more warnings
  set(CMAKE_CXX_FLAGS "-Wpointer-arith -Wcast-qual -Wwrite-strings -Wno-long-long	-Wno-variadic-macros -Wno-multichar -Woverloaded-virtual -Wsign-promo ${CMAKE_CXX_FLAGS}")
  # enable support for c++11
  set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
  # treat warnings as errors
  # set(CMAKE_CXX_FLAGS "-Werror ${CMAKE_CXX_FLAGS}")
  # set(CMAKE_CXX_FLAGS "-Weffc++ ${CMAKE_CXX_FLAGS}")
endif()

## Compiler flags for GCC only
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "-Wstrict-null-sentinel ${CMAKE_CXX_FLAGS}")
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/opencl)