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
      ${Boost_INCLUDE_DIR}
      ${OPENCL_INCLUDE_DIRS}
      ${OPENGL_INCLUDE_DIR}
      ${GLFW_INCLUDE_DIR}
      ${GLEW_INCLUDE_DIR}
      ${gtest_SOURCE_DIR}/include
      ${gtest_SOURCE_DIR}
)

if(VLD_FOUND)
      SET(INCLUDEDIRS ${VLD_INCLUDE_DIRS} ${INCLUDEDIRS})
endif()

include_directories(${INCLUDEDIRS})

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

SET(HEADERS
  stdafx.h
  util.h
  utilShader.h
  CLWrapper.h
  Simulator.h
  BasePlotter.h
  GnuPlotPlotter.h
  OpenGLPlotter.h
  Definitions.h
  gnuplot_i/gnuplot_i.hpp
)

SET(SOURCE
  main.cpp
  util.cpp
  utilShader.cpp
  CLWrapper.cpp
  Simulator.cpp
  GnuPlotPlotter.cpp
  OpenGLPlotter.cpp
  kernels.cl
  vertexShader.glsl
  fragmentShader.glsl
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

SET(LIBRARIES
  ${Boost_LIBRARIES}
  ${OPENCL_LIBRARIES}
  ${FFTW_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${GLFW_LIBRARY}
  ${GLEW_LIBRARY}
)

if(VLD_FOUND)
      SET(LIBRARIES ${VLD_LIBRARIES} ${LIBRARIES})
endif()

if(MSVC)
  target_link_libraries(opencl 
        ${LIBRARIES}
        Winmm
  )
else()
  target_link_libraries(opencl 
        ${LIBRARIES}
  )
endif()