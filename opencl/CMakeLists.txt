include_directories(
      ${Boost_INCLUDE_DIR}
      ${OPENCL_INCLUDE_DIRS}
)

add_executable(opencl 
      main.cpp
      opencl_fft/fft_execute.cpp
      opencl_fft/fft_setup.cpp
##      opencl_fft/main.cpp
      opencl_fft/fft_kernelstring.cpp
)

target_link_libraries(opencl 
      ${Boost_LIBRARIES}
      ${OPENCL_LIBRARIES}
)