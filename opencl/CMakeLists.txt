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

add_test( openclUnitTests openclUnitTests )