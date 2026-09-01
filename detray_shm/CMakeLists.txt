# Builds only the producer. The consumer side is the Triton backend itself,
# built by build.sh from ../../traccc-aaS/backend/traccc-gpu.
#
# shm_region.cpp/.hpp are staged in from the repo by build_producer.sh rather
# than kept here, so the producer and the backend compile against exactly the
# same region layout. A local copy would be free to drift, and a drifted header
# means the two processes disagree about the region silently.
cmake_minimum_required(VERSION 3.22)
project(d3_producer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(traccc REQUIRED)
find_package(vecmem REQUIRED)
find_package(detray REQUIRED)

add_executable(d3_producer producer.cpp shm_region.cpp)
target_include_directories(d3_producer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(d3_producer PRIVATE traccc::core traccc::io
                                          detray::core vecmem::core rt)
