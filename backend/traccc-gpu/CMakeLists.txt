# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.22)

project(TracccBackend LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(/usr/local/cuda/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../standalone/src)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because recommended backend
# doesn't use GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "r25.02" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r25.02" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r25.02" CACHE STRING "Tag for triton-inference-server/backend repo")

#
# Setting C++ min standard
#
set(TRITON_MIN_CXX_STANDARD 20 CACHE STRING "The minimum C++ standard whose features are requested to build this target.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set optimization flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Building in Release mode with optimizations")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
endif()

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

# 
# Traccc dependencies
#
find_package(covfie REQUIRED)
find_package(traccc REQUIRED)

# Work around an upstream ACTS packaging bug. 
if(TARGET Eigen3::Eigen)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${traccc_INCLUDE_DIR}/eigen3")
endif()

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_traccc.ldscript libtriton_traccc.ldscript COPYONLY)

add_library(
  triton-traccc-backend SHARED
  src/traccc.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../../standalone/src/shm_region.cpp
)

add_library(
  TraccBackend::triton-traccc-backend ALIAS triton-traccc-backend
)

target_include_directories(
  triton-traccc-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${traccc_INCLUDE_DIR}
)

target_compile_features(triton-traccc-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-traccc-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror
    $<$<CONFIG:Release>:-O3 -march=native -mtune=native>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g>
    $<$<CONFIG:Debug>:-O0 -g>>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor
    $<$<CONFIG:Release>:/O2 /Ob2 /DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:/O2 /Ob1 /Zi>
    $<$<CONFIG:Debug>:/Od /Zi>>
)

target_link_libraries(
  triton-traccc-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    # traccc deps
    vecmem::core 
    vecmem::cuda 
    traccc::io 
    traccc::performance
    traccc::core 
    traccc::device_common
    traccc::cuda
    traccc::cuda_utils
    traccc::options
    detray::io
    rt          # shm_open, for the detray-in-shared-memory path
)

# must set algrebra puglin include array
add_definitions(-DALGEBRA_PLUGINS_INCLUDE_ARRAY)

if(WIN32)
  set_target_properties(
    triton-traccc-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_traccc
  )
else()
  set_target_properties(
    triton-traccc-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_traccc
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_traccc.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_traccc.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TracccBackend)

install(
  TARGETS
    triton-traccc-backend
  EXPORT
    triton-traccc-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/traccc
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/traccc
)

install(DIRECTORY ${CMAKE_INSTALL_PREFIX}/../../models
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        USE_SOURCE_PERMISSIONS
)

install(
  TARGETS
    triton-traccc-backend
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/../../models/traccc-gpu
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/../../models/traccc-gpu
)

install(
  EXPORT
    triton-traccc-backend-targets
  FILE
    TracccBackendTargets.cmake
  NAMESPACE
    TracccBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TracccBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TracccBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TracccBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-traccc-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TracccBackendTargets.cmake
  NAMESPACE TracccBackend::
)

export(PACKAGE TracccBackend)
