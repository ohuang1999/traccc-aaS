# The producer, built against an ATLAS release rather than the .sif, so the
# region it writes carries the same vecmem/detray layout the release-built
# backend expects. shm_region.{hpp,cpp} are staged in from the repo by
# build_producer_release.sh -- one definition, compiled into both sides.
cmake_minimum_required(VERSION 3.22)
project(rel_producer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(traccc REQUIRED)
find_package(vecmem REQUIRED)
find_package(detray REQUIRED)

# Same ACTS packaging repair as the backend: the exported targets advertise an
# eigen3 directory inside the ACTS include tree that the release does not ship.
find_path(REAL_EIGEN3_INCLUDE_DIR signature_of_eigen3_matrix_library
          PATH_SUFFIXES eigen3)
if(NOT REAL_EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR "Eigen3 not found; cannot repair the ACTS include paths")
endif()
if(TARGET Eigen3::Eigen)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${REAL_EIGEN3_INCLUDE_DIR}")
endif()
foreach(_tgt traccc::core traccc::io detray::core detray::io)
  if(TARGET ${_tgt})
    get_target_property(_inc ${_tgt} INTERFACE_INCLUDE_DIRECTORIES)
    if(_inc)
      set(_fixed "")
      foreach(_dir ${_inc})
        if(_dir MATCHES "eigen3$" AND NOT EXISTS "${_dir}")
          list(APPEND _fixed "${REAL_EIGEN3_INCLUDE_DIR}")
        else()
          list(APPEND _fixed "${_dir}")
        endif()
      endforeach()
      set_target_properties(${_tgt} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_fixed}")
    endif()
  endif()
endforeach()

add_executable(rel_producer producer.cpp shm_region.cpp)
target_include_directories(rel_producer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rel_producer PRIVATE traccc::core traccc::io
                                           detray::core vecmem::core rt)
