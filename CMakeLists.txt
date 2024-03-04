cmake_minimum_required(VERSION 2.8.3)
project(unitree_legged_sdk)

#find H5Cpp library
#find_package(HDF5 COMPONENTS CXX HL)
find_package(realsense2 REQUIRED)


include_directories(include)
include_directories(include/lcm_types)
#include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${realsense2_INCLUDE_DIRS})
include_directories(./)

# Adding directory of the .h files
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

link_directories(lib/cpp/amd64)

add_compile_options(-std=c++17)

message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
  set(ARCH amd64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
  set(ARCH arm64)
endif()

link_directories(lib/cpp/${ARCH})
set(EXTRA_LIBS -pthread libunitree_legged_sdk.so lcm ${realsense2_LIBRARY})

set(CMAKE_CXX_FLAGS "-O3 -fPIC")

# add_executable(traj_recorder HDF5_recorder.h T265_reader.h traj_record.cpp)
# target_link_libraries(traj_recorder ${EXTRA_LIBS})

# add_executable(lcm_position HDF5_recorder.h T265_reader.h lcm_position.cpp)
# target_link_libraries(lcm_position ${EXTRA_LIBS})

add_executable(record_obs_from_lcm T265_reader.h record_obs_from_lcm.cpp)
target_link_libraries(record_obs_from_lcm ${EXTRA_LIBS})