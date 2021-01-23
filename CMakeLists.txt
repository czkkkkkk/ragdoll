cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(ragdoll LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Python
# find_package(PythonLibs REQUIRED)

# if(PYTHONLIBS_FOUND)
#   message(STATUS "Found python libs:")
#   message(STATUS "  (Headers)       ${PYTHON_INCLUDE_DIRS}")
#   message(STATUS "  (Library)       ${PYTHON_LIBRARIES}")
#   list(APPEND RAGDOLL_EXTERNAL_INCLUDES ${PYTHON_INCLUDE_DIRS})
#   list(APPEND RAGDOLL_EXTERNAL_LIB ${PYTHON_LIBRARIES})
# else()
#   message(FATAL_ERROR "Python development not found")
# endif()


# Torch
find_package(Torch REQUIRED)
# list(APPEND RAGDOLL_EXTERNAL_INCLUDES ${TORCH_INCLUDE_DIRS})
list(APPEND RAGDOLL_EXTERNAL_LIB ${TORCH_LIBRARIES})
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# message(STATUS "CXX FLAGS IS " ${TORCH_CXX_FLAGS})

include_directories(/usr/local/cuda/include)

# GLOG
find_path(glog_INCLUDE_DIR NAMES glog/logging.h)
find_library(glog_LIBRARY NAMES glog)
if(glog_INCLUDE_DIR AND glog_LIBRARY)
  message(STATUS "Found glog:")
  message(STATUS "  (Headers)       ${glog_INCLUDE_DIR}")
  message(STATUS "  (Library)       ${glog_LIBRARY}")
  list(APPEND RAGDOLL_EXTERNAL_INCLUDES ${glog_INCLUDE_DIR})
  list(APPEND RAGDOLL_EXTERNAL_LIB ${glog_LIBRARY})
else()
  message(FATAL_ERROR "glog not found")
endif()

# MPI
if(DEFINED ENV{MPI_HOME})
  find_library(mpi_LIBRARY NAMES mpi PATHS $ENV{MPI_HOME}/lib NO_DEFAULT_PATH)
  set(mpi_INCLUDE_DIR $ENV{MPI_HOME}/include)
  message(STATUS "Found mpi:")
  message(STATUS "  (Headers)       ${mpi_INCLUDE_DIR}")
  message(STATUS "  (Library)       ${mpi_LIBRARY}")
  list(APPEND RAGDOLL_EXTERNAL_INCLUDES ${mpi_INCLUDE_DIR})
  list(APPEND RAGDOLL_EXTERNAL_LIB ${mpi_LIBRARY})
else()
  message(FATAL_ERROR "MPI_HOME is not defined in environment")
endif()

if(DEFINED ENV{GCCL_HOME})
  find_library(gccl_LIBRARY NAMES gccl PATHS $ENV{GCCL_HOME}/lib NO_DEFAULT_PATH)
  set(gccl_INCLUDE_DIR $ENV{GCCL_HOME}/include)
  message(STATUS "Found GCCL HOME:")
  message(STATUS "  (Headers)       ${gccl_INCLUDE_DIR}")
  message(STATUS "  (Library)       ${gccl_LIBRARY}")
  list(APPEND RAGDOLL_EXTERNAL_INCLUDES ${gccl_INCLUDE_DIR})
  list(APPEND RAGDOLL_EXTERNAL_LIB ${gccl_LIBRARY})
else()
  message(FATAL_ERROR "GCCL_HOME is not defined in environment")
endif()


add_subdirectory(third_party/pybind11)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ragdoll ${RAGDOLL_EXTERNAL_INCLUDES})

file(GLOB ragdoll_src
    ragdoll/common/operations.cc)

# Torch ops
pybind11_add_module(ragdoll_torch_ops
    ${ragdoll_src}
    ragdoll/torch/ops.cc
)
target_link_libraries(ragdoll_torch_ops PRIVATE ${RAGDOLL_EXTERNAL_LIB})

# Ragdoll core
add_library(ragdoll_core SHARED ${ragdoll_src})
set_target_properties(ragdoll_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ragdoll_core PUBLIC ${RAGDOLL_EXTERNAL_LIB})

add_executable(simple examples/simple.cc)
target_link_libraries(simple PRIVATE ragdoll_core)
set_target_properties(simple PROPERTIES CXX_STANDARD ${CMAKE_CXX_STANDARD})