cmake_minimum_required( VERSION 3.10 )
cmake_policy( SET CMP0025 NEW )

# =============================================
set(CPU_BACKEND ON)
set(CUDA_BACKEND OFF)
set(VULKAN_BACKEND OFF)
set(USE_OPENMP ON)
# =============================================

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" )
set( CMAKE_DISABLE_IN_SOURCE_BUILD ON )

### Set a default build type in case none is passed
if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
    message( STATUS ">> Setting build type to 'Release' as none was specified." )
    set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
    # Set the possible values of build type for cmake-gui
    set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" )
elseif( CMAKE_BUILD_TYPE )
    message( STATUS ">> Did not need to set build type, using: ${CMAKE_BUILD_TYPE}" )
else()
    message( STATUS ">> Did not need to set build type. Configuration types: ${CMAKE_CONFIGURATION_TYPES}" )
endif()

if( APPLE )
    set( PLATFORM_NAME "Apple" )
elseif( UNIX )
    set( PLATFORM_NAME "UNIX" )
elseif( WIN32 )
    set( PLATFORM_NAME "Win32" )
endif()

include( ChooseCompiler )
project(spirit_toy_code)
include( CompilerFlags )

message( STATUS ">> Please check the CMAKE_CXX_COMPILER to make sure it's the right one" )
message( STATUS ">> CMAKE_C_COMPILER:   ${CMAKE_C_COMPILER}" )
message( STATUS ">> CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}" )

if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CXX_STANDARD 11)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set( SRC 
        ${PROJECT_SOURCE_DIR}/main.cpp
        ${PROJECT_SOURCE_DIR}/State.cpp
        ${PROJECT_SOURCE_DIR}/src/backend_cpu/backend_cpu.cpp
        ${PROJECT_SOURCE_DIR}/src/backend_cpu/backend_handle.cpp
        ${PROJECT_SOURCE_DIR}/src/backend_cuda/backend_cuda.cu
        ${PROJECT_SOURCE_DIR}/src/backend_cuda/backend_handle.cu
        ${PROJECT_SOURCE_DIR}/src/backend_vulkan/backend_vulkan.cpp
        ${PROJECT_SOURCE_DIR}/src/backend_vulkan/backend_handle.cpp
)

set( INCLUDE 
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/thirdparty
)

add_executable( run ${SRC})
add_custom_command(TARGET run 
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:run> ${PROJECT_SOURCE_DIR})

target_include_directories( run PUBLIC ${INCLUDE} )

set(COMPILE_OPTIONS ${COMPILE_OPTIONS} "-O3")
set(COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS})

if (${CPU_BACKEND})
        message("Building CPU backend")
        set(COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS} -DBACKEND_CPU)
        if(USE_OPENMP)
                find_package(OpenMP)
                if(OpenMP_CXX_FOUND)
                        message("Found OpenMP")
                        target_link_libraries(run PUBLIC OpenMP::OpenMP_CXX)
                        set(COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS} -DUSE_OPENMP)
                endif()
        endif()
elseif (${CUDA_BACKEND})
        message("Building CUDA backend")
        include(CheckLanguage)
        check_language(CUDA)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
                set(CMAKE_CUDA_STANDARD 11)
                set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()

        set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr --expt-extended-lambda" )
        set_target_properties( run PROPERTIES CUDA_SEPERABLE_COMPILATION ON )

        set(COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS} -DBACKEND_CUDA)

        set_source_files_properties(
                PROPERTIES LANGUAGE CUDA
        )
else()
        message("Building VULKAN backend")
        set(COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS} -DBACKEND_VULKAN)
endif()

target_compile_definitions( run PUBLIC ${COMPILE_DEFINITIONS})
target_compile_options( run PUBLIC ${COMPILE_OPTIONS})

enable_testing()
add_test( run  run )