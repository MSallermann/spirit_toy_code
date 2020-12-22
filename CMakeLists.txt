cmake_minimum_required( VERSION 3.10 )
cmake_policy( SET CMP0025 NEW )

# =============================================
set(CPU_BACKEND ON)
set(CUDA_BACKEND ON)
set(VULKAN_BACKEND OFF)
set(USE_OPENMP ON) # Use openMP to accelerate host code and CPU backend
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
project( spirit_toy_code )
set(META_PROJECT_NAME "spirit")
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
        ${PROJECT_SOURCE_DIR}/src/backend_cpu/backend_cpu.cpp
        ${PROJECT_SOURCE_DIR}/src/backend_cpu/Device_State.cpp
        ${PROJECT_SOURCE_DIR}/src/backend_cuda/backend_cuda.cu
        ${PROJECT_SOURCE_DIR}/src/backend_cuda/Device_State.cu
        # ${PROJECT_SOURCE_DIR}/src/backend_vulkan/backend_vulkan.cpp
        # ${PROJECT_SOURCE_DIR}/src/backend_vulkan/Device_State.cpp
)

set( INCLUDE 
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/thirdparty
        ${PROJECT_SOURCE_DIR}/include
)

set(COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS})

if (${CPU_BACKEND})
        message("Building CPU backend")
        list(APPEND EXECUTABLE_NAMES  ${META_PROJECT_NAME}_CPU)
        add_executable(${META_PROJECT_NAME}_CPU  ${SRC})
        target_compile_definitions( ${META_PROJECT_NAME}_CPU PUBLIC -DBACKEND_CPU )
endif()
if (${CUDA_BACKEND})
        message("Building CUDA backend")
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
                enable_language(CUDA)
                if(NOT DEFINED CMAKE_CUDA_STANDARD)
                        set(CMAKE_CUDA_STANDARD 11)
                        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
                endif()

                list (APPEND EXECUTABLE_NAMES  ${META_PROJECT_NAME}_CUDA )
                add_executable(${META_PROJECT_NAME}_CUDA  ${SRC})
                target_compile_definitions( ${META_PROJECT_NAME}_CUDA PUBLIC -DBACKEND_CUDA )

                set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr --expt-extended-lambda" )
                set_target_properties( ${META_PROJECT_NAME}_CUDA PROPERTIES CUDA_SEPERABLE_COMPILATION ON )
        else()
                message(STATUS "No CUDA support")
                set(CUDA_BACKEND OFF)
        endif()
endif()
# if()
#         message("Building VULKAN backend")
#         set(COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS} -DBACKEND_VULKAN)
# endif()

# Additional definitions for all execs
foreach(executable IN LISTS EXECUTABLE_NAMES)
        message(${executable})
        if(USE_OPENMP)
                find_package(OpenMP)
                if(OpenMP_CXX_FOUND)
                        message("Found OpenMP")
                        target_link_libraries(${executable} PUBLIC OpenMP::OpenMP_CXX)
                        target_compile_definitions(${executable} PUBLIC -DUSE_OPENMP)
                endif()
        endif()

        target_include_directories( ${executable} PUBLIC ${INCLUDE} )
        add_custom_command(TARGET ${executable} 
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${executable}> ${PROJECT_SOURCE_DIR})
        add_test( ${executable} ${executable} )
endforeach()

enable_testing()