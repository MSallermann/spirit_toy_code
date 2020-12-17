cmake_minimum_required( VERSION 3.10 )
cmake_policy( SET CMP0025 NEW )
project(spirit_toy_code)

set(CPU_BACKEND ON)
set(CUDA_BACKEND ON)
set(VULKAN_BACKEND OFF)
set(USE_OPENMP ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CXX_STANDARD 11)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set( SRC 
        ${PROJECT_SOURCE_DIR}/main.cpp
        ${PROJECT_SOURCE_DIR}/State.cpp
        ${PROJECT_SOURCE_DIR}/src/backend_cpu.cpp
        ${PROJECT_SOURCE_DIR}/src/backend_cuda.cu
        ${PROJECT_SOURCE_DIR}/src/backend_vulkan.cpp
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