cmake_minimum_required( VERSION 3.10 )
cmake_policy( SET CMP0025 NEW )
project(spirit_toy_code)

set(CPU_BACKEND OFF)
set(CUDA_BACKEND ON)
set(VULKAN_BACKEND OFF)

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
target_compile_options( run PUBLIC "-O3")

if (${CPU_BACKEND})
        target_compile_definitions( run PUBLIC -DBACKEND_CPU)

elseif(${CUDA_BACKEND})
        include(CheckLanguage)
        check_language(CUDA)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
                set(CMAKE_CUDA_STANDARD 11)
                set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()

        set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr --expt-extended-lambda" )
        set_target_properties( run PROPERTIES CUDA_SEPERABLE_COMPILATION ON )

        target_compile_definitions( run PUBLIC -DBACKEND_CUDA)

        set_source_files_properties(
                PROPERTIES LANGUAGE CUDA
        )
else()
        target_compile_definitions( run PUBLIC -DBACKEND_VULKAN)
endif()