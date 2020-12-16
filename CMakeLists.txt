cmake_minimum_required( VERSION 3.10 )
cmake_policy( SET CMP0025 NEW )
project(spirit_toy_code)

set(CPU_BACKEND ON)
set(CUDA_BACKEND OFF)
set(VULKAN_BACKEND OFF)

set( SRC 
        ${PROJECT_SOURCE_DIR}/main.cpp
        ${PROJECT_SOURCE_DIR}/State.cpp
        ${PROJECT_SOURCE_DIR}/src/backend_cpu.cpp
        ${PROJECT_SOURCE_DIR}/src/backend_cuda.cpp
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
        target_compile_definitions( run PUBLIC -DBACKEND_CUDA)
else()
        target_compile_definitions( run PUBLIC -DBACKEND_VULKAN)
endif()