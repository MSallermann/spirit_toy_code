macro( add_framework_test testName testSrc libSpiritCore)
    # Executable
    add_executable( ${testName} tests/main.cpp ${testSrc} )
    # Link to core library
    target_link_libraries( ${testName} PUBLIC ${libSpiritCore} )

    if(CMAKE_CUDA_COMPILER)
        set_source_files_properties( 
               ${testSrc}
               PROPERTIES LANGUAGE CUDA
        )  
    endif()

    # Include Directories
    target_include_directories( ${testName} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/test
        ${CMAKE_CURRENT_LIST_DIR}/thirdparty
        ${CMAKE_CURRENT_LIST_DIR}/include
     )

    # Apply public includes from the object library
    target_include_directories( ${testName} PUBLIC
        $<TARGET_PROPERTY:${libSpiritCore},INTERFACE_INCLUDE_DIRECTORIES>)

    # Coverage flags and linking if needed
    # Add the test
    add_test( NAME        ${testName}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND           ${testName} --use-colour=yes )
endmacro( add_framework_test testName testSrc libSpiritCore)
