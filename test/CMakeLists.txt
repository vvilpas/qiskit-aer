#target_include_directories(unitc_tests PRIVATE ${Catch2_PATH}
#                                 PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
#                                 PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})

function(aer_add_test target_name)
    cmake_parse_arguments(ARG "${options}" "" "" ${ARGN})
    add_executable(${target_name} ${ARG_UNPARSED_ARGUMENTS})
    target_compile_options(${target_name} PRIVATE "-DDEBUG=1"
                                          PRIVATE ${AER_COMPILER_DEFINITIONS})
    #if(APPLE)
    #    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    #        set(AER_LINKER_FLAGS "${AER_LINKER_FLAGS} -undefined dynamic_lookup")
    #    else()
    #        set(AER_LINKER_FLAGS "${AER_LINKER_FLAGS} -undefined dynamic_lookup -flat_namespace")
    #    endif()
    #    set_target_properties(${target_name} PROPERTIES MACOSX_RPATH ON)
    #endif()
    set_target_properties(${target_name} PROPERTIES
										 LINKER_LANGUAGE CXX
										 CXX_STANDARD 14
                                         COMPILE_FLAGS ${AER_COMPILER_FLAGS}
                                         LINK_FLAGS ${AER_LINKER_FLAGS}
                                         RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}
                                         RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR})

    target_include_directories(${target_name}
                               PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
                               PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
    target_link_libraries(${target_name}
                          PRIVATE CONAN_PKG::catch2
                          PRIVATE ${AER_LIBRARIES})
    add_test(${target_name} ${target_name})
endfunction()

#aer_add_test(test_snapshot "src/test_snapshot.cpp")
#aer_add_test(test_snapshot_bdd "src/test_snapshot_bdd.cpp")
#aer_add_test(test_utils "src/test_utils.cpp")
aer_add_test(test_linalg "src/test_linalg.cpp")
if(WIN32 AND NOT BLAS_LIB_PATH)
    add_custom_command(TARGET test_linalg POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${OPENBLAS_DLLs}
					   "${PROJECT_SOURCE_DIR}/test")
endif()
