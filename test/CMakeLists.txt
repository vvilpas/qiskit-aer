set(Catch2_PATH "third-party/")
#find_package(Catch2 REQUIRED)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(unitc_tests "src/test_linalg.cpp")

target_link_libraries(unitc_tests PRIVATE ${AER_LIBRARIES})
                            #PRIVATE Catch2::Catch2 )

if(APPLE)
    # It's quite common to have multiple copies of the same Python version
    # installed on one's system. E.g.: one copy from the OS and another copy
    # that's statically linked into an application like Blender or Maya.
    # If we link our plugin library against the OS Python here and import it
    # into Blender or Maya later on, this will cause segfaults when multiple
    # conflicting Python instances are active at the same time (even when they
    # are of the same version).

    # Windows is not affected by this issue since it handles DLL imports
    # differently. The solution for Linux and Mac OS is simple: we just don't
    # link against the Python library. The resulting shared library will have
    # missing symbols, but that's perfectly fine -- they will be resolved at
    # import time.
    # Set some general flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(AER_LINKER_FLAGS "${AER_LINKER_FLAGS} -undefined dynamic_lookup")
    else()
        # -flat_namespace linker flag is needed otherwise dynamic symbol resolution doesn't work as expected with GCC.
        # Symbols with the same name exist in different .so, so the loader just takes the first one it finds,
        # which is usually the one from the first .so loaded.
        # See: Two-Leve namespace symbol resolution
        set(AER_LINKER_FLAGS "${AER_LINKER_FLAGS} -undefined dynamic_lookup -flat_namespace")
    endif()
    set_target_properties(unitc_tests PROPERTIES MACOSX_RPATH ON)
endif()

set_target_properties(unitc_tests PROPERTIES LINKER_LANGUAGE CXX
									   CXX_STANDARD    14 
                                       COMPILE_FLAGS ${AER_COMPILER_FLAGS}
                                       LINK_FLAGS ${AER_LINKER_FLAGS}
                                       RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}
                                       RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(unitc_tests PRIVATE ${AER_COMPILER_DEFINITIONS})
target_include_directories(unitc_tests PRIVATE ${Catch2_PATH}
                                 PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
                                 PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})


# Don't forget to add your test target here
#add_custom_target(build_tests
#    test_snapshot
#    test_snapshot_bdd
#    test_utils)
