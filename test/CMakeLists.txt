set(Catch2_PATH "third-party/")
find_package(Catch2 REQUIRED)

add_executable(test_linalg "src/test_linalg.cpp")
set_target_properties(test_linalg PROPERTIES
										LINKER_LANGUAGE CXX
										CXX_STANDARD 14)
target_include_directories(test_linalg
                            PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
                            PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
target_link_libraries(test_linalg
                        PRIVATE Catch2::Catch
                        PRIVATE ${AER_LIBRARIES})
add_test(test_linalg test_linalg)

# Don't forget to add your test target here
#add_custom_target(build_tests
#    test_snapshot
#    test_snapshot_bdd
#    test_utils)
