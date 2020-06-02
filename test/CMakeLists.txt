#add_executable(test_snapshot "src/test_snapshot.cpp")
#set_target_properties(test_snapshot PROPERTIES
#										LINKER_LANGUAGE CXX
#										CXX_STANDARD 14)
#target_include_directories(test_snapshot
#                            PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
#                            PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
#target_link_libraries(test_snapshot
#                        PRIVATE CONAN_PKG::catch2
#                        PRIVATE ${AER_LIBRARIES})
#add_test(test_snapshot test_snapshot)


#add_executable(test_snapshot_bdd "src/test_snapshot_bdd.cpp")
#set_target_properties(test_snapshot_bdd PROPERTIES
#										LINKER_LANGUAGE CXX
#										CXX_STANDARD 14)
#target_include_directories(test_snapshot_bdd
#                            PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
#                            PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
#target_link_libraries(test_snapshot_bdd
#                        PRIVATE CONAN_PKG::catch2
#                        PRIVATE ${AER_LIBRARIES})
#add_test(test_snapshot_bdd test_snapshot_bdd)

macro(add_test_executable target_name)
	add_executable(${target_name} ${ARGN})
	set_target_properties(${target_name} PROPERTIES
			LINKER_LANGUAGE CXX
			CXX_STANDARD 14)
	target_include_directories(${target_name}
			PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
			PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
	target_link_libraries(${target_name}
			PRIVATE CONAN_PKG::catch2
			PRIVATE ${AER_LIBRARIES})
	add_test(${target_name} ${target_name})
endmacro()

add_executable(test_utils "src/test_utils.cpp")
set_target_properties(test_utils PROPERTIES
								LINKER_LANGUAGE CXX
								CXX_STANDARD 14)
target_include_directories(test_utils
                            PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
                            PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
target_link_libraries(test_utils
                        PRIVATE CONAN_PKG::catch2
                        PRIVATE ${AER_LIBRARIES})
add_test(test_utils test_utils)

#add_executable(test_sundials_vector "src/ode/test_sundials_vector.cpp")
#set_target_properties(test_sundials_vector PROPERTIES
#		LINKER_LANGUAGE CXX
#		CXX_STANDARD 14)
#target_include_directories(test_sundials_vector
#		PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
#		PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
#target_link_libraries(test_sundials_vector
#		PRIVATE CONAN_PKG::catch2
#		PRIVATE ${AER_LIBRARIES})
#add_test(test_sundials_vector test_sundials_vector)

find_package(Pybind11 REQUIRED)
add_test_executable(test_sundials_vector "src/ode/test_sundials_vector.cpp")

add_test_executable(test_sundials_cvode "src/ode/test_sundials_cvode.cpp")

# Don't forget to add your test target here
add_custom_target(build_tests
#    test_snapshot
#    test_snapshot_bdd
    test_utils
    test_sundials_vector
    test_sundials_cvode)