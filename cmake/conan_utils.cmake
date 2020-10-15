include(conan)

macro(_rename_conan_lib package)
    add_library(AER_DEPENDENCY_PKG::${package} INTERFACE IMPORTED)
    target_link_libraries(AER_DEPENDENCY_PKG::${package} PUBLIC INTERFACE CONAN_PKG::${package})
endmacro()

macro(setup_conan)

    # Right now every dependency shall be static
    set(CONAN_OPTIONS ${CONAN_OPTIONS} "*:shared=False")

    set(REQUIREMENTS nlohmann_json/3.1.1 spdlog/1.5.0)
    list(APPEND AER_CONAN_LIBS nlohmann_json spdlog)
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(REQUIREMENTS ${REQUIREMENTS} llvm-openmp/8.0.1)
        list(APPEND AER_CONAN_LIBS llvm-openmp)
        if(SKBUILD)
            set(CONAN_OPTIONS ${CONAN_OPTIONS} "llvm-openmp:shared=True")
        endif()
    endif()

    if(SKBUILD)
        set(REQUIREMENTS ${REQUIREMENTS} muparserx/4.0.8)
        list(APPEND AER_CONAN_LIBS muparserx)
        if(NOT MSVC)
            set(CONAN_OPTIONS ${CONAN_OPTIONS} "muparserx:fPIC=True")
        endif()
    endif()

    if(AER_THRUST_BACKEND AND NOT AER_THRUST_BACKEND STREQUAL "CUDA")
        set(REQUIREMENTS ${REQUIREMENTS} thrust/1.9.5)
        list(APPEND AER_CONAN_LIBS thrust)
        string(TOLOWER ${AER_THRUST_BACKEND} THRUST_BACKEND)
        set(CONAN_OPTIONS ${CONAN_OPTIONS} "thrust:device_system=${THRUST_BACKEND}")
        if(THRUST_BACKEND MATCHES "tbb")
            list(APPEND AER_CONAN_LIBS tbb)
        endif()
    endif()

    if(BUILD_TESTS)
        set(REQUIREMENTS ${REQUIREMENTS} catch2/2.12.1)
        list(APPEND AER_CONAN_LIBS catch2)
    endif()

    # Add Appleclang-12 until officially supported by Conan
    conan_config_install(ITEM ${PROJECT_SOURCE_DIR}/conan_settings)

    conan_cmake_run(REQUIRES ${REQUIREMENTS}
                    OPTIONS ${CONAN_OPTIONS}
                    ENV CONAN_CMAKE_PROGRAM=${CMAKE_COMMAND}
                    BASIC_SETUP
                    CMAKE_TARGETS
                    KEEP_RPATHS
                    BUILD missing)

    # Reassign targets from CONAN_PKG to AER_DEPENDENCY_PKG
    foreach(CONAN_LIB ${AER_CONAN_LIBS})
        _rename_conan_lib(${CONAN_LIB})
    endforeach()
endmacro()
