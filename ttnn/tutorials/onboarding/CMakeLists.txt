# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Onboarding Workshop - Auto-discovers exercises and builds modules
# Adding a new exercise just requires following the folder naming convention

if(NOT WITH_PYTHON_BINDINGS)
    return()
endif()

# Helper function to build an onboarding exercise module
function(add_onboarding_module TARGET_NAME)
    cmake_parse_arguments(ARG "" "" "SOURCES;NANOBIND_SOURCE" ${ARGN})

    add_library(${TARGET_NAME} SHARED)
    target_sources(
        ${TARGET_NAME}
        PRIVATE
            ${ARG_SOURCES}
            ${ARG_NANOBIND_SOURCE}
    )

    target_include_directories(
        ${TARGET_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/ttnn/api
            ${PROJECT_SOURCE_DIR}/ttnn/cpp
    )

    target_link_libraries(
        ${TARGET_NAME}
        PRIVATE
            TT::Nanobind
            TT::Metalium
            TTNN::CPP
    )

    nanobind_opt_size(${TARGET_NAME})
    nanobind_lto(${TARGET_NAME})
    nanobind_compile_options(${TARGET_NAME})
    nanobind_link_options(${TARGET_NAME})

    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES
            PREFIX
                ""
            BUILD_RPATH
                "${PROJECT_BINARY_DIR}/tt_metal;${PROJECT_BINARY_DIR}/ttnn"
            INSTALL_RPATH
                "$ORIGIN/../../../.."
    )

    add_custom_command(
        TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/.build
        COMMAND
            ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}/.build/${TARGET_NAME}.so
        COMMENT "Copying ${TARGET_NAME}.so to .build directory"
    )
endfunction()

# ============================================================================
# Auto-discover and build all exercises
# ============================================================================

# Find all exercise directories (e01_*, e02_*, etc.)
file(GLOB exercise_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "e[0-9][0-9]_*")

# Track all module targets for the main 'onboarding' target
set(all_modules)

foreach(exercise_dir ${exercise_dirs})
    # Extract exercise number (e.g., "03" from "e03_python_bindings")
    string(REGEX MATCH "^e([0-9]+)_" _ ${exercise_dir})
    set(exercise_num ${CMAKE_MATCH_1})

    # Build solution module if solution_cpp exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${exercise_dir}/solution_cpp")
        file(GLOB_RECURSE solution_all_sources "${exercise_dir}/solution_cpp/*.cpp")

        # Separate nanobind source from other sources
        set(solution_sources)
        set(solution_nanobind)
        foreach(src ${solution_all_sources})
            if(src MATCHES "_nanobind\\.cpp$")
                list(APPEND solution_nanobind ${src})
            else()
                list(APPEND solution_sources ${src})
            endif()
        endforeach()

        if(solution_nanobind)
            add_onboarding_module(_e${exercise_num}_solution
                SOURCES ${solution_sources}
                NANOBIND_SOURCE ${solution_nanobind}
            )
            list(APPEND all_modules _e${exercise_num}_solution)
        endif()
    endif()

    # Build exercise module if exercise_cpp exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${exercise_dir}/exercise_cpp")
        file(GLOB_RECURSE exercise_all_sources "${exercise_dir}/exercise_cpp/*.cpp")

        # Separate nanobind source from other sources
        set(exercise_sources)
        set(exercise_nanobind)
        foreach(src ${exercise_all_sources})
            if(src MATCHES "_nanobind\\.cpp$")
                list(APPEND exercise_nanobind ${src})
            else()
                list(APPEND exercise_sources ${src})
            endif()
        endforeach()

        if(exercise_nanobind)
            add_onboarding_module(_e${exercise_num}_exercise
                SOURCES ${exercise_sources}
                NANOBIND_SOURCE ${exercise_nanobind}
            )
            list(APPEND all_modules _e${exercise_num}_exercise)
        endif()
    endif()
endforeach()

# ============================================================================
# Kernel installation for JIT compilation
# ============================================================================

file(GLOB_RECURSE onboarding_kernels "e[0-9][0-9]_*/*/device/kernels/*")

if(onboarding_kernels)
    add_library(ttnn_onboarding_kernels OBJECT)
    target_sources(
        ttnn_onboarding_kernels
        PUBLIC
            FILE_SET kernels
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
            FILES ${onboarding_kernels}
    )

    install(
        TARGETS
            ttnn_onboarding_kernels
        FILE_SET
        kernels
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/tutorials/onboarding
            COMPONENT ttnn-runtime
    )
endif()

# ============================================================================
# Main target that builds all discovered exercises
# ============================================================================

add_custom_target(onboarding DEPENDS ${all_modules})

# Clean target for onboarding only
add_custom_target(
    onboarding-clean
    COMMAND
        ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/.build
    COMMAND
        ${CMAKE_COMMAND} -E rm -f ${CMAKE_CURRENT_BINARY_DIR}/_e*_solution.so
    COMMAND
        ${CMAKE_COMMAND} -E rm -f ${CMAKE_CURRENT_BINARY_DIR}/_e*_exercise.so
    COMMAND
        ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/_e*
    COMMENT "Cleaning onboarding build artifacts"
)

message(STATUS "Onboarding: discovered modules: ${all_modules}")
