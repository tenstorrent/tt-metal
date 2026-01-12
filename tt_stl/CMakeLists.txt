# Note: Don't call project() or include(GNUInstallDirs) here - tt_stl is always built
# as part of the main TT-Metal project, and calling these again can cause CMAKE_INSTALL_LIBDIR
# to be set differently than the parent project, leading to install path mismatches.

add_library(tt_stl SHARED)
add_library(TT::STL ALIAS tt_stl)

# FIXME: TT-STL should support down to C++17 (so that TT-Metalium can), but we have headers (eg: <reflect>) in our
#        public API that is pushing C++20 on consumers
target_compile_features(tt_stl INTERFACE cxx_std_20)

target_sources(
    tt_stl
    INTERFACE
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS .
        FILES
            tt_stl/aligned_allocator.hpp
            tt_stl/cleanup.hpp
            tt_stl/concepts.hpp
            tt_stl/indestructible.hpp
            tt_stl/optional_reference.hpp
            tt_stl/overloaded.hpp
            tt_stl/reflection.hpp
            tt_stl/small_vector.hpp
            tt_stl/llvm/llvm_small_vector.hpp
            tt_stl/span.hpp
            tt_stl/strong_type.hpp
            tt_stl/type_name.hpp
            tt_stl/unique_any.hpp
            tt_stl/assert.hpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tt_stl/llvm/llvm_small_vector.cpp
)

# These dependencies are currently brought in by dependencies/CMakeLists.txt
# However, it would probably be more appropriate to bring them in here so that tt_stl can be standalone header only library
# Fix this later
target_link_libraries(
    tt_stl
    INTERFACE
        Reflect::Reflect
        enchantum::enchantum
        nlohmann_json::nlohmann_json
        fmt::fmt-header-only
        tt-logger::tt-logger
)

target_precompile_headers(
    metal_common_pch # TT::CommonPCH
    PUBLIC
        <reflect>
        tt_stl/reflection.hpp
)

target_link_libraries(metal_common_pch PRIVATE tt_stl)

# Output library directly to build/lib (not build/lib64) for CI tar artifacts.
# This matches the pattern used by tt_metal and other core libraries.
# On Fedora/AlmaLinux (where CMAKE_INSTALL_LIBDIR=lib64), the top-level symlink
# lib64 -> lib ensures CMake's install(TARGETS) can find the file via the lib64 path.
# This approach:
# - Uses a single canonical location (build/lib/) to avoid ODR violations
# - Works with install(TARGETS) via the lib64 -> lib symlink on Fedora/AlmaLinux
# - Matches tar artifact layout expectations (build/lib/)
set_target_properties(
    tt_stl
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/lib"
)

# Set RPATH properties BEFORE install() calls
# Using BUILD_WITH_INSTALL_RPATH=TRUE embeds INSTALL_RPATH at build time, so install(FILES)
# copies the library with the correct RPATH already set (no post-install fixup needed).
# This works for tt_stl because all its dependencies are INTERFACE (header-only),
# so it doesn't need BUILD_RPATH to find libraries at build time.
set_target_properties(
    tt_stl
    PROPERTIES
        BUILD_WITH_INSTALL_RPATH
            TRUE
        INSTALL_RPATH_USE_LINK_PATH
            FALSE
        INSTALL_RPATH
            "$ORIGIN"
)

# Install target to Metalium export set (required because tt_metal depends on tt_stl)
# Uses install(FILES) workaround for CMake's LIBRARY_OUTPUT_DIRECTORY path resolution bug
if(TT_INSTALL)
    # Export and headers only - library installed separately due to CMake bug
    install(
        TARGETS
            tt_stl
        EXPORT Metalium
        FILE_SET
        api
            DESTINATION COMPONENT
            metalium-dev
    )

    # Install library using generator expression (works around CMake path resolution bug)
    # Since BUILD_WITH_INSTALL_RPATH=TRUE, the library already has $ORIGIN RPATH embedded
    install(FILES $<TARGET_FILE:tt_stl> DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT metalium-runtime)
endif()

if(TT_METAL_BUILD_TESTS)
    add_subdirectory(tests)
endif()
