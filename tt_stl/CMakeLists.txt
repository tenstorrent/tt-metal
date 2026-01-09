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

# Install EXPORT and FILE_SET via install(TARGETS) - but NOT the library itself
# because install(TARGETS) LIBRARY has path resolution issues when LIBRARY_OUTPUT_DIRECTORY is set
install(
    TARGETS
        tt_stl
    EXPORT Metalium
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT metalium-runtime
    FILE_SET
    api
        DESTINATION COMPONENT
        metalium-dev
)

# Set BUILD_RPATH to ensure tt_stl can find its dependencies during build
tt_set_library_rpath(tt_stl)

# Install the library file manually using install(CODE) to work around CMake's
# install(TARGETS) LIBRARY path resolution bug when LIBRARY_OUTPUT_DIRECTORY is set.
# The diagnostic proves the file exists at build/lib/libtt_stl.so, but install(TARGETS) fails.
#install(
#    CODE
#        "
#        set(_src \"${CMAKE_BINARY_DIR}/lib/libtt_stl.so\")
#        set(_dst \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
#        message(STATUS \"Installing tt_stl library: \${_src} -> \${_dst}/libtt_stl.so\")
#        file(INSTALL DESTINATION \"\${_dst}\" TYPE SHARED_LIBRARY FILES \"\${_src}\")
#        "
#    COMPONENT
#        metalium-runtime
#)

if(TT_METAL_BUILD_TESTS)
    add_subdirectory(tests)
endif()
