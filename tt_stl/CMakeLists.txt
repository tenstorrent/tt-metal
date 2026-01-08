project(tt_stl)

include(GNUInstallDirs)

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

# Set BUILD_RPATH to ensure tt_stl can find its dependencies during build
tt_set_library_rpath(tt_stl)

# Output library to build/${CMAKE_INSTALL_LIBDIR} (lib on Ubuntu, lib64 on Fedora/AlmaLinux).
# On Fedora/AlmaLinux, the top-level symlink (lib64 -> lib) ensures the file actually ends up
# in build/lib/ for tar artifact compatibility, while CMake's internal tracking uses the
# lib64 path that it expects. This approach:
# - Avoids install(TARGETS) path resolution issues (CMake's tracked path matches the setting)
# - Ensures a single canonical .so file (no copies, no ODR violations)
# - Works with both lib and lib64 systems via the symlink
set_target_properties(
    tt_stl
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
)

# Standard install with LIBRARY - this works because LIBRARY_OUTPUT_DIRECTORY matches
# CMake's expected path pattern (build/${CMAKE_INSTALL_LIBDIR})
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

if(TT_METAL_BUILD_TESTS)
    add_subdirectory(tests)
endif()
