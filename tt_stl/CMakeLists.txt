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

# Output library directly to build/lib (not build/lib64) for CI tar artifacts.
# This avoids creating a second physical copy of the .so (which can cause ODR
# violations if different processes dlopen different copies of the same library).
# Fedora installs to lib64 and Ubuntu installs to lib; the top-level symlink logic
# (lib64 -> lib when needed) keeps build-time lookups consistent.
set_target_properties(
    tt_stl
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/lib"
)

# Install library, EXPORT, and FILE_SET using TARGETS
# When LIBRARY_OUTPUT_DIRECTORY is set to a custom location, install(TARGETS) LIBRARY
# may fail to resolve the file path correctly on some systems. We use install(FILES)
# with a generator expression for the library file, while keeping install(TARGETS)
# for the EXPORT and FILE_SET to maintain proper CMake package configuration.
install(
    TARGETS
        tt_stl
    EXPORT Metalium
    # Note: We install the library file separately using install(FILES) below
    # because install(TARGETS) LIBRARY may fail to find the file when
    # LIBRARY_OUTPUT_DIRECTORY is set to a custom location
    FILE_SET
    api
        DESTINATION COMPONENT
        metalium-dev
)

# Install the library file using a generator expression that resolves at install time
# This avoids path resolution issues that can occur with install(TARGETS) LIBRARY
# when LIBRARY_OUTPUT_DIRECTORY is overridden
install(FILES $<TARGET_FILE:tt_stl> DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT metalium-runtime)

# Ensure the generated install target builds tt_stl first.
# This avoids intermittent "file INSTALL cannot find ..." failures in generators
# where the install target does not strictly depend on all library targets.
if(TT_INSTALL AND TARGET install)
    add_dependencies(install tt_stl)
endif()

if(TT_METAL_BUILD_TESTS)
    add_subdirectory(tests)
endif()
