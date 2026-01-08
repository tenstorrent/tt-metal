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

# Output library directly to build/lib (not build/lib64) for CI tar artifacts
# This prevents ODR violations in multiprocess tests where different processes
# might load the library from different locations (original vs copied)
# CI tar artifacts expect libraries in build/lib regardless of CMAKE_INSTALL_LIBDIR
set_target_properties(
    tt_stl
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/lib"
)

# Install using FILES instead of TARGETS to work around CI issue where
# install(TARGETS) can't find the library when LIBRARY_OUTPUT_DIRECTORY is set
# The generator expression $<TARGET_FILE:tt_stl> resolves to the actual file location
# at build time, which should be ${CMAKE_BINARY_DIR}/lib/libtt_stl.so
# This ensures the library is installed even if install(TARGETS) would fail
install(FILES $<TARGET_FILE:tt_stl> DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT metalium-runtime)

# Still install the EXPORT and FILE_SET using TARGETS for proper CMake package support
# The FILE_SET (headers) doesn't have the same issue as the library file
install(
    TARGETS
        tt_stl
    EXPORT Metalium
    FILE_SET
    api
        DESTINATION COMPONENT
        metalium-dev
    # Don't install LIBRARY here since we handle it with FILES above
    # This avoids the "file not found" error in CI
)

if(TT_METAL_BUILD_TESTS)
    add_subdirectory(tests)
endif()
