project(tt_stl)

add_library(tt_stl INTERFACE)
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
            tt_stl/caseless_comparison.hpp
            tt_stl/aligned_allocator.hpp
            tt_stl/any_range.hpp
            tt_stl/cleanup.hpp
            tt_stl/concepts.hpp
            tt_stl/indestructible.hpp
            tt_stl/overloaded.hpp
            tt_stl/reflection.hpp
            tt_stl/slotmap.hpp
            tt_stl/small_vector.hpp
            tt_stl/span.hpp
            tt_stl/strong_type.hpp
            tt_stl/type_name.hpp
            tt_stl/unique_any.hpp
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
        span
        small_vector
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

install(TARGETS tt_stl EXPORT Metalium FILE_SET api COMPONENT metalium-dev)

if(TT_METAL_BUILD_TESTS)
    add_subdirectory(tests)
endif()
