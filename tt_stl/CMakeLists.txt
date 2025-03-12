add_library(tt_stl INTERFACE)
add_library(TT:STL ALIAS tt_stl)

target_sources(
    tt_stl
    INTERFACE
        FILE_SET tt_stl
        TYPE HEADERS
        BASE_DIRS tt_stl
        FILES
            tt_stl/aligned_allocator.hpp
            tt_stl/any_range.hpp
            tt_stl/concepts.hpp
            tt_stl/indestructible.hpp
            tt_stl/overloaded.hpp
            tt_stl/reflection.hpp
            tt_stl/slotmap.hpp
            tt_stl/span.hpp
            tt_stl/strong_type.hpp
            tt_stl/type_name.hpp
            tt_stl/unique_any.hpp
)

target_include_directories(tt_stl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# These dependencies are currently brought in by dependencies/CMakeLists.txt
# However, it would probably be more appropriate to bring them in here so that tt_stl can be standalone header only library
# Fix this later
target_link_libraries(
    tt_stl
    INTERFACE
        Reflect::Reflect
        magic_enum::magic_enum
        nlohmann_json::nlohmann_json
)
