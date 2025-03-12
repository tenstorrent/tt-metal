add_library(tt_stl INTERFACE)
add_library(TT:STL ALIAS tt_stl)

target_sources(
    tt_stl
    PUBLIC
        FILE_SET tt_stl
        TYPE HEADERS
        BASE_DIRS
            tt_stl
            aligned_allocator.hpp
            any_range.hpp
            concepts.hpp
            indestructible.hpp
            overloaded.hpp
            reflection.hpp
            slotmap.hpp
            span.hpp
            strong_type.hpp
            type_name.hpp
            unique_any.hpp
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
