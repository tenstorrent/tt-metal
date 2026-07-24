# Split-TU fixture used by test_pimpl.cpp to test against an incomplete type.
add_subdirectory(test_indirect_util)

# Smoke tests (fast, not necessarily thorough)
add_library(unit_tests_stl_smoke OBJECT)
add_library(TT::Metalium::Test::STL::Smoke ALIAS unit_tests_stl_smoke)
#TT_ENABLE_UNITY_BUILD(unit_tests_stl_smoke)

target_sources(
    unit_tests_stl_smoke
    PRIVATE
        test_cleanup.cpp
        test_fmt.cpp
        test_hash.cpp
        test_indestructible.cpp
        test_indirect.cpp
        test_optional_reference.cpp
        test_pimpl.cpp
        test_reflection.cpp
        test_span.cpp
        test_strong_type.cpp
        test_small_vector.cpp
        test_tt_pause.cpp
        test_type_visitor.cpp
)
target_link_libraries(
    unit_tests_stl_smoke
    PRIVATE
        gmock
        gtest
        gtest_main
        TT::STL
        test_indirect_util
)

add_executable(unit_tests_stl)
target_link_libraries(unit_tests_stl PRIVATE TT::Metalium::Test::STL::Smoke)
