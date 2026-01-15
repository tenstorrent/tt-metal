# Smoke tests (fast, not necessarily thorough)
add_library(unit_tests_stl_smoke OBJECT)
add_library(TT::Metalium::Test::STL::Smoke ALIAS unit_tests_stl_smoke)
#TT_ENABLE_UNITY_BUILD(unit_tests_stl_smoke)

target_sources(
    unit_tests_stl_smoke
    PRIVATE
        test_cleanup.cpp
        test_indestructible.cpp
        test_optional_reference.cpp
        test_reflection.cpp
        test_span.cpp
        test_strong_type.cpp
        test_small_vector.cpp
)
target_link_libraries(
    unit_tests_stl_smoke
    PRIVATE
        gmock
        gtest
        gtest_main
        TT::STL
)

add_executable(unit_tests_stl)
target_link_libraries(unit_tests_stl PRIVATE TT::Metalium::Test::STL::Smoke)
