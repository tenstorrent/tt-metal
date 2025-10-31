# Telemetry unit tests

add_executable(telemetry_unit_tests)

target_sources(telemetry_unit_tests PRIVATE test_exception_handling.cpp)

target_include_directories(
    telemetry_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${UMD_HOME}
)

target_link_libraries(
    telemetry_unit_tests
    PRIVATE
        gmock
        gtest
        gtest_main
        tt_metal # Need this for tt::umd::Cluster destructor
        telemetry_libs
        fmt::fmt-header-only
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(telemetry_unit_tests)
