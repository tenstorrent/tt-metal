# Telemetry unit tests

add_executable(telemetry_unit_tests)

target_sources(
    telemetry_unit_tests
    PRIVATE
        test_exception_handling.cpp
        test_custom_labels.cpp
        test_physical_link_info.cpp
        test_ethernet_path_parsing.cpp
        test_topology_helper_api.cpp
)

target_include_directories(
    telemetry_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_BINARY_DIR}/tools/scaleout # For generated protobuf headers
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
        protobuf::libprotobuf # For protobuf runtime
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(telemetry_unit_tests)
