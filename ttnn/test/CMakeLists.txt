# Spell out the RPATH so that shlibdeps can find deps across package components.
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# Smoke
add_executable(tt-nn-validation-smoke)

target_link_libraries(tt-nn-validation-smoke PRIVATE TTNN::Test::Smoke)
target_link_libraries(tt-nn-validation-smoke PRIVATE TTNN::Test::CCL::Ops)

install(TARGETS tt-nn-validation-smoke RUNTIME COMPONENT ttnn-validation)

# Basic
add_executable(tt-nn-validation-basic)

target_link_libraries(
    tt-nn-validation-basic
    PRIVATE
        TTNN::Test::Basic
        TTNN::Test::Basic::Tensor
        TTNN::Test::CCL
)

install(TARGETS tt-nn-validation-basic RUNTIME COMPONENT ttnn-validation)

# Mock cluster descriptors used by configure_mock_mode() in tt-nn-validation-basic.
install(
    DIRECTORY
        ${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/tests/cluster_descriptor_examples/
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/third_party/umd/tests/cluster_descriptor_examples
    COMPONENT ttnn-validation
    FILES_MATCHING
    PATTERN
    "*.yaml"
)
