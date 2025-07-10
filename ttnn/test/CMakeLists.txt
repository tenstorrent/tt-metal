# Spell out the RPATH so that shlibdeps can find deps across package components.
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# Smoke
add_executable(tt-nn-validation-smoke)

target_link_libraries(tt-nn-validation-smoke PRIVATE TTNN::Test::Smoke)

install(TARGETS tt-nn-validation-smoke RUNTIME COMPONENT ttnn-validation)

# Basic
add_executable(tt-nn-validation-basic)

target_link_libraries(
    tt-nn-validation-basic
    PRIVATE
        TTNN::Test::Basic
        TTNN::Test::Basic::Tensor
)

install(TARGETS tt-nn-validation-basic RUNTIME COMPONENT ttnn-validation)
