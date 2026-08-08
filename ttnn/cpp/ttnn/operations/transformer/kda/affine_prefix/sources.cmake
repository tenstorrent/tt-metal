# Fixed-mode KDA affine-prefix registration.
set(TTNN_KDA_AFFINE_PREFIX_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/kda_affine_prefix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kda_affine_prefix_device_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kda_affine_prefix_program_factory.cpp
)
set(TTNN_KDA_AFFINE_PREFIX_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/kda_affine_prefix_nanobind.cpp)
