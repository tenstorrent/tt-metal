# Fixed-mode KDA affine-composition registration.
set(TTNN_KDA_AFFINE_COMPOSITION_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/kda_affine_composition.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kda_affine_composition_device_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kda_affine_composition_program_factory.cpp
)
set(TTNN_KDA_AFFINE_COMPOSITION_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/kda_affine_composition_nanobind.cpp)
