set(TTNN_OP_EXPERIMENTAL_KDA_REDUCE_AFFINE_TRANSFORMS_API_HEADERS reduce_affine_transforms.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_REDUCE_AFFINE_TRANSFORMS_SRCS
    reduce_affine_transforms.cpp
    device/reduce_affine_transforms_device_operation.cpp
    device/reduce_affine_transforms_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_REDUCE_AFFINE_TRANSFORMS_NANOBIND_SRCS
    reduce_affine_transforms_nanobind.cpp
    ../kda_nanobind.cpp
)
