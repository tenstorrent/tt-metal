# Source files for ttnn_op_experimental_multi_scale_deformable_attn.

set(TTNN_OP_EXPERIMENTAL_MSDA_API_HEADERS multi_scale_deformable_attn.hpp)

set(TTNN_OP_EXPERIMENTAL_MSDA_SRCS
    multi_scale_deformable_attn.cpp
    device/multi_scale_deformable_attn_device_operation.cpp
    device/multi_scale_deformable_attn_program_factory.cpp
)
