# Source files for ttnn_op_experimental_fused_rotate.

set(TTNN_OP_EXPERIMENTAL_FUSED_ROTATE_API_HEADERS fused_rotate.hpp fused_rotate_gc.hpp fused_ln_bw.hpp fused_gate.hpp)

set(TTNN_OP_EXPERIMENTAL_FUSED_ROTATE_SRCS
    device/fused_rotate_device_operation.cpp
    device/fused_rotate_program_factory.cpp
    fused_rotate.cpp
    device/gc_device_operation.cpp
    device/gc_program_factory.cpp
    fused_rotate_gc.cpp
    device/lnbw_device_operation.cpp
    device/lnbw_program_factory.cpp
    fused_ln_bw.cpp
    device/gate_device_operation.cpp
    device/gate_program_factory.cpp
    fused_gate.cpp
)
