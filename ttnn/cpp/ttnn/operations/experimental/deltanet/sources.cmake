# Source files for ttnn_op_experimental_deltanet.
# Trimmed to the decode_full op only (the recurrent single-token DeltaNet decode kernel used
# by the Qwen3.6-27B GDN decode path). The branch's prefill / plain-decode / chunked variants
# are intentionally not ported here (main uses gated_delta_attn for chunk-parallel prefill).

set(TTNN_OP_EXPERIMENTAL_DELTANET_SRCS
    device/deltanet_full_device_operation.cpp
    device/deltanet_full_program_factory.cpp
    deltanet_decode_full.cpp
)

set(TTNN_OP_EXPERIMENTAL_DELTANET_API_HEADERS
    deltanet_decode_full.hpp
)
