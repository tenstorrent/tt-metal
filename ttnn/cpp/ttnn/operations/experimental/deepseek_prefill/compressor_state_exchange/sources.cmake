# Source files for ttnn_op_experimental_deepseek_prefill_compressor_state_exchange.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_COMPRESSOR_STATE_EXCHANGE_API_HEADERS compressor_state_exchange.hpp)

# No NANOBIND_SRCS: this op has no Python binding. It is a C++-only building block, called by
# csa_compressor (see csa_compressor/csa_compressor.cpp), which is the op the model path uses and the
# one that is bound.
