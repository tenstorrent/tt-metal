# Source files for ttnn_op_experimental_deepseek_prefill_prefill_test.
# Module owners should update this file when adding/removing/renaming source files.
# NOTE: the *_nanobind.cpp is compiled into the ttnn nanobind module (see ttnn/sources.cmake),
# not into this op library — only the host-service impl lives here.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PREFILL_TEST_SRCS layer_completion_consumer.cpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PREFILL_TEST_API_HEADERS layer_completion_consumer.hpp)
