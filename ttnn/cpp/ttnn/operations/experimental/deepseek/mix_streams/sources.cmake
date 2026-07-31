# Source files for ttnn_op_experimental_deepseek_mix_streams.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MIX_STREAMS_API_HEADERS mix_streams.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MIX_STREAMS_SRCS
    mix_streams.cpp
    device/mix_streams_device_operation.cpp
    device/mix_streams_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MIX_STREAMS_NANOBIND_SRCS mix_streams_nanobind.cpp)
