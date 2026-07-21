# Source files for ttnn_op_experimental_dropout.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DROPOUT_SRCS
    device/dropout_device_operation.cpp
    device/dropout_program_factory.cpp
    dropout.cpp
)

set(TTNN_OP_EXPERIMENTAL_DROPOUT_API_HEADERS dropout.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/dropout/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DROPOUT_NANOBIND_SRCS dropout_nanobind.cpp)
