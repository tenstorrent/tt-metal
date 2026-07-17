# Source files for ttnn_op_conv.
# Module owners should update this file when adding/removing/renaming source files.
#
# NOTE: conv1d, conv_transpose2d, and the public conv2d Python binding
# (conv2d_nanobind) were nuked for the agent-regen baseline so `ttnn.conv2d` is
# no longer exposed. The conv2d C++ backend (op impl, device op, program
# factories, prepare_weights, and the conv2d_utils sharding helpers) is retained
# as shared infrastructure: pool/ and sliding_window/halo depend on conv2d_utils,
# and prepare_conv2d_weights depends on conv2d.cpp's get_conv2d_slice_attr.

set(TTNN_OP_CONV_SRCS
    conv2d/conv2d.cpp
    conv2d/conv2d_utils.cpp
    conv2d/conv2d_op_program_factory_common.cpp
    conv2d/device/conv2d_device_operation.cpp
    conv2d/device/conv2d_op_sharded_program_factory.cpp
    conv2d/device/conv2d_op_width_sharded_program_factory.cpp
    conv2d/prepare_conv2d_weights.cpp
)

set(TTNN_OP_CONV_API_HEADERS
    conv2d/conv2d.hpp
    conv2d/device/conv2d_device_operation_types.hpp
    conv2d/device/conv2d_device_operation.hpp
    conv2d/device/conv2d_op_sharded_program_factory.hpp
    conv2d/device/conv2d_op_width_sharded_program_factory.hpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/conv/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_CONV_NANOBIND_SRCS
    conv2d/conv2d_nanobind.cpp
    conv_nanobind.cpp
)
