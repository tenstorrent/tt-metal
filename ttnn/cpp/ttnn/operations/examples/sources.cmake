# Source files for ttnn_op_examples.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXAMPLES_SRCS
    bh_dram_read/bh_dram_read.cpp
    bh_dram_read/device/bh_dram_read_device_operation.cpp
    bh_dram_read/device/bh_dram_read_program_factory.cpp
    example/example.cpp
    example/device/example_device_operation.cpp
    example/device/multi_core_program_factory.cpp
    example/device/single_core_program_factory.cpp
    example_multiple_return/device/example_multiple_return_device_operation.cpp
    example_multiple_return/device/single_core_program_factory.cpp
    example_multiple_return/example_multiple_return.cpp
)
