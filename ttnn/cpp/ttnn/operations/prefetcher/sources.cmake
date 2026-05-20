# Source files for ttnn_op_prefetcher.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_PREFETCHER_SRCS
    prefetcher/device/dram_prefetcher_device_operation.cpp
    prefetcher/device/dram_prefetcher_program_factory.cpp
    prefetcher/dram_prefetcher.cpp
    prefetcher/dram_core_prefetcher.cpp
    prefetcher_consumer/dram_prefetcher_consumer.cpp
)
