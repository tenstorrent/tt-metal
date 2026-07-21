# Source files for ttnn_op_experimental_test.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_TEST_SRCS
    hang_device/hang_device_operation.cpp
    prefetcher_consumer/dram_prefetcher_consumer.cpp
    prefetcher_consumer/dram_prefetcher_validator.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/test/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_TEST_NANOBIND_SRCS
    hang_device/hang_device_operation_nanobind.cpp
    prefetcher_consumer/dram_prefetcher_consumer_nanobind.cpp
)
