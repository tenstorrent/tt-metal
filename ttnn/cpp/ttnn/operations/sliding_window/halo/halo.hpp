// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

 #include "ttnn/decorators.hpp"
 #include "ttnn/operations/sliding_window/sliding_window.hpp"
namespace ttnn::operations::sliding_window::halo {

// This is the main operation that will be called by the user
struct HaloOperation {
    // This how the user can call the operation
    static Tensor invoke(
                uint8_t queue_id,
                const Tensor& input_tensor,
                const SlidingWindowConfig& config,
                uint32_t pad_val = 0x0,
                bool remote_read = false,
                bool transpose_mcast = true,
                uint32_t reshard_num_cores_nhw = 0,
                MemoryConfig output_memory_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
                bool is_out_tiled = true);

    // invoke can be overloaded as many times as needed to provide all desired APIs
};

}  // namespace ttnn::operations::examples

namespace ttnn {

// Register the operation. The name, in this case, "ttnn::example" should match the namespace of the operation
// And the name will be directly mapped to python, where it will become "ttnn.example"
constexpr auto halo = ttnn::register_operation<"ttnn::halo", operations::sliding_window::halo::HaloOperation>();

// Alternatively, the operation can be registered as asynchronous
// constexpr auto example = ttnn::register_operation_with_auto_launch_op<"ttnn::example", operations::examples::ExampleOperation>();

}  // namespace ttnn
