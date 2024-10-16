// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tuple>

// #include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::sliding_window {
namespace halo {

struct HaloDeviceOperation {
    thread_local static std::unordered_map<std::size_t, std::uint32_t> sliding_window_max_out_nsticks_per_core;
    SlidingWindowConfig config_;
    ParallelConfig parallel_config_;
    uint32_t pad_val_;
    bool remote_read_;
    bool transpose_mcast_;
    uint32_t reshard_num_cores_nhw_;
    uint32_t max_out_nsticks_per_core_;
    MemoryConfig output_memory_config_;
    bool is_out_tiled_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor> &input_tensors,
                                                   std::vector<Tensor> &output_tensors) const;
    // const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("config_",
                                                            "parallel_config_",
                                                            "pad_val_",
                                                            "remote_read_",
                                                            "transpose_mcast_",
                                                            "reshard_num_cores_nhw_",
                                                            "max_out_nsticks_per_core_",
                                                            "output_memory_config_",
                                                            "is_out_tiled_");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(config_),
                               std::cref(parallel_config_),
                               std::cref(pad_val_),
                               std::cref(remote_read_),
                               std::cref(transpose_mcast_),
                               std::cref(reshard_num_cores_nhw_),
                               std::cref(max_out_nsticks_per_core_),
                               std::cref(output_memory_config_),
                               std::cref(is_out_tiled_));
    }
};

Tensor halo_op(const Tensor &input_tensor,
               const SlidingWindowConfig &config,
               uint32_t pad_val = 0x0,
               bool remote_read = false,
               bool transpose_mcast = true,
               uint32_t reshard_num_cores_nhw = 0,
               MemoryConfig output_memory_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
               bool is_out_tiled = true);

}  // namespace halo

}  // namespace ttnn::operations::sliding_window
