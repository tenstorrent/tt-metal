// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/sharding_utilities.hpp"

namespace tt {

namespace tt_metal {

#define MAX_PACK_UNTILIZE_WIDTH 8  // pack untilize currently does not support > 8 width

enum class UntilizeOpParallelizationStrategy { MULTI_CORE, SINGLE_CORE };

struct Untilize {
    const MemoryConfig output_mem_config;
    const bool use_multicore;
    const bool use_pack_untilize;
    const bool fp32_dest_acc_en;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    UntilizeOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};

enum class UntilizeWithUnpaddingOpParallelizationStrategy { MULTI_CORE, SINGLE_CORE };

struct UntilizeWithUnpadding {
    const Shape output_tensor_end;
    const MemoryConfig output_mem_config;
    const bool use_multicore;
    const bool use_pack_untilize;
    const bool fp32_dest_acc_en;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    UntilizeWithUnpaddingOpParallelizationStrategy get_parallelization_strategy(
        const std::vector<Tensor> &input_tensors) const;

};

operation::ProgramWithCallbacks untilize_multi_core(
    const Tensor &a, Tensor &output, bool use_pack_untilize = true, bool fp32_dest_acc_en = false);
operation::ProgramWithCallbacks untilize_single_core(
    const Tensor &a, Tensor &output, bool use_pack_untilize = true, bool fp32_dest_acc_en = false);

operation::ProgramWithCallbacks untilize_with_unpadding_multi_core(
    const Tensor &a, Tensor &output, bool use_pack_untilize = true, bool fp32_dest_acc_en = false);
operation::ProgramWithCallbacks untilize_with_unpadding_single_core(
    const Tensor &a, Tensor &output, bool use_pack_untilize = true, bool fp32_dest_acc_en = false);

Tensor untilize(
    const Tensor &a,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    bool use_multicore = true,
    bool use_pack_untilize = true);
Tensor untilize_with_unpadding(
    const Tensor &a,
    const Shape &output_tensor_end,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    bool use_multicore = false,
    bool use_pack_untilize = true);

// NOTE: UntilizeWithHalo is only for sharded input/output
struct UntilizeWithHalo {
    const uint32_t pad_val_;
    const uint32_t in_b;
    const uint32_t in_h;
    const uint32_t in_w;
    const int32_t max_out_nsticks_per_core_;
    const uint32_t stride_;
    const PoolConfig pc_;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
};
Tensor untilize_with_halo(
    const Tensor &a,
    const uint32_t pad_val,
    const uint32_t &in_b,
    const uint32_t &in_h,
    const uint32_t &in_w,
    const uint32_t stride = 1,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

struct UntilizeWithHaloV2 {
    const uint32_t pad_val_;
    const uint32_t ncores_nhw_;
    const uint32_t max_out_nsticks_per_core_;
    const MemoryConfig out_mem_config_;
    const bool remote_read_;
    const bool transpose_mcast_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

};

operation::ProgramWithCallbacks untilize_with_halo_multi_core_v2(
    Program &program,
    const Tensor& input_tensor,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t max_out_nsticks_per_core,
    const Tensor& padding_config,
    const Tensor& local_config,
    const Tensor& remote_config,
    const bool remote_read,
    const bool transpose_mcast,
    Tensor& output_tensor);

Tensor untilize_with_halo_v2(
    const Tensor &input_tensor,
    const Tensor &padding_config,
    const Tensor &local_config,
    const Tensor &remote_config,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t max_out_nsticks_per_core,
    const MemoryConfig &mem_config,
    const bool remote_read,
    const bool transpose_mcast);

namespace untilize_helpers {

uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks);

}
}  // namespace tt_metal
}  // namespace tt
