// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/deepseek_moe_post_combine_tilize_program_factory.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

DeepseekMoEPostCombineTilizeProgramFactory::cached_program_t DeepseekMoEPostCombineTilizeProgramFactory::create(
    const DeepseekMoEPostCombineTilizeParams&, const DeepseekMoEPostCombineTilizeInputs& tensor_args, ttnn::Tensor&) {
    auto program = Program();

    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_rank = input_shape.rank();

    /*
     * Number of cores to use
     */
    uint32_t upper_dims = 1;
    for (uint32_t dim = 0; dim < input_rank - 1; ++dim) {
        upper_dims *= input_shape[dim];
    }
    uint32_t total_tile_height = upper_dims / tt::constants::TILE_HEIGHT;

    auto grid_size = input_tensor.device()->compute_with_storage_grid_size();
    uint32_t total_cores = grid_size.x * grid_size.y;

    uint32_t cores_per_tile_height = total_cores / total_tile_height;

    uint32_t num_op_cores = total_tile_height * cores_per_tile_height;
    CoreRangeSet op_cores = tt::tt_metal::num_cores_to_corerangeset(num_op_cores, grid_size, true);

    // reader
    std::unordered_map<std::string, uint32_t> reader_named_ct_args = {

    };
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, "...", op_cores, tt::tt_metal::ComputeConfig{.named_compile_args = reader_named_ct_args});

    // compute
    std::unordered_map<std::string, uint32_t> compute_named_ct_args = {

    };
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program, "...", op_cores, tt::tt_metal::ComputeConfig{.named_compile_args = compute_named_ct_args});

    // writer
    std::unordered_map<std::string, uint32_t> writer_named_ct_args = {

    };
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program, "...", op_cores, tt::tt_metal::ComputeConfig{.named_compile_args = writer_named_ct_args});

    std::vector<tt::tt_metal::CoreCoord> cores = corerange_to_cores(op_cores, std::nullopt, true);
    for (uint32_t i = 0; i < num_op_cores; ++i) {
        const auto& core = cores[i];

        std::vector<uint32_t> reader_rt_args = {

        };
        SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        std::vector<uint32_t> compute_rt_args = {

        };
        SetRuntimeArgs(program, compute_kernel_id, core, compute_rt_args);

        std::vector<uint32_t> writer_rt_args = {

        };
        SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cores = cores}};
}

void DeepseekMoEPostCombineTilizeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const DeepseekMoEPostCombineTilizeParams&,
    const DeepseekMoEPostCombineTilizeInputs& tensor_args,
    ttnn::Tensor&) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;

    auto& program = cached_program.program;
    const tt::tt_metal::KernelHandle& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const tt::tt_metal::KernelHandle& compute_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const tt::tt_metal::KernelHandle& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const std::vector<tt::tt_metal::CoreCoord>& cores = cached_program.shared_variables.cores;

    for (uint32_t i = 0; i < cores.size(); ++i) {
        auto& reader_args = GetRuntimeArgs(program, reader_kernel_id, cores[i]);
        reader_args[0] = input_tensor.buffer()->address();

        auto& compute_args = GetRuntimeArgs(program, compute_kernel_id, cores[i]);
        compute_args[0] = input_tensor.buffer()->address();

        auto& writer_args = GetRuntimeArgs(program, writer_kernel_id, cores[i]);
        writer_args[0] = input_tensor.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
