// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generalized_moe_gate_program_descriptor_builder.hpp"

#include <cstring>

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>

#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate {

namespace {

constexpr const char* kGeneralizedMoeGateKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/device/kernels/"
    "generalized_moe_gate_kernel.cpp";

uint32_t float_bits_u32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void set_cb_page_size_for_tile(tt::tt_metal::CBDescriptor& cb_desc, const tt::tt_metal::Tensor& tensor) {
    const auto& spec = tensor.tensor_spec();
    const auto& tile = spec.tile();
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(spec.data_type());
    uint32_t tile_size = tile.get_tile_size(data_format);
    auto& fd = cb_desc.format_descriptors[0];
    fd.tile = tt::tt_metal::TileDescriptor(tile);
    fd.page_size = (cb_desc.total_size % tile_size == 0) ? tile_size : cb_desc.total_size;
}

}  // namespace

tt::tt_metal::ProgramDescriptor build_moe_gate_program_descriptor(
    const tensor_args_t& tensor_args, const operation_attributes_t& operation_attrs) {
    using tt::tt_metal::CoreRangeSet;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::DataMovementProcessor;
    using tt::tt_metal::DataType;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::NOC;
    using tt::tt_metal::NOC_MODE;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& bias_tensor = tensor_args.bias_tensor;
    const auto& input_indices_tensor = tensor_args.input_indices_tensor;
    const auto& output_tensor = tensor_args.output_tensor;
    const auto& output_indices_tensor = tensor_args.output_indices_tensor;

    TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "input_tensor must be BFLOAT16");
    TT_FATAL(bias_tensor.dtype() == DataType::BFLOAT16, "bias_tensor must be BFLOAT16");
    TT_FATAL(output_tensor.dtype() == DataType::BFLOAT16, "output_tensor must be BFLOAT16");

    TT_FATAL(input_tensor.is_sharded(), "input_tensor must be sharded");
    TT_FATAL(bias_tensor.is_sharded(), "bias_tensor must be sharded");
    TT_FATAL(input_indices_tensor.is_sharded(), "input_indices_tensor must be sharded");
    TT_FATAL(output_tensor.is_sharded(), "output_tensor must be sharded");
    TT_FATAL(output_indices_tensor.is_sharded(), "output_indices_tensor must be sharded");

    const auto& input_shard = input_tensor.shard_spec().value();
    const auto& bias_shard = bias_tensor.memory_config().shard_spec().value();
    const auto& in_indices_shard = input_indices_tensor.memory_config().shard_spec().value();
    const auto& output_shard = output_tensor.shard_spec().value();
    const auto& out_indices_shard = output_indices_tensor.memory_config().shard_spec().value();

    CoreRangeSet all_cores = input_shard.grid;

    // num_blocks = how many 32x32 tiles per input shard (one 256-expert block per tile). 256->1, 512->2.
    const uint32_t num_blocks = (input_shard.shape[0] / 32) * (input_shard.shape[1] / 32);
    TT_FATAL(num_blocks >= 1, "input shard must hold at least one 32x32 tile");
    TT_FATAL(num_blocks <= 2, "generalized_moe_gate supports up to 2 blocks (<=512 experts), got {}", num_blocks);
    // Grouped (DeepSeek) mode runs the single-256-block grouped gate only; the 2-block combine is ungrouped.
    TT_FATAL(
        !operation_attrs.grouped || num_blocks == 1,
        "grouped mode (DeepSeek gate) is single-256-block only; got num_blocks={}",
        num_blocks);

    TT_FATAL(input_shard.shape == bias_shard.shape, "Input and bias shard shapes must match");
    TT_FATAL(input_shard.orientation == bias_shard.orientation, "Input and bias shard orientations must match");
    TT_FATAL(bias_shard.grid.contains(all_cores), "Bias shard grid must contain input shard grid");

    TT_FATAL(
        in_indices_shard.shape[0] % 32 == 0 && in_indices_shard.shape[1] == 32,
        "Input-indices shard must be tile-aligned (one 32x32 tile per block; block b holds global ids)");
    TT_FATAL(
        input_shard.orientation == in_indices_shard.orientation,
        "Input and input-indices shard orientations must match");
    TT_FATAL(in_indices_shard.grid.contains(all_cores), "Input-indices shard grid must contain input shard grid");

    TT_FATAL(output_shard.grid == out_indices_shard.grid, "Output and output-indices shard grids must match");
    TT_FATAL(output_shard.grid.contains(all_cores), "Output shard grid must contain input compute grid");

    const auto& in_tile = input_tensor.tensor_spec().tile();
    const auto& out_tile = output_tensor.tensor_spec().tile();
    TT_FATAL(in_tile == bias_tensor.tensor_spec().tile(), "Input and bias tiles must match");
    TT_FATAL(in_tile == input_indices_tensor.tensor_spec().tile(), "Input and input-indices tiles must match");
    TT_FATAL(out_tile == output_indices_tensor.tensor_spec().tile(), "Output tiles must match");

    constexpr uint8_t input_cb = 0;
    constexpr uint8_t bias_cb = 1;
    constexpr uint8_t output_cb = 2;
    constexpr uint8_t input_indices_cb = 3;
    constexpr uint8_t output_indices_cb = 4;
    // Intermediate L1 stash CBs for the multi-block combine: each holds num_blocks per-block run tiles
    // for one field (scores/bias bf16, idx uint16). Only used when num_blocks > 1.
    constexpr uint8_t run_scores_cb = 5;
    constexpr uint8_t run_idx_cb = 6;
    constexpr uint8_t run_bias_cb = 7;
    // Scratch tiled CB for the L1-stash layout convert: a stashed run (row-major, via pack_untilize) is
    // tilize'd into here, then transpose_wh'd into DEST math layout. bf16 (scores/bias).
    constexpr uint8_t cb_tilize = 8;
    // Separate uint16 tilize scratch for the idx field: the expert id must be BIT-PRESERVED through the
    // round-trip (the SFPU reads it as a raw 16-bit id; a bf16 numeric convert would corrupt the bits).
    constexpr uint8_t cb_tilize_idx = 9;

    auto in_cb_desc = tt::tt_metal::cb_descriptor_from_sharded_tensor(input_cb, input_tensor);
    auto bias_cb_desc = tt::tt_metal::cb_descriptor_from_sharded_tensor(bias_cb, bias_tensor);
    auto out_cb_desc = tt::tt_metal::cb_descriptor_from_sharded_tensor(output_cb, output_tensor);
    auto in_indices_cb_desc = tt::tt_metal::cb_descriptor_from_sharded_tensor(input_indices_cb, input_indices_tensor);
    auto out_indices_cb_desc =
        tt::tt_metal::cb_descriptor_from_sharded_tensor(output_indices_cb, output_indices_tensor);

    set_cb_page_size_for_tile(in_cb_desc, input_tensor);
    set_cb_page_size_for_tile(bias_cb_desc, bias_tensor);
    set_cb_page_size_for_tile(out_cb_desc, output_tensor);
    set_cb_page_size_for_tile(in_indices_cb_desc, input_indices_tensor);
    set_cb_page_size_for_tile(out_indices_cb_desc, output_indices_tensor);

    // Build an intermediate (non-tensor) L1 CB sized for `num_tiles` tiles of fmt_tensor's format.
    auto make_run_cb = [&](uint8_t cb_id, const tt::tt_metal::Tensor& fmt_tensor, uint32_t num_tiles) {
        const auto& spec = fmt_tensor.tensor_spec();
        const auto& tile = spec.tile();
        auto df = tt::tt_metal::datatype_to_dataformat_converter(spec.data_type());
        uint32_t tsz = tile.get_tile_size(df);
        tt::tt_metal::CBDescriptor d;
        d.total_size = num_tiles * tsz;
        d.core_ranges = all_cores;
        d.format_descriptors.push_back(
            tt::tt_metal::CBFormatDescriptor{cb_id, df, tsz, tt::tt_metal::TileDescriptor(tile)});
        return d;
    };
    auto run_scores_cb_desc = make_run_cb(run_scores_cb, input_tensor, num_blocks);    // bf16 (score), 1/block
    auto run_idx_cb_desc = make_run_cb(run_idx_cb, input_indices_tensor, num_blocks);  // uint16 (idx), 1/block
    auto run_bias_cb_desc = make_run_cb(run_bias_cb, input_tensor, num_blocks);        // bf16 (bias), 1/block
    // cb_tilize holds the bf16 tilize-scratch for BOTH score+bias of every block (the combine tilizes all
    // fields before the merge acquire), so 2 bf16 tiles per block. cb_tilize_idx holds 1 uint16 idx/block.
    auto cb_tilize_desc = make_run_cb(cb_tilize, input_tensor, 2 * num_blocks);              // bf16 scratch
    auto cb_tilize_idx_desc = make_run_cb(cb_tilize_idx, input_indices_tensor, num_blocks);  // uint16 scratch

    KernelDescriptor::NamedCompileTimeArgs ncrisc_named = {
        {"moe_gate_input_cb", input_cb},
        {"moe_gate_bias_cb", bias_cb},
        {"moe_gate_input_indices_cb", input_indices_cb},
        {"moe_gate_num_blocks", num_blocks},
        {"moe_gate_is_active_core", 1},
    };
    KernelDescriptor::NamedCompileTimeArgs brisc_named = {
        {"moe_gate_output_cb", output_cb},
        {"moe_gate_output_indices_cb", output_indices_cb},
        {"moe_gate_is_active_core", 1},
    };
    KernelDescriptor::NamedCompileTimeArgs trisc_named = {
        {"moe_gate_input_cb", input_cb},
        {"moe_gate_bias_cb", bias_cb},
        {"moe_gate_input_indices_cb", input_indices_cb},
        {"moe_gate_output_cb", output_cb},
        {"moe_gate_output_indices_cb", output_indices_cb},
        {"moe_gate_eps", float_bits_u32(operation_attrs.eps)},
        {"moe_gate_scaling_factor", float_bits_u32(operation_attrs.scaling_factor)},
        {"moe_gate_enable_sigmoid", operation_attrs.enable_sigmoid ? 1u : 0u},
        {"moe_gate_topk", operation_attrs.topk},
        {"moe_gate_softmax", operation_attrs.output_softmax ? 1u : 0u},
        {"moe_gate_num_blocks", num_blocks},
        {"moe_gate_run_scores_cb", run_scores_cb},
        {"moe_gate_run_idx_cb", run_idx_cb},
        {"moe_gate_run_bias_cb", run_bias_cb},
        {"moe_gate_cb_tilize", cb_tilize},
        {"moe_gate_cb_tilize_idx", cb_tilize_idx},
        {"moe_gate_is_active_core", 1},
    };

    tt::tt_metal::ComputeConfigDescriptor compute_config{};
    compute_config.math_fidelity = MathFidelity::HiFi4;
    // Full DEST sync (single 16-tile bank; no double-buffer bank alternation across tile_regs_acquire).
    // The multi-block combine parks block0's run in DEST and reads it back in block1's SEPARATE acquire;
    // that only survives if acquire does not swap banks.
    compute_config.dst_full_sync_en = true;

    // Path-select for the unified kernel, injected as a compile DEFINE (NOT a named CT arg): the compute API
    // picks ungrouped-vs-grouped with `#if GMG_UNGROUPED_TOP8` at PREPROCESS time, before CT-arg constexprs
    // exist. =1 → ungrouped global top-k; =0 → DeepSeek grouped gate. Set on ALL THREE kernels: the single
    // .cpp is compiled for every RISC and includes the compute API header, whose `#error` guard requires the
    // macro to be defined (so a missing define is a hard compile error, never a silent grouped fallthrough).
    const KernelDescriptor::Defines gmg_defines = {{"GMG_UNGROUPED_TOP8", operation_attrs.grouped ? "0" : "1"}};

    KernelDescriptor reader{
        .kernel_source = std::string(kGeneralizedMoeGateKernelPath),
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .named_compile_time_args = std::move(ncrisc_named),
        .config =
            DataMovementConfigDescriptor{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_0_default,
                .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
            },
    };

    KernelDescriptor writer{
        .kernel_source = std::string(kGeneralizedMoeGateKernelPath),
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .named_compile_time_args = std::move(brisc_named),
        .config =
            DataMovementConfigDescriptor{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_1_default,
                .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
            },
    };

    KernelDescriptor compute_k{
        .kernel_source = std::string(kGeneralizedMoeGateKernelPath),
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = all_cores,
        .named_compile_time_args = std::move(trisc_named),
        .config = compute_config,
    };

    reader.defines = gmg_defines;
    writer.defines = gmg_defines;
    compute_k.defines = gmg_defines;

    tt::tt_metal::ProgramDescriptor program_desc;
    program_desc.kernels.reserve(3);
    program_desc.kernels.push_back(std::move(reader));
    program_desc.kernels.push_back(std::move(writer));
    program_desc.kernels.push_back(std::move(compute_k));

    program_desc.cbs.reserve(8);
    program_desc.cbs.push_back(std::move(in_cb_desc));
    program_desc.cbs.push_back(std::move(bias_cb_desc));
    program_desc.cbs.push_back(std::move(out_cb_desc));
    program_desc.cbs.push_back(std::move(in_indices_cb_desc));
    program_desc.cbs.push_back(std::move(out_indices_cb_desc));
    program_desc.cbs.push_back(std::move(run_scores_cb_desc));
    program_desc.cbs.push_back(std::move(run_idx_cb_desc));
    program_desc.cbs.push_back(std::move(run_bias_cb_desc));
    program_desc.cbs.push_back(std::move(cb_tilize_desc));
    program_desc.cbs.push_back(std::move(cb_tilize_idx_desc));

    return program_desc;
}

std::uint64_t hash_moe_gate_program_structure(const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    if (program_descriptor.custom_program_hash.has_value()) {
        return static_cast<std::uint64_t>(program_descriptor.custom_program_hash.value());
    }

    auto hash_kernel = [&](const tt::tt_metal::KernelDescriptor& kernel) -> size_t {
        return ttsl::hash::hash_objects_with_default_seed(
            kernel.kernel_source,
            kernel.source_type,
            kernel.core_ranges,
            kernel.compile_time_args,
            kernel.named_compile_time_args,
            kernel.defines,
            kernel.common_runtime_args.size(),
            kernel.runtime_args.size(),
            kernel.config.index(),
            kernel.config);
    };

    auto hash_cb_format_descriptor = [&](const tt::tt_metal::CBFormatDescriptor& format_descriptor) -> size_t {
        return ttsl::hash::hash_objects_with_default_seed(
            format_descriptor.buffer_index,
            format_descriptor.data_format,
            format_descriptor.page_size,
            format_descriptor.tile);
    };

    auto hash_circular_buffer = [&](const tt::tt_metal::CBDescriptor& cb) -> size_t {
        size_t hash = cb.total_size;
        for (const auto& core_range : cb.core_ranges.ranges()) {
            ttsl::hash::hash_combine(hash, core_range);
        }
        ttsl::hash::hash_combine(hash, cb.format_descriptors.size());
        for (const auto& format_descriptor : cb.format_descriptors) {
            ttsl::hash::hash_combine(hash, hash_cb_format_descriptor(format_descriptor));
        }
        ttsl::hash::hash_combine(hash, cb.remote_format_descriptors.size());
        for (const auto& format_descriptor : cb.remote_format_descriptors) {
            ttsl::hash::hash_combine(hash, hash_cb_format_descriptor(format_descriptor));
        }
        ttsl::hash::hash_combine(hash, cb.buffer != nullptr);
        ttsl::hash::hash_combine(hash, cb.global_circular_buffer != nullptr);
        return hash;
    };

    auto hash_semaphore = [&](const tt::tt_metal::SemaphoreDescriptor& semaphore) -> size_t {
        return ttsl::hash::hash_objects_with_default_seed(
            semaphore.core_ranges, semaphore.core_type, semaphore.initial_value);
    };

    size_t hash = 0;
    for (const auto& kernel : program_descriptor.kernels) {
        ttsl::hash::hash_combine(hash, hash_kernel(kernel));
    }
    for (const auto& cb : program_descriptor.cbs) {
        ttsl::hash::hash_combine(hash, hash_circular_buffer(cb));
    }
    for (const auto& semaphore : program_descriptor.semaphores) {
        ttsl::hash::hash_combine(hash, hash_semaphore(semaphore));
    }
    return static_cast<std::uint64_t>(hash);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate
