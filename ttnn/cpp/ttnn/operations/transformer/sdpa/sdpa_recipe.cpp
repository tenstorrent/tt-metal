// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_recipe.hpp"

#include <algorithm>
#include <array>
#include "sdpa_numerics.hpp"
#include <bit>
#include <cmath>
#include <set>

#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/generic/generic_op.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::transformer::sdpa::detail {
using namespace tt::tt_metal;

RecipeSelection select_recipe(ttnn::transformer::SDPAPrecision precision, DataType kv_type) {
    using ttnn::transformer::SDPAPrecision;
    Recipe recipe;
    switch (precision) {
        case SDPAPrecision::FAST: recipe = Recipe::A; break;
        case SDPAPrecision::COMPENSATED: recipe = Recipe::B; break;
        case SDPAPrecision::BALANCED: recipe = Recipe::C; break;
        case SDPAPrecision::ACCURATE: recipe = Recipe::D; break;
        case SDPAPrecision::LOW_PRECISION: recipe = Recipe::E; break;
        default: TT_THROW("Unknown SDPA precision recipe");
    }
    const auto storage = kv_type == DataType::BFLOAT4_B   ? KVStorage::BFP4
                         : kv_type == DataType::BFLOAT8_B ? KVStorage::BFP8
                                                          : KVStorage::BF16;
    return {recipe, storage};
}

uint32_t recipe_q_tiles(const std::optional<SDPAProgramConfig>& program_config) {
    const uint32_t q_chunk = program_config ? program_config->q_chunk_size : 256;
    // Ten query tile rows bound the recurrent-state arrays. Compensated BF16
    // recipes additionally need an even tile count (checked with the policy).
    TT_FATAL(
        q_chunk % 32 == 0 && q_chunk >= 128 && q_chunk <= 320,
        "Named SDPA recipes support Q chunks from 128 to 320 rows in 32-row steps, got {}",
        q_chunk);
    return q_chunk / 32;
}

uint32_t recipe_k_tiles(const std::optional<SDPAProgramConfig>& program_config) {
    const uint32_t k_chunk = program_config ? program_config->k_chunk_size : 512;
    // QK/PV subblocks are four tiles wide; the early max-reduce trigger needs at least two of them.
    TT_FATAL(
        k_chunk == 256 || k_chunk == 384 || k_chunk == 512,
        "Named SDPA recipes support K chunks of 256, 384 or 512 rows, got {}",
        k_chunk);
    return k_chunk / 32;
}

ProgramDescriptor recipe_compute_program(
    const PrecisionPolicy& policy, const CoreRangeSet& grid, uint32_t k_chunks, uint32_t q_tiles, uint32_t k_tiles) {
    const bool fp32 = policy.fp32_destination;
    const bool compensated = policy.recurrent_state == RecurrentState::CompensatedBF16;
    const uint32_t stride = compensated ? 2 : 1;
    const auto state_format = fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t state_bytes = fp32 ? 4096 : 2048;
    const auto kv_type = policy.selection.kv_storage == KVStorage::BF16   ? DataType::BFLOAT16
                         : policy.selection.kv_storage == KVStorage::BFP8 ? DataType::BFLOAT8_B
                                                                          : DataType::BFLOAT4_B;
    const auto kv_format = datatype_to_dataformat_converter(kv_type);
    const uint32_t kv_bytes = kv_type == DataType::BFLOAT16 ? 2048 : kv_type == DataType::BFLOAT8_B ? 1088 : 576;
    ProgramDescriptor program;
    auto add_cb = [&](uint8_t index, uint32_t count, uint32_t page, tt::DataFormat format) {
        CBDescriptor cb{
            .total_size = count * page,
            .core_ranges = grid,
            .format_descriptors = {{.buffer_index = index, .data_format = format, .page_size = page}}};
        if (index == 6 && fp32) {
            cb.format_descriptors.push_back({.buffer_index = 7, .data_format = format, .page_size = page});
        }
        program.cbs.push_back(std::move(cb));
    };
    // Preserve the selected families' actual buffer depths: Q double buffered;
    // K/V one slot for FP32, two slots for BF16. No hidden geometry retuning.
    // Q-row buffers scale with the Q chunk; K/V depths and per-row state do not change.
    add_cb(0, 2 * q_tiles * 4, 2048, tt::DataFormat::Float16_b);
    add_cb(1, k_tiles * 4 * (fp32 ? 1 : 2), kv_bytes, kv_format);
    add_cb(2, k_tiles * 4 * (fp32 ? 1 : 2), kv_bytes, kv_format);
    add_cb(3, 1, 2048, tt::DataFormat::Float16_b);
    add_cb(4, 1, 2048, tt::DataFormat::Float16_b);
    add_cb(5, 1, state_bytes, state_format);
    add_cb(6, q_tiles * k_tiles, state_bytes, state_format);
    for (uint8_t index : {8, 9}) {
        add_cb(index, q_tiles * 4 * stride, state_bytes, state_format);
    }
    for (uint8_t index : {10, 11}) {
        add_cb(index, q_tiles, 2048, tt::DataFormat::Float16_b);
    }
    for (uint8_t index : {12, 13}) {
        add_cb(index, q_tiles * stride, state_bytes, state_format);
    }
    add_cb(14, q_tiles, state_bytes, state_format);
    add_cb(16, fp32 ? 8 : 16, 2048, tt::DataFormat::Float16_b);
    ComputeConfigDescriptor compute_config{
        .math_fidelity = policy.pv_fidelity,
        .fp32_dest_acc_en = fp32,
        .dst_full_sync_en = false,
        .math_approx_mode = true};
    if (fp32) {
        compute_config.unpack_to_dest_mode.resize(64, UnpackToDestMode::Default);
        for (uint32_t cb : {5, 7, 8, 9, 12, 13, 14}) {
            compute_config.unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
        }
    }
    KernelDescriptor compute{
        .kernel_source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa_recipe.cpp",
        .core_ranges = grid,
        .compile_time_args = {k_chunks, std::bit_cast<uint32_t>(1.0f / std::sqrt(128.0f)), q_tiles, k_tiles},
        .defines =
            {{"EXP_APPROX_MODE", "1"},
             {"STATS_GRANULARITY", fp32 ? "4" : "8"},
             {"SUB_EXP_GRANULARITY", fp32 ? "4" : "8"},
             {"MUL_BCAST_GRANULARITY", fp32 ? "4" : "8"},
             {"DHT_GRANULARITY", "4"},
             {"REDUCE_GRANULARITY", fp32 ? "2" : "4"}},
        .config = compute_config};
    if (fp32) {
        compute.defines.emplace_back("SDPA_RECIPE_FP32", "1");
    }
    if (policy.selection.recipe == Recipe::D) {
        compute.defines.emplace_back("SDPA_RECIPE_ACCURATE", "1");
    }
    if (policy.selection.recipe == Recipe::E) {
        compute.defines.emplace_back("SDPA_RECIPE_LOFI", "1");
    }
    if (policy.selection.recipe == Recipe::A) {
        compute.defines.emplace_back("SDPA_RECIPE_BASELINE", "1");
    }
    program.kernels.push_back(std::move(compute));
    return program;
}

PrecisionPolicy resolve_recipe_policy(
    const Tensor& q,
    const Tensor& k,
    ttnn::transformer::SDPAPrecision precision,
    bool inputs_prepared,
    std::optional<float> scale,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<SDPAProgramConfig>& program_config) {
    TT_FATAL(q.storage_type() == StorageType::DEVICE, "SDPA recipes require device inputs");
    TT_FATAL(!scale || *scale == 1.0f / std::sqrt(128.0f), "SDPA recipes currently require the default D128 scale");
    const auto selection = select_recipe(precision, k.dtype());
    TT_FATAL(
        inputs_prepared == (selection.recipe == Recipe::E),
        "LOW_PRECISION requires inputs_prepared=True and explicit SDPA preparation; other recipes use ordinary inputs");
    return *resolve_numerics(
                q.device()->arch(),
                selection,
                compute_kernel_config,
                program_config ? program_config->exp_approx_mode : std::nullopt)
                .policy;
}

// Reject a recipe CB layout that cannot fit the device's unreserved L1.
static void check_recipe_l1_fit(
    const ProgramDescriptor& program, IDevice& device, uint32_t q_chunk, uint32_t k_chunk) {
    uint64_t bytes = 0;
    for (const auto& cb : program.cbs) {
        bytes += cb.total_size;
    }
    const uint64_t available =
        device.l1_size_per_core() - device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    TT_FATAL(
        bytes <= available,
        "SDPA recipe needs {} bytes of L1 per core at Q{}/K{}, but only {} are available; use a smaller Q or K chunk",
        bytes,
        q_chunk,
        k_chunk,
        available);
}

static std::vector<Tensor> run_recipe_segments(
    const std::vector<std::array<Tensor, 3>>& segments,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config) {
    const auto& [q, k, v] = segments.front();
    std::vector<Tensor> io;
    for (const auto& segment : segments) {
        io.insert(io.end(), segment.begin(), segment.end());
    }
    for (const auto& input : io) {
        const Tensor* tensor = &input;
        TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "SDPA recipes require device inputs");
        TT_FATAL(tensor->device() == q.device(), "SDPA recipe inputs must belong to the same device");
        TT_FATAL(tensor->device()->arch() == tt::ARCH::BLACKHOLE, "SDPA recipes currently support Blackhole only");
        TT_FATAL(tensor->layout() == Layout::TILE, "SDPA recipes require tiled inputs");
        TT_FATAL(tensor->memory_config() == DRAM_MEMORY_CONFIG, "SDPA recipes require interleaved DRAM inputs");
        TT_FATAL(tensor->logical_shape().rank() == 4, "SDPA recipes require rank-four inputs");
        const auto& shape = tensor->logical_shape();
        const auto& padded = tensor->padded_shape();
        TT_FATAL(
            shape[0] == padded[0] && shape[1] == padded[1] && shape[3] == padded[3] &&
                padded[2] == ((shape[2] + 31) / 32) * 32,
            "SDPA recipes only support minimal sequence-axis tile padding");
        TT_FATAL(tensor->tensor_spec().tile() == Tile({32, 32}), "SDPA recipes require standard 32x32 tiles");
    }
    const auto& qs = q.logical_shape();
    const DataType kv_type = policy.selection.kv_storage == KVStorage::BF16   ? DataType::BFLOAT16
                             : policy.selection.kv_storage == KVStorage::BFP8 ? DataType::BFLOAT8_B
                                                                              : DataType::BFLOAT4_B;
    uint32_t q_length = 0, k_length = 0;
    for (const auto& [sq, sk, sv] : segments) {
        const auto& qshape = sq.logical_shape();
        const auto& kshape = sk.logical_shape();
        TT_FATAL(
            qshape[0] > 0 && qshape[0] == qs[0] && qshape[1] > 0 && qshape[1] == qs[1] && qshape[3] == 128,
            "SDPA recipe segments require Q [B,H,Q,128] with matching positive batch/head counts");
        TT_FATAL(
            kshape == sv.logical_shape() && kshape[0] == qs[0] && kshape[1] > 0 && kshape[1] == k.logical_shape()[1] &&
                qs[1] % kshape[1] == 0 && kshape[3] == 128,
            "SDPA recipes require matching K/V [B,Hkv,K,128] and Q heads divisible by KV heads");
        TT_FATAL(qshape[2] > 0 && kshape[2] > 0, "SDPA recipe segments require positive sequence lengths");
        TT_FATAL(
            sq.dtype() == DataType::BFLOAT16 && sk.dtype() == kv_type && sv.dtype() == kv_type,
            "SDPA input types do not match the selected recipe");
        q_length += sq.padded_shape()[2];
        k_length += sk.padded_shape()[2];
    }
    const uint32_t joint_q_rows = segments.size() == 2 ? segments[1][0].logical_shape()[2] : 0;
    const uint32_t joint_k_rows = segments.size() == 2 ? segments[1][1].logical_shape()[2] : 0;
    const auto hardware = q.device()->compute_with_storage_grid_size();
    const auto grid_size = program_config ? program_config->compute_with_storage_grid_size : hardware;
    TT_FATAL(
        grid_size.x > 0 && grid_size.y > 0 && grid_size.x <= hardware.x && grid_size.y <= hardware.y,
        "SDPA recipe compute grid must fit the device");
    const uint32_t q_tiles = recipe_q_tiles(program_config);
    const uint32_t q_chunk = q_tiles * 32;
    TT_FATAL(
        q_tiles % 2 == 0 || policy.recurrent_state != RecurrentState::CompensatedBF16,
        "COMPENSATED and LOW_PRECISION recipes pair query tile rows and need a Q chunk that is a multiple of 64, got {}",
        q_chunk);
    const uint32_t k_tiles = recipe_k_tiles(program_config);
    const uint32_t k_chunk = k_tiles * 32;
    if (program_config) {
        TT_FATAL(!program_config->sub_core_grids.has_value(), "SDPA recipes do not yet support sub_core_grids");
        TT_FATAL(program_config->max_cores_per_head_batch > 0, "SDPA max_cores_per_head_batch must be positive");
    }
    const uint32_t jobs_per_head = (q_length + q_chunk - 1) / q_chunk;
    const uint32_t k_chunks = (k_length + k_chunk - 1) / k_chunk;
    TT_FATAL(
        qs[1] <= grid_size.x * grid_size.y && qs[0] <= grid_size.x * grid_size.y / qs[1],
        "SDPA recipes require at least one compute core per batch/query head");
    const uint32_t batch_heads = qs[0] * qs[1];
    const uint32_t chain = std::min<uint32_t>(
        {jobs_per_head,
         static_cast<uint32_t>(grid_size.x * grid_size.y / batch_heads),
         program_config ? program_config->max_cores_per_head_batch : 16u});
    TT_FATAL(chain > 0, "SDPA recipes require at least one compute core per head");
    const uint32_t cores = chain * batch_heads;
    std::vector<CoreCoord> coordinates;
    std::set<CoreRange> ranges;
    for (uint32_t i = 0; i < cores; ++i) {
        const CoreCoord core(i % grid_size.x, i / grid_size.x);
        coordinates.push_back(core);
        ranges.emplace(core, core);
    }
    const CoreRangeSet grid(ranges);
    std::vector<Tensor> outputs;
    for (const auto& segment : segments) {
        outputs.push_back(create_device_tensor(segment[0].tensor_spec(), q.device()));
    }
    const auto& output = outputs.front();
    auto program = recipe_compute_program(policy, grid, k_chunks, q_tiles, k_tiles);
    check_recipe_l1_fit(program, *q.device(), q_chunk, k_chunk);
    if (k_length % k_chunk != 0 || k.logical_shape()[2] % 32 != 0 || joint_k_rows % 32 != 0) {
        program.kernels.front().defines.emplace_back(
            "SDPA_RECIPE_K_PRIMARY_ROWS", std::to_string(k.logical_shape()[2]));
        program.kernels.front().defines.emplace_back("SDPA_RECIPE_K_JOINT_ROWS", std::to_string(joint_k_rows));
    }
    for (uint32_t i = 0; i < 3; ++i) {
        program.semaphores.push_back({.id = i, .core_ranges = grid, .initial_value = i == 2 ? 1u : 0u});
    }
    const std::string prefix = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    KernelDescriptor reader{
        .kernel_source = prefix + "dataflow/reader_recipe.cpp",
        .core_ranges = grid,
        .compile_time_args =
            {q_tiles, k_chunks, jobs_per_head, q.logical_shape()[2], joint_q_rows, k.logical_shape()[2], joint_k_rows},
        .config = ReaderConfigDescriptor{}};
    if (segments.size() == 2) {
        reader.defines.emplace_back("SDPA_JOINT", "1");
    }
    reader.defines.emplace_back("SDPA_K_CHUNK_TILES", std::to_string(k_tiles));
    if (qs[1] != k.logical_shape()[1]) {
        reader.defines.emplace_back("SDPA_RECIPE_Q_PER_KV_HEAD", std::to_string(qs[1] / k.logical_shape()[1]));
    }
    for (const auto& tensor : io) {
        TensorAccessorArgs(tensor.buffer()).append_to(reader.compile_time_args);
    }
    KernelDescriptor writer{
        .kernel_source = prefix + "dataflow/writer_recipe.cpp",
        .core_ranges = grid,
        .compile_time_args = {q_tiles, q.logical_shape()[2], joint_q_rows},
        .config = WriterConfigDescriptor{}};
    if (segments.size() == 2) {
        writer.defines.emplace_back("SDPA_JOINT", "1");
    }
    for (const auto& tensor : outputs) {
        TensorAccessorArgs(tensor.buffer()).append_to(writer.compile_time_args);
    }
    auto compute = std::move(program.kernels.front());
    for (uint32_t i = 0; i < cores; ++i) {
        const auto core = coordinates[i];
        const uint32_t head = i / chain, rank = i % chain;
        const uint32_t count = jobs_per_head / chain + (rank < jobs_per_head % chain);
        const uint32_t offset =
            head * jobs_per_head + rank * (jobs_per_head / chain) + std::min(rank, jobs_per_head % chain);
        const auto prev = rank ? q.device()->worker_core_from_logical_core(coordinates[i - 1]) : CoreCoord(0, 0);
        const auto next =
            rank + 1 < chain ? q.device()->worker_core_from_logical_core(coordinates[i + 1]) : CoreCoord(0, 0);
        const uint32_t next_count = rank + 1 < chain ? jobs_per_head / chain + (rank + 1 < jobs_per_head % chain) : 0;
        reader.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                q.buffer()->address(),
                k.buffer()->address(),
                v.buffer()->address(),
                offset,
                count,
                rank,
                chain,
                prev.x,
                prev.y,
                next.x,
                next.y,
                next_count});
        writer.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{output.buffer()->address(), offset, count});
        if (segments.size() == 2) {
            for (const auto& tensor : segments[1]) {
                reader.runtime_args.back().second.push_back(tensor.buffer()->address());
            }
            writer.runtime_args.back().second.push_back(outputs[1].buffer()->address());
        }
        compute.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{count});
    }
    program.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    io.insert(io.end(), outputs.begin(), outputs.end());
    ttnn::generic_op(io, program);
    return outputs;
}

Tensor run_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config) {
    return run_recipe_segments({{q, k, v}}, policy, program_config).front();
}

std::tuple<Tensor, Tensor> run_joint_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& joint_q,
    const Tensor& joint_k,
    const Tensor& joint_v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config) {
    auto outputs = run_recipe_segments({{q, k, v}, {joint_q, joint_k, joint_v}}, policy, program_config);
    return {outputs[0], outputs[1]};
}

}  // namespace ttnn::operations::transformer::sdpa::detail
