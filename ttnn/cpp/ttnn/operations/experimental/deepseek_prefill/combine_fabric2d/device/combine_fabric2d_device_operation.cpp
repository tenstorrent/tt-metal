// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <set>
#include <string>

#include "combine_fabric2d_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace {

// Both regions are plain interleaved uint32 ROW_MAJOR DRAM tensors: one row is exactly one page is
// exactly one token, which is what lets the kernels address them by page index.
void validate_region(const ttnn::Tensor& t, uint32_t min_tokens, uint32_t token_size_bytes, const char* what) {
    TT_FATAL(t.storage_type() == tt::tt_metal::StorageType::DEVICE, "combine_fabric2d: {} must be on device", what);
    TT_FATAL(
        t.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM &&
            t.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "combine_fabric2d: {} must be an interleaved DRAM tensor",
        what);
    TT_FATAL(
        t.dtype() == tt::tt_metal::DataType::UINT32,
        "combine_fabric2d: {} must be UINT32 (the op moves raw bytes; {} would only confuse the check)",
        what,
        t.dtype());
    TT_FATAL(
        t.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "combine_fabric2d: {} must be ROW_MAJOR so one row is exactly one page (= one token)",
        what);
    const auto shape = t.logical_shape();
    TT_FATAL(shape.rank() == 2, "combine_fabric2d: {} must be rank 2 (tokens x token elements)", what);
    TT_FATAL(
        shape[-1] * sizeof(uint32_t) == token_size_bytes,
        "combine_fabric2d: {} row is {} B but token_size_bytes is {}",
        what,
        shape[-1] * sizeof(uint32_t),
        token_size_bytes);
    // A sharded mesh tensor's logical shape is its PER-DEVICE shard, so rows are directly the tokens one
    // device holds. Extra rows are harmless (the tail simply goes untouched), too few is not.
    TT_FATAL(
        shape[0] >= min_tokens,
        "combine_fabric2d: {} holds {} tokens per device but the movements need at least {}",
        what,
        shape[0],
        min_tokens);
}

}  // namespace

void CombineFabric2dDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(args.device != nullptr, "combine_fabric2d requires a mesh device in attributes");
    TT_FATAL(args.num_links >= 1 && args.num_links <= 4, "num_links must be between 1 and 4 (got {})", args.num_links);
    TT_FATAL(args.tokens_per_movement >= 1, "tokens_per_movement must be >= 1 (got {})", args.tokens_per_movement);
    TT_FATAL(
        args.num_l1_slots >= 2,
        "num_l1_slots must be >= 2 for the reader and producer to overlap (got {})",
        args.num_l1_slots);
    TT_FATAL(
        args.token_size_bytes % sizeof(uint32_t) == 0,
        "combine_fabric2d: token_size_bytes {} must be a multiple of 4",
        args.token_size_bytes);
    TT_FATAL(!args.movements.empty(), "combine_fabric2d: no movements given, so the op would do nothing");

    const auto mesh_dims = args.device->shape().dims();
    // ---- MOVEMENT-SEMANTICS-DEPENDENT BLOCK ------------------------------------------------------
    // Everything from here to the end of this function reads `in_base_token` / `out_base_token` /
    // `tokens_per_movement` as "a contiguous run of tokens". Rewrite it as one piece if the descriptor's
    // addressing changes; the PROPERTIES it enforces should survive any such change:
    //   * a coordinate of the wrong rank can never name a device;
    //   * no output token is claimed twice on the same destination — last writer wins, and which one that
    //     is depends on packet timing, so an overlapping list is a nondeterministic answer, not an answer;
    //   * every source device's input coverage is gap-free from token 0 — a hole means the caller's plan
    //     silently drops data it believes it asked to move.
    // Input OVERLAP is deliberately allowed: reads are idempotent, so several movements may legitimately
    // source the same tokens. Only gaps are an error.
    // The remaining checks (dst actually reachable, regions in range) need the placement or the buffers,
    // so they live in the program factory — which re-checks range against the real page count.
    std::map<std::vector<uint32_t>, std::set<uint32_t>> claimed_out_tokens;
    std::map<std::vector<uint32_t>, std::set<uint32_t>> claimed_in_tokens;
    for (const auto& m : args.movements) {
        TT_FATAL(
            m.src.size() == mesh_dims && m.dst.size() == mesh_dims,
            "combine_fabric2d: movement src {} / dst {} must both have {} coordinate(s) for this {} mesh",
            movement_coord_str(m.src),
            movement_coord_str(m.dst),
            mesh_dims,
            args.device->shape());
        TT_FATAL(
            m.src != m.dst,
            "combine_fabric2d: movement src and dst are both {}; a device does not send to itself",
            movement_coord_str(m.src));
        auto& claimed = claimed_out_tokens[m.dst];
        auto& sourced = claimed_in_tokens[m.src];
        for (uint32_t t = 0; t < args.tokens_per_movement; t++) {
            const auto [_, fresh] = claimed.insert(m.out_base_token + t);
            TT_FATAL(
                fresh,
                "combine_fabric2d: two movements both write output token {} of device {} (this one is src {} "
                "out_base_token {}). Overlapping destination ranges make the result depend on packet timing.",
                m.out_base_token + t,
                movement_coord_str(m.dst),
                movement_coord_str(m.src),
                m.out_base_token);
            sourced.insert(m.in_base_token + t);
        }
    }

    // Input coverage: each source device's claimed tokens must be exactly [0, N). A set whose size equals
    // its largest element + 1 is gap-free, and starting at 0 pins the base — together that rules out both
    // a hole in the middle and an off-by-one at the front.
    for (const auto& [src, sourced] : claimed_in_tokens) {
        const uint32_t lowest = *sourced.begin();
        const uint32_t highest = *sourced.rbegin();
        TT_FATAL(
            lowest == 0,
            "combine_fabric2d: device {} sources input tokens starting at {}, not 0 — tokens [0, {}) are "
            "never moved by any movement",
            movement_coord_str(src),
            lowest,
            lowest);
        TT_FATAL(
            sourced.size() == static_cast<size_t>(highest) + 1,
            "combine_fabric2d: device {} sources {} distinct input tokens but its highest is {}, so its "
            "coverage of [0, {}] has {} hole(s) — some input the caller staged is never moved",
            movement_coord_str(src),
            sourced.size(),
            highest,
            highest,
            highest + 1 - sourced.size());
    }

    // Smallest region each buffer must cover, given where the movements point.
    uint32_t min_in = 0;
    uint32_t min_out = 0;
    for (const auto& m : args.movements) {
        min_in = std::max(min_in, m.in_base_token + args.tokens_per_movement);
        min_out = std::max(min_out, m.out_base_token + args.tokens_per_movement);
    }
    validate_region(tensor_args.input, min_in, args.token_size_bytes, "input");
    validate_region(tensor_args.output, min_out, args.token_size_bytes, "output");
    // ---- END movement-semantics-dependent block --------------------------------------------------
}

void CombineFabric2dDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {}

CombineFabric2dDeviceOperation::spec_return_value_t CombineFabric2dDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // The output region is caller-owned (so a test can zero it before the run and read it back after),
    // so the op has nothing to size or allocate — it writes into the tensor it was handed.
    return tensor_args.output.tensor_spec();
}

CombineFabric2dDeviceOperation::tensor_return_value_t CombineFabric2dDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d

namespace ttnn::prim {
ttnn::Tensor combine_fabric2d(
    ttnn::MeshDevice* device,
    const ttnn::Tensor& input,
    const ttnn::Tensor& output,
    const std::vector<ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dMovement>&
        movements,
    uint32_t num_links,
    uint32_t tokens_per_movement,
    uint32_t token_size_bytes,
    uint32_t axis,
    uint32_t num_l1_slots,
    uint32_t stall_telemetry,
    tt::tt_fabric::Topology topology) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .device = device,
            .num_links = num_links,
            .tokens_per_movement = tokens_per_movement,
            .token_size_bytes = token_size_bytes,
            .axis = axis,
            .num_l1_slots = num_l1_slots,
            .stall_telemetry = stall_telemetry,
            .topology = topology,
            .movements = movements},
        OperationType::tensor_args_t{.input = input, .output = output});
}
}  // namespace ttnn::prim
