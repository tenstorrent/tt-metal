// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ag_local_reduce_device_operation.hpp"

#include "moe_ag_common.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;
using namespace ttnn::operations::experimental::deepseek_prefill::moe_ag;

namespace ttnn::prim {

void MoeAgLocalReduceDeviceOperation::validate_on_program_cache_miss(
    const MoeAgLocalReduceParams& args, const MoeAgLocalReduceInputs& t) {
    constexpr const char* op = "moe_ag_local_reduce";
    check_row_major(t.y, DataType::BFLOAT16, op, "y");
    check_row_major(t.y_slot, DataType::UINT32, op, "y_slot");
    check_single_row(t.y_slot, op, "y_slot");
    check_row_major(t.weights, DataType::BFLOAT16, op, "weights");
    check_row_major(t.chip_info, DataType::UINT32, op, "chip_info");
    TT_FATAL(t.chip_info.logical_shape()[-1] == 16, "{}: chip_info must be [.., 1, 16] uint32", op);
    const uint32_t H = t.y.logical_shape()[-1], K = t.weights.logical_shape()[-1], T = rm_rows(t.weights);
    const uint32_t S = args.chunk_size_per_chip;
    TT_FATAL(H % 32 == 0, "{}: hidden {} must be a multiple of 32", op, H);
    TT_FATAL(
        K >= 1 && K <= MOE_AG_LOCAL_REDUCE_MAX_K, "{}: top-k {} must be in [1, {}]", op, K, MOE_AG_LOCAL_REDUCE_MAX_K);
    TT_FATAL(
        t.y_slot.logical_shape()[-1] == T * K,
        "{}: y_slot length {} != tokens {} x top-k {}",
        op,
        t.y_slot.logical_shape()[-1],
        T,
        K);
    TT_FATAL(S >= 1 && T % S == 0, "{}: tokens {} must be a multiple of chunk_size_per_chip {}", op, T, S);
    TT_FATAL(
        (S * K) % 16 == 0, "{}: a mesh row's y_slot block (S {} x top-k {} uint32) must be 64 B aligned", op, S, K);
    TT_FATAL(args.phase <= 2, "{}: phase must be 0, 1 or 2", op);
    TT_FATAL(args.pairs_depth >= 1, "{}: pairs_depth must be >= 1", op);
    if (args.phase == 0) {
        TT_FATAL(!(args.split && args.tiled), "{}: split and tiled are exclusive", op);
        TT_FATAL(!args.split || T == 2 * S, "{}: split needs two mesh rows (tokens {} == 2 x {})", op, T, S);
        TT_FATAL(!args.tiled || T % 32 == 0, "{}: tiled partials need tokens {} % 32 == 0", op, T);
        TT_FATAL(!t.peer.has_value(), "{}: peer is only used by phase 2", op);
    } else {
        TT_FATAL(!args.split, "{}: phases 1 / 2 write one [S, H] partial", op);
        TT_FATAL(
            !args.tiled || (args.phase == 2 && S % 32 == 0),
            "{}: only phase 2 may write tiles (phase 1's partial is gathered row-major; S {} % 32 == 0)",
            op,
            S);
        TT_FATAL(T == 2 * S, "{}: phases 1 / 2 need two mesh rows (tokens {} == 2 x {})", op, T, S);
        TT_FATAL(t.peer.has_value() == (args.phase == 2), "{}: peer is required by (only) phase 2", op);
        if (t.peer.has_value()) {
            check_row_major(*t.peer, DataType::BFLOAT16, op, "peer");
            TT_FATAL(
                t.peer->logical_shape()[-1] == H && rm_rows(*t.peer) == 2 * S,
                "{}: peer must be [.., 2 S, H] = [{}, {}], got {}",
                op,
                2 * S,
                H,
                t.peer->logical_shape());
        }
    }
    const auto& outs = t.preallocated_outputs;
    const auto specs = compute_output_specs(args, t);
    TT_FATAL(
        outs.empty() || outs.size() == specs.size(), "{}: preallocated_outputs must be empty or {}", op, specs.size());
    for (size_t i = 0; i < outs.size(); ++i) {
        check_dram_interleaved(outs[i], op, "output");
        TT_FATAL(
            outs[i].logical_shape() == specs[i].logical_shape() && outs[i].layout() == specs[i].layout() &&
                outs[i].dtype() == DataType::BFLOAT16,
            "{}: preallocated output {} must be bf16 {} {}",
            op,
            i,
            specs[i].logical_shape(),
            specs[i].layout());
    }
}

std::vector<TensorSpec> MoeAgLocalReduceDeviceOperation::compute_output_specs(
    const MoeAgLocalReduceParams& args, const MoeAgLocalReduceInputs& t) {
    const uint32_t H = t.y.logical_shape()[-1], T = rm_rows(t.weights), S = args.chunk_size_per_chip;
    auto spec = [&](uint32_t rows, Layout layout) {
        return TensorSpec(
            ttnn::Shape({1, 1, rows, H}), TensorLayout(DataType::BFLOAT16, PageConfig(layout), DRAM_MEMORY_CONFIG));
    };
    if (args.phase != 0) {
        return {spec(S, args.phase == 2 && args.tiled ? Layout::TILE : Layout::ROW_MAJOR)};
    }
    if (args.split) {
        return {spec(S, Layout::ROW_MAJOR), spec(S, Layout::ROW_MAJOR)};
    }
    return {spec(T, args.tiled ? Layout::TILE : Layout::ROW_MAJOR)};
}

std::vector<Tensor> MoeAgLocalReduceDeviceOperation::create_output_tensors(
    const MoeAgLocalReduceParams& args, const MoeAgLocalReduceInputs& t) {
    if (!t.preallocated_outputs.empty()) {
        return t.preallocated_outputs;
    }
    std::vector<Tensor> out;
    for (const auto& spec : compute_output_specs(args, t)) {
        out.push_back(create_device_tensor(spec, t.y.device()));
    }
    return out;
}

std::vector<Tensor> moe_ag_local_reduce(
    const Tensor& y,
    const Tensor& y_slot,
    const Tensor& weights,
    const Tensor& chip_info,
    uint32_t chunk_size_per_chip,
    uint32_t phase,
    bool split,
    bool tiled,
    const std::optional<Tensor>& peer,
    const std::vector<Tensor>& preallocated_outputs) {
    using OperationType = MoeAgLocalReduceDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .phase = phase, .chunk_size_per_chip = chunk_size_per_chip, .split = split, .tiled = tiled},
        OperationType::tensor_args_t{
            .y = y,
            .y_slot = y_slot,
            .weights = weights,
            .chip_info = chip_info,
            .peer = peer,
            .preallocated_outputs = preallocated_outputs});
}

}  // namespace ttnn::prim
