// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ag_route_plan_device_operation.hpp"

#include <algorithm>

#include "moe_ag_common.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;
using namespace ttnn::operations::experimental::deepseek_prefill::moe_ag;

namespace ttnn::prim {

void MoeAgRoutePlanDeviceOperation::validate_on_program_cache_miss(
    const MoeAgRoutePlanParams& args, const MoeAgRoutePlanInputs& tensor_args) {
    constexpr const char* op = "moe_ag_route_plan";
    const auto& idx = tensor_args.topk_indices;
    const auto& lmap = tensor_args.local_slot_map;
    check_row_major(idx, DataType::UINT16, op, "topk_indices");
    check_row_major(lmap, DataType::UINT32, op, "local_slot_map");
    check_single_row(lmap, op, "local_slot_map");
    const uint32_t K = idx.logical_shape()[-1], T = rm_rows(idx), NG = lmap.logical_shape()[-1];
    TT_FATAL(K >= 1 && K <= MOE_AG_ROUTE_PLAN_MAX_K, "{}: top-k {} must be in [1, {}]", op, K, MOE_AG_ROUTE_PLAN_MAX_K);
    TT_FATAL(T >= 1, "{}: no tokens", op);
    TT_FATAL(
        args.experts_per_chip >= 1 && args.experts_per_chip <= MOE_AG_ROUTE_PLAN_CORES,
        "{}: experts_per_chip {} must be in [1, {}] (one expert per plan core)",
        op,
        args.experts_per_chip,
        MOE_AG_ROUTE_PLAN_CORES);
    TT_FATAL(args.experts_per_chip <= NG, "{}: experts_per_chip {} > global experts {}", op, args.experts_per_chip, NG);
    TT_FATAL(NG <= 0xFFFF, "{}: global expert ids must fit uint16, got {}", op, NG);
    TT_FATAL(
        args.num_rows >= 32 && args.num_rows % 32 == 0,
        "{}: num_rows {} must be a positive multiple of 32",
        op,
        args.num_rows);
    // Worst case of the flat space: every token's min(K, EPC) pairs local, each active expert's region 32-padded
    // (sum_e ceil(c_e / 32) <= ceil(P / 32) + n - 1). The plan never drops a pair, so a smaller space would let
    // adversarial routing overrun token_index / y.
    const uint32_t pairs = T * std::min(K, args.experts_per_chip);
    const uint32_t worst = (pairs + 31) / 32 * 32 + 32 * (std::min(pairs, args.experts_per_chip) - 1);
    TT_FATAL(
        args.num_rows >= worst,
        "{}: num_rows {} < the worst-case flat rows {} (tokens {} x min(top-k {}, experts_per_chip {}) + region "
        "padding)",
        op,
        args.num_rows,
        worst,
        T,
        K,
        args.experts_per_chip);
    const auto grid = idx.device()->compute_with_storage_grid_size();
    TT_FATAL(grid.x >= 8 && grid.y >= 8, "{}: needs an 8 x 8 worker grid, got {}", op, grid);
    const auto& outs = tensor_args.preallocated_outputs;
    TT_FATAL(outs.empty() || outs.size() == 4, "{}: preallocated_outputs must be empty or 4 tensors", op);
    if (!outs.empty()) {
        const auto specs = compute_output_specs(args, tensor_args);
        const char* names[] = {"counts", "regions", "token_index", "y_slot"};
        for (size_t i = 0; i < 4; ++i) {
            check_row_major(outs[i], DataType::UINT32, op, names[i]);
            TT_FATAL(
                outs[i].logical_shape() == specs[i].logical_shape(),
                "{}: preallocated {} shape {} != expected {}",
                op,
                names[i],
                outs[i].logical_shape(),
                specs[i].logical_shape());
        }
    }
}

std::vector<TensorSpec> MoeAgRoutePlanDeviceOperation::compute_output_specs(
    const MoeAgRoutePlanParams& args, const MoeAgRoutePlanInputs& tensor_args) {
    const auto& idx = tensor_args.topk_indices;
    const uint32_t NG = tensor_args.local_slot_map.logical_shape()[-1];
    const uint32_t TK = rm_rows(idx) * idx.logical_shape()[-1];
    auto spec = [](uint32_t n) {
        return TensorSpec(
            ttnn::Shape({1, n}), TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), DRAM_MEMORY_CONFIG));
    };
    return {spec(NG), spec(NG), spec(args.num_rows), spec(TK)};
}

std::vector<Tensor> MoeAgRoutePlanDeviceOperation::create_output_tensors(
    const MoeAgRoutePlanParams& args, const MoeAgRoutePlanInputs& tensor_args) {
    if (!tensor_args.preallocated_outputs.empty()) {
        return tensor_args.preallocated_outputs;
    }
    std::vector<Tensor> out;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        out.push_back(create_device_tensor(spec, tensor_args.topk_indices.device()));
    }
    return out;
}

std::vector<Tensor> moe_ag_route_plan(
    const Tensor& topk_indices,
    const Tensor& local_slot_map,
    uint32_t experts_per_chip,
    uint32_t num_rows,
    const std::vector<Tensor>& preallocated_outputs) {
    using OperationType = MoeAgRoutePlanDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.experts_per_chip = experts_per_chip, .num_rows = num_rows},
        OperationType::tensor_args_t{
            .topk_indices = topk_indices,
            .local_slot_map = local_slot_map,
            .preallocated_outputs = preallocated_outputs});
}

}  // namespace ttnn::prim
