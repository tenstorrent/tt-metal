// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_device_operation.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert {

namespace {
bool dram_interleaved(const Tensor& t) {
    const auto& mc = t.memory_config();
    return mc.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mc.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}
}  // namespace

void FlatRoutedExpertDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& cfg, const tensor_args_t& t) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;
    TT_FATAL(
        t.x.layout() == Layout::ROW_MAJOR && t.x.dtype() == DataType::BFLOAT16 && dram_interleaved(t.x),
        "flat_routed_expert: x must be a row-major bf16 DRAM interleaved dispatch buffer");
    const uint32_t xw = t.x.logical_shape()[-1];
    TT_FATAL(
        cfg.hidden % xw == 0 && (cfg.hidden / xw) * xw * 2 >= 1024 && ((xw * 2) % 1024 == 0 || cfg.hidden == xw),
        "flat_routed_expert: x width {} must be hidden {} / pages per row, pages a multiple of 1 KB",
        xw,
        cfg.hidden);
    for (const Tensor* r : {&t.counts, &t.regions}) {
        TT_FATAL(
            r->layout() == Layout::ROW_MAJOR && (r->dtype() == DataType::UINT32 || r->dtype() == DataType::INT32) &&
                dram_interleaved(*r) && r->logical_volume() == cfg.num_global_experts &&
                r->logical_shape()[-1] == cfg.num_global_experts,
            "flat_routed_expert: counts / regions must be [1, {}] uint32 row-major DRAM interleaved rows",
            cfg.num_global_experts);
    }
    TT_FATAL(
        t.global_expert_ids.layout() == Layout::ROW_MAJOR &&
            t.global_expert_ids.logical_volume() == cfg.experts_per_chip && dram_interleaved(t.global_expert_ids),
        "flat_routed_expert: global_expert_ids must be [{}] uint32 row-major DRAM interleaved",
        cfg.experts_per_chip);
    if (t.token_index) {
        TT_FATAL(
            t.token_index->layout() == Layout::ROW_MAJOR &&
                (t.token_index->dtype() == DataType::UINT32 || t.token_index->dtype() == DataType::INT32) &&
                dram_interleaved(*t.token_index) &&
                t.token_index->logical_volume() == t.token_index->logical_shape()[-1],
            "flat_routed_expert: token_index must be a [1, rows] uint32 row-major DRAM interleaved row");
    }
    const uint32_t rows =
        t.token_index ? t.token_index->logical_shape()[-1] : t.x.logical_shape()[-2] / (cfg.hidden / xw);
    const bool out_ok = cfg.y_row_major
                            ? t.output.layout() == Layout::ROW_MAJOR && t.output.dtype() == DataType::BFLOAT16
                            : t.output.layout() == Layout::TILE && t.output.dtype() == DataType::BFLOAT8_B;
    TT_FATAL(
        out_ok && dram_interleaved(t.output) && t.output.logical_shape()[-2] == rows &&
            t.output.logical_shape()[-1] == cfg.hidden,
        "flat_routed_expert: output must be a {} DRAM tensor with the flat (dispatch buffer / token_index) rows",
        cfg.y_row_major ? "row-major bf16" : "bfp8 TILE");
}

void FlatRoutedExpertDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& cfg, const tensor_args_t& t) {
    TT_FATAL(cfg.hidden % t.x.logical_shape()[-1] == 0, "flat_routed_expert: x width");
}

FlatRoutedExpertDeviceOperation::spec_return_value_t FlatRoutedExpertDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.output.tensor_spec();
}

FlatRoutedExpertDeviceOperation::tensor_return_value_t FlatRoutedExpertDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.output;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert

namespace ttnn::prim {

ttnn::Tensor flat_routed_expert(
    const ttnn::operations::experimental::deepseek_prefill::flat_routed_expert::FlatRoutedExpertConfig& config,
    const ttnn::operations::experimental::deepseek_prefill::flat_routed_expert::FlatRoutedExpertInputs& inputs) {
    using Op = ttnn::operations::experimental::deepseek_prefill::flat_routed_expert::FlatRoutedExpertDeviceOperation;
    return ttnn::device_operation::launch<Op>(config, inputs);
}

}  // namespace ttnn::prim
