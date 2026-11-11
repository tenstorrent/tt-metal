// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "engram_gate_prefill_device_operation.hpp"

#include <bit>
#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>

#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::engram_gate_prefill {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

void check_tile_operand(const Tensor& t, const char* name, DataType dtype) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "engram_gate_prefill: {} must be on device", name);
    TT_FATAL(t.dtype() == dtype, "engram_gate_prefill: {} must be {}, got {}", name, dtype, t.dtype());
    TT_FATAL(t.layout() == Layout::TILE, "engram_gate_prefill: {} must be TILE layout", name);
    const auto& tile = t.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == TILE_HEIGHT && tile.get_width() == TILE_WIDTH,
        "engram_gate_prefill: {} must use 32x32 tiles",
        name);
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "engram_gate_prefill: {} must be interleaved (the program streams it from DRAM)",
        name);
}

void validate(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& streams = tensor_args.streams;
    const auto& kv = tensor_args.kv;
    const auto& gate = tensor_args.gate_weight;
    check_tile_operand(streams, "streams", DataType::BFLOAT16);
    check_tile_operand(kv, "kv", DataType::BFLOAT16);
    check_tile_operand(gate, "gate_weight", DataType::FLOAT32);

    const auto& shape = streams.logical_shape();
    TT_FATAL(shape.rank() == 4 && shape[0] == 1, "engram_gate_prefill: streams must be [1, hc, T, D], got {}", shape);
    const uint32_t hc = shape[1], tokens = shape[2], d = shape[3];
    TT_FATAL(
        tokens % TILE_HEIGHT == 0,
        "engram_gate_prefill: T must be a multiple of {} (a block is one tile row), got {}",
        TILE_HEIGHT,
        tokens);
    TT_FATAL(d % TILE_WIDTH == 0, "engram_gate_prefill: D ({}) must be a multiple of {}", d, TILE_WIDTH);
    // Every copy needs its own gate page, and a block's 32 rows x hc copies must fill one tile row per stream.
    TT_FATAL(
        hc <= TILE_HEIGHT, "engram_gate_prefill: hc {} must be at most {} (a block is one tile row)", hc, TILE_HEIGHT);

    const auto& kv_shape = kv.logical_shape();
    TT_FATAL(
        kv_shape.rank() == 4 && kv_shape[0] == 1 && kv_shape[1] == 1 && kv_shape[2] == tokens &&
            kv_shape[3] == (hc + 1) * d,
        "engram_gate_prefill: kv must be [1, 1, {}, {}], got {}",
        tokens,
        (hc + 1) * d,
        kv_shape);

    // One full-D gate vector per copy, each repeated down a tile's 32 rows: [1, hc, 32, D]. Repeating is what lets
    // the kernel treat the gate as a plain tile-wise operand against a 32-token block.
    const auto& gate_shape = gate.logical_shape();
    TT_FATAL(
        gate_shape.rank() == 4 && gate_shape[0] == 1 && gate_shape[1] == hc && gate_shape[2] == TILE_HEIGHT &&
            gate_shape[3] == d,
        "engram_gate_prefill: gate_weight must be [1, {}, {}, {}], got {}",
        hc,
        TILE_HEIGHT,
        d,
        gate_shape);

    TT_FATAL(
        args.output_mem_config == streams.memory_config(),
        "engram_gate_prefill: output memory config must equal the streams memory config");

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(streams.device()->arch(), args.compute_kernel_config);
    TT_FATAL(fp32_dest_acc_en, "engram_gate_prefill: needs fp32_dest_acc_en (its tiles are fp32)");
}

}  // namespace

EngramGatePrefillDeviceOperation::program_factory_t EngramGatePrefillDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return EngramGatePrefillProgramFactory{};
}

void EngramGatePrefillDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate(args, tensor_args);
}

void EngramGatePrefillDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate(args, tensor_args);
}

EngramGatePrefillDeviceOperation::spec_return_value_t EngramGatePrefillDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& streams = tensor_args.streams;
    return TensorSpec(
        streams.logical_shape(),
        TensorLayout(streams.dtype(), PageConfig(Layout::TILE, streams.tensor_spec().tile()), args.output_mem_config));
}

EngramGatePrefillDeviceOperation::tensor_return_value_t EngramGatePrefillDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.streams.device());
}

}  // namespace ttnn::operations::experimental::deepseek::engram_gate_prefill

namespace ttnn::prim {

ttnn::Tensor engram_gate_prefill(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& gate_weight,
    float eps,
    float clamp_value,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType =
        ttnn::operations::experimental::deepseek::engram_gate_prefill::EngramGatePrefillDeviceOperation;

    const auto kernel_config_val = init_device_compute_kernel_config(
        streams.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    OperationType::operation_attributes_t attrs{
        .eps = eps,
        .clamp_value = clamp_value,
        .output_mem_config = memory_config.value_or(streams.memory_config()),
        .compute_kernel_config = kernel_config_val,
    };
    OperationType::tensor_args_t tensor_args{.streams = streams, .kv = kv, .gate_weight = gate_weight};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
