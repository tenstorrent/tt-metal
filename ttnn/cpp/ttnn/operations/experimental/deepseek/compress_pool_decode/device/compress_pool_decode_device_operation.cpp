// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compress_pool_decode_device_operation.hpp"

#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::compress_pool_decode {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

void validate_row_input(const Tensor& t, const char* name, uint32_t head_dim) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "compress_pool_decode: {} must be on device", name);
    TT_FATAL(
        t.layout() == Layout::ROW_MAJOR && t.dtype() == DataType::BFLOAT16,
        "compress_pool_decode: {} must be ROW_MAJOR BFLOAT16",
        name);
    const auto& shape = t.logical_shape();
    TT_FATAL(
        shape[-1] == head_dim && shape.volume() == head_dim,
        "compress_pool_decode: {} must have shape [1, 1, 1, {}] (single row), got {}",
        name,
        head_dim,
        shape);
    if (t.is_sharded()) {
        TT_FATAL(
            t.buffer()->buffer_type() == BufferType::L1 && t.memory_config().shard_spec().has_value(),
            "compress_pool_decode: sharded {} must be in L1 with a shard_spec",
            name);
        const auto layout = t.memory_config().memory_layout();
        const auto& shard = t.memory_config().shard_spec().value();
        TT_FATAL(
            layout == TensorMemoryLayout::WIDTH_SHARDED ||
                (layout == TensorMemoryLayout::HEIGHT_SHARDED && shard.grid.num_cores() == 1),
            "compress_pool_decode: sharded {} must be WIDTH_SHARDED (or single-core HEIGHT_SHARDED)",
            name);
        TT_FATAL(
            shard.shape[0] == 1 && shard.shape[1] % TILE_WIDTH == 0 &&
                shard.grid.num_cores() * shard.shape[1] == head_dim,
            "compress_pool_decode: sharded {} must have shard [1, w] with w % 32 == 0 covering {}, got {} on {} cores",
            name,
            head_dim,
            shard.shape,
            shard.grid.num_cores());
    } else {
        TT_FATAL(
            t.buffer()->buffer_type() == BufferType::DRAM || t.buffer()->buffer_type() == BufferType::L1,
            "compress_pool_decode: {} must be in DRAM or L1",
            name);
    }
}

void validate_slot_tensor(const Tensor& t, const char* name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "compress_pool_decode: {} must be on device", name);
    TT_FATAL(
        t.dtype() == DataType::INT32 && t.layout() == Layout::ROW_MAJOR && !t.is_sharded() &&
            t.logical_shape().volume() >= 1,
        "compress_pool_decode: {} must be interleaved ROW_MAJOR INT32 with at least 1 element, got {} {} {}",
        name,
        t.dtype(),
        t.layout(),
        t.logical_shape());
}

void validate_window_cache(const Tensor& t, const char* name, uint32_t head_dim) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "compress_pool_decode: {} must be on device", name);
    TT_FATAL(
        t.layout() == Layout::TILE && t.dtype() == DataType::BFLOAT16 && !t.is_sharded() &&
            t.buffer()->buffer_type() == BufferType::DRAM && t.tensor_spec().tile().get_height() == TILE_HEIGHT &&
            t.tensor_spec().tile().get_width() == TILE_WIDTH,
        "compress_pool_decode: {} must be DRAM-interleaved bfloat16 32x32 TILE",
        name);
    const auto& shape = t.logical_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[0] == 1 && shape[1] == 1 && shape[2] >= 1 && shape[3] == head_dim,
        "compress_pool_decode: {} must have shape [1, 1, rows, {}], got {}",
        name,
        head_dim,
        shape);
}

}  // namespace

CompressPoolDecodeDeviceOperation::program_factory_t CompressPoolDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return CompressPoolDecodeProgramFactory{};
}

void CompressPoolDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    auto* device = t.kv.device();
    TT_FATAL(device->arch() == tt::ARCH::BLACKHOLE, "compress_pool_decode: Blackhole-only op");
    const uint32_t head_dim = args.head_dim;
    TT_FATAL(
        head_dim > 0 && head_dim % TILE_WIDTH == 0,
        "compress_pool_decode: head_dim ({}) must be a positive multiple of {}",
        head_dim,
        TILE_WIDTH);
    const uint32_t num_cores = head_dim / TILE_WIDTH;
    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        num_cores <= grid.x * grid.y,
        "compress_pool_decode: needs {} worker cores, device grid has {}",
        num_cores,
        grid.x * grid.y);

    validate_row_input(t.kv, "kv", head_dim);

    TT_FATAL(
        t.norm_weight.storage_type() == StorageType::DEVICE, "compress_pool_decode: norm_weight must be on device");
    TT_FATAL(
        t.norm_weight.dtype() == DataType::BFLOAT16 && !t.norm_weight.is_sharded() &&
            t.norm_weight.buffer()->buffer_type() == BufferType::DRAM &&
            t.norm_weight.logical_shape()[-1] == head_dim && t.norm_weight.logical_shape().volume() == head_dim,
        "compress_pool_decode: norm_weight must be DRAM-interleaved bfloat16 [1, 1, 1, {}], got {}",
        head_dim,
        t.norm_weight.logical_shape());
    if (t.norm_weight.layout() == Layout::TILE) {
        TT_FATAL(
            t.norm_weight.tensor_spec().tile().get_height() == TILE_HEIGHT &&
                t.norm_weight.tensor_spec().tile().get_width() == TILE_WIDTH,
            "compress_pool_decode: tiled norm_weight must use 32x32 tiles");
    } else {
        TT_FATAL(
            t.norm_weight.layout() == Layout::ROW_MAJOR, "compress_pool_decode: norm_weight must be TILE or ROW_MAJOR");
    }

    const bool any_window =
        t.gate.has_value() || t.win_kv.has_value() || t.win_gate.has_value() || t.win_slot.has_value();
    const bool all_window =
        t.gate.has_value() && t.win_kv.has_value() && t.win_gate.has_value() && t.win_slot.has_value();
    TT_FATAL(
        any_window == all_window,
        "compress_pool_decode: gate, win_kv, win_gate, and win_slot must all be provided or all omitted");
    if (all_window) {
        validate_row_input(*t.gate, "gate", head_dim);
        validate_window_cache(*t.win_kv, "win_kv", head_dim);
        validate_window_cache(*t.win_gate, "win_gate", head_dim);
        TT_FATAL(
            t.win_kv->logical_shape() == t.win_gate->logical_shape(),
            "compress_pool_decode: win_kv and win_gate must have the same shape");
        validate_slot_tensor(*t.win_slot, "win_slot");
        if (t.prev_slot.has_value()) {
            validate_slot_tensor(*t.prev_slot, "prev_slot");
        }
    } else {
        TT_FATAL(!t.prev_slot.has_value(), "compress_pool_decode: prev_slot requires window caches");
        TT_FATAL(args.do_pool, "compress_pool_decode: do_pool=false requires window caches");
    }

    TT_FATAL(
        t.cos.has_value() == t.sin.has_value(),
        "compress_pool_decode: cos and sin must both be provided or both omitted");
    if (t.cos.has_value()) {
        for (const auto& [tensor, name] : {std::pair{&(*t.cos), "cos"}, std::pair{&(*t.sin), "sin"}}) {
            TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "compress_pool_decode: {} must be on device", name);
            TT_FATAL(
                tensor->layout() == Layout::TILE && tensor->dtype() == DataType::BFLOAT16 && !tensor->is_sharded() &&
                    tensor->buffer()->buffer_type() == BufferType::DRAM &&
                    tensor->tensor_spec().tile().get_height() == TILE_HEIGHT &&
                    tensor->tensor_spec().tile().get_width() == TILE_WIDTH,
                "compress_pool_decode: {} must be DRAM-interleaved bfloat16 32x32 TILE",
                name);
            const auto& s = tensor->logical_shape();
            TT_FATAL(
                s.rank() == 4 && s[0] == 1 && s[1] == 1 && s[2] >= 1 && s[3] == args.rope_dim,
                "compress_pool_decode: {} must be [1, 1, >=1, {}], got {}",
                name,
                args.rope_dim,
                s);
        }
        TT_FATAL(
            args.rope_dim > 0 && args.rope_dim % TILE_WIDTH == 0 && args.rope_dim <= head_dim,
            "compress_pool_decode: rope_dim ({}) must be a multiple of {} in (0, {}]",
            args.rope_dim,
            TILE_WIDTH,
            head_dim);
    }
}

CompressPoolDecodeDeviceOperation::spec_return_value_t CompressPoolDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t&) {
    if (!args.do_pool) {
        return {};
    }
    const MemoryConfig dram_mem(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const ttnn::Shape out_shape({1, 1, 1, args.head_dim});
    std::vector<TensorSpec> specs;
    specs.reserve(args.has_rope ? 2 : 1);
    specs.emplace_back(out_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), dram_mem));
    if (args.has_rope) {
        specs.emplace_back(out_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), dram_mem));
    }
    return specs;
}

CompressPoolDecodeDeviceOperation::tensor_return_value_t CompressPoolDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& t) {
    auto specs = compute_output_specs(args, t);
    std::vector<Tensor> outputs;
    outputs.reserve(specs.size());
    for (const auto& spec : specs) {
        outputs.push_back(ttnn::create_device_tensor(spec, t.kv.device()));
    }
    return outputs;
}

}  // namespace ttnn::operations::experimental::deepseek::compress_pool_decode

namespace ttnn::prim {

std::vector<ttnn::Tensor> compress_pool_decode(
    const ttnn::Tensor& kv,
    const ttnn::Tensor& norm_weight,
    float epsilon,
    const std::optional<ttnn::Tensor>& gate,
    const std::optional<ttnn::Tensor>& win_kv,
    const std::optional<ttnn::Tensor>& win_gate,
    const std::optional<ttnn::Tensor>& win_slot,
    const std::optional<ttnn::Tensor>& prev_slot,
    const std::optional<ttnn::Tensor>& cos,
    const std::optional<ttnn::Tensor>& sin,
    bool do_pool) {
    using OperationType =
        ttnn::operations::experimental::deepseek::compress_pool_decode::CompressPoolDecodeDeviceOperation;
    const uint32_t head_dim = kv.logical_shape()[-1];
    const bool has_window = win_kv.has_value();
    const bool has_rope = cos.has_value();
    const uint32_t rope_dim = has_rope ? cos->logical_shape()[-1] : 0u;
    auto attrs = OperationType::operation_attributes_t{
        .epsilon = epsilon,
        .do_pool = do_pool,
        .has_window = has_window,
        .has_prev_slot = prev_slot.has_value(),
        .has_rope = has_rope,
        .head_dim = head_dim,
        .rope_dim = rope_dim,
        .kv_sharded = kv.is_sharded(),
        .gate_sharded = gate.has_value() && gate->is_sharded(),
        .norm_weight_is_tile = norm_weight.layout() == tt::tt_metal::Layout::TILE,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .kv = kv,
        .norm_weight = norm_weight,
        .gate = gate,
        .win_kv = win_kv,
        .win_gate = win_gate,
        .win_slot = win_slot,
        .prev_slot = prev_slot,
        .cos = cos,
        .sin = sin,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
