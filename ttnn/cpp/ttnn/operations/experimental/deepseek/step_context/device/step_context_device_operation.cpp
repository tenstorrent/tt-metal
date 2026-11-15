// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "step_context_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::step_context {

namespace {

using namespace tt::tt_metal;

bool is_dram_interleaved(const Tensor& t) {
    return t.memory_config().buffer_type() == BufferType::DRAM &&
           t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;
}

uint32_t volume(const std::vector<uint32_t>& shape) {
    uint32_t v = 1;
    for (uint32_t d : shape) {
        v *= d;
    }
    return v;
}

void validate(const StepContextParams& attrs, const StepContextInputs& args) {
    const auto& pkt = args.packet;
    TT_FATAL(
        pkt.storage_type() == StorageType::DEVICE && pkt.buffer() != nullptr,
        "decode_step_context: packet must be on device");
    TT_FATAL(pkt.dtype() == DataType::INT32, "decode_step_context: packet must be INT32");
    TT_FATAL(pkt.layout() == Layout::ROW_MAJOR, "decode_step_context: packet must be ROW_MAJOR");
    TT_FATAL(is_dram_interleaved(pkt), "decode_step_context: packet must be DRAM interleaved");
    const auto& ps = pkt.logical_shape();
    TT_FATAL(
        ps.rank() == 4 && ps[0] == 1 && ps[1] == 1 && ps[2] == 1,
        "decode_step_context: packet must be [1, 1, 1, W], got {}",
        ps);
    const auto width = static_cast<int32_t>(ps[3]);

    const auto n_int = attrs.int_shapes.size();
    TT_FATAL(n_int >= 1, "decode_step_context: needs at least one integer output");
    TT_FATAL(
        attrs.int_dtypes.size() == n_int && attrs.int_terms.size() == n_int,
        "decode_step_context: int_shapes, int_dtypes and int_terms must have one entry per output");
    auto check_field = [&](int32_t field) {
        TT_FATAL(
            field >= -1 && field < width, "decode_step_context: packet field {} out of range [-1, {})", field, width);
    };
    for (size_t i = 0; i < n_int; ++i) {
        const uint32_t n = volume(attrs.int_shapes[i]);
        TT_FATAL(
            !attrs.int_shapes[i].empty() && n >= 1 && n <= kMaxIntElems,
            "decode_step_context: integer output {} must have 1..{} elements",
            i,
            kMaxIntElems);
        TT_FATAL(
            attrs.int_dtypes[i] == DataType::INT32 || attrs.int_dtypes[i] == DataType::UINT32,
            "decode_step_context: integer output {} must be INT32 or UINT32",
            i);
        TT_FATAL(
            attrs.int_terms[i].size() == n * kTermWords,
            "decode_step_context: integer output {} needs {} term words, got {}",
            i,
            n * kTermWords,
            attrs.int_terms[i].size());
        for (uint32_t e = 0; e < n; ++e) {
            const auto* t = &attrs.int_terms[i][e * kTermWords];
            const uint32_t w0 = static_cast<uint32_t>(t[0]);
            check_field(static_cast<int32_t>(w0 & 0xFF) - 1);
            check_field(static_cast<int32_t>((w0 >> 16) & 0xFF) - 1);
            const uint32_t mode = (w0 >> 8) & 0xF;
            TT_FATAL(mode <= 2, "decode_step_context: term mode must be 0, 1 or 2");
            TT_FATAL(mode == 0 || t[2] > 0, "decode_step_context: a dividing term needs a positive divisor");
        }
    }

    const auto n_rope = attrs.rope_tables.size();
    TT_FATAL(
        attrs.rope_fields.size() == n_rope && attrs.rope_offsets.size() == n_rope,
        "decode_step_context: rope_table_ids, rope_fields and rope_offsets must have one entry per RoPE output");
    for (size_t j = 0; j < n_rope; ++j) {
        TT_FATAL(attrs.rope_tables[j] < args.rope_tables.size(), "decode_step_context: RoPE table id out of range");
        check_field(static_cast<int32_t>(attrs.rope_fields[j]));
        TT_FATAL(attrs.rope_offsets[j] >= -1, "decode_step_context: a RoPE position offset must be >= -1");
    }
    for (const auto& table : args.rope_tables) {
        TT_FATAL(
            table.dtype() == DataType::INT32 && table.layout() == Layout::ROW_MAJOR,
            "decode_step_context: RoPE tables must be INT32 ROW_MAJOR");
        TT_FATAL(is_dram_interleaved(table), "decode_step_context: RoPE tables must be DRAM interleaved");
        const auto& s = table.logical_shape();
        TT_FATAL(
            s.rank() == 4 && s[0] == 1 && s[1] == 1 && s[2] > kRopeLoRows &&
                s[3] % (2 * tt::constants::TILE_WIDTH) == 0,
            "decode_step_context: a RoPE table must be [1, 1, H + {}, 2 * Rd] with Rd a multiple of {}, got {}",
            kRopeLoRows,
            tt::constants::TILE_WIDTH,
            s);
    }
    TT_FATAL(n_rope <= 16, "decode_step_context: at most 16 RoPE outputs");
}

}  // namespace

void StepContextDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate(attributes, tensor_args);
}

void StepContextDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {}

StepContextDeviceOperation::spec_return_value_t StepContextDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    spec_return_value_t specs;
    for (size_t i = 0; i < attributes.int_shapes.size(); ++i) {
        ttsl::SmallVector<uint32_t> dims(attributes.int_shapes[i].begin(), attributes.int_shapes[i].end());
        specs.emplace_back(
            ttnn::Shape(dims),
            TensorLayout(attributes.int_dtypes[i], PageConfig(Layout::ROW_MAJOR), attributes.output_mem_config));
    }
    for (size_t j = 0; j < attributes.rope_tables.size(); ++j) {
        const uint32_t rd = tensor_args.rope_tables[attributes.rope_tables[j]].logical_shape()[3] / 2;
        for (int k = 0; k < 3; ++k) {
            specs.emplace_back(
                ttnn::Shape({1, 1, 1, rd}),
                TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attributes.output_mem_config));
        }
    }
    return specs;
}

StepContextDeviceOperation::tensor_return_value_t StepContextDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    tensor_return_value_t outs;
    for (const auto& spec : compute_output_specs(attributes, tensor_args)) {
        outs.push_back(create_device_tensor(spec, tensor_args.packet.device()));
    }
    return outs;
}

}  // namespace ttnn::operations::experimental::deepseek::step_context

namespace ttnn::prim {

std::vector<Tensor> decode_step_context(
    const Tensor& packet,
    const std::vector<Tensor>& rope_tables,
    const std::vector<std::vector<uint32_t>>& int_shapes,
    const std::vector<DataType>& int_dtypes,
    const std::vector<std::vector<int32_t>>& int_terms,
    const std::vector<uint32_t>& rope_table_ids,
    const std::vector<uint32_t>& rope_fields,
    const std::vector<int32_t>& rope_offsets) {
    using OperationType = ttnn::operations::experimental::deepseek::step_context::StepContextDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{
        .int_shapes = int_shapes,
        .int_dtypes = int_dtypes,
        .int_terms = int_terms,
        .rope_tables = rope_table_ids,
        .rope_fields = rope_fields,
        .rope_offsets = rope_offsets,
        .output_mem_config = ttnn::DRAM_MEMORY_CONFIG,
    };
    auto tensor_args = OperationType::tensor_args_t{.packet = packet, .rope_tables = rope_tables};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
