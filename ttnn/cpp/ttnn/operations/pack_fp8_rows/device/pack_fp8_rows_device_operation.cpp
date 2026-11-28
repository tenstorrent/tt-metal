// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pack_fp8_rows_device_operation.hpp"

#include <string_view>
#include <utility>

#include <fmt/format.h>

#include "kernels/pack_fp8_rows_args.hpp"

namespace ttnn::operations::pack_fp8_rows {

using namespace tt::tt_metal;

namespace {

// The message is built only on a refusal: the public entry runs these checks on every call.
template <typename MakeMessage>
void refuse_unless(bool ok, MakeMessage&& make_message) {
    if (!ok) {
        throw PackFp8RowsAssertion(std::forward<MakeMessage>(make_message)());
    }
}

void check_interleaved_dram(const MemoryConfig& mc, std::string_view what) {
    refuse_unless(mc.buffer_type() == BufferType::DRAM && mc.memory_layout() == TensorMemoryLayout::INTERLEAVED, [&] {
        return fmt::format("pack_fp8_rows: {} must be DRAM interleaved, got {}", what, mc);
    });
}

ttnn::Shape packed_shape(const Tensor& e4m3) {
    ttnn::Shape shape = e4m3.logical_shape();
    shape[-1] = ::pack_fp8_rows::ROW_BYTES;
    return shape;
}

bool leading_dims_equal(const ttnn::Shape& a, const ttnn::Shape& b) {
    if (a.rank() != b.rank()) {
        return false;
    }
    for (size_t i = 0; i + 1 < a.rank(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

void check_support(const PackFp8RowsParams& attrs, const PackFp8RowsInputs& t) {
    refuse_unless(t.e4m3.dtype() == DataType::FP8_E4M3 && t.e4m3.layout() == Layout::ROW_MAJOR, [&] {
        return fmt::format(
            "pack_fp8_rows: e4m3 must be FP8_E4M3 ROW_MAJOR, got {} {}", t.e4m3.dtype(), t.e4m3.layout());
    });
    refuse_unless(t.scale.dtype() == DataType::FLOAT32 && t.scale.layout() == Layout::ROW_MAJOR, [&] {
        return fmt::format(
            "pack_fp8_rows: scale must be FLOAT32 ROW_MAJOR, got {} {}", t.scale.dtype(), t.scale.layout());
    });
    const auto& es = t.e4m3.logical_shape();
    const auto& ss = t.scale.logical_shape();
    refuse_unless(es[-1] == ::pack_fp8_rows::VALUES && ss[-1] == 1, [&] {
        return fmt::format(
            "pack_fp8_rows: values [.., M, {}] and one scale a row, got {} / {}", ::pack_fp8_rows::VALUES, es, ss);
    });
    refuse_unless(leading_dims_equal(es, ss), [&] { return fmt::format("({}, {})", es, ss); });
    check_interleaved_dram(t.e4m3.memory_config(), "e4m3");
    check_interleaved_dram(t.scale.memory_config(), "scale");
    if (t.output.has_value()) {
        const Tensor& o = *t.output;
        const ttnn::Shape shape = packed_shape(t.e4m3);
        refuse_unless(
            o.dtype() == DataType::UINT8 && o.layout() == Layout::ROW_MAJOR && o.logical_shape() == shape, [&] {
                return fmt::format(
                    "pack_fp8_rows: output_tensor must be uint8 ROW_MAJOR {}, got {} {} {}",
                    shape,
                    o.dtype(),
                    o.layout(),
                    o.logical_shape());
            });
    }
    check_interleaved_dram(attrs.output_memory_config, "output");
}

PackFp8RowsDeviceOperation::program_factory_t PackFp8RowsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return PackFp8RowsProgramFactory{};
}

void PackFp8RowsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // The public entry already checked; a direct prim caller has not.
    check_support(attrs, tensor_args);
}

void PackFp8RowsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

ttsl::hash::hash_t PackFp8RowsDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // The output spec is the packed shape of e4m3 in attrs.output_memory_config (the caller's output is refused
    // otherwise), so the key needs neither the output nor whether it was given.
    return tt::tt_metal::operation::hash_operation<PackFp8RowsDeviceOperation>(
        attrs.output_memory_config,
        attrs.grid_x,
        attrs.grid_y,
        tensor_args.e4m3.tensor_spec(),
        tensor_args.scale.tensor_spec());
}

PackFp8RowsDeviceOperation::spec_return_value_t PackFp8RowsDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->tensor_spec();
    }
    return TensorSpec(
        packed_shape(tensor_args.e4m3),
        TensorLayout(DataType::UINT8, PageConfig(Layout::ROW_MAJOR), attrs.output_memory_config));
}

PackFp8RowsDeviceOperation::tensor_return_value_t PackFp8RowsDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return *tensor_args.output;
    }
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.e4m3.device());
}

}  // namespace ttnn::operations::pack_fp8_rows

namespace ttnn::prim {

Tensor pack_fp8_rows(
    const Tensor& e4m3,
    const Tensor& scale,
    const std::optional<Tensor>& output,
    const ttnn::operations::pack_fp8_rows::PackFp8RowsParams& params) {
    using OperationType = ttnn::operations::pack_fp8_rows::PackFp8RowsDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.e4m3 = e4m3, .scale = scale, .output = output});
}

}  // namespace ttnn::prim
