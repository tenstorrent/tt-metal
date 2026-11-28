// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unpack_fp8_rows_device_operation.hpp"

#include <algorithm>
#include <mutex>
#include <utility>

#include <fmt/format.h>

#include <tt_stl/indestructible.hpp>

#include "kernels/unpack_fp8_rows_args.hpp"

namespace ttnn::operations::unpack_fp8_rows {

using namespace tt::tt_metal;

namespace {

// The message is built only on a refusal: the public entry runs these checks on every call.
template <typename MakeMessage>
void refuse_if_not(bool ok, MakeMessage&& make_message) {
    if (!ok) {
        throw UnpackFp8RowsAssertion(std::forward<MakeMessage>(make_message)());
    }
}

struct PreparedRegistry {
    std::mutex mutex;
    std::vector<UnpackFp8RowsPreparedKey> keys;
};

PreparedRegistry& prepared_registry() {
    static ttsl::Indestructible<PreparedRegistry> registry;
    return registry.get();
}

std::vector<uint32_t> shape_dims(const ttnn::Shape& shape) {
    std::vector<uint32_t> dims;
    dims.reserve(shape.rank());
    for (size_t i = 0; i < shape.rank(); ++i) {
        dims.push_back(shape[i]);
    }
    return dims;
}

void record_prepared(const UnpackFp8RowsParams& attrs, const UnpackFp8RowsInputs& t) {
    UnpackFp8RowsPreparedKey key{
        .mesh_id = t.rows.device()->id(),
        .rows_shape = shape_dims(t.rows.logical_shape()),
        .out_shape = shape_dims(t.out.logical_shape()),
        .rows_mc = fmt::format("{}", t.rows.memory_config()),
        .out_mc = fmt::format("{}", t.out.memory_config()),
        .ncores = attrs.ncores,
        .grid_x = attrs.grid_x,
        .grid_y = attrs.grid_y,
        .common_values = attrs.common_values,
    };
    auto& registry = prepared_registry();
    const std::lock_guard lock(registry.mutex);
    if (std::find(registry.keys.begin(), registry.keys.end(), key) == registry.keys.end()) {
        registry.keys.push_back(std::move(key));
    }
}

}  // namespace

void check_support(const UnpackFp8RowsInputs& t) {
    const auto& rs = t.rows.logical_shape();
    refuse_if_not(
        t.rows.dtype() == DataType::UINT8 && t.rows.layout() == Layout::ROW_MAJOR &&
            rs[-1] == ::unpack_fp8_rows::ROW_BYTES,
        [&] {
            return fmt::format(
                "unpack_fp8_rows: rows must be uint8 ROW_MAJOR [.., {}], got {} {} {}",
                ::unpack_fp8_rows::ROW_BYTES,
                t.rows.dtype(),
                t.rows.layout(),
                rs);
        });
    const auto& os = t.out.logical_shape();
    refuse_if_not(
        t.out.dtype() == DataType::BFLOAT8_B && t.out.layout() == Layout::TILE && os[-1] == ::unpack_fp8_rows::VALUES,
        [&] {
            return fmt::format(
                "unpack_fp8_rows: out must be bfloat8_b TILE [.., {}], got {} {} {}",
                ::unpack_fp8_rows::VALUES,
                t.out.dtype(),
                t.out.layout(),
                os);
        });
    refuse_if_not(t.rows.memory_config().buffer_type() == BufferType::DRAM, [] {
        return std::string("unpack_fp8_rows: rows must be in DRAM");
    });
    const MemoryConfig& omc = t.out.memory_config();
    // (The Python's shared helper names pack_fp8_rows in this message; kept.)
    refuse_if_not(omc.buffer_type() == BufferType::DRAM && omc.memory_layout() == TensorMemoryLayout::INTERLEAVED, [&] {
        return fmt::format("pack_fp8_rows: out must be DRAM interleaved, got {}", omc);
    });
}

UnpackFp8RowsDeviceOperation::program_factory_t UnpackFp8RowsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return UnpackFp8RowsProgramFactory{};
}

void UnpackFp8RowsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // The public entry already checked; a direct prim caller has not.
    check_support(tensor_args);
    TT_FATAL(attrs.nbands > 0, "unpack_fp8_rows: nbands must be > 0");
    // unpack_program's assert.
    TT_FATAL(
        attrs.ncores >= 1 && attrs.ncores <= attrs.grid_x * attrs.grid_y,
        "unpack_fp8_rows: ({}, {}, {})",
        attrs.ncores,
        attrs.grid_x,
        attrs.grid_y);
    // A miss builds the program: recorded for the package's _PREPARED view.
    record_prepared(attrs, tensor_args);
}

void UnpackFp8RowsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

ttsl::hash::hash_t UnpackFp8RowsDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // src_page0, out_page0 and nbands are left out: override_runtime_arguments re-applies them on every hit, in the
    // per-core or common slots common_values (hashed here) selects.
    return tt::tt_metal::operation::hash_operation<UnpackFp8RowsDeviceOperation>(
        attrs.grid_x,
        attrs.grid_y,
        attrs.ncores,
        attrs.common_values,
        tensor_args.rows.tensor_spec(),
        tensor_args.out.tensor_spec());
}

UnpackFp8RowsDeviceOperation::spec_return_value_t UnpackFp8RowsDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return tensor_args.out.tensor_spec();
}

UnpackFp8RowsDeviceOperation::tensor_return_value_t UnpackFp8RowsDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return tensor_args.out;  // written in place
}

std::vector<UnpackFp8RowsPreparedKey> unpack_fp8_rows_prepared_keys() {
    auto& registry = prepared_registry();
    const std::lock_guard lock(registry.mutex);
    return registry.keys;
}

}  // namespace ttnn::operations::unpack_fp8_rows

namespace ttnn::prim {

Tensor unpack_fp8_rows(
    const Tensor& rows, const Tensor& out, const ttnn::operations::unpack_fp8_rows::UnpackFp8RowsParams& params) {
    using OperationType = ttnn::operations::unpack_fp8_rows::UnpackFp8RowsDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.rows = rows, .out = out});
}

}  // namespace ttnn::prim
