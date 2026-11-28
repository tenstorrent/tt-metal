// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ckv_bfp4_rows_device_operation.hpp"

#include <utility>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>

#include "kernels/ckv_bfp4_rows_args.hpp"

namespace ttnn::operations::ckv_bfp4_rows {

using namespace tt::tt_metal;

namespace {

// The message is formatted only when the check fails: a passing call builds no string.
template <typename... Args>
void assert_that(bool ok, fmt::format_string<Args...> format, Args&&... args) {
    if (!ok) {
        throw ContractAssertion(fmt::format(format, std::forward<Args>(args)...));
    }
}

void validate_output_tensor(const CkvBfp4RowsInputs& t) {
    if (!t.output.has_value()) {
        return;
    }
    TT_FATAL(
        t.output->tensor_spec() == t.rows.tensor_spec(),
        "ckv_bfp4_rows: output_tensor spec {} differs from rows' {} (the output is the record table itself)",
        t.output->tensor_spec(),
        t.rows.tensor_spec());
    TT_FATAL(
        t.output->buffer() == t.rows.buffer(),
        "ckv_bfp4_rows: output_tensor buffer must be rows' buffer (the records are written in place)");
}

}  // namespace

void check_support(const Tensor& src, const Tensor& rows, uint32_t src_row0, uint32_t n, uint32_t row0) {
    using ::ckv_bfp4_rows::ROW_BYTES;
    using ::ckv_bfp4_rows::WIDTH;
    assert_that(
        src.dtype() == DataType::BFLOAT16 && src.layout() == Layout::ROW_MAJOR && src.logical_shape()[-1] == WIDTH,
        "({}, {}, {})",
        src.dtype(),
        src.layout(),
        src.logical_shape());
    assert_that(
        rows.dtype() == DataType::UINT8 && rows.layout() == Layout::ROW_MAJOR && rows.logical_shape()[-1] == ROW_BYTES,
        "({}, {}, {})",
        rows.dtype(),
        rows.layout(),
        rows.logical_shape());
    const uint64_t total_src = src.padded_shape().volume() / WIDTH;
    const uint64_t total_rows = rows.padded_shape().volume() / ROW_BYTES;
    assert_that(n % tt::constants::TILE_HEIGHT == 0, "('ckv_bfp4_rows: n must be a multiple of 32', {})", n);
    assert_that(
        uint64_t{src_row0} + n <= total_src, "('ckv_bfp4_rows: source rows', {}, {}, {})", src_row0, n, total_src);
    assert_that(uint64_t{row0} + n <= total_rows, "('ckv_bfp4_rows: record rows', {}, {}, {})", row0, n, total_rows);
}

CkvBfp4RowsDeviceOperation::program_factory_t CkvBfp4RowsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return CkvBfp4RowsProgramFactory{};
}

void CkvBfp4RowsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // The public entry already checked; a direct prim caller has not.
    TT_FATAL(attrs.num_bands > 0, "ckv_bfp4_rows: num_bands must be positive");
    check_support(
        tensor_args.src, tensor_args.rows, attrs.src_row0, attrs.num_bands * tt::constants::TILE_HEIGHT, attrs.row0);
    validate_output_tensor(tensor_args);
}

void CkvBfp4RowsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // dtype / layout / width are pinned by the key; the row ranges vary per call.
    const uint64_t n = uint64_t{attrs.num_bands} * tt::constants::TILE_HEIGHT;
    TT_FATAL(
        attrs.src_row0 + n <= tensor_args.src.padded_shape().volume() / ::ckv_bfp4_rows::WIDTH,
        "ckv_bfp4_rows: source rows");
    TT_FATAL(
        attrs.row0 + n <= tensor_args.rows.padded_shape().volume() / ::ckv_bfp4_rows::ROW_BYTES,
        "ckv_bfp4_rows: record rows");
    validate_output_tensor(tensor_args);
}

ttsl::hash::hash_t CkvBfp4RowsDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // src_row0, num_bands and row0 are left out: override_runtime_arguments re-applies them on every hit (common
    // args). The key holds the grid and each tensor's geometry (the Python's _tensor_geo): dtype, layout, memory
    // config, page width, and the full shape when sharded (an ND-sharded accessor's page mapping holds it).
    const auto& src = tensor_args.src;
    const auto& rows = tensor_args.rows;
    auto hash = tt::tt_metal::operation::hash_operation<CkvBfp4RowsDeviceOperation>(
        attrs.grid_x,
        attrs.grid_y,
        src.dtype(),
        src.layout(),
        src.memory_config(),
        src.logical_shape()[-1],
        rows.dtype(),
        rows.layout(),
        rows.memory_config(),
        rows.logical_shape()[-1]);
    if (src.memory_config().is_sharded()) {
        hash = ttsl::hash::hash_objects(hash, src.logical_shape());
    }
    if (rows.memory_config().is_sharded()) {
        hash = ttsl::hash::hash_objects(hash, rows.logical_shape());
    }
    return hash;
}

CkvBfp4RowsDeviceOperation::spec_return_value_t CkvBfp4RowsDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return tensor_args.rows.tensor_spec();
}

CkvBfp4RowsDeviceOperation::tensor_return_value_t CkvBfp4RowsDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return tensor_args.output.has_value() ? *tensor_args.output : tensor_args.rows;
}

}  // namespace ttnn::operations::ckv_bfp4_rows

namespace ttnn::prim {

Tensor ckv_bfp4_rows(
    const Tensor& src,
    const Tensor& rows,
    const std::optional<Tensor>& output,
    const ttnn::operations::ckv_bfp4_rows::CkvBfp4RowsParams& params) {
    using OperationType = ttnn::operations::ckv_bfp4_rows::CkvBfp4RowsDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.src = src, .rows = rows, .output = output});
}

}  // namespace ttnn::prim
