// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/deepseek_v4_prefill_write_rows_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/hal.hpp>

namespace ttnn::prim {

namespace {
void check_dram(const Tensor& t, const char* name) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr,
        "deepseek_v4_prefill_write_rows: {} must be on device",
        name);
    TT_FATAL(t.layout() == Layout::ROW_MAJOR, "deepseek_v4_prefill_write_rows: {} must be ROW_MAJOR", name);
    TT_FATAL(
        t.memory_config().buffer_type() == BufferType::DRAM,
        "deepseek_v4_prefill_write_rows: {} must be in DRAM",
        name);
    TT_FATAL(!t.memory_config().is_sharded(), "deepseek_v4_prefill_write_rows: {} must be interleaved", name);
}

// Everything but the input shape and dtypes (which the program hash keys on), checked on every dispatch.
void validate_non_hashed(const DeepseekV4PrefillWriteRowsInputs& t) {
    check_dram(t.input, "input");
    check_dram(t.output, "output");
    check_dram(t.rows, "rows");
    TT_FATAL(t.output.device() == t.input.device(), "deepseek_v4_prefill_write_rows: output must be on input's device");
    TT_FATAL(t.rows.device() == t.input.device(), "deepseek_v4_prefill_write_rows: rows must be on input's device");
    const auto is = t.input.logical_shape();
    const auto os = t.output.logical_shape();
    TT_FATAL(
        os.rank() == 4 && os[0] == 1 && os[1] == 1 && os[3] == is[3],
        "deepseek_v4_prefill_write_rows: output must be [1, 1, R, {}] (got {})",
        is[3],
        os);
    TT_FATAL(t.output.padded_shape() == os, "deepseek_v4_prefill_write_rows: output must not be padded");
}
}  // namespace

DeepseekV4PrefillWriteRowsOperation::program_factory_t DeepseekV4PrefillWriteRowsOperation::select_program_factory(
    const DeepseekV4PrefillWriteRowsParams& /*attrs*/, const DeepseekV4PrefillWriteRowsInputs& /*t*/) {
    return ProgramFactory{};
}

void DeepseekV4PrefillWriteRowsOperation::validate_on_program_cache_hit(
    const DeepseekV4PrefillWriteRowsParams& /*attrs*/, const DeepseekV4PrefillWriteRowsInputs& t) {
    validate_non_hashed(t);
}

void DeepseekV4PrefillWriteRowsOperation::validate_on_program_cache_miss(
    const DeepseekV4PrefillWriteRowsParams& /*attrs*/, const DeepseekV4PrefillWriteRowsInputs& t) {
    validate_non_hashed(t);
    const auto is = t.input.logical_shape();
    TT_FATAL(
        is.rank() == 4 && is[0] == 1 && is[1] == 1 && is[2] > 0,
        "deepseek_v4_prefill_write_rows: input must be [1, 1, n, W] with n > 0 (got {})",
        is);
    TT_FATAL(t.input.padded_shape() == is, "deepseek_v4_prefill_write_rows: input must not be padded");
    TT_FATAL(t.output.dtype() == t.input.dtype(), "deepseek_v4_prefill_write_rows: output must have input's dtype");
    TT_FATAL(
        t.rows.dtype() == DataType::UINT32 || t.rows.dtype() == DataType::INT32,
        "deepseek_v4_prefill_write_rows: rows must be uint32 or int32");
    const auto rs = t.rows.logical_shape();
    TT_FATAL(
        t.rows.logical_volume() == is[2] && rs[-1] == is[2] && t.rows.padded_shape() == rs,
        "deepseek_v4_prefill_write_rows: rows must hold {} indices in its last dimension (got {})",
        is[2],
        rs);
    TT_FATAL(
        t.input.buffer()->aligned_page_size() == t.input.buffer()->page_size(),
        "deepseek_v4_prefill_write_rows: input row bytes must be DRAM aligned");
}

DeepseekV4PrefillWriteRowsOperation::spec_return_value_t DeepseekV4PrefillWriteRowsOperation::compute_output_specs(
    const DeepseekV4PrefillWriteRowsParams& /*attrs*/, const DeepseekV4PrefillWriteRowsInputs& t) {
    return t.output.tensor_spec();
}

DeepseekV4PrefillWriteRowsOperation::tensor_return_value_t DeepseekV4PrefillWriteRowsOperation::create_output_tensors(
    const DeepseekV4PrefillWriteRowsParams& /*attrs*/, const DeepseekV4PrefillWriteRowsInputs& t) {
    return t.output;
}

ttsl::hash::hash_t DeepseekV4PrefillWriteRowsOperation::compute_program_hash(
    const DeepseekV4PrefillWriteRowsParams& /*attrs*/, const DeepseekV4PrefillWriteRowsInputs& t) {
    // The output's row count R rides on a runtime arg, so any buffer of the same width reuses the program.
    return tt::tt_metal::operation::hash_operation<DeepseekV4PrefillWriteRowsOperation>(
        t.input.logical_shape(), t.input.dtype(), t.rows.dtype());
}

Tensor deepseek_v4_prefill_write_rows(const Tensor& input, const Tensor& output, const Tensor& rows) {
    using OperationType = ttnn::prim::DeepseekV4PrefillWriteRowsOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{},
        OperationType::tensor_args_t{.input = input, .output = output, .rows = rows});
}

}  // namespace ttnn::prim
