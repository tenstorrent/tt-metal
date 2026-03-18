// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_ng_device_operation.hpp"

#include "ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.hpp"

#include <algorithm>
#include <map>

#include <fmt/format.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::unary_ng {

namespace {

using namespace tt::tt_metal;
using unary::EltwiseUnaryWithParam;
using unary::UnaryOpType;

static const std::string kComputeRoot = "ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/compute/";
static const std::string kReaderUnary =
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary.cpp";

void apply_input_dtype_defines(DataType dtype, std::map<std::string, std::string>& defines) {
    if (dtype == DataType::FLOAT32) {
        defines["INP_FLOAT32"] = "1";
    } else if (dtype == DataType::INT32) {
        defines["INP_INT32"] = "1";
    } else if (dtype == DataType::UINT32) {
        defines["INP_UINT32"] = "1";
    } else {
        defines["INP_FLOAT"] = "1";
    }
}

void pack_first_op_scalars(
    const EltwiseUnaryWithParam& op,
    DataType input_dtype,
    uint32_t& packed_scalar1,
    uint32_t& packed_scalar2,
    std::map<std::string, std::string>& unary_defines) {
    if (op.empty()) {
        return;
    }
    switch (op.type()) {
        case UnaryOpType::HARDSHRINK:
        case UnaryOpType::MISH: packed_scalar1 = pack_scalar_runtime_arg(op, 0, input_dtype); break;
        case UnaryOpType::WHERE_TSS:
            packed_scalar1 = pack_scalar_runtime_arg(op, 0, input_dtype);
            packed_scalar2 = pack_scalar_runtime_arg(op, 1, input_dtype);
            break;
        case UnaryOpType::LOGIT: {
            float value1 = *op.get_param_if<float>(0);
            float value2 = 1.0f - value1;
            packed_scalar1 = pack_scalar_runtime_arg_impl(value1, input_dtype);
            packed_scalar2 = pack_scalar_runtime_arg_impl(value2, input_dtype);
            if (value1 > 0.5f) {
                const char* data_format = (input_dtype == DataType::FLOAT32) ? "Float32" : "Float16_b";
                unary_defines["WHERE"] = fmt::format("where_tile<DataFormat::{0}>", data_format);
                unary_defines["CLAMP"] = "clamp_tile";
            } else if (value1 >= 0.0f) {
                unary_defines["CLAMP"] = "clamp_tile";
            }
            break;
        }
        default: break;
    }
}

bool needs_tmp0_cb(UnaryOpType t) { return t == UnaryOpType::HARDSHRINK || t == UnaryOpType::LOGIT; }

}  // namespace

UnaryNgDeviceOperation::ProgramFactory::cached_program_t UnaryNgDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using tt::CBIndex;
    using tt::DataFormat;

    const auto& input = tensor_args.input;
    const auto& ops_chain = operation_attributes.op_chain;
    TT_FATAL(!ops_chain.empty(), "UnaryNg: op_chain must not be empty");

    uint32_t packed_scalar1 = 0;
    uint32_t packed_scalar2 = 0;
    Program program = CreateProgram();

    DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tile_size(cb_data_format);
    DataFormat cb_data_format_output = datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tile_size(cb_data_format_output);

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    const uint32_t src0_cb_index = CBIndex::c_0;
    const uint32_t tmp0_cb_index = CBIndex::c_1;
    if (operation_attributes.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[tmp0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    const bool math_approx_mode =
        std::all_of(ops_chain.begin(), ops_chain.end(), [](const auto& u) { return get_op_approx_mode(u.type()); });
    std::map<std::string, std::string> unary_defines = get_block_defines(ops_chain, "0", "0", input.dtype());
    apply_input_dtype_defines(input.dtype(), unary_defines);
    pack_first_op_scalars(ops_chain[0], input.dtype(), packed_scalar1, packed_scalar2, unary_defines);

    const std::string compute_path =
        fmt::format("{}/{}", kComputeRoot, get_compute_kernel_path(ops_chain[0].type(), input.dtype()));

    const uint32_t num_pages = src_buffer->num_pages();
    const bool row_major = input.layout() == Layout::ROW_MAJOR;

    // split_work_to_cores(CoreRangeSet) uses num_cores_to_corerangeset_in_subcoregrids, which fails on
    // fragmented worker_cores() layouts (e.g. harvested devices). Use the rectangular compute grid for
    // normal interleaved tensors, like UnaryProgramFactory. Explicit sub_core_grids keeps CoreRangeSet split.
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t n_per_g1 = 0;
    uint32_t n_per_g2 = 0;
    std::vector<CoreCoord> core_order;

    if (!input.is_sharded() && !operation_attributes.sub_core_grids.has_value()) {
        CoreCoord grid_size = input.device()->compute_with_storage_grid_size();
        std::tie(num_cores, all_cores, core_group_1, core_group_2, n_per_g1, n_per_g2) =
            split_work_to_cores(grid_size, num_pages, row_major);
        core_order = grid_to_cores(num_cores, grid_size.x, grid_size.y, row_major);
    } else {
        const auto& worker_grid = operation_attributes.worker_grid;
        std::tie(num_cores, all_cores, core_group_1, core_group_2, n_per_g1, n_per_g2) =
            split_work_to_cores(worker_grid, num_pages, row_major);
        core_order = corerange_to_cores(worker_grid, num_cores, row_major);
    }

    const uint32_t input_cb_page_size = row_major ? src_buffer->page_size() : single_tile_size;
    const uint32_t output_cb_page_size = row_major ? dst_buffer->page_size() : single_tile_size_output;
    const uint32_t num_input_tiles = 2;

    DataFormat cb_data_format_for_input =
        (ops_chain[0].type() == UnaryOpType::BITCAST) ? cb_data_format_output : cb_data_format;

    CircularBufferConfig cb_src0 =
        CircularBufferConfig(num_input_tiles * input_cb_page_size, {{src0_cb_index, cb_data_format_for_input}})
            .set_page_size(src0_cb_index, input_cb_page_size);
    CreateCircularBuffer(program, all_cores, cb_src0);

    if (needs_tmp0_cb(ops_chain[0].type())) {
        CircularBufferConfig cb_tmp0 =
            CircularBufferConfig(num_input_tiles * input_cb_page_size, {{tmp0_cb_index, cb_data_format}})
                .set_page_size(tmp0_cb_index, input_cb_page_size);
        CreateCircularBuffer(program, all_cores, cb_tmp0);
    }

    const uint32_t output_cb_index = CBIndex::c_2;
    const uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_out =
        CircularBufferConfig(num_output_tiles * output_cb_page_size, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, output_cb_page_size);
    CreateCircularBuffer(program, all_cores, cb_out);

    std::map<std::string, std::string> reader_defines;
    reader_defines["SRC_SHARDED"] = "0";
    std::vector<uint32_t> reader_cta = {src0_cb_index};
    TensorAccessorArgs(*src_buffer).append_to(reader_cta);

    KernelHandle reader_id =
        CreateKernel(program, kReaderUnary, all_cores, ReaderDataMovementConfig(reader_cta, reader_defines));

    std::vector<uint32_t> writer_cta = {static_cast<uint32_t>(output_cb_index)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_cta);
    KernelHandle writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig(writer_cta));

    std::vector<uint32_t> compute_args_g1 = {n_per_g1, 1u};
    KernelHandle compute_g1 = CreateKernel(
        program,
        compute_path,
        core_group_1,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = operation_attributes.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_args_g1,
            .defines = unary_defines});

    KernelHandle compute_g2{};
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_g2 = {n_per_g2, 1u};
        compute_g2 = CreateKernel(
            program,
            compute_path,
            core_group_2,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .bfp8_pack_precise = operation_attributes.bfp8_pack_precise,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_args_g2,
                .defines = unary_defines});
    }

    std::vector<CoreCoord> cores_with_runtime_args;
    uint32_t pages_written = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = core_order[i];
        uint32_t npc = 0;
        KernelHandle compute_id = compute_g1;
        if (core_group_1.contains(core)) {
            npc = n_per_g1;
        } else if (core_group_2.contains(core)) {
            npc = n_per_g2;
            compute_id = compute_g2;
        } else {
            TT_FATAL(false, "UnaryNg: core ({},{}) not in split groups", core.x, core.y);
        }

        SetRuntimeArgs(program, reader_id, core, {src_buffer->address(), npc, pages_written});
        SetRuntimeArgs(program, writer_id, core, {dst_buffer->address(), npc, pages_written});
        SetRuntimeArgs(program, compute_id, core, {packed_scalar1, packed_scalar2});
        cores_with_runtime_args.push_back(core);
        pages_written += npc;
    }

    return {std::move(program), {reader_id, writer_id, compute_g1, std::move(cores_with_runtime_args)}};
}

void UnaryNgDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    Program& program = cached_program.program;
    const KernelHandle reader_id = cached_program.shared_variables.reader_kernel_id;
    const KernelHandle writer_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores_with_runtime_args;

    auto* src = tensor_args.input.buffer();
    auto* dst = output.buffer();

    auto& reader_by_core = GetRuntimeArgs(program, reader_id);
    auto& writer_by_core = GetRuntimeArgs(program, writer_id);
    for (const CoreCoord& core : cores) {
        reader_by_core[core.x][core.y][0] = src->address();
        writer_by_core[core.x][core.y][0] = dst->address();
    }
}

}  // namespace ttnn::operations::unary_ng
