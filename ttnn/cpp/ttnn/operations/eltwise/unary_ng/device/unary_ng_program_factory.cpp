// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_ng_device_operation.hpp"

#include "ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.hpp"
#include "ttnn/operations/eltwise/unary_ng/common/unary_ng_utils.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include <algorithm>
#include <map>

#include <fmt/format.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace tt::tt_metal;
using namespace ttnn::operations::unary_ng;
using ttnn::operations::unary::EltwiseUnaryWithParam;
using ttnn::operations::unary::UnaryOpType;

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

uint32_t get_shards_per_width(const ShardSpec& shard_spec, TensorMemoryLayout memory_layout) {
    auto num_cores = shard_spec.grid.num_cores();
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        return 1;
    }
    if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        return num_cores;
    }
    const auto& bbox = shard_spec.grid.bounding_box();
    const auto& start = bbox.start_coord;
    const auto& end = bbox.end_coord;
    return (shard_spec.orientation == ShardOrientation::ROW_MAJOR ? end.x - start.x : end.y - start.y) + 1;
}

std::uint32_t* copy_common_runtime_args(const Buffer& buffer, std::uint32_t* dst) {
    const auto src =
        TensorAccessorArgs(buffer, tensor_accessor::ArgConfig::RuntimeTensorShape).get_common_runtime_args();
    return std::copy(src.begin(), src.end(), dst);
}

template <typename F>
void set_or_update_runtime_arguments(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    KernelHandle compute_kernel_id,
    CBHandle cb_src,
    CBHandle cb_out,
    const UnaryNgDeviceOperation::operation_attributes_t& operation_attributes,
    const UnaryNgDeviceOperation::tensor_args_t& tensor_args,
    UnaryNgDeviceOperation::tensor_return_value_t& output,
    uint32_t packed_scalar1,
    uint32_t packed_scalar2,
    F handle_args) {
    const auto& input = tensor_args.input;

    const auto shard_specs = get_shard_specs(input.tensor_spec(), output.tensor_spec());
    const bool has_sharding = shard_specs.has_value();
    auto grid = has_sharding ? shard_specs->input_shard_spec.grid : CoreRangeSet{};

    const auto row_major =
        has_sharding ? shard_specs->input_shard_spec.orientation == ShardOrientation::ROW_MAJOR : true;

    bool zero_start_grid = false;
    CoreCoord compute_with_storage_grid;
    const auto& all_device_cores = operation_attributes.worker_grid;
    if (all_device_cores.size() == 1) {
        const auto& cr = *all_device_cores.ranges().begin();
        if (cr.start_coord.x == 0 && cr.start_coord.y == 0) {
            if (has_sharding) {
                const auto& shard_start_coord = grid.ranges()[0].start_coord;
                if (shard_start_coord.x == 0 && shard_start_coord.y == 0) {
                    zero_start_grid = true;
                    compute_with_storage_grid = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
                }
            } else {
                zero_start_grid = true;
                compute_with_storage_grid = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
            }
        }
    }
    const uint32_t num_cores_total =
        zero_start_grid ? compute_with_storage_grid.x * compute_with_storage_grid.y : all_device_cores.num_cores();

    uint32_t num_tiles_per_core_group_1{}, num_tiles_per_core_group_2{};
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores;
    std::vector<CoreCoord> cores;

    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;
    const uint32_t tile_height = output.tensor_spec().tile().get_height();
    const uint32_t tile_width = output.tensor_spec().tile().get_width();
    const uint32_t tile_hw = tile_height * tile_width;
    const uint32_t out_num_tiles = is_row_major ? output.buffer()->num_pages() : output.physical_volume() / tile_hw;
    uint32_t out_shard_height{}, out_shard_width{}, num_shards_per_width{};

    const auto [oD, oN, oC, oHt, oWt] = [&]() -> std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> {
        const auto& shape = output.padded_shape();
        const auto& tile = output.tensor_spec().tile();
        return {
            shape.rank() >= 5 ? shape[-5] : 1,
            shape[-4],
            shape[-3],
            shape[-2] / tile.get_height(),
            shape[-1] / tile.get_width()};
    }();

    if (has_sharding) {
        core_group_1 = grid;
        out_shard_height = shard_specs->output_shard_spec.shape[0] / tile_height;
        out_shard_width = shard_specs->output_shard_spec.shape[1] / tile_width;
        auto out_memory_layout = output.memory_config().is_sharded() ? output.memory_config().memory_layout()
                                                                     : input.memory_config().memory_layout();
        num_shards_per_width = get_shards_per_width(shard_specs->output_shard_spec, out_memory_layout);

        auto compute_shard_tiles = [&](const ShardSpec& spec,
                                       const auto& tensor) -> std::function<uint32_t(CoreCoord)> {
            auto end_core = spec.grid.ranges().rbegin()->end_coord;
            bool rm = spec.orientation == ShardOrientation::ROW_MAJOR;
            auto mem_layout = tensor.memory_config().memory_layout();
            uint32_t sh = tt::round_up(spec.shape[0], tile_height) / tile_height;
            uint32_t sw = tt::round_up(spec.shape[1], tile_width) / tile_width;
            const auto& pshape = tensor.padded_shape();
            uint32_t D = pshape.rank() >= 5 ? pshape[-5] : 1;
            uint32_t N = pshape[-4], C = pshape[-3];
            uint32_t Ht = pshape[-2] / tile_height, Wt = pshape[-1] / tile_width;
            uint32_t unrolled_Ht = D * N * C * Ht;
            uint32_t last_h = sh - (tt::round_up(unrolled_Ht, sh) - unrolled_Ht);
            uint32_t last_w = sw - (tt::round_up(Wt, sw) - Wt);

            return [=](CoreCoord core) -> uint32_t {
                uint32_t h = sh, w = sw;
                if (mem_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
                    mem_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                    if (core == end_core) {
                        h = last_h;
                        w = last_w;
                    }
                } else {
                    if (rm) {
                        if (core.x == end_core.x) {
                            w = last_w;
                        }
                        if (core.y == end_core.y) {
                            h = last_h;
                        }
                    } else {
                        if (core.y == end_core.y) {
                            w = last_w;
                        }
                        if (core.x == end_core.x) {
                            h = last_h;
                        }
                    }
                }
                return h * w;
            };
        };

        auto in_shard_tiles = compute_shard_tiles(shard_specs->input_shard_spec, input);
        auto out_shard_tiles = compute_shard_tiles(shard_specs->output_shard_spec, output);

        if (zero_start_grid) {
            auto bbox = core_group_1.bounding_box();
            cores = grid_to_cores_with_noop(
                bbox.end_coord.x,
                bbox.end_coord.y,
                compute_with_storage_grid.x,
                compute_with_storage_grid.y,
                row_major);
        } else {
            cores = grid_to_cores_with_noop(core_group_1, all_device_cores, row_major);
        }

        for (uint32_t i = 0; i < num_cores_total; ++i) {
            const auto& core = cores[i];
            if (!core_group_1.contains(core)) {
                handle_args(program, reader_kernel_id, core, std::array<uint32_t, 3>{0});
                handle_args(program, writer_kernel_id, core, std::array<uint32_t, 3>{0});
                handle_args(program, compute_kernel_id, core, std::array<uint32_t, 3>{0});
                continue;
            }
            uint32_t in_tiles = in_shard_tiles(core);
            uint32_t o_tiles = out_shard_tiles(core);
            uint32_t out_start_id =
                (i / num_shards_per_width) * (out_shard_height * oWt) + (i % num_shards_per_width) * out_shard_width;

            std::array reader_runtime_args = {input.buffer()->address(), in_tiles, out_start_id};
            handle_args(program, reader_kernel_id, core, reader_runtime_args);

            std::array writer_runtime_args = {output.buffer()->address(), o_tiles, out_start_id};
            handle_args(program, writer_kernel_id, core, writer_runtime_args);

            std::array compute_runtime_args = {o_tiles, packed_scalar1, packed_scalar2};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        }

        if (input.is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, cb_src, *input.buffer());
        }
        if (output.is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, cb_out, *output.buffer());
        }
    } else {
        if (zero_start_grid) {
            std::tie(
                num_cores,
                all_cores,
                core_group_1,
                core_group_2,
                num_tiles_per_core_group_1,
                num_tiles_per_core_group_2) = split_work_to_cores(compute_with_storage_grid, out_num_tiles, row_major);
            cores = grid_to_cores(num_cores_total, compute_with_storage_grid.x, compute_with_storage_grid.y, row_major);
        } else {
            std::tie(
                num_cores,
                all_cores,
                core_group_1,
                core_group_2,
                num_tiles_per_core_group_1,
                num_tiles_per_core_group_2) = split_work_to_cores(all_device_cores, out_num_tiles, row_major);
            cores = corerange_to_cores(all_device_cores, {}, row_major);
        }

        for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; ++i) {
            const auto& core = cores[i];

            uint32_t npc = 0;
            if (core_group_1.contains(core)) {
                npc = num_tiles_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                npc = num_tiles_per_core_group_2;
            } else {
                handle_args(program, reader_kernel_id, core, std::array<uint32_t, 3>{0});
                handle_args(program, writer_kernel_id, core, std::array<uint32_t, 3>{0});
                handle_args(program, compute_kernel_id, core, std::array<uint32_t, 3>{0});
                continue;
            }

            std::array reader_runtime_args = {input.buffer()->address(), npc, start_tile_id};
            handle_args(program, reader_kernel_id, core, reader_runtime_args);

            std::array writer_runtime_args = {output.buffer()->address(), npc, start_tile_id};
            handle_args(program, writer_kernel_id, core, writer_runtime_args);

            std::array compute_runtime_args = {npc, packed_scalar1, packed_scalar2};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);

            start_tile_id += npc;
        }
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::unary_ng {

UnaryNgDeviceOperation::ProgramFactory::cached_program_t UnaryNgDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& ops_chain = operation_attributes.op_chain;
    TT_FATAL(!ops_chain.empty(), "UnaryNg: op_chain must not be empty");

    uint32_t packed_scalar1 = 0;
    uint32_t packed_scalar2 = 0;
    Program program = CreateProgram();

    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tile_size(cb_data_format);
    DataFormat cb_data_format_output = datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tile_size(cb_data_format_output);

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    const uint32_t input_cb_page_size = is_row_major ? src_buffer->page_size() : single_tile_size;
    const uint32_t output_cb_page_size = is_row_major ? dst_buffer->page_size() : single_tile_size_output;

    const auto shard_specs = get_shard_specs(input.tensor_spec(), output.tensor_spec());
    const bool has_sharding = shard_specs.has_value();
    const bool src_sharded = has_sharding && input.is_sharded();
    const bool dst_sharded = has_sharding && output.is_sharded();
    const auto src_num_tiles_per_shard =
        src_sharded ? std::optional<uint32_t>(
                          shard_specs->input_shard_spec.numel() / (input.tensor_spec().tile().get_tile_hw()))
                    : std::nullopt;
    const auto dst_num_tiles_per_shard =
        dst_sharded ? std::optional<uint32_t>(
                          shard_specs->output_shard_spec.numel() / (output.tensor_spec().tile().get_tile_hw()))
                    : std::nullopt;

    const auto& all_device_cores = operation_attributes.worker_grid;

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
    CMAKE_UNIQUE_NAMESPACE::apply_input_dtype_defines(input.dtype(), unary_defines);
    CMAKE_UNIQUE_NAMESPACE::pack_first_op_scalars(
        ops_chain[0], input.dtype(), packed_scalar1, packed_scalar2, unary_defines);

    const std::string compute_path = fmt::format(
        "ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/compute/{}",
        get_compute_kernel_path(ops_chain[0].type(), input.dtype()));

    DataFormat cb_data_format_for_input =
        (ops_chain[0].type() == unary::UnaryOpType::BITCAST) ? cb_data_format_output : cb_data_format;

    // --- Circular Buffers ---
    auto [src0_cb, src0_cb_handle] = create_cb(
        src0_cb_index,
        program,
        all_device_cores,
        input_cb_page_size,
        src_num_tiles_per_shard.value_or(2),
        cb_data_format_for_input,
        src_sharded ? src_buffer : nullptr);

    if (CMAKE_UNIQUE_NAMESPACE::needs_tmp0_cb(ops_chain[0].type())) {
        create_cb(tmp0_cb_index, program, all_device_cores, input_cb_page_size, 2, cb_data_format);
    }

    const uint32_t output_cb_index = CBIndex::c_2;
    auto [out_cb, out_cb_handle] = create_cb(
        output_cb_index,
        program,
        all_device_cores,
        output_cb_page_size,
        dst_num_tiles_per_shard.value_or(2),
        cb_data_format_output,
        dst_sharded ? dst_buffer : nullptr);

    // --- Reader Kernel ---
    std::map<std::string, std::string> reader_defines;
    reader_defines["SRC_SHARDED"] = src_sharded ? "1" : "0";

    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> reader_common_runtime_args;
    TensorAccessorArgs(*src_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_compile_time_args, reader_common_runtime_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/dataflow/reader_unary_ng.cpp",
        all_device_cores,
        ReaderDataMovementConfig(reader_compile_time_args, reader_defines));
    SetCommonRuntimeArgs(program, reader_kernel_id, reader_common_runtime_args);

    // --- Writer Kernel ---
    std::map<std::string, std::string> writer_defines;
    writer_defines["DST_SHARDED"] = dst_sharded ? "1" : "0";

    std::vector<uint32_t> writer_compile_time_args;
    std::vector<uint32_t> writer_common_runtime_args;
    TensorAccessorArgs(*dst_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_compile_time_args, writer_common_runtime_args);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/dataflow/writer_unary_ng.cpp",
        all_device_cores,
        WriterDataMovementConfig(writer_compile_time_args, writer_defines));
    SetCommonRuntimeArgs(program, writer_kernel_id, writer_common_runtime_args);

    // --- Compute Kernel (single instance, runtime tile count, no compile-time args) ---
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        compute_path,
        all_device_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = operation_attributes.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
            .defines = unary_defines});

    // --- Per-core runtime args ---
    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        SetRuntimeArgs(program, kernel_id, core, args);
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        src0_cb_handle,
        out_cb_handle,
        operation_attributes,
        tensor_args,
        output,
        packed_scalar1,
        packed_scalar2,
        set_runtime_args);

    return {std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, src0_cb_handle, out_cb_handle}};
}

void UnaryNgDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    auto reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    {
        auto* src = tensor_args.input.buffer();
        auto* dst = output.buffer();
        auto* args = GetCommonRuntimeArgs(program, reader_kernel_id).data();
        CMAKE_UNIQUE_NAMESPACE::copy_common_runtime_args(*src, args);
        args = GetCommonRuntimeArgs(program, writer_kernel_id).data();
        CMAKE_UNIQUE_NAMESPACE::copy_common_runtime_args(*dst, args);
    }

    const auto& ops_chain = operation_attributes.op_chain;
    uint32_t packed_scalar1 = 0;
    uint32_t packed_scalar2 = 0;
    std::map<std::string, std::string> unused_defines;
    CMAKE_UNIQUE_NAMESPACE::pack_first_op_scalars(
        ops_chain[0], tensor_args.input.dtype(), packed_scalar1, packed_scalar2, unused_defines);

    auto update_args = [](Program& prog, KernelHandle kid, CoreCoord core, auto&& args) {
        auto& all_args = GetRuntimeArgs(prog, kid);
        auto& core_args = all_args.at(core.x).at(core.y);
        std::copy(args.begin(), args.end(), core_args.data());
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        cached_program.shared_variables.compute_kernel_id,
        cached_program.shared_variables.cb_src,
        cached_program.shared_variables.cb_out,
        operation_attributes,
        tensor_args,
        output,
        packed_scalar1,
        packed_scalar2,
        update_args);
}

}  // namespace ttnn::operations::unary_ng
