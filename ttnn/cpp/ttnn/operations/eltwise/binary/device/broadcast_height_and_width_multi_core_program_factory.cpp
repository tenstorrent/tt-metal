// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "binary_device_operation.hpp"
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/device_operation.hpp"
#include "tt_metal/common/constants.hpp"


namespace ttnn::operations::binary {

static const BcastOpMath binary_op_type_to_bcast_op_math(const BinaryOpType binary_op_type) {
    switch (binary_op_type) {
        case BinaryOpType::ADD: return BcastOpMath::ADD;
        case BinaryOpType::SUB: return BcastOpMath::SUB;
        case BinaryOpType::MUL: return BcastOpMath::MUL;
        default: TT_THROW("BinaryOpType cannot be mapped to BcastOpMath");
    }
}

BinaryDeviceOperation::BroadcastHeightAndWidthMultiCore::cached_program_t
BinaryDeviceOperation::BroadcastHeightAndWidthMultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;
    auto bcast_math = binary_op_type_to_bcast_op_math(operation_attributes.binary_op_type);
    const auto ashape = a.get_legacy_shape();
    const auto bshape = b.get_legacy_shape();
    uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    uint32_t H = ashape[-2];
    uint32_t W = ashape[-1];
    uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    uint32_t bH = bshape[-2];
    uint32_t bW = bshape[-1];
    uint32_t NC = N * C;
    uint32_t HW = H * W;

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    uint32_t num_tensor_tiles = NC * Ht * Wt;

    uint32_t bnc1 = (bN * bC == 1);

    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::Device* device = a.device();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool src0_sharded = a.memory_config().is_sharded();
    bool output_sharded = output.memory_config().is_sharded();
    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (output_sharded) {
        shard_spec = output.shard_spec().value();
    }

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    bool row_major = false;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    }
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    auto src0_buffer = a.buffer();
    auto src1_buffer = b.buffer();
    auto dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    uint32_t num_tiles_per_shard = 0;
    if (shard_spec.has_value()) {
        num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        num_tiles_per_core_group_1 = num_tiles_per_shard;
        num_tiles_per_core_group_2 = 0;
        all_cores = shard_spec.value().grid;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet({});
    }

    uint32_t num_input_tiles_cb0 = src0_sharded ? num_tiles_per_shard : num_input_tiles;

    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(
            num_input_tiles_cb0 * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    if (src0_sharded) {
        src0_cb_config = src0_cb_config.set_globally_allocated_address(*a.buffer());
    }
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_device_cores, src0_cb_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
            .set_page_size(src1_cb_index, src1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_device_cores, src1_cb_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = output_sharded ? num_tiles_per_shard : 2;
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    if (output_sharded) {
        output_cb_config = output_cb_config.set_globally_allocated_address(*output.buffer());
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_device_cores, output_cb_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    std::map<string, string> reader_defines;
    std::map<string, string> bcast_compute_defines = bcast_op_utils::get_defines(BcastOpDim::HW, bcast_math);
    if (bnc1) {
        reader_defines["BCAST_SCALAR"] = "1";
        bcast_compute_defines["BCAST_SCALAR"] = "1";
    }
    if (src0_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    }
    KernelHandle binary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/reader_bcast_hw_interleaved_partitioned.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    std::map<string, string> writer_defines;
    if (output_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }
    KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    auto bcast_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw.cpp",
        all_device_cores,
        tt_metal::ComputeConfig{.compile_args = {}, .defines = bcast_compute_defines});

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_tensor_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, std::vector<uint32_t>(7, 0));
            tt_metal::SetRuntimeArgs(program, bcast_kernel_id, core, {1, 1, 0});
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::vector<uint32_t>(3, 0));
            continue;
        }

        tt_metal::SetRuntimeArgs(
            program,
            binary_reader_kernel_id,
            core,
            {a.buffer()->address(),  // 0
             b.buffer()->address(),
             num_tensor_tiles_per_core,
             HtWt,
             num_tiles_read / HtWt * HtWt,
             num_tiles_read % HtWt,
             bnc1 ? 0 : num_tiles_read / HtWt});

        tt_metal::SetRuntimeArgs(
            program,
            bcast_kernel_id,
            core,
            {
                1,                         // B
                1,                         // Ht
                num_tensor_tiles_per_core  // Wt
            });

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                output.buffer()->address(),
                num_tensor_tiles_per_core,
                num_tiles_read,
            });
        num_tiles_read += num_tensor_tiles_per_core;
    }

    return {
        std::move(program),
        {binary_reader_kernel_id,
         unary_writer_kernel_id,
         bcast_kernel_id,
         compute_with_storage_grid_size,
         cb_src0,
         src0_single_tile_size,
         src1_single_tile_size,
         dst_single_tile_size,
         cb_output}};
}

void BinaryDeviceOperation::BroadcastHeightAndWidthMultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    auto& binary_reader_kernel_id = cached_program.shared_variables.binary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& bcast_kernel_id = cached_program.shared_variables.bcast_kernel_id;
    auto& compute_with_storage_grid_size = cached_program.shared_variables.compute_with_storage_grid_size;
    auto& cb_src0 = cached_program.shared_variables.cb_src0;
    auto& src0_single_tile_size = cached_program.shared_variables.src0_single_tile_size;
    auto& src1_single_tile_size = cached_program.shared_variables.src1_single_tile_size;
    auto& dst_single_tile_size = cached_program.shared_variables.dst_single_tile_size;
    auto& cb_output = cached_program.shared_variables.cb_output;

    auto& program = cached_program.program;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;

    auto src_buffer_a = input_tensor_a.buffer();
    auto src_dram_buffer_b = input_tensor_b.buffer();
    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool src0_sharded = input_tensor_a.memory_config().is_sharded();
    bool out_sharded = output_tensor.memory_config().is_sharded();

    if (src0_sharded) {
        shard_spec = input_tensor_a.shard_spec().value();
    } else if (out_sharded) {
        shard_spec = output_tensor.shard_spec().value();
    }

    auto dst_buffer = output_tensor.buffer();

    const auto ashape = input_tensor_a.get_legacy_shape();
    const auto bshape = input_tensor_b.get_legacy_shape();
    uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    uint32_t H = ashape[-2];
    uint32_t W = ashape[-1];
    uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    uint32_t bH = bshape[-2];
    uint32_t bW = bshape[-1];
    uint32_t NC = N * C;
    uint32_t HW = H * W;

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    uint32_t num_tensor_tiles = NC * Ht * Wt;

    uint32_t bnc1 = (bN * bC == 1);

    bool row_major = false;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    }
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    if (shard_spec.has_value()) {
        uint32_t num_tiles_per_shard = 0;
        num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        num_tiles_per_core_group_1 = num_tiles_per_shard;
        num_tiles_per_core_group_2 = 0;
        all_cores = shard_spec.value().grid;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet({});
    }

    auto& cached_reader_args = GetRuntimeArgs(program, binary_reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(program, bcast_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, unary_writer_kernel_id);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_tensor_tiles_per_core;

        auto& binary_reader_args = cached_reader_args.at(core.x).at(core.y);
        auto& bcast_kernel_args = cached_eltwise_args.at(core.x).at(core.y);
        auto& unary_writer_args = cached_writer_args.at(core.x).at(core.y);

        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            binary_reader_args[2] = 0;
            bcast_kernel_args[2] = 0;
            unary_writer_args[1] = 0;
            continue;
        }

        binary_reader_args[0] = src_buffer_a->address();
        binary_reader_args[1] = src_dram_buffer_b->address();
        binary_reader_args[2] = num_tensor_tiles_per_core;
        binary_reader_args[3] = HtWt;
        binary_reader_args[4] = num_tiles_read / HtWt * HtWt;
        binary_reader_args[5] = num_tiles_read % HtWt;
        binary_reader_args[6] = bnc1 ? 0 : num_tiles_read / HtWt;

        bcast_kernel_args[2] = num_tensor_tiles_per_core; // Wt

        unary_writer_args[0] = dst_buffer->address();
        unary_writer_args[1] = num_tensor_tiles_per_core;
        unary_writer_args[2] = num_tiles_read;

        num_tiles_read += num_tensor_tiles_per_core;
    }

    if (src0_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer_a);
        UpdateCircularBufferTotalSize(program, cb_src0, num_tiles_per_core_group_1 * src0_single_tile_size);
    }

    if (out_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
        UpdateCircularBufferTotalSize(program, cb_output, num_tiles_per_core_group_1 * dst_single_tile_size);
    }
}

}  // namespace ttnn::operations::binary
