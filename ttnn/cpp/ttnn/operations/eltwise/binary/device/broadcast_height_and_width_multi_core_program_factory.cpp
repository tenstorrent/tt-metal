// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "binary_device_operation.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::binary {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
BcastOpMath binary_op_type_to_bcast_op_math(const BinaryOpType binary_op_type) {
    switch (binary_op_type) {
        case BinaryOpType::ADD: return BcastOpMath::ADD;
        case BinaryOpType::SUB: return BcastOpMath::SUB;
        case BinaryOpType::MUL: return BcastOpMath::MUL;
        default: TT_THROW("BinaryOpType cannot be mapped to BcastOpMath");
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

BinaryDeviceOperation::BroadcastHeightAndWidthMultiCore::cached_program_t
BinaryDeviceOperation::BroadcastHeightAndWidthMultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;
    auto bcast_math = binary_op_type_to_bcast_op_math(operation_attributes.binary_op_type);
    const auto ashape = a.padded_shape();
    const auto bshape = b.has_value() ? b->padded_shape() : ttnn::Shape({1, 1});
    uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    uint32_t H = ashape[-2];
    uint32_t W = ashape[-1];
    uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    uint32_t NC = N * C;
    uint32_t HW = H * W;

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    uint32_t num_tensor_tiles = NC * Ht * Wt;

    bool bnc1 = (bN * bC == 1);

    auto program = tt_metal::CreateProgram();

    tt_metal::IDevice* device = a.device();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool src0_sharded = a.memory_config().is_sharded();
    bool output_sharded = output.memory_config().is_sharded();
    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (output_sharded) {
        shard_spec = output.shard_spec().value();
    }

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat src1_cb_data_format =
        b.has_value() ? tt_metal::datatype_to_dataformat_converter(b->dtype()) : tt::DataFormat::Float16_b;
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    bool row_major = shard_spec.has_value() ? shard_spec->orientation == ShardOrientation::ROW_MAJOR : false;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    auto* src0_buffer = a.buffer();
    auto* src1_buffer = b.has_value() ? b->buffer() : nullptr;
    auto* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_input_tiles = 2;
    uint32_t num_tiles_per_shard = 0;
    if (shard_spec.has_value()) {
        num_tiles_per_shard = shard_spec->shape[0] * shard_spec->shape[1] / TILE_HW;
        num_tiles_per_core_group_1 = num_tiles_per_shard;
        num_tiles_per_core_group_2 = 0;
        all_cores = shard_spec->grid;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
    }

    uint32_t num_input_tiles_cb0 = src0_sharded ? num_tiles_per_shard : num_input_tiles;

    auto* cb_src0_buffer = src0_sharded ? src0_buffer : nullptr;
    auto [cb_src0, cb_handle_src0] = create_cb(
        tt::CBIndex::c_0,
        program,
        all_device_cores,
        src0_single_tile_size,
        num_input_tiles_cb0,
        src0_cb_data_format,
        cb_src0_buffer);

    uint32_t num_input_tiles_cb1 = src1_buffer != nullptr ? num_input_tiles : 1;
    create_cb(
        tt::CBIndex::c_1, program, all_device_cores, src1_single_tile_size, num_input_tiles_cb1, src1_cb_data_format);

    uint32_t num_output_tiles = output_sharded ? num_tiles_per_shard : 2;
    auto* cb_output_buffer = output_sharded ? dst_buffer : nullptr;
    auto [cb_output, cb_handle_output] = create_cb(
        tt::CBIndex::c_2,
        program,
        all_device_cores,
        dst_single_tile_size,
        num_output_tiles,
        dst_cb_data_format,
        cb_output_buffer);

    auto src0_is_dram = static_cast<uint32_t>(src0_buffer->buffer_type() == tt_metal::BufferType::DRAM);
    auto dst_is_dram = static_cast<uint32_t>(dst_buffer->buffer_type() == tt_metal::BufferType::DRAM);

    std::map<string, string> reader_defines;
    std::map<string, string> bcast_compute_defines = bcast_op_utils::get_defines(BcastOpDim::HW, bcast_math);
    if (bnc1) {
        reader_defines["BCAST_SCALAR"] = "1";
        bcast_compute_defines["BCAST_SCALAR"] = "1";
    }
    if (src0_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    }

    KernelHandle binary_reader_kernel_id{};

    if (src1_buffer != nullptr) {
        auto src1_is_dram = static_cast<uint32_t>(src1_buffer->buffer_type() == tt_metal::BufferType::DRAM);
        binary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/"
            "reader_bcast_hw_interleaved_partitioned.cpp",
            all_device_cores,
            tt_metal::ReaderDataMovementConfig({src0_is_dram, src1_is_dram}, reader_defines));
    } else {
        binary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/"
            "reader_bcast_scalar_interleaved_partitioned.cpp",
            all_device_cores,
            tt_metal::ReaderDataMovementConfig({src0_is_dram}, reader_defines));
    }

    std::map<string, string> writer_defines;
    if (output_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }
    KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig({cb_output, dst_is_dram}, writer_defines));

    auto bcast_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_hw.cpp",
        all_device_cores,
        tt_metal::ComputeConfig{.compile_args = {}, .defines = bcast_compute_defines});

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_tensor_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, std::vector<uint32_t>(7, 0));
            tt_metal::SetRuntimeArgs(program, bcast_kernel_id, core, {1, 1, 0});
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::vector<uint32_t>(3, 0));
            continue;
        }

        std::array binary_reader_args = {
            src0_buffer->address(),  // 0
            0u,
            num_tensor_tiles_per_core,
            HtWt,
            num_tiles_read / HtWt * HtWt,
            num_tiles_read % HtWt,
            bnc1 ? 0 : num_tiles_read / HtWt};

        if (src1_buffer != nullptr) {
            binary_reader_args[1] = src1_buffer->address();
        } else {
            class bfloat16 bfloat_scalar(*operation_attributes.scalar);
            uint32_t packed_scalar = pack_two_bfloat16_into_uint32({bfloat_scalar, bfloat_scalar});
            binary_reader_args[1] = packed_scalar;
        }

        tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, binary_reader_args);

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
                dst_buffer->address(),
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
         cb_handle_src0,
         src0_single_tile_size,
         src1_single_tile_size,
         dst_single_tile_size,
         cb_handle_output}};
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
    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool src0_sharded = input_tensor_a.memory_config().is_sharded();
    bool out_sharded = output_tensor.memory_config().is_sharded();

    if (src0_sharded) {
        shard_spec = input_tensor_a.shard_spec().value();
    } else if (out_sharded) {
        shard_spec = output_tensor.shard_spec().value();
    }

    auto dst_buffer = output_tensor.buffer();

    const auto ashape = input_tensor_a.padded_shape();
    const auto bshape = input_tensor_b.has_value() ? input_tensor_b->padded_shape() : ttnn::Shape({1, 1});
    uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    uint32_t H = ashape[-2];
    uint32_t W = ashape[-1];
    uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    uint32_t NC = N * C;
    uint32_t HW = H * W;

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    uint32_t num_tensor_tiles = NC * Ht * Wt;

    auto bnc1 = static_cast<uint32_t>(bN * bC == 1);

    bool row_major = false;
    if (shard_spec.has_value()) {
        row_major = shard_spec->orientation == ShardOrientation::ROW_MAJOR;
    }
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    if (shard_spec.has_value()) {
        uint32_t num_tiles_per_shard = 0;
        num_tiles_per_shard = shard_spec->shape[0] * shard_spec->shape[1] / TILE_HW;
        num_tiles_per_core_group_1 = num_tiles_per_shard;
        num_tiles_per_core_group_2 = 0;
        all_cores = shard_spec->grid;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
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

        if (core_group_1.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            binary_reader_args[2] = 0;
            bcast_kernel_args[2] = 0;
            unary_writer_args[1] = 0;
            continue;
        }

        binary_reader_args[0] = src_buffer_a->address();

        if (input_tensor_b.has_value()) {
            binary_reader_args[1] = input_tensor_b->buffer()->address();
        } else {
            class bfloat16 bfloat_scalar(*operation_attributes.scalar);
            uint32_t packed_scalar = pack_two_bfloat16_into_uint32({bfloat_scalar, bfloat_scalar});
            binary_reader_args[1] = packed_scalar;
        }
        binary_reader_args[2] = num_tensor_tiles_per_core;
        binary_reader_args[3] = HtWt;
        binary_reader_args[4] = num_tiles_read / HtWt * HtWt;
        binary_reader_args[5] = num_tiles_read % HtWt;
        binary_reader_args[6] = bnc1 ? 0 : num_tiles_read / HtWt;

        bcast_kernel_args[2] = num_tensor_tiles_per_core;  // Wt

        unary_writer_args[0] = dst_buffer->address();
        unary_writer_args[1] = num_tensor_tiles_per_core;
        unary_writer_args[2] = num_tiles_read;

        num_tiles_read += num_tensor_tiles_per_core;
    }

    if (src0_sharded) {
        UpdateDynamicCircularBufferAddressAndTotalSize(
            program, cb_src0, *src_buffer_a, num_tiles_per_core_group_1 * src0_single_tile_size);
    }

    if (out_sharded) {
        UpdateDynamicCircularBufferAddressAndTotalSize(
            program, cb_output, *dst_buffer, num_tiles_per_core_group_1 * dst_single_tile_size);
    }
}

}  // namespace ttnn::operations::binary
