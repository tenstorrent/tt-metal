// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::reduction::detail::program {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

bool is_tensor_divisible_by_shard(const ttnn::Shape& tensor_shape, const ttnn::Shape& shard_shape) {
    // Only compare common (end) dimensions. Any extra front dimensions would be
    // divisible by the implied 1 in the non-existent dimensions in shard_shape.
    // Use negative dimensions to compare the end of both shapes.
    for (int i = 1; i <= shard_shape.size(); i++) {
        if (shard_shape[-i] == 0 || tensor_shape[-i] % shard_shape[-i] != 0) {
            return false;
        }
    }
    return true;
}
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_and_scale_spatial_dims(
    const ttnn::Shape& shape, uint32_t dim) {
    const auto rank = shape.rank();

    TT_FATAL(rank >= 2, "Shape must have at least two dims.");
    uint32_t Wt = shape[-1] / TILE_WIDTH;
    uint32_t Ht = shape[-2] / TILE_HEIGHT;

    uint32_t reduce_dim = shape[dim];
    uint32_t inner_dims_product = 1;
    for (auto i = dim + 1; i < rank - 2; ++i) {
        inner_dims_product *= shape[i];
    }

    uint32_t inner_tile_size = inner_dims_product * Ht * Wt;
    uint32_t reduce_tile_size = reduce_dim * inner_tile_size;

    return {Wt, Ht, inner_tile_size, reduce_tile_size};
}

}  // namespace

FastReduceNCProgramFactory::cached_program_t FastReduceNCProgramFactory::create(
    const DetailParams& operation_attributes, const DetailInputs& tensor_args, Tensor& tensor_return_value) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = tensor_args.input.device();
    auto program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto single_tile_size = tt::tile_size(cb_data_format);
    const auto cb_1_data_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const auto cb_1_tile_size = tt::tile_size(cb_1_data_format);

    const auto& input_shape = tensor_args.input.padded_shape();
    const auto [Wt, Ht, inner_tile_size, reduce_tile_size] =
        extract_and_scale_spatial_dims(input_shape, static_cast<uint32_t>(operation_attributes.dim));
    const auto num_reduce_input_tile = input_shape[operation_attributes.dim];
    const auto num_output_tiles = tensor_return_value.physical_volume() / TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tensor_args.input.device()->arch(), operation_attributes.compute_kernel_config);
    // Choose granularity as the largest factor of num_reduce_input_tile that is less than or equal to 8.
    // Helps with locality and increases work unit for better performance.
    uint32_t input_granularity;
    for (input_granularity = 8; input_granularity > 1; --input_granularity) {
        if (num_reduce_input_tile % input_granularity == 0) {
            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    const auto num_cores_y = grid.y;

    const uint32_t in0_t = input_granularity * 2;  // input
    const uint32_t in1_t = 1;                      // zero
    const uint32_t intermed0_t = 1;                // accumulated sum
    const uint32_t out0_t = 2;                     // output
    uint32_t shard_factor = 1;

    // When dim=0, nd sharded, tile sizes are the same, the shards are compatible
    // with the kernel accesses, tensor shape is divisible by shard shape, and
    // number of shards is larger than core count, divide the work by shards.
    uint32_t output_shard_size = 1;
    auto input_tile = tensor_args.input.tensor_spec().tile().get_tile_shape();
    auto output_tile = tensor_return_value.tensor_spec().tile().get_tile_shape();
    bool nd_sharded = tensor_args.input.nd_shard_spec().has_value() && tensor_return_value.nd_shard_spec().has_value();
    bool same_tiles = input_tile[0] == output_tile[0] && input_tile[1] == output_tile[1];
    bool divide_by_shards = false;
    const auto& dspec = *tensor_return_value.buffer()->buffer_distribution_spec();
    if (nd_sharded && same_tiles && operation_attributes.dim == 0) {
        const NdShardSpec& input_nd_shard_spec = tensor_args.input.nd_shard_spec().value();
        const NdShardSpec& output_nd_shard_spec = tensor_return_value.nd_shard_spec().value();
        const Shape& input_shard_shape = input_nd_shard_spec.shard_shape;
        bool compatible_shards =
            input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR &&
            input_nd_shard_spec.shard_distribution_strategy == ShardDistributionStrategy::ROUND_ROBIN_1D &&
            output_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR &&
            output_nd_shard_spec.shard_distribution_strategy == ShardDistributionStrategy::ROUND_ROBIN_1D;
        if (compatible_shards && is_tensor_divisible_by_shard(input_shape, input_shard_shape)) {
            uint32_t num_output_shards = dspec.num_shards();
            output_shard_size = dspec.shard_shape_in_pages().volume();
            bool more_shards_than_cores = num_output_shards > (num_cores_x * num_cores_y);
            if (more_shards_than_cores) {
                divide_by_shards = true;
                shard_factor = output_shard_size;
            }
        }
    }
    auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] =
            divide_by_shards ? dspec.core_groups_tuple()
                             : tt::tt_metal::split_work_to_cores(grid, num_output_tiles, /*row_wise=*/true);
    num_cols_per_core_group_1 *= shard_factor;
    num_cols_per_core_group_2 *= shard_factor;

    const auto intermed_cb_data_format = (fp32_dest_acc_en) ? tt::DataFormat::Float32 : cb_data_format;
    const auto intermed_cb_single_tile_size = (fp32_dest_acc_en) ? single_tile_size * 2 : single_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::CircularBufferConfig cb_scr0_config =
        tt_metal::CircularBufferConfig(in0_t * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_scr0_config);

    tt_metal::CircularBufferConfig cb_scr1_config =
        tt_metal::CircularBufferConfig(in1_t * cb_1_tile_size, {{CBIndex::c_1, cb_1_data_format}})
            .set_page_size(CBIndex::c_1, cb_1_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_scr1_config);

    tt_metal::CircularBufferConfig cb_intermed0_config =
        tt_metal::CircularBufferConfig(
            intermed0_t * intermed_cb_single_tile_size, {{CBIndex::c_24, intermed_cb_data_format}})
            .set_page_size(CBIndex::c_24, intermed_cb_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed0_config);

    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(out0_t * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args = {input_granularity, shard_factor, num_cores_to_be_used};
    TensorAccessorArgs(*tensor_args.input.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {shard_factor, num_cores_to_be_used};
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_compile_time_args);

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reader_reduce_nc.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/writer_reduce_nc.cpp";

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program, reader_kernel_file, all_cores, tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program, writer_kernel_file, all_cores, tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1 = {
        num_cols_per_core_group_1, num_reduce_input_tile, input_granularity};
    std::map<std::string, std::string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }
    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp";
    tt_metal::CreateKernel(
        program,
        compute_kernel_file,
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_args_group_1,
            .defines = compute_defines});

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2 = {
            num_cols_per_core_group_2, num_reduce_input_tile, input_granularity};
        compute_kernel_2_id = tt_metal::CreateKernel(
            program,
            compute_kernel_file,
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_args_group_2,
                .defines = compute_defines});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Each core is assigned an output work unit in a row wise round robin
    // fashion. For a given core, the first index is i, and all subsequent
    // indicies are increments of num_cores_to_be_used. The total number of
    // units is num_tiles_per_group times num_cores_to_be_used.
    // For example, with 130 output tiles to be processed and no shards (shard
    // factor is 1) on an 8x8 grid
    // - the increment is 64
    // - the first 2 cores will have num_tiles_per_core 3 and the rest 2
    // - core x=0,y=0 will process output tiles 0, 64, and 128
    // - core x=1,y=0 will process output tiles 1, 65, and 129
    // - core x=2,y=0 will process output tiles 2 and 66
    // - core x=3,y=0 will process output tiles 3 and 67
    // - etc
    // The first tile that needs to be reduced has the same as the output tile.
    // That is the starting point for the reader, which then processes all
    // subsequent tiles to be reduced. The increment for the input indicies is
    // the size of the inner dimensions in tiles (inner_tile_size). The number
    // of tiles to process is the size of the reduce dimension in tiles
    // (reduce_tile_size).
    // The shard factor is used to iterate over shards instead of tiles.
    // It is taken into account in the num_cols_per_core_group variables and
    // the tile_offset is incremented by it for the reader to adjust it's
    // reading pattern.
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {tensor_args.input.buffer()->address(),
             num_reduce_input_tile,
             /*id_range_length=*/num_tiles_per_core * num_cores_to_be_used,
             tile_offset,
             static_cast<uint32_t>(operation_attributes.dim),
             reduce_tile_size,
             inner_tile_size});

        tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {tensor_return_value.buffer()->address(),
             /*id_range_length=*/num_tiles_per_core * num_cores_to_be_used,
             tile_offset});

        tile_offset += shard_factor;
    }

    return cached_program_t{
        std::move(program),
        {/* reader_kernel_id = */ reader_kernel_id,
         /* writer_kernel_id = */ writer_kernel_id,
         /* num_cores_to_be_used = */ num_cores_to_be_used,
         /* num_cores_x = */ num_cores_x}};
}

void FastReduceNCProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const DetailParams&,
    const DetailInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto* input_buffer = tensor_args.input.buffer();
    const auto* output_buffer = tensor_return_value.buffer();
    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& num_cores_to_be_used = cached_program.shared_variables.num_cores_to_be_used;
    const auto& num_cores_x = cached_program.shared_variables.num_cores_x;

    auto& reader_kernel_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_kernel_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
    for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};
        auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
        reader_kernel_args[0] = input_buffer->address();
        auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
        writer_kernel_args[0] = output_buffer->address();
    }
}

}  // namespace ttnn::operations::experimental::reduction::detail::program
