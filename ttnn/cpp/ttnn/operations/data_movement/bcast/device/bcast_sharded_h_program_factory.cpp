// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_sharded_h_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"

namespace ttnn::operations::data_movement::bcast::program {

using namespace tt::tt_metal;
using namespace tt::constants;

BcastShardedHProgramFactory::cached_program_t BcastShardedHProgramFactory::create(
    const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& a = tensor_args.input_a;
    const Tensor& b = tensor_args.input_b;
    Tensor& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    const uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    const uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const uint32_t NC = N * C;

    Program program = CreateProgram();
    IDevice* device = a.device();

    const auto shard_spec = a.shard_spec().value();
    const auto all_cores = shard_spec.grid;
    const uint32_t ncores = shard_spec.num_cores();

    uint32_t ncores_x = device->compute_with_storage_grid_size().x;

    const auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == ncores,
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        ncores);

    const auto act_df = datatype_to_dataformat_converter(a.dtype());
    const auto b_df = datatype_to_dataformat_converter(b.dtype());
    const auto out_df = datatype_to_dataformat_converter(output.dtype());

    const uint32_t input_tile_size = tt::tile_size(act_df);
    const uint32_t input1_tile_size = tt::tile_size(b_df);
    const uint32_t output_tile_size = tt::tile_size(out_df);

    TT_FATAL(input_tile_size == output_tile_size, "Input and output tile size should be same");

    const uint32_t ntiles_along_width = std::ceil(shard_spec.shape[1] / (float)TILE_WIDTH);
    const uint32_t ntiles_along_height = std::ceil(shard_spec.shape[0] / (float)TILE_HEIGHT);
    const uint32_t num_tile_per_core = ntiles_along_width * ntiles_along_height;

    uint32_t Wt, Ht;
    if (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        ncores_x = all_cores.ranges().begin()->end_coord.y + 1;
        Wt = shard_spec.shape[1] / TILE_WIDTH;
        Ht = shard_spec.shape[0] / TILE_HEIGHT;
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        Wt = shard_spec.shape[1] / TILE_WIDTH;
        Ht = shard_spec.shape[0] / TILE_HEIGHT;
        TT_ASSERT(
            (shard_spec.shape[0] % (bN * TILE_HEIGHT) == 0),
            "Shard height per batch must be divisible by TILE_HEIGHT {} {} {} ",
            shard_spec.shape[0],
            bN,
            TILE_HEIGHT);
    } else {
        TT_THROW("Unsupported memory layout");
    }

    TT_ASSERT(
        (shard_spec.shape[0] % TILE_HEIGHT == 0) && (shard_spec.shape[0] % TILE_WIDTH == 0),
        "Shard shapes must be multiple of TILE_HEIGHT ");

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t aligned_input_tile_nbytes =
        round_up_to_mul32(input_tile_size);  // will have issue if the page is not multiple of 32
    const uint32_t in_cb_pagesize = aligned_input_tile_nbytes;
    CircularBufferConfig src0_cb_config =
        CircularBufferConfig(aligned_input_tile_nbytes * num_tile_per_core, {{src0_cb_index, act_df}})
            .set_page_size(src0_cb_index, in_cb_pagesize)
            .set_globally_allocated_address(*a.buffer());
    const auto cb_src0 = CreateCircularBuffer(program, all_cores, src0_cb_config);

    const uint32_t output_cb_index = tt::CBIndex::c_16;
    CircularBufferConfig output_cb_config =
        CircularBufferConfig(aligned_input_tile_nbytes * num_tile_per_core, {{output_cb_index, out_df}})
            .set_page_size(output_cb_index, in_cb_pagesize)
            .set_globally_allocated_address(*output.buffer());
    const auto out_cb = CreateCircularBuffer(program, all_cores, output_cb_config);

    const uint32_t num_input_tiles = (b.padded_shape()[-1] * output.element_size() + TILE_HW - 1) / TILE_HW;
    const uint32_t src1_cb_index = tt::CBIndex::c_1;
    CircularBufferConfig src1_cb_config =
        CircularBufferConfig(num_input_tiles * input1_tile_size, {{src1_cb_index, b_df}})
            .set_page_size(src1_cb_index, input1_tile_size);
    CreateCircularBuffer(program, all_cores, src1_cb_config);

    Buffer* src1_buffer = b.buffer();
    Buffer* dst_buffer = output.buffer();
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index};
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    const bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    const KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/reader_bcast_h_sharded.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    const std::map<std::string, std::string> bcast_defines =
        bcast_op_utils::get_defines(BcastOpDim::H, operation_attributes.math_op);
    const auto bcast_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_h.cpp",
        all_cores,
        ComputeConfig{.compile_args = {}, .defines = bcast_defines});

    const uint32_t ncores_y = ncores / ncores_x;
    for (uint32_t i = 0; i < ncores; i++) {
        CoreCoord core;
        uint32_t offset = 0;
        uint32_t Ht_per_core = 0;
        if (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            core = {i / ncores_x, i % ncores_x};
            Ht_per_core = Wt * Ht;
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                offset = Wt * (i / ncores_x) + Wt * ncores_y * ((i % ncores_x) / (ncores_x / bN));
            } else {
                offset = Wt * (i % ncores_x) + Wt * ncores_x * ((i / ncores_x) / (ncores_y / bN));
            }
        } else if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            core = {i % ncores_x, i / ncores_x};
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                offset = Wt * (core.x + core.y * ncores_x);
            } else {
                offset = Wt * (ncores_y * core.x + core.y);
                if (core.y == ncores_y) {
                    offset = Wt * (ncores_y * ncores_x + core.x);
                }
            }
            Ht_per_core = Ht / bN;
        }
        const uint32_t tile_offset = Wt * ncores;  // used in multi batch weight for block sharded
        SetRuntimeArgs(
            program,
            binary_reader_kernel_id,
            core,
            {
                b.buffer()->address(),  // 0
                Ht,                     // 1
                Wt,                     // 2
                offset,                 // 3
                Ht_per_core,            // 4
                tile_offset,            // 5
            });

        SetRuntimeArgs(
            program,
            bcast_kernel_id,
            core,
            {
                NC,  // B
                Ht,  // Hbatch  for block sharded
                Wt,  // Wt
            });
    }

    return cached_program_t{std::move(program), {binary_reader_kernel_id, bcast_kernel_id, cb_src0, out_cb, ncores_x}};
}

void BcastShardedHProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const BcastParams& /*operation_attributes*/,
    const BcastInputs& tensor_args,
    Tensor& tensor_return_value) {
    Buffer* src_buffer = tensor_args.input_a.buffer();
    Buffer* dst_buffer = tensor_return_value.buffer();
    UpdateDynamicCircularBufferAddress(cached_program.program, cached_program.shared_variables.cb_src0, *src_buffer);
    UpdateDynamicCircularBufferAddress(cached_program.program, cached_program.shared_variables.out_cb, *dst_buffer);
    const auto& a = tensor_args.input_a;
    const auto& b = tensor_args.input_b;
    const auto shard_spec = a.shard_spec().value();
    const auto all_cores = shard_spec.grid;
    const uint32_t ncores = shard_spec.num_cores();
    uint32_t Wt = 0, Ht = 0;
    const auto ashape = tensor_args.input_a.padded_shape();
    const uint32_t N = ashape[0];
    const uint32_t C = ashape[1];
    const uint32_t bN = tensor_args.input_b.padded_shape()[0];
    const uint32_t NC = N * C;
    switch (a.memory_config().memory_layout()) {
        case TensorMemoryLayout::BLOCK_SHARDED:
        case TensorMemoryLayout::WIDTH_SHARDED:
            Wt = shard_spec.shape[1] / TILE_WIDTH;
            Ht = shard_spec.shape[0] / TILE_HEIGHT;
            break;
        default: TT_THROW("Unsupported memory layout");
    }

    const uint32_t ncores_y = ncores / cached_program.shared_variables.ncores_x;
    for (uint32_t i = 0; i < ncores; i++) {
        CoreCoord core;
        uint32_t offset = 0;
        uint32_t Ht_per_core = 0;
        if (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            core = {i / cached_program.shared_variables.ncores_x, i % cached_program.shared_variables.ncores_x};
            Ht_per_core = Wt * Ht;
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                offset = Wt * (i / cached_program.shared_variables.ncores_x) +
                         Wt * ncores_y *
                             ((i % cached_program.shared_variables.ncores_x) /
                              (cached_program.shared_variables.ncores_x / bN));
            } else {
                offset = Wt * (i % cached_program.shared_variables.ncores_x) +
                         Wt * cached_program.shared_variables.ncores_x *
                             ((i / cached_program.shared_variables.ncores_x) / (ncores_y / bN));
            }
        } else if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            core = {i % cached_program.shared_variables.ncores_x, i / cached_program.shared_variables.ncores_x};
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                offset = Wt * (core.x + core.y * cached_program.shared_variables.ncores_x);
            } else {
                offset = Wt * (ncores_y * core.x + core.y);
                if (core.y == ncores_y) {
                    offset = Wt * (ncores_y * cached_program.shared_variables.ncores_x + core.x);
                }
            }
            Ht_per_core = Ht / bN;
        }
        const uint32_t tile_offset = Wt * ncores;

        SetRuntimeArgs(
            cached_program.program,
            cached_program.shared_variables.binary_reader_kernel_id,
            core,
            {
                b.buffer()->address(),  // 0
                Ht,                     // 1
                Wt,                     // 2
                offset,                 // 3
                Ht_per_core,            // 4
                tile_offset,            // 5
            });

        SetRuntimeArgs(
            cached_program.program,
            cached_program.shared_variables.bcast_kernel_id,
            core,
            {
                NC,  // B
                Ht,  // Ht
                Wt,  // Wt
            });
    }
}

}  // namespace ttnn::operations::data_movement::bcast::program
