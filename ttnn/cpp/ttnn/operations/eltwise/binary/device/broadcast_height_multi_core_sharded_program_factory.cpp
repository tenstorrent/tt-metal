// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
// #include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
// #include "ttnn/device_operation.hpp"

namespace ttnn::operations::binary {

static BcastOpMath binary_op_type_to_bcast_op_math(const BinaryOpType binary_op_type) {
    switch (binary_op_type) {
        case BinaryOpType::ADD: return BcastOpMath::ADD;
        case BinaryOpType::SUB: return BcastOpMath::SUB;
        case BinaryOpType::MUL: return BcastOpMath::MUL;
        default: TT_THROW("BinaryOpType cannot be mapped to BcastOpMath");
    }
}

tt::tt_metal::ProgramDescriptor BinaryDeviceOperation::BroadcastHeightMultiCoreSharded::create(
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

    const auto ashape = a.padded_shape();
    const auto bshape = b->padded_shape();
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

    // uint32_t Wt = W / TILE_WIDTH;
    // uint32_t Ht = H / TILE_HEIGHT;

    // uint32_t num_tensor_tiles = NC * Ht * Wt;
    uint32_t num_btensor_tiles = NC * bH * bW / TILE_HW;

    uint32_t bnc1 = (bN * bC == 1) ? 1 : 0;

    tt_metal::ProgramDescriptor program;

    tt_metal::IDevice* device = a.device();

    auto shard_spec = a.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();

    uint32_t ncores_x = device->compute_with_storage_grid_size().x;

    auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == ncores,
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        ncores);

    tt::DataFormat act_df = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat b_df = tt_metal::datatype_to_dataformat_converter(b->get_dtype());
    tt::DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    uint32_t input_tile_size = tt::tt_metal::detail::TileSize(act_df);
    uint32_t input1_tile_size = tt::tt_metal::detail::TileSize(b_df);
    uint32_t output_tile_size = tt::tt_metal::detail::TileSize(out_df);

    TT_FATAL(input_tile_size == output_tile_size, "Input and output tile size should be same");
    uint32_t shard_size_in_bytes = shard_spec.numel() * a.element_size();

    uint32_t num_tile_per_core = (shard_size_in_bytes + input_tile_size - 1) / TILE_HW;  // ceil value
    TT_FATAL(input_tile_size <= shard_size_in_bytes, "Input tile size should be less than shard size");

    uint32_t Wt, Ht;
    if (a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        ncores_x = all_cores.ranges().begin()->end_coord.y + 1;
        Wt = shard_spec.shape[1] / TILE_WIDTH;
        Ht = shard_spec.shape[0] / TILE_HEIGHT;
    } else if (a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
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

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t aligned_input_tile_nbytes =
        round_up_to_mul32(input_tile_size);  // will have issue if the page is not multiple of 32
    uint32_t in_cb_pagesize = aligned_input_tile_nbytes;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_tile_per_core * aligned_input_tile_nbytes,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = act_df,
            .page_size = in_cb_pagesize,
        }}},
        .buffer = a.buffer(),
    });

    uint32_t output_cb_index = tt::CBIndex::c_2;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_tile_per_core * aligned_input_tile_nbytes,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = out_df,
            .page_size = in_cb_pagesize,
        }}},
        .buffer = output.buffer(),
    });

    uint32_t num_input_tiles = (b->padded_shape()[-1] * output.element_size() + TILE_HW - 1) / TILE_HW;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input1_tile_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = b_df,
            .page_size = input1_tile_size,
        }}},
    });

    auto src0_buffer = a.buffer();
    auto src1_buffer = b->buffer();
    auto dst_buffer = output.buffer();
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    static constexpr size_t num_kernels = 2;
    program.kernels.resize(num_kernels);

    auto& binary_reader_kernel = program.kernels[0];
    binary_reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/reader_bcast_h_sharded.cpp";
    binary_reader_kernel.core_ranges = all_cores.ranges();
    binary_reader_kernel.compile_time_args = {(uint32_t)src0_cb_index, (uint32_t)src1_is_dram};
    binary_reader_kernel.config = tt_metal::ReaderConfigDescriptor{};
    binary_reader_kernel.reserve_runtime_args();

    auto& bcast_kernel = program.kernels[1];
    bcast_kernel.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/bcast_h.cpp";
    bcast_kernel.core_ranges = all_cores.ranges();
    bcast_kernel.compile_time_args = {(uint32_t)dst_is_dram};
    bcast_kernel.defines = bcast_op_utils::get_defines_vec(BcastOpDim::H, bcast_math);
    bcast_kernel.config = tt_metal::ComputeConfigDescriptor{};
    bcast_kernel.reserve_runtime_args();
    uint32_t ncores_y = ncores / ncores_x;
    log_debug(
        "ncores {}, ncores_x {}, Wt {}, Ht {}, src0_cb_index {}, src1_cb_index {}, output_cb_index {}, src1_is_dram "
        "{}, dst_is_dram {}",
        ncores,
        ncores_x,
        Wt,
        Ht,
        src0_cb_index,
        src1_cb_index,
        output_cb_index,
        src1_is_dram,
        dst_is_dram);
    for (uint32_t i = 0; i < ncores; i++) {
        CoreCoord core;
        uint32_t offset = 0;
        uint32_t Ht_per_core = 0;
        if (a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            core = {i / ncores_x, i % ncores_x};
            Ht_per_core = Wt * Ht;
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                offset = Wt * (i / ncores_x) + Wt * ncores_y * ((i % ncores_x) / (ncores_x / bN));
            } else {
                offset = Wt * (i % ncores_x) + Wt * ncores_x * ((i / ncores_x) / (ncores_y / bN));
            }
        } else if (a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
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
        uint32_t tile_offset = Wt * ncores;  // used in multi batch weight for block sharded

        auto& binary_reader_args = binary_reader_kernel.runtime_args[core.x][core.y];
        auto& bcast_args = bcast_kernel.runtime_args[core.x][core.y];

        binary_reader_args = {
            b->buffer()->address(),  // 0
            Ht,                      // 1
            Wt,                      // 2
            offset,                  // 3
            Ht_per_core,             // 4
            tile_offset,             // 5
        };

        bcast_args = {
            NC,  // B
            NC,  // B
            Ht,  // Hbatch  for block shardeshardedt
            Wt,  // Wt
        };
    }

    return program;
}

}  // namespace ttnn::operations::binary
