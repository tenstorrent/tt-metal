// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>

#include "detail/util.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/pool/maxpool/max_pool.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"  // for reduce_op_utils

#include "tt_dnn/op_library/sharding_utilities.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/sliding_window/utils.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace tt_metal {

namespace max_pool_helpers {

CoreCoord get_ncores_hw(uint32_t h, uint32_t w, uint32_t avail_cores_h, uint32_t avail_cores_w) {
    CoreCoord cores_shape(0, 0);
    uint32_t total_cores = avail_cores_h * avail_cores_w;
    if (h >= total_cores) {
        TT_ASSERT("Too many cores ({}) :P. Case not yet implemented", total_cores);
    } else {
        // h < total_cores
        if (h == 56) {
            // NOTE: hardcoded for resnet50 maxpool output shape. TODO [AS]: Generalize
            // 56 = 7 * 8
            cores_shape.x = 7;
            cores_shape.y = 8;
        } else {
            TT_ASSERT("This case not handled yet!!");
        }
    }
    return cores_shape;
}

using ttnn::operations::sliding_window::SlidingWindowConfig;
using ttnn::operations::sliding_window::ParallelConfig;

std::tuple<CoreRange, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t> get_decomposition_h(
    uint32_t out_h, uint32_t ncores_h, uint32_t ncores_w) {
    uint32_t out_h_per_core = out_h / (ncores_h * ncores_w);
    uint32_t out_h_per_core_cliff = out_h % (ncores_h * ncores_w);
    std::set<CoreRange> core_range, core_range_cliff;
    if (out_h_per_core_cliff == 0) {
        // no cliff, distribute evenly, corerange is full core rectangle
        core_range.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));
    } else {
        // all but last row
        core_range.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 2, ncores_h - 1)));
        // last row but last core, only the last core is cliff (1D, not 2D)
        core_range.insert(CoreRange(CoreCoord(0, ncores_h - 1), CoreCoord(ncores_w - 2, ncores_h - 1)));
        core_range_cliff.insert(
            CoreRange(CoreCoord(ncores_w - 1, ncores_h - 1), CoreCoord(ncores_w - 1, ncores_h - 1)));
    }
    CoreRange all_cores(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1));
    return std::make_tuple(all_cores, core_range, core_range_cliff, out_h_per_core, out_h_per_core_cliff);
}

// uint32_t get_num_cores(CoreCoord grid_size, uint32_t out_nhw, uint32_t nbatch) {
uint32_t get_num_cores(const Device* device, uint32_t out_nhw, uint32_t nbatch) {
    using namespace tt::constants;
    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t avail_ncores = grid_size.x * grid_size.y;
    uint32_t ncores = 0;
    if (device->arch() == ARCH::GRAYSKULL) {
        // resnet50 shapes
        switch (out_nhw) {
            case 1024:  // test case
                ncores = 32;
                break;
            case 2048:   // test case
            case 4096:   // test case
            case 8192:   // test case
            case 16384:  // test case
            case 32768:  // test case
                ncores = 64;
                break;
            case 3136:   // nbatch = 1
            case 6272:   // nbatch = 2
            case 12544:  // nbatch = 4
            case 25088:  // nbatch = 8
            case 50176:  // nbatch = 16
            case 62720:  // nbatch = 20
                ncores = 98;
                break;
            case 784:  // test case
                ncores = 49;
                break;
            default:
                // TT_ASSERT(false, "General case is not yet handled! Only RN50 shapes supported in multicore.");
                uint32_t out_nhw_per_core = (uint32_t)std::ceil((float)out_nhw / avail_ncores);
                ncores = out_nhw / out_nhw_per_core;
                while (avail_ncores > 0) {
                    if (out_nhw % avail_ncores == 0 && (out_nhw / avail_ncores) % TILE_HEIGHT == 0) {
                        ncores = avail_ncores;
                        break;
                    }
                    --avail_ncores;
                }
                ncores = std::max(avail_ncores, (uint32_t)1);
                break;
        }
    } else if (device->arch() == ARCH::WORMHOLE_B0) {
        uint32_t out_nhw_per_core = (uint32_t)std::ceil((float)out_nhw / avail_ncores);
        ncores = out_nhw / out_nhw_per_core;
        while (avail_ncores > 0) {
            if (out_nhw % avail_ncores == 0 && (out_nhw / avail_ncores) % TILE_HEIGHT == 0) {
                ncores = avail_ncores;
                break;
            }
            --avail_ncores;
        }
        ncores = std::max(avail_ncores, (uint32_t)1);
    } else {
        TT_THROW("Unsupported device arch: {}", device->arch());
    }
    if (ncores == 0)
        TT_THROW("ncores = 0!");
    return ncores;
}

// decompose along height = N * H * W
std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t, uint32_t, uint32_t>
get_decomposition_nhw(const Device* device, uint32_t in_nhw, uint32_t out_nhw, uint32_t nbatch) {
    std::set<CoreRange> all_cores, core_range, core_range_cliff;
    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t avail_ncores = grid_size.x * grid_size.y;
    // // generic decomposition:
    // uint32_t ncores = out_nhw / out_nhw_per_core;

    // hardcoded for resnet shapes:
    uint32_t ncores = 0, out_nhw_per_core = 0, in_nhw_per_core = 0;
    ncores = get_num_cores(device, out_nhw, nbatch);

    out_nhw_per_core = out_nhw / ncores;
    in_nhw_per_core = in_nhw / ncores;
    uint32_t ncores_w = grid_size.x;  // 12
    uint32_t ncores_h = ncores / ncores_w;
    uint32_t ncores_cliff_h = 0;
    if (ncores % ncores_w != 0)
        ncores_cliff_h = 1;
    uint32_t ncores_cliff_w = ncores % ncores_w;
    // NOTE: Cliff core is not yet handled, assuming (out_nhw / ncores) is a whole number.
    uint32_t in_nhw_per_core_cliff = 0;
    uint32_t out_nhw_per_core_cliff = 0;

    // all but last row
    core_range.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));
    all_cores.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));
    // last row
    if (ncores_cliff_h > 0) {
        core_range.insert(CoreRange(CoreCoord(0, ncores_h), CoreCoord(ncores_cliff_w - 1, ncores_h)));
        all_cores.insert(CoreRange(CoreCoord(0, ncores_h), CoreCoord(ncores_cliff_w - 1, ncores_h)));
    }

    return std::make_tuple(
        ncores,
        all_cores,
        core_range,
        core_range_cliff,
        in_nhw_per_core,
        in_nhw_per_core_cliff,
        out_nhw_per_core,
        out_nhw_per_core_cliff);
}

}  // namespace max_pool_helpers

// this version uses distribution along height = N * H * W
operation::ProgramWithCallbacks max_pool_2d_multi_core_generic(
    const Tensor& input,
    Tensor& output,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t kernel_size_h,
    uint32_t kernel_size_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    const MemoryConfig& out_mem_config,
    uint32_t nblocks) {
    Program program = CreateProgram();

    // This should allocate a DRAM buffer on the device
    Device* device = input.device();
    Buffer* src_dram_buffer = input.buffer();
    Buffer* dst_dram_buffer = output.buffer();

    tt::tt_metal::LegacyShape input_shape = input.get_legacy_shape();
    tt::tt_metal::LegacyShape output_shape = output.get_legacy_shape();

    // NOTE: input is assumed to be in {N, 1, H * W, C }

    // TODO [AS]: Support other data formats??
    DataFormat in_df = datatype_to_dataformat_converter(input.get_dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);
    uint32_t in_nbytes_c = input_shape[3] * in_nbytes;                                      // row of input (channels)
    uint32_t out_nbytes_c = output_shape[3] * out_nbytes;                                   // row of output (channels)
    TT_ASSERT((in_nbytes_c & (in_nbytes_c - 1)) == 0, "in_nbytes_c should be power of 2");  // in_nbytes_c is power of 2
    TT_ASSERT(
        (out_nbytes_c & (out_nbytes_c - 1)) == 0, "out_nbytes_c should be power of 2");  // out_nbytes_c is power of 2

    uint32_t nbatch = input_shape[0];
    TT_ASSERT(nbatch == output_shape[0], "Mismatch in N for input and output!!");

    uint32_t kernel_size_hw = kernel_size_w * kernel_size_h;  // number of valid rows, to read
    uint32_t kernel_size_hw_padded = ceil_multiple_of(kernel_size_hw, constants::TILE_HEIGHT);
    uint32_t in_ntiles_hw = (uint32_t)std::ceil((float)kernel_size_hw_padded / constants::TILE_HEIGHT);
    uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / constants::TILE_WIDTH);
    uint32_t out_ntiles_hw = (uint32_t)std::ceil((float)output_shape[2] / constants::TILE_HEIGHT);
    uint32_t out_ntiles_c = (uint32_t)std::ceil((float)output_shape[3] / constants::TILE_WIDTH);

    uint32_t out_nelems = nblocks;  // TODO [AS]: Remove hard coding after identifying optimal param val
                                    // Also ensure the calculated ncores is good
    uint32_t out_w_loop_count = std::ceil((float)out_w / out_nelems);

    uint32_t in_hw = in_h * in_w;
    uint32_t in_nhw = in_hw * nbatch;
    uint32_t out_hw = out_h * out_w;
    uint32_t out_nhw = out_hw * nbatch;

    // distributing out_hw across the grid
    auto grid_size = device->compute_with_storage_grid_size();
    auto
        [ncores,
         all_cores,
         core_range,
         core_range_cliff,
         in_nhw_per_core,
         in_nhw_per_core_cliff,
         out_nhw_per_core,
         out_nhw_per_core_cliff] = max_pool_helpers::get_decomposition_nhw(device, in_nhw, out_nhw, nbatch);
    if (input.memory_config().is_sharded()) {
        all_cores = input.shard_spec().value().grid;
        uint32_t ncores = all_cores.num_cores();
        core_range = all_cores;
        core_range_cliff = CoreRangeSet({});
        in_nhw_per_core = input.shard_spec().value().shape[0];
        in_nhw_per_core_cliff = 0;
        out_nhw_per_core = out_nhw / ncores;
        out_nhw_per_core_cliff = 0;
    }
    uint32_t ncores_w = grid_size.x;

    // TODO: support generic nblocks
    TT_ASSERT(
        out_nhw_per_core % nblocks == 0,
        "number of sticks per core ({}) should be divisible by nblocks ({})",
        out_nhw_per_core,
        nblocks);
    // TODO: support generic values for in_nhw_per_core
    TT_ASSERT(
        (in_nhw_per_core & (in_nhw_per_core - 1)) == 0, "in_nhw_per_core {} needs to be power of 2!", in_nhw_per_core);

    uint32_t in_nhw_per_core_rem_mask = in_nhw_per_core - 1;  // NOTE: assuming in_nhw_per_core is power of 2

    // CBs
    uint32_t multi_buffering_factor = 2;

    // scalar CB as coefficient of reduce
    uint32_t in_scalar_cb_id = CB::c_in4;
    uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    uint32_t in_scalar_cb_npages = 1;
    CircularBufferConfig in_scalar_cb_config =
        CircularBufferConfig(in_scalar_cb_npages * in_scalar_cb_pagesize, {{in_scalar_cb_id, in_df}})
            .set_page_size(in_scalar_cb_id, in_scalar_cb_pagesize);
    auto in_scalar_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_scalar_cb_config);

    CBHandle raw_in_cb = 0;
    if (input.memory_config().is_sharded()) {
        // incoming data is the input cb instead of raw l1/dram addr
        auto raw_in_cb_id = CB::c_in2;
        uint32_t raw_in_cb_npages = in_nhw_per_core;
        uint32_t raw_in_cb_pagesize = in_nbytes_c;
        CircularBufferConfig raw_in_cb_config =
            CircularBufferConfig(raw_in_cb_npages * raw_in_cb_pagesize, {{raw_in_cb_id, in_df}})
                .set_page_size(raw_in_cb_id, raw_in_cb_pagesize)
                .set_globally_allocated_address(*input.buffer());
        raw_in_cb = CreateCircularBuffer(program, all_cores, raw_in_cb_config);
    }

    // reader output == input to tilize
    uint32_t in_cb_id = CB::c_in0;  // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_page_nelems_padded = ceil_multiple_of(
        input_shape[3] * kernel_size_hw_padded,
        constants::TILE_HW);  // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    uint32_t in_cb_pagesize = in_nbytes * in_cb_page_nelems_padded;
    uint32_t in_cb_npages = multi_buffering_factor * out_nelems;
    CircularBufferConfig in_cb_config = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id, in_df}})
                                            .set_page_size(in_cb_id, in_cb_pagesize);
    auto in_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config);

    // output of tilize == input to reduce
    uint32_t in_tiled_cb_id = CB::c_intermed0;  // tiled input
    uint32_t in_tiled_cb_pagesize = tile_size(in_df);
    uint32_t in_tiled_cb_npages = in_ntiles_c * in_ntiles_hw * out_nelems;
    CircularBufferConfig in_tiled_cb_config =
        CircularBufferConfig(in_tiled_cb_npages * in_tiled_cb_pagesize, {{in_tiled_cb_id, in_df}})
            .set_page_size(in_tiled_cb_id, in_tiled_cb_pagesize);
    auto in_tiled_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_tiled_cb_config);

    // output of reduce == writer to write
    uint32_t out_cb_id = CB::c_out0;  // output rows in RM
    uint32_t out_cb_pagesize = tile_size(out_df) * out_ntiles_c * out_nelems;
    uint32_t out_cb_npages = multi_buffering_factor;
    CircularBufferConfig cb_out_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
                                             .set_page_size(out_cb_id, out_cb_pagesize);
    auto cb_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    CBHandle cb_sharded_out = 0;
    if (output.memory_config().is_sharded()) {
        uint32_t sharded_out_cb_id = CB::c_out1;  // output rows in RM

        uint32_t sharded_out_num_pages = output.shard_spec().value().shape[0];

        uint32_t sharded_out_cb_page_size =
            output.shard_spec().value().shape[1] * out_nbytes;  // there is just one row of channels after reduction
        CircularBufferConfig cb_sharded_out_config =
            CircularBufferConfig(sharded_out_num_pages * sharded_out_cb_page_size, {{sharded_out_cb_id, out_df}})
                .set_page_size(sharded_out_cb_id, sharded_out_cb_page_size)
                .set_globally_allocated_address(*output.buffer());
        cb_sharded_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_sharded_out_config);
    }

    // Construct const buffer with -INF
    // uint32_t const_buffer_size = 32;
    uint32_t const_buffer_size = input_shape[3];  // set it equal to 1 row
    auto minus_inf_const_buffer = owned_buffer::create(std::vector<bfloat16>(const_buffer_size, bfloat16(0xf7ff)));
    const Tensor minus_inf_const_tensor =
        Tensor(
            OwnedStorage{minus_inf_const_buffer},
            tt::tt_metal::LegacyShape({1, 1, 1, const_buffer_size}),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR)
            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1});
    auto minus_inf_const_tensor_addr = minus_inf_const_tensor.buffer()->address();

#if 0
    {   // debug
        log_debug(LogOp, "in_cb :: PS = {}, NP = {}", in_cb_pagesize, in_cb_npages);
        log_debug(LogOp, "in_scalar_cb :: PS = {}, NP = {}", in_scalar_cb_pagesize, in_scalar_cb_npages);
        log_debug(LogOp, "in_tiled_cb :: PS = {}, NP = {}", in_tiled_cb_pagesize, in_tiled_cb_npages);
        log_debug(LogOp, "out_cb :: PS = {}, NP = {}", out_cb_pagesize, out_cb_npages);
        log_debug(LogOp, "in_addr: {}", src_dram_buffer->address());
        log_debug(LogOp, "out_addr: {}", dst_dram_buffer->address());
        log_debug(LogOp, "nbatch: {}", nbatch);
        log_debug(LogOp, "kernel_size_h: {}", kernel_size_h);
        log_debug(LogOp, "kernel_size_w: {}", kernel_size_w);
        log_debug(LogOp, "kernel_size_hw: {}", kernel_size_hw);
        log_debug(LogOp, "kernel_size_hw_padded: {}", kernel_size_hw_padded);
        log_debug(LogOp, "stride_h: {}", stride_h);
        log_debug(LogOp, "stride_w: {}", stride_w);
        log_debug(LogOp, "pad_h: {}", pad_h);
        log_debug(LogOp, "pad_w: {}", pad_w);
        log_debug(LogOp, "out_h: {}", out_h);
        log_debug(LogOp, "out_w: {}", out_w);
        log_debug(LogOp, "out_hw: {}", output_shape[2]);
        log_debug(LogOp, "out_c: {}", output_shape[3]);
        log_debug(LogOp, "in_nbytes_c: {}", in_nbytes_c);
        log_debug(LogOp, "out_nbytes_c: {}", out_nbytes_c);
        log_debug(LogOp, "in_h: {}", in_h);
        log_debug(LogOp, "in_w: {}", in_w);
        log_debug(LogOp, "in_hw: {}", input_shape[2]);
        log_debug(LogOp, "in_hw_padded: {}", in_hw);
        log_debug(LogOp, "in_c: {}", input_shape[3]);
        log_debug(LogOp, "out_hw_padded: {}", out_hw);
        log_debug(LogOp, "out_ntiles_hw: {}", out_ntiles_hw);
        log_debug(LogOp, "out_ntiles_c: {}", out_ntiles_c);
        log_debug(LogOp, "out_nelems: {}", out_nelems);
        log_debug(LogOp, "out_w_loop_count: {}", out_w_loop_count);
        log_debug(LogOp, "out_hw: {}", out_hw);
        log_debug(LogOp, "minus_inf_const_tensor_addr: {}", minus_inf_const_tensor_addr);
        log_debug(LogOp, "minus_inf_const_tensor_size: {}", minus_inf_const_buffer.size());
        log_debug(LogOp, "ncores: {}", ncores);
        log_debug(LogOp, "in_nhw_per_core: {}", in_nhw_per_core);
        log_debug(LogOp, "out_nhw_per_core: {}", out_nhw_per_core);
        log_debug(LogOp, "in_nhw_per_core_rem_mask: {}", in_nhw_per_core_rem_mask);
        log_debug(LogOp, "is_in_sharded: {}", input.memory_config().is_sharded());
        log_debug(LogOp, "is_out_sharded: {}", output.memory_config().is_sharded());
    }
#endif

    const uint32_t reader_noc = 0;
    const uint32_t writer_noc = 1;

    std::map<CoreCoord, CoreCoord> left_neighbor_core, right_neighbor_core;
    if (input.memory_config().is_sharded()) {
        utils::init_neighbor_core_xy_mapping(
            grid_size,
            left_neighbor_core,
            right_neighbor_core,
            input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
    }

    /**
     * Reader Kernel: input rows -> input cb
     */
    float one = 1.;
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t*>(&one);
    std::vector<uint32_t> reader_ct_args = {
        input.memory_config().buffer_type == BufferType::DRAM ? (uint)1 : (uint)0,
        out_mem_config.buffer_type == BufferType::DRAM ? (uint)1 : (uint)0,
        bf16_one_u32,
        out_nelems,
        static_cast<uint32_t>(((in_nbytes_c & (in_nbytes_c - 1)) == 0) ? 1 : 0),  // is in_nbytes_c power of 2
        stride_h,
        stride_w,
        reader_noc,
        writer_noc};
    uint32_t in_log_base_2_of_page_size = (uint32_t)std::log2((float)in_nbytes_c);
    std::vector<uint32_t> reader_rt_args = {
        src_dram_buffer->address(),
        dst_dram_buffer->address(),
        kernel_size_h,
        kernel_size_w,
        kernel_size_hw,
        kernel_size_hw_padded,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        out_h,
        out_w,
        output_shape[2],
        output_shape[3],
        in_nbytes_c,
        out_nbytes_c,
        in_h,
        in_w,
        input_shape[2],
        input_shape[3],
        out_ntiles_hw,
        out_ntiles_c,
        in_cb_pagesize,
        out_cb_pagesize,
        in_cb_page_nelems_padded,
        out_w_loop_count,
        in_log_base_2_of_page_size,
        nbatch,
        in_hw,
        out_hw,
        // these are set later in the following
        0,  // start_out_h_i
        0,  // end_out_h_i
        0,  // base_start_h
        0,  // start_out_row_id
        minus_inf_const_tensor_addr,
        const_buffer_size * in_nbytes,
        (in_cb_page_nelems_padded * out_nelems * 2) >> 5,  // TODO: generalize num rows to fill in in_cb
        0,                                                 // core_offset_in_row_id
        0,                                                 // core_out_w_i_start
        0,                                                 // core_out_h_i_start
        out_nhw_per_core,                                  // nsticks_per_core
        0,                                                 // core_offset_out_row_id
        out_nhw_per_core / nblocks,                        // loop count with blocks
        // the following are for sharded input
        0,                // 43: local_out_stick_start
        out_hw,           // out_nsticks_per_batch
        0,                // local_in_stick_start
        0,                // local_in_stick_end
        in_hw,            // in_nsticks_per_batch
        in_nhw_per_core,  // in_nsticks_per_core
        0,                // has_left
        0,                // left_noc_x
        0,                // left_noc_y
        0,                // has_right
        0,                // right_noc_x
        0,                // right_noc_y
        in_nhw_per_core_rem_mask,
        0,  // 56: has_left_left,
        0,  // left_left_noc_x,
        0,  // left_left_noc_y,
        0,  // has_right_right,
        0,  // right_right_noc_x,
        0,  // right_right_noc_y,
        0,  // left_in_stick_start,
        0,  // right_in_stick_end,
        0,  // my_core
    };
    auto reader_config = ReaderDataMovementConfig(reader_ct_args);
    std::string reader_kernel_fname;
    if (input.memory_config().is_sharded()) {
        // sharded, without halo
        reader_kernel_fname =
            std::string("ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/dataflow/reader_max_pool_2d_multi_core_sharded.cpp");
    } else {
        reader_kernel_fname =
            std::string("ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/dataflow/reader_max_pool_2d_multi_core.cpp");
    }
    auto reader_kernel = CreateKernel(program, reader_kernel_fname, all_cores, reader_config);

    /**
     * Writer Kernel: output cb -> output rows
     */
    std::map<string, string> writer_defines;
    if (output.memory_config().is_sharded()) {
        writer_defines["SHARDED_OUT"] = "1";
    }
    std::vector<uint32_t> writer_ct_args = reader_ct_args;
    auto writer_config = WriterDataMovementConfig(writer_ct_args, writer_defines);
    std::string writer_kernel_fname(
        "ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/dataflow/writer_max_pool_2d_multi_core.cpp");
    auto writer_kernel = CreateKernel(program, writer_kernel_fname, all_cores, writer_config);

    /**
     * Compute Kernel: input cb -> tilize_block -> input tiles -> reduce_h max -> output tiles -> untilize_block ->
     * output cb
     */
    std::vector<uint32_t> compute_ct_args = {
        in_ntiles_hw,
        in_ntiles_c,
        in_ntiles_hw * in_ntiles_c,
        kernel_size_hw,
        out_h,
        out_w,
        (uint32_t)std::ceil((float)output_shape[2] / constants::TILE_HEIGHT),
        (uint32_t)std::ceil((float)output_shape[3] / constants::TILE_WIDTH),
        out_nelems,
        out_w_loop_count,
        nbatch,
        out_nhw_per_core,
        0,                           // Split reader
        out_nhw_per_core / nblocks,  // loop count with blocks
        input_shape[3],
    };
    auto compute_ct_args_cliff = compute_ct_args;
    auto reduce_op = ReduceOpMath::MAX;
    auto reduce_dim = ReduceOpDim::H;
    auto compute_config = ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = compute_ct_args,
        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
    std::string compute_kernel_fname("ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/compute/max_pool_multi_core.cpp");
    auto compute_kernel = CreateKernel(program, compute_kernel_fname, core_range, compute_config);

    if (out_nhw_per_core_cliff > 0) {
        TT_ASSERT(false, "The cliff core case is not yet handled");  // TODO
        // there is a cliff core
        compute_ct_args_cliff[11] = out_nhw_per_core_cliff;
        auto compute_config_cliff = ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_ct_args_cliff,
            .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
        auto compute_kernel_cliff = CreateKernel(program, compute_kernel_fname, core_range_cliff, compute_config);
    }

    // calculate and set the start/end h_i for each core
    // for all but last core (cliff)
    const auto& cores = grid_to_cores(ncores, ncores_w, grid_size.y, true);
    uint32_t core_out_h_i = 0;
    uint32_t core_out_w_i = 0;
    int32_t curr_start_h = -pad_h;
    if (out_nhw_per_core_cliff > 0) {
        // TODO? ... not yet handled
        TT_ASSERT(false, "The cliff core case is not yet handled");  // TODO
    } else {
        uint32_t core_batch_offset = 0;
        uint32_t curr_out_stick_id = 0;  // track output sticks with batch folded in
        int32_t curr_in_stick_id = 0;    // track input sticks with batch folded in
        uint32_t core_out_w_i_start = 0;
        uint32_t core_out_h_i_start = 0;
        for (int32_t i = 0; i < ncores; ++i) {
            const CoreCoord& core_coord = cores[i];  // logical
            reader_rt_args[37] = (curr_in_stick_id / in_hw) * in_hw;
            core_out_w_i_start = curr_out_stick_id % out_w;
            core_out_h_i_start = (curr_out_stick_id / out_w) % out_h;
            reader_rt_args[38] = core_out_w_i_start;
            reader_rt_args[39] = core_out_h_i_start;
            reader_rt_args[41] = curr_out_stick_id;

            if (input.memory_config().is_sharded()) {
                reader_rt_args[43] = curr_out_stick_id;
                reader_rt_args[45] = curr_in_stick_id;
                reader_rt_args[46] = curr_in_stick_id + in_nhw_per_core;

                reader_rt_args[64] = i;  // my_core

                if (left_neighbor_core.count(core_coord) > 0) {
                    CoreCoord left_core = left_neighbor_core.at(core_coord);
                    CoreCoord left_noc = device->worker_core_from_logical_core(left_core);
                    reader_rt_args[49] = 1;
                    reader_rt_args[50] = (uint32_t)left_noc.x;
                    reader_rt_args[51] = (uint32_t)left_noc.y;

                    // left-left
                    if (left_neighbor_core.count(left_core) > 0) {
                        CoreCoord left_left_core = left_neighbor_core.at(left_core);
                        CoreCoord left_left_noc = device->worker_core_from_logical_core(left_left_core);
                        reader_rt_args[56] = 1;
                        reader_rt_args[57] = (uint32_t)left_left_noc.x;
                        reader_rt_args[58] = (uint32_t)left_left_noc.y;
                        reader_rt_args[62] = (uint32_t)(curr_in_stick_id - (int32_t)in_nhw_per_core);
                    } else {
                        reader_rt_args[56] = 0;
                    }
                } else {
                    reader_rt_args[49] = 0;
                }
                if (right_neighbor_core.count(core_coord) > 0) {
                    CoreCoord right_core = right_neighbor_core.at(core_coord);
                    CoreCoord right_noc = device->worker_core_from_logical_core(right_core);
                    reader_rt_args[52] = 1;
                    reader_rt_args[53] = (uint32_t)right_noc.x;
                    reader_rt_args[54] = (uint32_t)right_noc.y;

                    // right-right
                    if (right_neighbor_core.count(right_core) > 0) {
                        CoreCoord right_right_core = right_neighbor_core.at(right_core);
                        CoreCoord right_right_noc = device->worker_core_from_logical_core(right_right_core);
                        reader_rt_args[59] = 1;
                        reader_rt_args[60] = (uint32_t)right_right_noc.x;
                        reader_rt_args[61] = (uint32_t)right_right_noc.y;
                        reader_rt_args[63] = (uint32_t)(curr_in_stick_id + 2 * in_nhw_per_core);
                    } else {
                        reader_rt_args[59] = 0;
                    }
                } else {
                    reader_rt_args[52] = 0;
                }
            }

            SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
            std::vector<uint32_t> writer_rt_args = reader_rt_args;
            SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);

            curr_out_stick_id += out_nhw_per_core;
            curr_in_stick_id += in_nhw_per_core;
        }
    }

    auto override_runtime_arguments_callback =
        [reader_kernel, writer_kernel, raw_in_cb, cb_sharded_out, cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors.at(0).buffer();
            bool input_sharded = input_tensors.at(0).is_sharded();

            auto dst_buffer = output_tensors.at(0).buffer();
            bool out_sharded = output_tensors.at(0).is_sharded();

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel);
            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = src_buffer->address();
                    runtime_args[1] = dst_buffer->address();
                }
                {
                    auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = src_buffer->address();
                    runtime_args[1] = dst_buffer->address();
                }
            }
            if (input_sharded) {
                UpdateDynamicCircularBufferAddress(program, raw_in_cb, *src_buffer);
            }
            if (out_sharded) {
                UpdateDynamicCircularBufferAddress(program, cb_sharded_out, *dst_buffer);
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

// this version uses distribution along height = N * H * W
operation::ProgramWithCallbacks max_pool_2d_multi_core_sharded_with_halo_v2_impl(
    Program& program,
    const Tensor& input,
    const Tensor& reader_indices,
    Tensor& output,
    uint32_t in_n,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t kernel_size_h,
    uint32_t kernel_size_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    const MemoryConfig& out_mem_config,
    uint32_t nblocks) {
    // This should allocate a DRAM buffer on the device
    Device* device = input.device();
    Buffer* src_dram_buffer = input.buffer();
    Buffer* reader_indices_buffer = reader_indices.buffer();
    Buffer* dst_dram_buffer = output.buffer();

    tt::tt_metal::LegacyShape input_shape = input.get_legacy_shape();
    tt::tt_metal::LegacyShape output_shape = output.get_legacy_shape();

    DataFormat in_df = datatype_to_dataformat_converter(input.get_dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);
    uint32_t in_nbytes_c = input_shape[3] * in_nbytes;                                      // row of input (channels)
    uint32_t out_nbytes_c = output_shape[3] * out_nbytes;                                   // row of output (channels)
    TT_ASSERT((in_nbytes_c & (in_nbytes_c - 1)) == 0, "in_nbytes_c should be power of 2");  // in_nbytes_c is power of 2
    TT_ASSERT(
        (out_nbytes_c & (out_nbytes_c - 1)) == 0, "out_nbytes_c should be power of 2");  // out_nbytes_c is power of 2

    DataFormat indices_df = DataFormat::RawUInt16;  // datatype_to_dataformat_converter(reader_indices.get_dtype());
    uint32_t indices_nbytes = datum_size(indices_df);

    uint32_t kernel_size_hw = kernel_size_w * kernel_size_h;  // number of valid rows, to read
    uint32_t kernel_size_hw_padded = ceil_multiple_of(kernel_size_hw, constants::TILE_HEIGHT);
    uint32_t in_ntiles_hw = (uint32_t)std::ceil((float)kernel_size_hw_padded / constants::TILE_HEIGHT);
    uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / constants::TILE_WIDTH);
    uint32_t out_ntiles_c = (uint32_t)std::ceil((float)output_shape[3] / constants::TILE_WIDTH);

    TT_ASSERT(nblocks == 1, "Multiple blocks not yet supported");

    uint32_t tile_w = constants::TILE_WIDTH;
    if (input_shape[3] < constants::TILE_WIDTH) {
        TT_FATAL(input_shape[3] == 16);
        tile_w = constants::FACE_WIDTH;
    }
    uint32_t out_w_loop_count = std::ceil((float)out_w / nblocks);

    // distributing out_hw across the grid
    auto grid_size = device->compute_with_storage_grid_size();
    auto all_cores = input.shard_spec().value().grid;
    uint32_t ncores = all_cores.num_cores();
    auto core_range = all_cores;
    auto core_range_cliff = CoreRangeSet({});
    uint32_t in_nhw_per_core = input.shard_spec()->shape[0];
    uint32_t in_nhw_per_core_cliff = 0;
    uint32_t out_nhw_per_core = output.shard_spec()->shape[0];

    uint32_t ncores_w = grid_size.x;

    // TODO: support generic nblocks
    TT_ASSERT(
        out_nhw_per_core % nblocks == 0,
        "number of sticks per core ({}) should be divisible by nblocks ({})",
        out_nhw_per_core,
        nblocks);

    // CBs
    uint32_t multi_buffering_factor = 2;

    uint32_t split_reader = 1;

    // scalar CB as coefficient of reduce
    uint32_t in_scalar_cb_id = CB::c_in4;
    uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    uint32_t in_scalar_cb_npages = 1;
    CircularBufferConfig in_scalar_cb_config =
        CircularBufferConfig(in_scalar_cb_npages * in_scalar_cb_pagesize, {{in_scalar_cb_id, in_df}})
            .set_page_size(in_scalar_cb_id, in_scalar_cb_pagesize);
    auto in_scalar_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_scalar_cb_config);
    log_debug(LogOp, "CB {} :: PS = {}, NP = {}", in_scalar_cb_id, in_scalar_cb_pagesize, in_scalar_cb_npages);

    // incoming data is the input cb instead of raw l1/dram addr
    // this input shard has halo and padding inserted.
    auto raw_in_cb_id = CB::c_in2;
    uint32_t raw_in_cb_npages = input.shard_spec().value().shape[0];
    uint32_t raw_in_cb_pagesize = in_nbytes_c;
    CircularBufferConfig raw_in_cb_config =
        CircularBufferConfig(raw_in_cb_npages * raw_in_cb_pagesize, {{raw_in_cb_id, in_df}})
            .set_page_size(raw_in_cb_id, raw_in_cb_pagesize)
            .set_globally_allocated_address(*input.buffer());
    auto raw_in_cb = CreateCircularBuffer(program, all_cores, raw_in_cb_config);
    log_debug(LogOp, "CB {} :: PS = {}, NP = {}", raw_in_cb_id, raw_in_cb_pagesize, raw_in_cb_npages);

    // reader indices
    auto in_reader_indices_cb_id = CB::c_in3;
    uint32_t in_reader_indices_cb_pagesize =
        round_up(out_nhw_per_core * indices_nbytes, 4);  // pagesize needs to be multiple of 4
    uint32_t in_reader_indices_cb_npages = 1;
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}",
        in_reader_indices_cb_id,
        in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages);
    CircularBufferConfig in_reader_indices_cb_config =
        CircularBufferConfig(
            in_reader_indices_cb_npages * in_reader_indices_cb_pagesize, {{in_reader_indices_cb_id, indices_df}})
            .set_page_size(in_reader_indices_cb_id, in_reader_indices_cb_pagesize)
            .set_globally_allocated_address(*reader_indices_buffer);
    auto in_reader_indices_cb = CreateCircularBuffer(program, all_cores, in_reader_indices_cb_config);

    // reader output == input to tilize
    uint32_t in_cb_id_0 = CB::c_in0;  // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_id_1 = CB::c_in1;  // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_page_padded = ceil_multiple_of(
        input_shape[3] * kernel_size_hw_padded,
        constants::TILE_HW);  // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    uint32_t in_cb_pagesize = in_nbytes * in_cb_page_padded;
    uint32_t in_cb_npages = multi_buffering_factor * nblocks;

    CircularBufferConfig in_cb_config_0 = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id_0, in_df}})
                                              .set_page_size(in_cb_id_0, in_cb_pagesize);
    auto in_cb_0 = tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config_0);
    log_debug(LogOp, "CB {} :: PS = {}, NP = {}", in_cb_id_0, in_cb_pagesize, in_cb_npages);

    if (split_reader) {
        CircularBufferConfig in_cb_config_1 = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id_1, in_df}})
                                                  .set_page_size(in_cb_id_1, in_cb_pagesize);
        auto in_cb_1 = tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config_1);
        log_debug(LogOp, "CB {} :: PS = {}, NP = {}", in_cb_id_1, in_cb_pagesize, in_cb_npages);
    }

    // output of tilize == input to reduce
    uint32_t in_tiled_cb_id = CB::c_intermed0;  // tiled input
    uint32_t in_tiled_cb_pagesize = tile_size(in_df);
    uint32_t in_tiled_cb_npages = in_ntiles_c * in_ntiles_hw * nblocks;
    CircularBufferConfig in_tiled_cb_config =
        CircularBufferConfig(in_tiled_cb_npages * in_tiled_cb_pagesize, {{in_tiled_cb_id, in_df}})
            .set_page_size(in_tiled_cb_id, in_tiled_cb_pagesize);
    auto in_tiled_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_tiled_cb_config);
    log_debug(LogOp, "CB {} :: PS = {}, NP = {}", in_tiled_cb_id, in_tiled_cb_pagesize, in_tiled_cb_npages);

    // output of reduce == writer to write
    uint32_t out_cb_id = CB::c_out0;  // output rows in RM
    // after reduction
    uint32_t out_cb_pagesize =
        output.shard_spec().value().shape[1] * out_nbytes;  // there is just one row of channels after reduction
    uint32_t out_cb_npages = output.shard_spec().value().shape[0];
    CircularBufferConfig cb_out_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
                                             .set_page_size(out_cb_id, out_cb_pagesize)
                                             .set_globally_allocated_address(*output.buffer());
    ;
    auto cb_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);
    log_debug(LogOp, "CB {} :: PS = {}, NP = {}", out_cb_id, out_cb_pagesize, out_cb_npages);

    TT_FATAL(output.memory_config().is_sharded());

    #if 1
    {  // debug
        log_debug(LogOp, "raw_in_cb :: PS = {}, NP = {}", raw_in_cb_pagesize, raw_in_cb_npages);
        log_debug(LogOp, "in_cb :: PS = {}, NP = {}", in_cb_pagesize, in_cb_npages);
        log_debug(
            LogOp,
            "in_reader_indices_cb :: PS = {}, NP = {}",
            in_reader_indices_cb_pagesize,
            in_reader_indices_cb_npages);
        log_debug(LogOp, "in_scalar_cb :: PS = {}, NP = {}", in_scalar_cb_pagesize, in_scalar_cb_npages);
        log_debug(LogOp, "in_tiled_cb :: PS = {}, NP = {}", in_tiled_cb_pagesize, in_tiled_cb_npages);
        log_debug(LogOp, "out_cb :: PS = {}, NP = {}", out_cb_pagesize, out_cb_npages);
        log_debug(LogOp, "in_addr: {}", src_dram_buffer->address());
        log_debug(LogOp, "in_reader_indices_addr: {}", reader_indices_buffer->address());
        log_debug(LogOp, "out_addr: {}", dst_dram_buffer->address());
        log_debug(LogOp, "kernel_size_h: {}", kernel_size_h);
        log_debug(LogOp, "kernel_size_w: {}", kernel_size_w);
        log_debug(LogOp, "kernel_size_hw: {}", kernel_size_hw);
        log_debug(LogOp, "kernel_size_hw_padded: {}", kernel_size_hw_padded);
        log_debug(LogOp, "stride_h: {}", stride_h);
        log_debug(LogOp, "stride_w: {}", stride_w);
        log_debug(LogOp, "pad_h: {}", pad_h);
        log_debug(LogOp, "pad_w: {}", pad_w);
        log_debug(LogOp, "out_h: {}", out_h);
        log_debug(LogOp, "out_w: {}", out_w);
        log_debug(LogOp, "out_w_loop_count: {}", out_w_loop_count);
        log_debug(LogOp, "out_c: {}", output_shape[3]);
        log_debug(LogOp, "out_nbytes_c: {}", out_nbytes_c);
        log_debug(LogOp, "in_h: {}", in_h);
        log_debug(LogOp, "in_w: {}", in_w);
        log_debug(LogOp, "in_c: {}", input_shape[3]);
        log_debug(LogOp, "in_nbytes_c: {}", in_nbytes_c);
        log_debug(LogOp, "out_ntiles_c: {}", out_ntiles_c);
        log_debug(LogOp, "nblocks: {}", nblocks);
        log_debug(LogOp, "ncores: {}", ncores);
        log_debug(LogOp, "in_nhw_per_core: {}", in_nhw_per_core);
        log_debug(LogOp, "out_nhw_per_core: {}", out_nhw_per_core);
        log_debug(LogOp, "is_in_sharded: {}", input.memory_config().is_sharded());
        log_debug(LogOp, "is_out_sharded: {}", output.memory_config().is_sharded());
    }
    #endif

    /**
     * Reader Kernel: input rows -> input cb
     */
    float one = 1.;
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t*>(&one);
    uint32_t in_nbytes_c_log2 = (uint32_t)std::log2((float)in_nbytes_c);
    std::vector<uint32_t> reader0_ct_args = {
        out_nhw_per_core,
        kernel_size_h,
        kernel_size_w,
        pad_w,
        in_nbytes_c,
        in_nbytes_c_log2,
        in_w,
        in_cb_page_padded * in_cb_npages / tile_w,
        input_shape[3],
        nblocks,
        split_reader,  // enable split reader
        0,             // split reader id
        bf16_one_u32};

    std::vector<uint32_t> reader1_ct_args = {
        out_nhw_per_core,
        kernel_size_h,
        kernel_size_w,
        pad_w,
        in_nbytes_c,
        in_nbytes_c_log2,
        in_w,
        in_cb_page_padded * in_cb_npages / tile_w,
        input_shape[3],
        nblocks,
        split_reader,  // enable split reader
        1,             // split reader id
        bf16_one_u32};

    std::string reader_kernel_fname(
        "ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/dataflow/reader_max_pool_2d_multi_core_sharded_with_halo_v2.cpp");

    auto reader0_config = DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader0_ct_args};

    auto reader0_kernel = CreateKernel(program, reader_kernel_fname, all_cores, reader0_config);

    auto reader1_config = DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader1_ct_args};
    auto reader1_kernel = split_reader ? CreateKernel(program, reader_kernel_fname, all_cores, reader1_config) : 0;

    /**
     * Compute Kernel: input cb -> tilize_block -> input tiles -> reduce_h max -> output tiles -> untilize_block ->
     * output cb
     */
    std::vector<uint32_t> compute_ct_args = {
        in_ntiles_hw,
        in_ntiles_c,
        in_ntiles_hw * in_ntiles_c,
        kernel_size_hw,
        out_h,
        out_w,
        div_up(output_shape[2], constants::TILE_HEIGHT),
        div_up(output_shape[3], constants::TILE_WIDTH),
        nblocks,
        out_w_loop_count,
        1,
        out_nhw_per_core,
        split_reader,                // enable split reader
        out_nhw_per_core / nblocks,  // loop count with blocks
        input_shape[3],
    };
    auto compute_ct_args_cliff = compute_ct_args;
    auto reduce_op = ReduceOpMath::MAX;
    auto reduce_dim = ReduceOpDim::H;
    auto compute_config = ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = compute_ct_args,
        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
    std::string compute_kernel_fname("ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/compute/max_pool_multi_core.cpp");
    auto compute_kernel = CreateKernel(program, compute_kernel_fname, core_range, compute_config);

    auto override_runtime_arguments_callback =
        [
            // reader_kernel, writer_kernel, raw_in_cb, in_reader_indices_cb, cb_sharded_out, ncores, ncores_w
            reader0_kernel,
            reader1_kernel,
            raw_in_cb,
            in_reader_indices_cb,
            cb_out,
            ncores,
            ncores_w](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors.at(0).buffer();
            bool input_sharded = input_tensors.at(0).is_sharded();
            auto reader_indices_buffer = input_tensors.at(1).buffer();

            auto dst_buffer = output_tensors.at(0).buffer();
            bool out_sharded = output_tensors.at(0).is_sharded();

            if (input_sharded) {
                UpdateDynamicCircularBufferAddress(program, raw_in_cb, *src_buffer);
                UpdateDynamicCircularBufferAddress(program, in_reader_indices_cb, *reader_indices_buffer);
            }
            if (out_sharded) {
                UpdateDynamicCircularBufferAddress(program, cb_out, *dst_buffer);
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks max_pool_2d_multi_core_sharded_with_halo_v2(
    const Tensor& input,
    const Tensor& reader_indices,
    Tensor& output,
    uint32_t in_n,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t kernel_size_h,
    uint32_t kernel_size_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    const MemoryConfig& out_mem_config,
    uint32_t nblocks) {
    Program program = CreateProgram();
    return max_pool_2d_multi_core_sharded_with_halo_v2_impl(
        program,
        input,
        reader_indices,
        output,
        in_n,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_size_h,
        kernel_size_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        out_mem_config,
        nblocks);
}


}  // namespace tt_metal
}  // namespace tt
