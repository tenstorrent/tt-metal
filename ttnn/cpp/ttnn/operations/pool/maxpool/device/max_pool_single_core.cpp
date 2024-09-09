// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>

#include "detail/util.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/pool/maxpool/max_pool.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp" // for reduce_op_utils
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace tt_metal {

operation::ProgramWithCallbacks max_pool_2d_single_core(const Tensor &input, Tensor& output,
                                                        uint32_t in_h, uint32_t in_w,
                                                        uint32_t out_h, uint32_t out_w,
                                                        uint32_t kernel_size_h, uint32_t kernel_size_w,
                                                        uint32_t stride_h, uint32_t stride_w,
                                                        uint32_t pad_h, uint32_t pad_w,
                                                        uint32_t dilation_h, uint32_t dilation_w,
                                                        const MemoryConfig& out_mem_config,
                                                        uint32_t nblocks) {
    Program program = CreateProgram();
    CoreRange cores({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device
    Device *device = input.device();
    Buffer *src_dram_buffer = input.buffer();
    Buffer *dst_dram_buffer = output.buffer();

    Shape input_shape = input.get_legacy_shape();
    Shape output_shape = output.get_legacy_shape();

    log_debug("SHAPES: input = {}, output = {}", input_shape, output_shape);

    // NOTE: input is assumed to be in {N, 1, H * W, C }

    // TODO [AS]: Support other data formats??
    DataFormat in_df = datatype_to_dataformat_converter(input.get_dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);
    uint32_t in_nbytes_c = input_shape[3] * in_nbytes;      // row of input (channels)
    uint32_t out_nbytes_c = output_shape[3] * out_nbytes;   // row of output (channels)
    TT_ASSERT((in_nbytes_c & (in_nbytes_c - 1)) == 0, "in_nbytes_c should be power of 2");    // in_nbytes_c is power of 2
    TT_ASSERT((out_nbytes_c & (out_nbytes_c - 1)) == 0, "out_nbytes_c should be power of 2"); // out_nbytes_c is power of 2

    uint32_t nbatch = input_shape[0];
    TT_ASSERT(nbatch == output_shape[0], "Mismatch in N for input and output!!");

    uint32_t kernel_size_hw = kernel_size_w * kernel_size_h;    // number of valid rows, to read
    uint32_t kernel_size_hw_padded = ceil_multiple_of(kernel_size_hw, constants::TILE_HEIGHT);
    uint32_t in_ntiles_hw = (uint32_t) std::ceil((float) kernel_size_hw_padded / constants::TILE_HEIGHT);
    uint32_t in_ntiles_c = (uint32_t) std::ceil((float) input_shape[3] / constants::TILE_WIDTH);
    uint32_t out_ntiles_hw = (uint32_t) std::ceil((float) output_shape[2] / constants::TILE_HEIGHT);
    uint32_t out_ntiles_c = (uint32_t) std::ceil((float) output_shape[3] / constants::TILE_WIDTH);

    uint32_t out_nelems = nblocks;     // TODO [AS]: Remove hard coding after identifying optimal param val
    uint32_t out_w_loop_count = std::ceil((float) out_w / out_nelems);

    uint32_t in_hw = in_h * in_w;
    uint32_t out_hw = out_h * out_w;

    // CBs
    uint32_t multi_buffering_factor = 2;

    // scalar CB as coefficient of reduce
    uint32_t in_scalar_cb_id = CB::c_in4;
    uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    uint32_t in_scalar_cb_npages = 1;
    CircularBufferConfig in_scalar_cb_config = CircularBufferConfig(in_scalar_cb_npages * in_scalar_cb_pagesize, {{in_scalar_cb_id, in_df}})
		.set_page_size(in_scalar_cb_id, in_scalar_cb_pagesize);
    auto in_scalar_cb = tt_metal::CreateCircularBuffer(program, cores, in_scalar_cb_config);

    // reader output == input to tilize
    uint32_t in_cb_id = CB::c_in0;          // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_page_nelems_padded = ceil_multiple_of(input_shape[3] * kernel_size_hw_padded, constants::TILE_HW);    // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    uint32_t in_cb_pagesize = in_nbytes * in_cb_page_nelems_padded;
    uint32_t in_cb_npages = multi_buffering_factor * out_nelems;
    CircularBufferConfig in_cb_config = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id, in_df}})
		.set_page_size(in_cb_id, in_cb_pagesize);
    auto in_cb = tt_metal::CreateCircularBuffer(program, cores, in_cb_config);

    // output of tilize == input to reduce
    uint32_t in_tiled_cb_id = CB::c_intermed0;  // tiled input
    uint32_t in_tiled_cb_pagesize = tile_size(in_df);
    uint32_t in_tiled_cb_npages = in_ntiles_c * in_ntiles_hw * out_nelems;
    CircularBufferConfig in_tiled_cb_config = CircularBufferConfig(in_tiled_cb_npages * in_tiled_cb_pagesize, {{in_tiled_cb_id, in_df}})
		.set_page_size(in_tiled_cb_id, in_tiled_cb_pagesize);
    auto in_tiled_cb = tt_metal::CreateCircularBuffer(program, cores, in_tiled_cb_config);

    // output of reduce == writer to write
    uint32_t out_cb_id = CB::c_out0;            // output rows in RM
    uint32_t out_cb_pagesize = tile_size(out_df);
    uint32_t out_cb_npages = out_ntiles_c * out_nelems * multi_buffering_factor;    // there is just one row of channels after reduction
    CircularBufferConfig cb_out_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
		.set_page_size(out_cb_id, out_cb_pagesize);
    auto cb_out = tt_metal::CreateCircularBuffer(program, cores, cb_out_config);

    // Construct const buffer with -INF
    // uint32_t const_buffer_size = 32;
    uint32_t const_buffer_size = input_shape[3];    // set it equal to 1 row
    auto minus_inf_const_buffer = owned_buffer::create(std::vector<bfloat16>(const_buffer_size, bfloat16(0xf7ff)));
    const Tensor minus_inf_const_tensor = Tensor(OwnedStorage{minus_inf_const_buffer},
                                                 Shape({1, 1, 1, const_buffer_size}),
                                                 DataType::BFLOAT16,
                                                 Layout::ROW_MAJOR)
                                            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED,
                                                                     .buffer_type = BufferType::L1});
    auto minus_inf_const_tensor_addr = minus_inf_const_tensor.buffer()->address();

    /**
     * Reader Kernel: input rows -> input cb
     */
    float one = 1.;
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t*>(&one);
    std::vector<uint32_t> reader_ct_args = {input.memory_config().buffer_type == BufferType::DRAM ? (uint) 1 : (uint) 0,
                                            out_mem_config.buffer_type == BufferType::DRAM ? (uint) 1 : (uint) 0,
                                            bf16_one_u32,
                                            out_nelems,
                                            static_cast<uint32_t>(((in_nbytes_c & (in_nbytes_c - 1)) == 0) ? 1 : 0),    // is in_nbytes_c power of 2
                                            stride_h,
                                            stride_w};
    uint32_t in_log_base_2_of_page_size = (uint32_t) std::log2((float) in_nbytes_c);
    std::vector<uint32_t> reader_rt_args = {src_dram_buffer->address(),
                                            dst_dram_buffer->address(),
                                            kernel_size_h, kernel_size_w, kernel_size_hw, kernel_size_hw_padded,
                                            stride_h, stride_w,
                                            pad_h, pad_w,
                                            out_h, out_w, output_shape[2], output_shape[3],
                                            in_nbytes_c, out_nbytes_c,
                                            in_h, in_w, input_shape[2], input_shape[3],
                                            out_ntiles_hw, out_ntiles_c,
                                            in_cb_pagesize, out_cb_pagesize,
                                            in_cb_page_nelems_padded, out_w_loop_count,
                                            in_log_base_2_of_page_size,
                                            nbatch,
                                            in_hw,
                                            out_hw,
                                            0,                              // start_out_h_i
                                            out_h,                          // end_out_h_i
                                            static_cast<uint32_t>(-pad_h),  // base_start_h
                                            0,                              // start_out_row_id
                                            minus_inf_const_tensor_addr,
                                            const_buffer_size * in_nbytes,
                                            (in_cb_page_nelems_padded * out_nelems * 2) >> 5    // TODO: generalize num rows to fill in in_cb
                                            };
    auto reader_config = ReaderDataMovementConfig(reader_ct_args);
    std::string reader_kernel_fname("ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/dataflow/reader_max_pool_2d_single_core.cpp");
    auto reader_kernel = CreateKernel(program,
                                                  reader_kernel_fname,
                                                  cores,
                                                  reader_config);
    SetRuntimeArgs(program, reader_kernel, cores, reader_rt_args);

    #if 0
    {   // debug
        log_debug("in_cb :: PS = {}, NP = {}", in_cb_pagesize, in_cb_npages);
        log_debug("in_scalar_cb :: PS = {}, NP = {}", in_scalar_cb_pagesize, in_scalar_cb_npages);
        log_debug("in_tiled_cb :: PS = {}, NP = {}", in_tiled_cb_pagesize, in_tiled_cb_npages);
        log_debug("out_cb :: PS = {}, NP = {}", out_cb_pagesize, out_cb_npages);
        log_debug("in_addr: {}", src_dram_buffer->address());
        log_debug("out_addr: {}", dst_dram_buffer->address());
        log_debug("nbatch: {}", nbatch);
        log_debug("kernel_size_h: {}", kernel_size_h);
        log_debug("kernel_size_w: {}", kernel_size_w);
        log_debug("kernel_size_hw: {}", kernel_size_hw);
        log_debug("kernel_size_hw_padded: {}", kernel_size_hw_padded);
        log_debug("stride_h: {}", stride_h);
        log_debug("stride_w: {}", stride_w);
        log_debug("pad_h: {}", pad_h);
        log_debug("pad_w: {}", pad_w);
        log_debug("out_h: {}", out_h);
        log_debug("out_w: {}", out_w);
        log_debug("out_hw: {}", output_shape[2]);
        log_debug("out_c: {}", output_shape[3]);
        log_debug("in_nbytes_c: {}", in_nbytes_c);
        log_debug("out_nbytes_c: {}", out_nbytes_c);
        log_debug("in_h: {}", in_h);
        log_debug("in_w: {}", in_w);
        log_debug("in_hw: {}", input_shape[2]);
        log_debug("in_hw_padded: {}", in_hw);
        log_debug("in_c: {}", input_shape[3]);
        log_debug("out_hw_padded: {}", out_hw);
        log_debug("out_ntiles_hw: {}", out_ntiles_hw);
        log_debug("out_ntiles_c: {}", out_ntiles_c);
        log_debug("out_nelems: {}", out_nelems);
        log_debug("out_w_loop_count: {}", out_w_loop_count);
    }
    #endif

    /**
     * Writer Kernel: output cb -> output rows
     */
    std::vector<uint32_t> writer_ct_args = reader_ct_args;
    std::vector<uint32_t> writer_rt_args = reader_rt_args;
    auto writer_config = WriterDataMovementConfig(writer_ct_args);
    std::string writer_kernel_fname("ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/dataflow/writer_max_pool_2d_single_core.cpp");
    auto writer_kernel = CreateKernel(program,
                                                  writer_kernel_fname,
                                                  cores,
                                                  writer_config);
    SetRuntimeArgs(program, writer_kernel, cores, writer_rt_args);

    /**
     * Compute Kernel: input cb -> tilize_block -> input tiles -> reduce_h max -> output tiles -> untilize_block -> output cb
     */
    auto reduce_op = ReduceOpMath::MAX;
    auto reduce_dim = ReduceOpDim::H;
    auto compute_config = ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                        .fp32_dest_acc_en = false,
                                        .math_approx_mode = false,
                                        .compile_args = {in_ntiles_hw,
                                                         in_ntiles_c,
                                                         in_ntiles_hw * in_ntiles_c,
                                                         kernel_size_hw_padded,
                                                         out_h,
                                                         out_w,
                                                         (uint32_t) std::ceil((float) output_shape[2] / constants::TILE_HEIGHT),
                                                         (uint32_t) std::ceil((float) output_shape[3] / constants::TILE_WIDTH),
                                                         out_nelems,
                                                         out_w_loop_count,
                                                         nbatch,
                                                         out_h},    // out_h_per_core
                                        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
    std::string compute_kernel_fname("ttnn/cpp/ttnn/operations/pool/maxpool/device/kernels/compute/max_pool.cpp");
    auto compute_kernel = CreateKernel(program,
                                              compute_kernel_fname,
                                              cores,
                                              compute_config);

    auto override_runtime_args_callback =
        [reader_kernel, writer_kernel](const Program& program,
                                       const std::vector<Buffer*>& input_buffers,
                                       const std::vector<Buffer*>& output_buffers) {
        auto src_dram_buffer = input_buffers.at(0);
        auto dst_dram_buffer = output_buffers.at(0);
        CoreCoord core = {0, 0};
        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel, core);
            runtime_args[0] = src_dram_buffer->address();
            runtime_args[1] = dst_dram_buffer->address();
        }
        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel, core);
            runtime_args[0] = src_dram_buffer->address();
            runtime_args[1] = dst_dram_buffer->address();
        }
    };
    return {std::move(program), override_runtime_args_callback};
}

} // namespace tt_metal
} // namespace tt
