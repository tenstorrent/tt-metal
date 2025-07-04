// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tanh_accurate_sharded_pgm_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::unary::program {

using namespace tt::constants;
using namespace tt::tt_metal;

TanhAccurateShardedProgramFactory::cached_program_t TanhAccurateShardedProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;

    tt::tt_metal::Program program = CreateProgram();
    tt::tt_metal::IDevice* device = input.device();

    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();

    auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == ncores,
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        ncores);

    tt::DataFormat act_df = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    uint32_t input_tile_size = tt::tt_metal::detail::TileSize(act_df);
    uint32_t output_tile_size = tt::tt_metal::detail::TileSize(out_df);

    TT_FATAL(input_tile_size == output_tile_size, "Input and output tile size should be same");

    uint32_t num_tile_per_core = 0;

    if (input.dtype() == DataType::BFLOAT8_B) {
        uint32_t ntiles_along_width = std::ceil(shard_spec.shape[1] / (float)tt::constants::TILE_WIDTH);
        uint32_t ntiles_along_height = std::ceil(shard_spec.shape[0] / (float)tt::constants::TILE_HEIGHT);
        num_tile_per_core = ntiles_along_width * ntiles_along_height;
    } else {
        TT_FATAL(
            (shard_spec.shape[1] * datum_size(act_df)) % hal::get_l1_alignment() == 0,
            "Shard width should be multiple of {} to satisfy L1 alignment",
            hal::get_l1_alignment());
        size_t shard_height = shard_spec.shape[0];
        size_t shard_width = shard_spec.shape[1];
        size_t shard_size_in_bytes = shard_height * shard_width * datum_size(act_df);
        TT_FATAL(shard_size_in_bytes % input_tile_size == 0, "Shard Size must be multiple of input_tile_size");
        num_tile_per_core = (shard_size_in_bytes + input_tile_size - 1) / input_tile_size;  // ceil value
    }

    uint32_t in_cb_id = tt::CBIndex::c_0;
    uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs since its sharded
    uint32_t aligned_input_tile_nbytes =
        round_up_to_mul32(input_tile_size);  // will have issue if the page is not multiple of 32
    uint32_t in_cb_pagesize = aligned_input_tile_nbytes;
    uint32_t in_cb_npages = num_tile_per_core * buffering_factor;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{in_cb_id, act_df}})
            .set_page_size(in_cb_id, in_cb_pagesize)
            .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // intermediate buffers

    // LUT tanh(x)
    uint32_t im1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_im1_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{im1_cb_index, act_df}})
            .set_page_size(im1_cb_index, in_cb_pagesize);
    auto cb_im1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im1_config);

    // exp(2x)
    uint32_t im2_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_im2_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{im2_cb_index, act_df}})
            .set_page_size(im2_cb_index, in_cb_pagesize);
    auto cb_im2 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im2_config);

    // exp(2x) - 1
    uint32_t im3_cb_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_im3_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{im3_cb_index, act_df}})
            .set_page_size(im3_cb_index, in_cb_pagesize);
    auto cb_im3 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im3_config);

    // recip(exp(2x) + 1)
    uint32_t im4_cb_index = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig cb_im4_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{im4_cb_index, act_df}})
            .set_page_size(im4_cb_index, in_cb_pagesize);
    auto cb_im4 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im4_config);

    // exp(2x) - 1 * recip(exp(2x) - 1)
    uint32_t im5_cb_index = tt::CBIndex::c_6;
    tt::tt_metal::CircularBufferConfig cb_im5_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{im5_cb_index, act_df}})
            .set_page_size(im5_cb_index, in_cb_pagesize);
    auto cb_im5 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im5_config);

    // output for x > 3.5
    uint32_t im6_cb_index = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig cb_im6_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{im6_cb_index, act_df}})
            .set_page_size(im6_cb_index, in_cb_pagesize);
    auto cb_im6 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im6_config);

    // output for x <= 3.5
    uint32_t im7_cb_index = tt::CBIndex::c_8;
    tt::tt_metal::CircularBufferConfig cb_im7_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{im7_cb_index, act_df}})
            .set_page_size(im7_cb_index, in_cb_pagesize);
    auto cb_im7 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_im7_config);

    // output sharded CB
    uint32_t out_cb_id = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig out_cb_config =
        tt::tt_metal::CircularBufferConfig(in_cb_pagesize * in_cb_npages, {{out_cb_id, out_df}})
            .set_page_size(out_cb_id, in_cb_pagesize)
            .set_globally_allocated_address(*output.buffer());
    auto out_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, out_cb_config);

    log_debug(tt::LogOp, "input_cb: {}, npages: {}, pagesize: {}", in_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(tt::LogOp, "input_tile_size: {}", input_tile_size);

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(src_is_dram == 0, "Input buffer should be in L1");
    std::vector<uint32_t> reader_compile_time_args = {
        in_cb_id,
    };

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(dst_is_dram == 0, "Output buffer should be in L1");

    std::map<std::string, std::string> kernel_defines;
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        1,                 // per_core_block_cnt
        num_tile_per_core  // per_core_block_size
    };

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[in_cb_id] = UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = false;
    std::map<std::string, std::string> unary_defines;
    auto path = "ttnn/cpp/ttnn/operations/eltwise/unary/tanh_accurate/device/kernels/compute/tanh_accurate.cpp";

    auto eltwise_unary_kernel_group_1_id = tt::tt_metal::CreateKernel(
        program,
        path,
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1,
            .defines = unary_defines});

    tt::tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        all_cores,
        {
            (uint32_t)(num_tile_per_core),
        });

    return cached_program_t{std::move(program), {cb_src0, out_cb}};
}

void TanhAccurateShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    const auto& cb_src0 = cached_program.shared_variables.cb_src0;
    const auto& out_cb = cached_program.shared_variables.out_cb;

    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = output.buffer();
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
}

}  // namespace ttnn::operations::unary::program
