// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_halo_v2_program_factory.hpp"

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks untilize_with_halo_multi_core_v2(
    Program& program,
    const Tensor& input_tensor,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t max_out_nsticks_per_core,
    const Tensor& padding_config,
    const Tensor& local_config,
    const Tensor& remote_config,
    const bool remote_read,
    const bool transpose_mcast,
    Tensor& output_tensor,
    const bool capture_buffers) {
    IDevice* device = input_tensor.device();
    Buffer* src_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool skip_untilize = input_tensor.get_layout() == Layout::ROW_MAJOR;

    auto input_shape = input_tensor.get_padded_shape();
    auto output_shape = output_tensor.get_padded_shape();

    tt::DataFormat in_df = datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t out_nbytes = datum_size(out_df);

    CoreRangeSet all_cores = output_tensor.shard_spec().value().grid;
    auto input_shard_shape = output_tensor.shard_spec().value().shape;
    auto output_shard_shape = output_tensor.shard_spec().value().shape;
    TT_ASSERT(input_shard_shape[1] == output_shard_shape[1]);
    uint32_t input_nhw_height = input_shape[0] * input_shape[1] * input_shape[2];
    uint32_t remapped_input_shard_shape_for_output_grid = tt::div_up(input_nhw_height, ncores_nhw);
    uint32_t ntiles_per_block = tt::div_up(input_shard_shape[1], TILE_WIDTH);
    uint32_t input_nblocks_per_core = tt::div_up(remapped_input_shard_shape_for_output_grid, TILE_HEIGHT);
    uint32_t input_npages = ntiles_per_block * input_nblocks_per_core;

    uint32_t out_stick_nbytes = output_shard_shape[1] * out_nbytes;

    uint32_t in_page_size = tt::tt_metal::detail::TileSize(in_df);
    uint32_t out_tile_size = tt::tt_metal::detail::TileSize(out_df);

    if (skip_untilize) {
        uint32_t in_nbytes = datum_size(in_df);
        in_page_size = input_shard_shape[1] * in_nbytes;
        input_npages = remapped_input_shard_shape_for_output_grid;
    }

    // Construct CBs
    // //

    uint32_t src_cb_id = tt::CBIndex::c_0;
    uint32_t pad_cb_id = tt::CBIndex::c_1;
    uint32_t untilize_out_cb_id = tt::CBIndex::c_16;
    uint32_t out_cb_id = tt::CBIndex::c_17;

    // input CB (sharded)
    auto src_cb_config = CircularBufferConfig(input_npages * in_page_size, {{src_cb_id, in_df}})
                             .set_page_size(src_cb_id, in_page_size)
                             .set_globally_allocated_address(*src_buffer);
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);
    log_debug(tt::LogOp, "CB {} :: npages = {}, pagesize = {}", src_cb_id, input_npages, in_page_size);

    uint32_t input_to_writer_cb_id = src_cb_id;
    if (!skip_untilize) {
        input_to_writer_cb_id = untilize_out_cb_id;

        // output of untilize from compute kernel goes into this CB
        uint32_t output_ntiles = ntiles_per_block * input_nblocks_per_core;
        auto untilize_out_cb_config =
            CircularBufferConfig(output_ntiles * out_tile_size, {{untilize_out_cb_id, out_df}})
                .set_page_size(untilize_out_cb_id, out_tile_size);
        auto untilize_out_cb = CreateCircularBuffer(program, all_cores, untilize_out_cb_config);
        log_debug(tt::LogOp, "CB {} :: npages = {}, pagesize = {}", untilize_out_cb_id, output_ntiles, out_tile_size);
    }

    // output shard, after inserting halo and padding, goes into this CB as input to next op.
    uint32_t out_cb_pagesize = out_stick_nbytes;
    uint32_t out_cb_npages = max_out_nsticks_per_core;
    auto out_cb_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
                             .set_page_size(out_cb_id, out_cb_pagesize)
                             .set_globally_allocated_address(*dst_buffer);
    auto out_cb = CreateCircularBuffer(program, all_cores, out_cb_config);
    log_debug(tt::LogOp, "CB {} :: npages = {}, pagesize = {}", out_cb_id, out_cb_npages, out_cb_pagesize);

    // CB for pad val buffer (stick sized)
    uint32_t pad_cb_pagesize = out_stick_nbytes;
    uint32_t pad_cb_npages = 1;
    auto pad_cb_config = CircularBufferConfig(pad_cb_pagesize * pad_cb_npages, {{pad_cb_id, out_df}})
                             .set_page_size(pad_cb_id, pad_cb_pagesize);
    auto pad_cb = CreateCircularBuffer(program, all_cores, pad_cb_config);
    log_debug(tt::LogOp, "CB {} :: npages = {}, pagesize = {}", pad_cb_id, pad_cb_npages, pad_cb_pagesize);

    // Additional CBs for sharded data kernel configs
    uint32_t padding_config_cb_id = tt::CBIndex::c_2;
    uint32_t local_config_cb_id = tt::CBIndex::c_3;
    uint32_t remote_config_cb_id = tt::CBIndex::c_4;

    tt::DataFormat kernel_config_df = tt::DataFormat::RawUInt16;  // NOTE: UInt16 is not supported for CB types
    uint32_t config_nbytes =
        tt::datum_size(kernel_config_df) * 2;  // each config is a pair "start, size", so double the size
    uint32_t pagesize = 0;

    // Gather data
    if (!skip_untilize) {
        // compute kernel
        std::vector<uint32_t> compute_ct_args = {input_nblocks_per_core, ntiles_per_block};
        std::string compute_kernel(
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
        if (ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH) {
            log_debug(
                tt::LogOp,
                "Falling back to slow untilize since ntiles_per_block {} > MAX_PACK_UNTILIZE_WIDTH {}",
                ntiles_per_block,
                MAX_PACK_UNTILIZE_WIDTH);
            compute_kernel = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
        }
        KernelHandle untilize_kernel_id =
            CreateKernel(program, compute_kernel, all_cores, ComputeConfig{.compile_args = compute_ct_args});
    }

    TT_ASSERT(padding_config.get_dtype() == DataType::UINT16);
    TT_ASSERT(local_config.get_dtype() == DataType::UINT16);
    TT_ASSERT(remote_config.get_dtype() == DataType::UINT16);

    auto padding_config_buffer = padding_config.device_buffer();
    const uint32_t num_cores = all_cores.num_cores();
    auto padding_config_cb_config =
        CircularBufferConfig(padding_config_buffer->size() / num_cores, {{padding_config_cb_id, kernel_config_df}})
            .set_page_size(padding_config_cb_id, padding_config_buffer->page_size())
            .set_globally_allocated_address(*padding_config_buffer);
    CBHandle padding_config_cb = CreateCircularBuffer(program, all_cores, padding_config_cb_config);

    auto local_config_buffer = local_config.device_buffer();
    auto local_config_cb_config =
        CircularBufferConfig(local_config_buffer->size() / num_cores, {{local_config_cb_id, kernel_config_df}})
            .set_page_size(local_config_cb_id, local_config_buffer->page_size())
            .set_globally_allocated_address(*local_config_buffer);
    CBHandle local_config_cb = CreateCircularBuffer(program, all_cores, local_config_cb_config);

    auto remote_config_buffer = remote_config.device_buffer();
    auto remote_config_cb_config =
        CircularBufferConfig(remote_config_buffer->size() / num_cores, {{remote_config_cb_id, kernel_config_df}})
            .set_page_size(remote_config_cb_id, remote_config_buffer->page_size())
            .set_globally_allocated_address(*remote_config_buffer);
    CBHandle remote_config_cb = CreateCircularBuffer(program, all_cores, remote_config_cb_config);

    const bool is_block_sharded = input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_width_sharded = input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED;

    auto aligned_input_nstick_nbytes = out_stick_nbytes;
    log_debug(tt::LogOp, "out_stick_nbytes = {}", out_stick_nbytes);
    log_debug(tt::LogOp, "input_tensor.buffer()->alignment() = {}", input_tensor.buffer()->alignment());

    if (out_stick_nbytes % input_tensor.buffer()->alignment() != 0) {
        aligned_input_nstick_nbytes = tt::round_up(out_stick_nbytes, input_tensor.buffer()->alignment());
    }
    // reader kernel
    std::vector<uint32_t> reader_ct_args = {
        0,  // padding_config_cb_id
        0,  // local_config_cb_id
        0,  // remote_config_cb_id
        src_cb_id,
        input_to_writer_cb_id,
        out_cb_id,
        pad_cb_id,
        pad_val,
        input_npages,
        out_stick_nbytes,
        is_block_sharded,
        remote_read,
        (uint32_t)(transpose_mcast ? 1 : 0),
        is_width_sharded,
        aligned_input_nstick_nbytes,
        true  // split_remote_reader
    };

    reader_ct_args[0] = 0;
    reader_ct_args[1] = local_config_cb_id;
    reader_ct_args[2] = remote_config_cb_id;

    KernelHandle reader_kernel_id0 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_halo_v2/device/kernels/dataflow/halo_gather.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct_args});

    reader_ct_args[0] = padding_config_cb_id;
    reader_ct_args[1] = 0;
    reader_ct_args[2] = remote_config_cb_id;

    KernelHandle reader_kernel_id1 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_halo_v2/device/kernels/dataflow/halo_gather.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    if (!capture_buffers) {
        padding_config_buffer = nullptr;
        local_config_buffer = nullptr;
        remote_config_buffer = nullptr;
    }
    // Capture padding_config_buffer, local_config_buffer, remote_config_buffer to cache this with the program
    auto override_runtime_arguments_callback = [src_cb,
                                                out_cb,
                                                padding_config_cb,
                                                local_config_cb,
                                                remote_config_cb,
                                                padding_config_buffer,
                                                local_config_buffer,
                                                remote_config_buffer](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, src_cb, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::data_movement::detail
