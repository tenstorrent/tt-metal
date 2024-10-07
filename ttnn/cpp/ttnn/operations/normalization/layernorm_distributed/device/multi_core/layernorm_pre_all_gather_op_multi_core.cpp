// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_op.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include <optional>
#include <variant>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::normalization {

inline bool is_dram(const Tensor& input_tensor) { return input_tensor.memory_config().buffer_type == BufferType::DRAM; }
inline bool is_dram(const std::optional<const Tensor> input_tensor) {
     return input_tensor.has_value() ? is_dram(input_tensor.value()) : true;
}
inline bool is_dram(const Buffer* b) { return b->buffer_type() == BufferType::DRAM; }

inline uint16_t bfloat16(float float_num) {
    uint32_t uint32_data;
    TT_ASSERT (sizeof float_num == sizeof uint32_data);

    uint32_data = *reinterpret_cast<uint32_t*>(&float_num);
    // just move upper 16 to lower 16 (truncate)
    uint32_data = (uint32_data >> 16);

    // store lower 16 as 16-bit uint
    return (uint16_t)uint32_data;
}
inline uint32_t pack_two_bfloat16_into_uint32(std::pair<uint16_t, uint16_t> two_bfloats) {
    // first -> lower 16
    // second -> upper 16
    return (uint32_t)two_bfloats.first | ((uint32_t)two_bfloats.second << 16);
}

operation::ProgramWithCallbacks layernorm_pre_allgather_multi_core(
    const Tensor &a,
    Tensor& output,
    LayerNormDistributedType norm_type,
    DeviceComputeKernelConfig compute_kernel_config
) {
    const bool is_rmsnorm = norm_type == LayerNormDistributedType::RMSNORM;
    const auto shape = a.get_legacy_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H*W;
    const uint32_t NC = a.volume() / HW;


    // Kernels are configured to support BFLOAT8_B, but bad pcc so we need mixed precision support in compute
    const auto& a_dtype = a.get_dtype();

    const uint32_t Wt = W/TILE_WIDTH;
    const uint32_t Ht = H/TILE_HEIGHT;
    const uint32_t tile_cols_per_device = is_rmsnorm ? 1 : 2;

    uint32_t num_tile_rows = NC * Ht;

    tt::log_debug("is_rmsnorm: {}", is_rmsnorm);
    tt::log_debug("W: {}", W);
    tt::log_debug("H: {}", H);
    tt::log_debug("num_tile_rows: {}", num_tile_rows);
    tt::log_debug("Wt: {}", Wt);
    tt::log_debug("Ht: {}", Ht);


    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    //////////////////////////////////////////////////////////////////////////
    Device *device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                Circular Buffer Data Format Setup
    //////////////////////////////////////////////////////////////////////////
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t block_size = 1; // find_max_divisor(Wt, 8);
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tt_metal::detail::TileSize(in_data_format);
    uint32_t out_single_tile_size = tt::tt_metal::detail::TileSize(out_data_format);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t bfloat16_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    tt::log_debug("in_data_format: {}", in_data_format);
    tt::log_debug("out_data_format: {}", out_data_format);

    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;

    auto a_addr = a.buffer()->address();
    auto dst_addr = output.buffer()->address();

    uint32_t num_tiles = a.volume()/TILE_HW;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    /*
    in0_cb: a
    in1_cb: 1 (reduction scalar)

    output CB is packed such that the first tile is for x**2 stats, second tile is for x stats
    in RMSNorm, only first tile has valid data.

    intermed0_cb: xˆ2
    out0_cb: [sum(xˆ2), sum(x)]  # For layernorm
    out0_cb: [sum(xˆ2)]  # RMSNorm

    */
    const uint32_t double_buffer_constant = 2;
    const uint32_t in0_tiles = Wt * double_buffer_constant;
    const uint32_t in1_tiles = 1; // reduce scalar

    const uint32_t intermed0_tiles = Wt * double_buffer_constant; // xˆ2
    uint32_t out0_tiles = 1;
    if (!is_rmsnorm) {
        out0_tiles = 2;
    }

    TT_ASSERT(W <= TILE_WIDTH*in0_tiles && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    TT_ASSERT(in0_tiles % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");
    TT_ASSERT(intermed0_tiles % block_size == 0 && "Size of buffer must be divisible by the size of block used by the reader and compute kernel.");


    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tile_rows_per_core_group_1, num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    tt::log_debug("num_cores: {}", num_cores);
    tt::log_debug("grid_size: {}", grid_size);
    tt::log_debug("core_group_1: {}", core_group_1.str());
    tt::log_debug("num_tile_rows_per_core_group_1: {}", num_tile_rows_per_core_group_1);
    tt::log_debug("core_group_2: {}", core_group_2.str());
    tt::log_debug("num_tile_rows_per_core_group_2: {}", num_tile_rows_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    auto program = CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) is_dram(a),
        (std::uint32_t) block_size,
    };

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) is_dram(output),
        (std::uint32_t) writer_block_size
    };


    bool tile_dtype_is_bfloat16 = a.get_dtype() == tt::tt_metal::DataType::BFLOAT16;
    std::map<string, string> compute_defines;

    if (is_rmsnorm) {
        compute_defines["RMSNORM"] = "1";
    }

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args)
    );

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args)
    );

    vector<uint32_t> compute_args = { Wt, block_size };

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compute_args, .defines = compute_defines}
    );

    // Create circular buffers
    // c_in0 -> a
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_tiles*in_single_tile_size, {{tt::CB::c_in0, in_data_format}}).set_page_size(tt::CB::c_in0, in_single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_src0_config );
    // c_in1 -> reduce scalar
    CircularBufferConfig cb_reduce_config = CircularBufferConfig(in1_tiles*bfloat16_tile_size, {{tt::CB::c_in1, cb_data_format}}).set_page_size(tt::CB::c_in1, bfloat16_tile_size);
    CreateCircularBuffer( program, all_cores, cb_reduce_config );

    // LN and RMS shared intermediates //
    // c_intermed0 -> xˆ2
    CircularBufferConfig cb_intermed0_config = CircularBufferConfig(intermed0_tiles*single_tile_size, {{tt::CB::c_intermed0, cb_data_format}}).set_page_size(tt::CB::c_intermed0, single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_intermed0_config );

    CircularBufferConfig cb_out0_config = CircularBufferConfig(out0_tiles*out_single_tile_size, {{tt::CB::c_out0, out_data_format}}).set_page_size(tt::CB::c_out0, out_single_tile_size);
    CreateCircularBuffer( program, all_cores, cb_out0_config );

    uint32_t curr_row = 0;
    float winv =  1.0f;
    auto bfloat_winv_value = bfloat16(winv);
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv_value, bfloat_winv_value});
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t in_tile_offset = curr_row * Wt;
        uint32_t out_tile_offset = curr_row * out0_tiles;

        SetRuntimeArgs(program, reader_kernels_id, core,
            { a_addr, num_tile_rows_per_core, Wt, in_tile_offset, packed_winv_value }
        );
        SetRuntimeArgs(program, compute_kernels_id, core, { num_tile_rows_per_core });
        SetRuntimeArgs(program, writer_kernels_id, core, { dst_addr, num_tile_rows_per_core * out0_tiles, out_tile_offset } );
        curr_row += num_tile_rows_per_core;
    }

    auto override_runtime_arguments_callback = [
            reader_kernel_id=reader_kernels_id,
            writer_kernel_id=writer_kernels_id,
            num_cores,
            grid_size
        ]
    (
        const void* operation,
        ProgramHandle program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        const auto& input_tensor = input_tensors.at(0);

        const auto input_addr = input_tensor.buffer()->address();

        const auto& output_tensor = output_tensors.at(0);
        const auto output_addr = output_tensor.buffer()->address();

        auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
        auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);

        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord core = {i % grid_size.x, i / grid_size.x};

            {
                auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);

                reader_args[0] = input_addr;
            }

            {
                auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
                writer_args[0] = output_addr;
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}


}  // namespace ttnn::operations::normalization
