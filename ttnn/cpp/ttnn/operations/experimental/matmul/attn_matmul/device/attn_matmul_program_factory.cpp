// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_matmul_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::matmul {

using namespace tt::constants;
using namespace tt;

operation::ProgramWithCallbacks multi_core_attn_matmul(const Tensor &a, const Tensor &b, Tensor& output, std::optional<const uint32_t> num_tokens, std::optional<const bool> transpose_hw, CoreCoord compute_with_storage_grid_size, ttnn::DeviceComputeKernelConfig compute_kernel_config) {

    tt::tt_metal::Program program{};

    const auto& ashape = a.get_legacy_shape(), bshape = b.get_legacy_shape();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device *device = a.device();

    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;
    bool packer_l1_acc;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = false;
            packer_l1_acc = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(ttnn::device::is_wormhole_or_blackhole(device->arch()), "kernel config is not for wormhole_b0 or blackhole");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
            packer_l1_acc = compute_kernel_config.packer_l1_acc;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    tt::DataFormat interm_data_format = fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t in0_single_tile_size = tt::tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt::tt_metal::detail::TileSize(in1_data_format);
    uint32_t interm_single_tile_size = tt::tt_metal::detail::TileSize(interm_data_format);
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_data_format);

    if (in0_data_format == tt::DataFormat::Float32 or in1_data_format == tt::DataFormat::Float32 or output_data_format == tt::DataFormat::Float32) {
        TT_ASSERT(fp32_dest_acc_en == true, "when inputs/output are in fp32 format, fp32_dest_acc_en must be set");
    }

    log_debug("math_fidelity: {}", math_fidelity);
    log_debug("math_approx_mode: {}", math_approx_mode);
    log_debug("fp32_dest_acc_en: {}", fp32_dest_acc_en);
    log_debug("packer_l1_acc: {}", packer_l1_acc);
    log_debug("in0_data_format: {}", in0_data_format);
    log_debug("in1_data_format: {}", in1_data_format);
    log_debug("interm_data_format: {}", interm_data_format);
    log_debug("output_data_format: {}", output_data_format);

    tt::tt_metal::Buffer *src0_buffer = a.buffer();
    tt::tt_metal::Buffer *src1_buffer = b.buffer();

    auto cshape = output.get_legacy_shape();

    // A block of work is one MtNt
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_output_blocks_total = ashape[1]; // ashape[1] is Q num_heads; only parallelize on this
    auto [num_cores, all_cores, core_group_1, core_group_2, num_output_blocks_per_core_group_1, num_output_blocks_per_core_group_2] = tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_blocks_total);

    auto all_device_cores = CoreRange({0, 0}, {a.device()->compute_with_storage_grid_size().x - 1, a.device()->compute_with_storage_grid_size().y - 1});
    auto total_num_cores = a.device()->compute_with_storage_grid_size().x * a.device()->compute_with_storage_grid_size().y;

    tt::tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // C = torch.matmul(A.transpose(0, 2) * B).transpose(0, 2)
    // MN = MK*KN
    // Note, in1 K may not be the same as in0 K. We will read up to in0 K from in1 K for matmul.
    const bool transpose_hw_bool = transpose_hw.value_or(false);
    const uint32_t num_tokens_val = num_tokens.value_or(0); // should not be nullopt if transpose_hw=true
    constexpr uint32_t num_rows_in_one_tile = 32;

    uint32_t B = ashape[1];  // ashape[0] is q_len
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    // For transpose_hw=true, in1_Kt is same as in0_Kt but on bshape[3]
    // For transpose_hw=false, in1_Kt is on bshape[2] but represents the max cache length to read from (ie. may not equal in0_Kt)
    uint32_t in1_Kt = transpose_hw_bool ? Kt : bshape[2]/TILE_HEIGHT;
    uint32_t Nt = transpose_hw_bool ? num_tokens_val/TILE_HEIGHT : bshape[3]/TILE_WIDTH;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;
    // For transpose_hw=true, in1_Kt is max cache length
    // For transpose_hw=false, bshape[2] is max cache length
    uint32_t in1_KtNt_stride = transpose_hw_bool ? bshape[2]/TILE_HEIGHT * in1_Kt : in1_Kt * Nt;
    uint32_t in1_KtNt_skip = transpose_hw_bool ? (bshape[2]/TILE_HEIGHT - 1) * in1_Kt : (in1_Kt - Kt) * Nt;

    uint32_t src0_addr = src0_buffer->address();
    uint32_t src1_addr = src1_buffer->address();
    uint32_t dst_addr = dst_buffer->address();

    uint32_t src0_cb_index = tt::CB::c_in0;
    uint32_t cb0_num_input_tiles = Kt * 2;
    tt::tt_metal::CircularBufferConfig src0_cb_config = tt::tt_metal::CircularBufferConfig(cb0_num_input_tiles * in0_single_tile_size, {{src0_cb_index, in0_data_format}})
		.set_page_size(src0_cb_index, in0_single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, src0_cb_config);

    uint32_t src1_cb_index = tt::CB::c_in1;
    uint32_t cb1_num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src1_config = tt::tt_metal::CircularBufferConfig(cb1_num_input_tiles * in1_single_tile_size, {{src1_cb_index, in1_data_format}})
		.set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src1_config);

    uint32_t cb_intermed0_index = tt::CB::c_intermed0;
    tt::tt_metal::CircularBufferConfig cb_interm0_config = tt::tt_metal::CircularBufferConfig(1 * interm_single_tile_size, {{cb_intermed0_index, interm_data_format}})
		.set_page_size(cb_intermed0_index, interm_single_tile_size);
    auto cb_interm0 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm0_config);

    uint32_t cb_intermed1_index = tt::CB::c_intermed1;
    tt::tt_metal::CircularBufferConfig cb_interm1_config = tt::tt_metal::CircularBufferConfig(1 * interm_single_tile_size, {{cb_intermed1_index, interm_data_format}})
		.set_page_size(cb_intermed1_index, interm_single_tile_size);
    auto cb_interm1 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm1_config);

    uint32_t cb_intermed2_index = tt::CB::c_intermed2;
    tt::tt_metal::CircularBufferConfig cb_interm2_config = tt::tt_metal::CircularBufferConfig(1 * interm_single_tile_size, {{cb_intermed2_index, interm_data_format}})
		.set_page_size(cb_intermed2_index, interm_single_tile_size);
    auto cb_interm2 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm2_config);

    uint32_t output_cb_index = tt::CB::c_out0; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config = tt::tt_metal::CircularBufferConfig(num_output_tiles * output_single_tile_size, {{output_cb_index, output_data_format}})
		.set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_output_config);

    const bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    const bool src1_is_dram = src1_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t) src0_is_dram,
        (uint32_t) src1_is_dram,
        (uint32_t) transpose_hw_bool,
        (uint32_t) (fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32)
    };

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    auto reader_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/dataflow/reader_transformer_attn_matmul.cpp",
        all_device_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_device_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_args = {
        (uint32_t) transpose_hw_bool, // transpose_hw for matmul_init
    }; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

    auto eltwise_binary_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/compute/transformer_attn_matmul.cpp",
        all_device_cores,
        tt::tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args}
    );

    uint32_t num_output_blocks_per_core;
    for (uint32_t i = 0, num_blocks_written = 0; i < total_num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_2;
        } else {
            tt::tt_metal::SetRuntimeArgs(program, reader_id, core, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
            tt::tt_metal::SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {0, 0, 0, 0});
            tt::tt_metal::SetRuntimeArgs(program, writer_id, core, {0, 0, 0});
            continue;
        }

        tt::tt_metal::SetRuntimeArgs(
            program, reader_id, core,
            {
                src0_addr,
                src1_addr,
                Mt,
                Kt,
                Nt,
                MtKt,
                in1_KtNt_skip, // Skip to get next batch for in1 after reading in0 Kt
                in1_KtNt_stride * num_rows_in_one_tile, // itileB stride; skips 32 * KtNt in bshape[0] for one block of MtNt
                num_output_blocks_per_core,
                num_blocks_written * MtKt, // itileA_start
                0, // itileB_start; always read in same in1 per core TODO: multi-cast
            }
        );


        tt::tt_metal::SetRuntimeArgs(
            program,
            eltwise_binary_kernel_id,
            core,
            {
                1, // B
                1, // Mt
                Kt, // Kt
                num_output_blocks_per_core * MtNt, // Nt
            }
        );

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_id,
            core,
            {
                dst_addr,
                num_output_blocks_per_core * MtNt,
                num_blocks_written * MtNt,
            }
        );
        num_blocks_written += num_output_blocks_per_core;
    }

    auto override_runtime_arguments_callback = [
            reader_id,
            writer_id,
            eltwise_binary_kernel_id,
            total_num_cores,
            in0_single_tile_size,
            cb_src0,
            src0_cb_index
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        auto transpose_hw = static_cast<const AttnMatmulDeviceOperation *>(operation)->transpose_hw;
        auto num_tokens = static_cast<const AttnMatmulDeviceOperation *>(operation)->num_tokens;
        auto compute_with_storage_grid_size = static_cast<const AttnMatmulDeviceOperation *>(operation)->compute_with_storage_grid_size;

        uint32_t num_cores_y = compute_with_storage_grid_size.y;

        auto src_dram_buffer_a = input_tensors.at(0).buffer();
        auto src_dram_buffer_b = input_tensors.at(1).buffer();

        auto dst_dram_buffer = output_tensors.at(0).buffer();

        auto ashape = input_tensors.at(0).get_legacy_shape();
        auto bshape = input_tensors.at(1).get_legacy_shape();

        // C = torch.matmul(A.transpose(0, 2) * B).transpose(0, 2)
        // MN = MK*KN
        // Note, in1 K may not be the same as in0 K. We will read up to in0 K from in1 K for matmul.
        const bool transpose_hw_bool = transpose_hw.value_or(false);
        const uint32_t num_tokens_val = num_tokens.value_or(0); // should not be nullopt if transpose_hw=true
        constexpr uint32_t num_rows_in_one_tile = 32;

        uint32_t B = ashape[1];  // ashape[0] is q_len
        uint32_t Mt = ashape[2]/TILE_HEIGHT;
        uint32_t Kt = ashape[3]/TILE_WIDTH;
        // For transpose_hw=true, in1_Kt is same as in0_Kt but on bshape[3]
        // For transpose_hw=false, in1_Kt is on bshape[2] but represents the max cache length to read from (ie. may not equal in0_Kt)
        uint32_t in1_Kt = transpose_hw_bool ? Kt : bshape[2]/TILE_HEIGHT;
        uint32_t Nt = transpose_hw_bool ? num_tokens_val/TILE_HEIGHT : bshape[3]/TILE_WIDTH;
        uint32_t MtKt = Mt * Kt;
        uint32_t MtNt = Mt * Nt;
        // For transpose_hw=true, in1_Kt is max cache length
        // For transpose_hw=false, bshape[2] is max cache length
        uint32_t in1_KtNt_stride = transpose_hw_bool ? bshape[2]/TILE_HEIGHT * in1_Kt : in1_Kt * Nt;
        uint32_t in1_KtNt_skip = transpose_hw_bool ? (bshape[2]/TILE_HEIGHT - 1) * in1_Kt : (in1_Kt - Kt) * Nt;

        UpdateCircularBufferTotalSize(program, cb_src0, Kt * in0_single_tile_size);
        UpdateCircularBufferPageSize(program, cb_src0, src0_cb_index, in0_single_tile_size);

        auto num_output_blocks_total = ashape[1]; // ashape[1] is Q num_heads; only parallelize on this
        auto [num_cores, all_cores, core_group_1, core_group_2, num_output_blocks_per_core_group_1, num_output_blocks_per_core_group_2] = tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_blocks_total);

        uint32_t num_output_blocks_per_core;
        for (uint32_t i = 0, num_blocks_written = 0; i < total_num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_output_blocks_per_core = num_output_blocks_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
                num_output_blocks_per_core = num_output_blocks_per_core_group_2;
            } else {
                tt::tt_metal::SetRuntimeArgs(program, reader_id, core, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
                tt::tt_metal::SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {0, 0, 0, 0});
                tt::tt_metal::SetRuntimeArgs(program, writer_id, core, {0, 0, 0});
                continue;
            }

            tt::tt_metal::SetRuntimeArgs(
                program, reader_id, core,
                {
                    src_dram_buffer_a->address(),
                    src_dram_buffer_b->address(),
                    Mt,
                    Kt,
                    Nt,
                    MtKt,
                    in1_KtNt_skip, // Skip to get next batch for in1 after reading in0 Kt
                    in1_KtNt_stride * num_rows_in_one_tile, // itileB stride; skips 32 * KtNt in bshape[0] for one block of MtNt
                    num_output_blocks_per_core,
                    num_blocks_written * MtKt, // itileA_start
                    0, // itileB_start; always read in same in1 per core TODO: multi-cast
                }
            );


            tt::tt_metal::SetRuntimeArgs(
                program,
                eltwise_binary_kernel_id,
                core,
                {
                    1, // B
                    1, // Mt
                    Kt, // Kt
                    num_output_blocks_per_core * MtNt, // Nt
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program,
                writer_id,
                core,
                {
                    dst_dram_buffer->address(),
                    num_output_blocks_per_core * MtNt,
                    num_blocks_written * MtNt,
                }
            );
            num_blocks_written += num_output_blocks_per_core;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // ttnn::operations::experimental::transformer
