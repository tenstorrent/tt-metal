// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <ttnn/metal_v2_artifacts.hpp>

#include <bit>
#include <string>
#include <utility>
#include <cstdint>

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts
SoftmaxDeviceOperation::SoftmaxProgramFactoryAttentionOptimized::create_program_artifacts(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    log_debug(tt::LogMetal, "SoftmaxProgramFactoryAttentionOptimized selected");

    // Constants
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& shape = input_tensor.padded_shape();
    const std::uint32_t W = shape[-1], H = (input_tensor.physical_volume() / (shape[0] * shape[-1])), NC = shape[0];
    const std::uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const std::uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const std::uint32_t Wt = W / tile_width;
    const std::uint32_t Ht = H / tile_height;
    const auto& shape_unpadded = input_tensor.logical_shape();

    const bool has_mask = tensor_args.mask.has_value();
    // is_causal_mask without an actual mask is contradictory and unsupported. Fail cleanly here rather
    // than letting a CAUSAL_MASK-without-FUSED_SCALE_MASK kernel reach JIT and fail opaquely (the host
    // used to throw std::bad_optional_access on this input before the mask-arg reads were guarded).
    TT_FATAL(
        has_mask || !attributes.is_causal_mask,
        "Causal-mask softmax requires an attention mask (is_causal_mask=true with no mask is unsupported).");

    bool mask_padded_data = false;
    std::uint32_t num_datum_padded = 0;
    std::uint32_t W_unpadded = shape_unpadded[-1];
    if (W > W_unpadded) {
        mask_padded_data = true;
        num_datum_padded = W - W_unpadded;
    }

    std::uint32_t mask_H = H;
    if (has_mask) {
        mask_H = tensor_args.mask.value().padded_shape()[2];
    }
    const std::uint32_t mask_Ht = mask_H / tile_height;

    auto* device = input_tensor.device();
    const auto arch = device->arch();

    const tt::DataFormat in0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const std::uint32_t in0_tile_size = tt::tile_size(in0_cb_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, attributes.compute_kernel_config);

    const tt::DataFormat max_scaler_cb_data_format =
        in0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t max_scaler_tile_size = tt::tile_size(max_scaler_cb_data_format);
    const std::uint32_t fused_attention_scale_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    const tt::DataFormat out0_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    const std::uint32_t out0_tile_size = tt::tile_size(out0_cb_data_format);

    const tt::DataFormat mask_cb_data_format =
        has_mask ? datatype_to_dataformat_converter(tensor_args.mask.value().dtype()) : tt::DataFormat::Float16_b;
    const std::uint32_t mask_tile_size = tt::tile_size(mask_cb_data_format);

    const tt::DataFormat im_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t im_tile_size = tt::tile_size(im_cb_data_format);

    const tt::DataFormat sum_scaler_cb_data_format =
        im_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t sum_scaler_tile_size = tt::tile_size(sum_scaler_cb_data_format);

    // Fixed block size that maximizes dest register usage: 4 with fp32 accumulation, 8 otherwise.
    // Widths not divisible by block_size are handled by the kernels via a clamped final block.
    std::uint32_t block_size = fp32_dest_acc_en ? 4 : 8;

    // Prefer an exact divisor of Wt when it costs no extra register batches: uniform blocks keep every
    // CB capacity a multiple of the block, so no fifo needs realignment between tile rows.
    if (Wt % block_size != 0) {
        const std::uint32_t divisor = static_cast<std::uint32_t>(tt::tt_metal::find_max_divisor(Wt, block_size));
        if (tt::div_up(Wt, block_size) == Wt / divisor) {
            block_size = divisor;
        }
    }

    // calc_numeric_stable() in softmax.cpp uses indexed access over Wt tiles of its input CB and a
    // WaitUpfrontNoPop reduce, so whichever CB it consumes must be sized to Wt:
    //   - no mask, no padding: it consumes cb_in0 directly.
    //   - fused scale-mask path or mask-padded path: it consumes cb_x (cb_in0 is streamed/popped per block).
    const bool small_kernel_numeric_stable_uses_cb_in0_at_wt =
        attributes.numeric_stable && !has_mask && !mask_padded_data;
    const bool small_kernel_numeric_stable_uses_cb_x_at_wt =
        attributes.numeric_stable && (has_mask || mask_padded_data);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    std::uint32_t in0_t =
        small_kernel_numeric_stable_uses_cb_in0_at_wt ? tt::div_up(Wt, block_size) * block_size : block_size * 2;
    std::uint32_t out0_t = block_size * 2;
    std::uint32_t im1_t = 1;  // 1/sum(exp(x))
    std::uint32_t in2_t = 1;  // scaler for reduce coming from reader
    std::uint32_t in3_t = 1;  // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    std::uint32_t in4_t =
        tt::div_up(Wt, block_size) * block_size;  // attention mask (N,C,32,W) - Wt is reused for each Ht, NC is cycled
    std::uint32_t in5_t = 1;
    // numeric_stable cb max
    std::uint32_t im2_t = 1;
    std::uint32_t im4_t = tt::div_up(Wt, block_size) * block_size;

    // cb_exps - keeps exps in tt::CBIndex in L1 to avoid recomputing (rounded up to a multiple of block_size)
    std::uint32_t im0_t = block_size * tt::div_up(Wt, block_size);

    // used for buffering scale-mask
    // can't easily reuse im0_t because cumulative wait for Wt needs to have Wt tiles contiguous free
    std::uint32_t im3_t = block_size * (tt::div_up(Wt, block_size) + 1);

    std::uint32_t dfb_length = in0_t;
    bool use_large_kernel = false;
    // Noisy CB estimator, if the cbs used take up 90% of L1 switch to large kernel implementation
    constexpr std::uint32_t single_tile_cb_count = 5;  // approximate
    std::uint32_t cb_size_sum_bytes = (in0_t * in0_tile_size) + (im0_t * im_tile_size) + (out0_t * out0_tile_size) +
                                      (single_tile_cb_count * im_tile_size);
    if (small_kernel_numeric_stable_uses_cb_x_at_wt) {
        // cb_x (c_10) is only allocated for the small kernel when calc_numeric_stable consumes it.
        // In the no-mask numeric_stable path the kernel aliases cb_x to cb_exps, so no extra CB is needed.
        cb_size_sum_bytes += im4_t * im_tile_size;
    }
    if (has_mask) {
        cb_size_sum_bytes +=
            (im3_t * im_tile_size) + (in3_t * fused_attention_scale_tile_size) + (in4_t * mask_tile_size);
    }

    // Program specific checks
    if ((device->l1_size_per_core() * 0.9) < cb_size_sum_bytes) {
        use_large_kernel = true;
        std::uint32_t large_kernel_cb_size = (80 / block_size) * block_size;
        dfb_length = large_kernel_cb_size;
        in0_t = large_kernel_cb_size;
        im4_t = large_kernel_cb_size;
        im0_t = large_kernel_cb_size;
        im3_t = large_kernel_cb_size;
        // c_4 is streamed a pass at a time too, and both large kernels align its pad to cb_length_t.
        in4_t = large_kernel_cb_size;
    }
    if (!use_large_kernel) {
        TT_FATAL(
            im3_t >= Wt + block_size,
            "im3_t {} must be >= Width in tiles {} + num_dest_regs to use {}",
            im3_t,
            Wt,
            block_size);
    }
    TT_FATAL(
        im0_t % block_size == 0,
        "Size of cb: {} must be divisible by the size of block used by the reader and compute kernel.",
        im0_t);
    TT_FATAL(
        out0_t % block_size == 0,
        "Size of cb: {} must be divisible by the size of block used by the reader and compute kernel.",
        out0_t);
    TT_FATAL(
        in4_t % block_size == 0,
        "Size of cb: {} must be divisible by the size of block used by the reader and compute kernel.",
        in4_t);

    // Work split
    const std::uint32_t num_tile_rows = NC * Ht;
    const auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    // ---- Resource names (program-scope; local to avoid unity-build symbol clashes) ----
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};
    const TensorParamName MASK{"mask"};

    const DFBSpecName IN0{"in0"};
    const DFBSpecName OUT0{"out0"};
    const DFBSpecName MAX_SCALER{"max_scaler"};
    const DFBSpecName SUM_SCALER{"sum_scaler"};
    const DFBSpecName FUSED_SCALE{"fused_scale"};
    const DFBSpecName FUSED_ATTN{"fused_attn"};
    const DFBSpecName MASK_PADDED{"mask_padded"};
    const DFBSpecName EXPS{"exps"};
    const DFBSpecName SCALE_MASK{"scale_mask"};
    const DFBSpecName RECIP_SUM_EXPS{"recip_sum_exps"};
    const DFBSpecName MAX{"max"};
    const DFBSpecName X{"x"};
    const DFBSpecName PREV_REDUCE{"prev_reduce"};
    const DFBSpecName PREV_MAX{"prev_max"};
    const DFBSpecName RECIP{"recip"};

    // ---- DataflowBuffers (mirrors the legacy CB allocation set exactly) ----
    Group<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = in0_tile_size,
        .num_entries = in0_t,
        .data_format_metadata = in0_cb_data_format});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = OUT0,
        .entry_size = out0_tile_size,
        .num_entries = out0_t,
        .data_format_metadata = out0_cb_data_format});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = RECIP_SUM_EXPS,
        .entry_size = im_tile_size,
        .num_entries = im1_t,
        .data_format_metadata = im_cb_data_format});
    if (use_large_kernel) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = PREV_REDUCE,
            .entry_size = im_tile_size,
            .num_entries = im1_t,
            .data_format_metadata = im_cb_data_format});
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = PREV_MAX,
            .entry_size = im_tile_size,
            .num_entries = im1_t,
            .data_format_metadata = im_cb_data_format});
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = RECIP,
            .entry_size = im_tile_size,
            .num_entries = im1_t,
            .data_format_metadata = im_cb_data_format});
    }
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = MAX_SCALER,
        .entry_size = max_scaler_tile_size,
        .num_entries = in2_t,
        .data_format_metadata = max_scaler_cb_data_format});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = SUM_SCALER,
        .entry_size = sum_scaler_tile_size,
        .num_entries = in2_t,
        .data_format_metadata = sum_scaler_cb_data_format});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = EXPS,
        .entry_size = im_tile_size,
        .num_entries = im0_t,
        .data_format_metadata = im_cb_data_format});
    if (has_mask) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = SCALE_MASK,
            .entry_size = im_tile_size,
            .num_entries = im3_t,
            .data_format_metadata = im_cb_data_format});
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = FUSED_SCALE,
            .entry_size = fused_attention_scale_tile_size,
            .num_entries = in3_t,
            .data_format_metadata = tt::DataFormat::Float16_b});
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = FUSED_ATTN,
            .entry_size = mask_tile_size,
            .num_entries = in4_t,
            .data_format_metadata = mask_cb_data_format});
    }
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = MASK_PADDED,
        .entry_size = mask_tile_size,
        .num_entries = in5_t,
        .data_format_metadata = mask_cb_data_format});
    if (attributes.numeric_stable) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = MAX,
            .entry_size = im_tile_size,
            .num_entries = im2_t,
            .data_format_metadata = im_cb_data_format});
    }
    // cb_x: only needed when calc_numeric_stable consumes a separate post-mask buffer (mask or
    // mask-padded path), or when the streaming large kernel is selected. In the no-mask
    // numeric_stable path softmax.cpp aliases cb_x to cb_exps, so we skip the allocation.
    const bool alloc_cb_x = small_kernel_numeric_stable_uses_cb_x_at_wt || use_large_kernel;
    if (alloc_cb_x) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = X,
            .entry_size = im_tile_size,
            .num_entries = im4_t,
            .data_format_metadata = im_cb_data_format});
    }

    // ---- Defines ----
    KernelSpec::CompilerOptions::Defines base_defines;
    if (has_mask) {
        base_defines["FUSED_SCALE_MASK"] = "1";
    }
    if (attributes.is_causal_mask) {
        base_defines["CAUSAL_MASK"] = "1";
    }
    if (attributes.numeric_stable) {
        base_defines["NUMERIC_STABLE"] = "1";
    }

    KernelSpec::CompilerOptions::Defines compute_defines = base_defines;
    compute_defines["EXP_APPROX"] = math_approx_mode ? "1" : "0";
    compute_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
    // The small compute kernel gates the pad-mask (c_5) intermediate (c_10) path on the host-known
    // MASK_PADDED_DATA define; the large compute keeps mask_padded_data as a runtime arg.
    if (!use_large_kernel && mask_padded_data) {
        compute_defines["MASK_PADDED_DATA"] = "1";
    }

    // ---- Reader kernel ----
    Group<DFBBinding> reader_bindings = {
        DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = MAX_SCALER, .accessor_name = "max_scaler", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = SUM_SCALER, .accessor_name = "sum_scaler", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    Group<TensorBinding> reader_tensor_bindings = {TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"}};
    std::vector<std::string> reader_rta_names = {"blk", "num_rows", "tile_offset", "Wt"};
    if (has_mask) {
        reader_bindings.push_back(DFBBinding{
            .dfb_spec_name = FUSED_ATTN, .accessor_name = "fused_attn", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_bindings.push_back(DFBBinding{
            .dfb_spec_name = FUSED_SCALE, .accessor_name = "fused_scale", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = MASK, .accessor_name = "mask"});
        reader_rta_names.push_back("pre_scale");
        reader_rta_names.push_back("Ht");
        reader_rta_names.push_back("start_ht");
        reader_rta_names.push_back("start_mask_id");
        if (attributes.is_causal_mask) {
            reader_rta_names.push_back("mask_start_ht");
            reader_rta_names.push_back("mask_offset");
        }
    }
    KernelSpec::CompileTimeArgs reader_cta;
    if (use_large_kernel) {
        // The large reader pads in0/fused_attn so each streamed pass ends on the fifo base.
        reader_cta.insert({"dfb_length", dfb_length});
    } else {
        // The small reader pads in0 up to this capacity so its fifo ends each tile row on the base.
        reader_cta.insert({"in0_t", in0_t});
    }
    if (has_mask && attributes.is_causal_mask) {
        const std::uint32_t num_tiles_causal_mask = tensor_args.mask.value().padded_shape()[-1] *
                                                    tensor_args.mask.value().padded_shape()[-2] / tile_width /
                                                    tile_height;
        reader_cta.insert({"num_tiles_causal_mask", num_tiles_causal_mask});
    }
    KernelSpec reader{
        .unique_id = READER,
        .source = std::string(SOFTMAX_KERNEL_PATH_ATTENTION) +
                  (use_large_kernel ? "/dataflow/reader_unary_interleaved_sm_large_tensor.cpp"
                                    : "/dataflow/reader_unary_interleaved_sm.cpp"),
        .compiler_options = {.defines = base_defines},
        .dfb_bindings = reader_bindings,
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args = reader_cta,
        .runtime_arg_schema = {.runtime_arg_names = reader_rta_names},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    // ---- Writer kernel ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/writer_unary_interleaved_start_id_blocked_sm.cpp",
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = MASK_PADDED,
                 .accessor_name = "mask_padded",
                 .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"}},
        .compile_time_args = {{"num_datum_padded", num_datum_padded}, {"tile_hw", tile_height * tile_width}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "tile_offset", "blk", "mask_padded_data", "Wt"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    // for broadcasting in H direction we need to
    // NCHt, Nt, Wt
    // if wtpc < Ht then since we pass tpc to the kernel as Ht, the broadcasts should be correct
    // if wtpc >= Ht then tpc should be a multiple of Ht

    // ---- Compute kernel ----
    Group<DFBBinding> compute_bindings = {
        DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = MAX_SCALER, .accessor_name = "max_scaler", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = SUM_SCALER, .accessor_name = "sum_scaler", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = MASK_PADDED, .accessor_name = "mask_padded", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::PRODUCER},
        // exps and recip_sum_exps are compute-internal (self-loop: PRODUCER + CONSUMER).
        DFBBinding{.dfb_spec_name = EXPS, .accessor_name = "exps", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = EXPS, .accessor_name = "exps", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = RECIP_SUM_EXPS,
            .accessor_name = "recip_sum_exps",
            .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = RECIP_SUM_EXPS,
            .accessor_name = "recip_sum_exps",
            .endpoint_type = DFBEndpointType::CONSUMER},
    };
    auto add_self_loop = [&](const DFBSpecName& name, const char* accessor) {
        compute_bindings.push_back(
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::PRODUCER});
        compute_bindings.push_back(
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::CONSUMER});
    };
    if (has_mask) {
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = FUSED_SCALE, .accessor_name = "fused_scale", .endpoint_type = DFBEndpointType::CONSUMER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = FUSED_ATTN, .accessor_name = "fused_attn", .endpoint_type = DFBEndpointType::CONSUMER});
        add_self_loop(SCALE_MASK, "scale_mask");
    }
    if (attributes.numeric_stable) {
        add_self_loop(MAX, "max");
    }
    if (alloc_cb_x) {
        add_self_loop(X, "x");
    }
    if (use_large_kernel) {
        add_self_loop(PREV_REDUCE, "prev_reduce");
        add_self_loop(PREV_MAX, "prev_max");
        add_self_loop(RECIP, "recip");
    }

    std::vector<std::string> compute_rta_names;
    if (use_large_kernel) {
        compute_rta_names = {"num_rows", "Wt", "blk", "mask_padded_data", "dfb_length"};
    } else {
        compute_rta_names = {"num_rows", "Ht", "Wt", "blk", "start_ht"};
    }

    // Compute hardware config (Style A). Legacy defaulted unpack_to_dest_mode (=> UnpackToSrc); Metal 2.0
    // requires an explicit entry for every Float32 DFB the compute consumes under enable_32_bit_dest.
    auto compute_hw = ttnn::to_compute_hardware_config(arch, attributes.compute_kernel_config);
    if (fp32_dest_acc_en) {
        auto& gen1 = std::get<ComputeGen1Config>(compute_hw);
        auto add_unpack = [&](const DFBSpecName& name, tt::DataFormat fmt) {
            if (fmt == tt::DataFormat::Float32) {
                gen1.unpack_modes.insert({name, tt::tt_metal::UnpackMode::UnpackToSrc});
            }
        };
        add_unpack(IN0, in0_cb_data_format);
        add_unpack(MAX_SCALER, max_scaler_cb_data_format);
        add_unpack(SUM_SCALER, sum_scaler_cb_data_format);
        add_unpack(MASK_PADDED, mask_cb_data_format);
        add_unpack(EXPS, im_cb_data_format);
        add_unpack(RECIP_SUM_EXPS, im_cb_data_format);
        if (has_mask) {
            add_unpack(FUSED_SCALE, tt::DataFormat::Float16_b);
            add_unpack(FUSED_ATTN, mask_cb_data_format);
            add_unpack(SCALE_MASK, im_cb_data_format);
        }
        if (attributes.numeric_stable) {
            add_unpack(MAX, im_cb_data_format);
        }
        if (alloc_cb_x) {
            add_unpack(X, im_cb_data_format);
        }
        if (use_large_kernel) {
            add_unpack(PREV_REDUCE, im_cb_data_format);
            add_unpack(PREV_MAX, im_cb_data_format);
            add_unpack(RECIP, im_cb_data_format);
        }
    }

    KernelSpec::CompileTimeArgs compute_cta;
    if (!use_large_kernel) {
        // The small compute drains the pad the reader pushed to in0, so both agree on the capacity.
        compute_cta.insert({"in0_t", in0_t});
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(SOFTMAX_KERNEL_PATH_ATTENTION) +
                  (use_large_kernel ? "/compute/softmax_large_tensor.cpp" : "/compute/softmax.cpp"),
        .compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings = compute_bindings,
        .compile_time_args = compute_cta,
        .runtime_arg_schema = {.runtime_arg_names = compute_rta_names},
        .hw_config = compute_hw,
    };

    // ---- Assemble spec ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = SRC, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = DST, .spec = output_tensor.tensor_spec()}};
    if (has_mask) {
        tensor_parameters.push_back(TensorParameter{.unique_id = MASK, .spec = tensor_args.mask.value().tensor_spec()});
    }

    Group<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{.name = "wu", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores});

    ProgramSpec spec{
        .name = "softmax_attention_optimized",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    // ---- Run args ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};
    KernelRunArgs compute_ra{.kernel = COMPUTE};

    const std::uint32_t scale_value = std::bit_cast<std::uint32_t>(attributes.scale.value_or(1.0f));

    std::uint32_t curr_row = 0;
    for (std::uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        std::uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        std::uint32_t tile_offset = curr_row * Wt;
        std::uint32_t curr_ht = curr_row % Ht;
        std::uint32_t mask_curr_ht = curr_ht % mask_Ht;            // the start offset for causal mask
        std::uint32_t mask_offset = curr_row / Ht * mask_Ht * Wt;  // causal mask batch offset
        std::uint32_t mask_id = attributes.is_causal_mask ? ((mask_curr_ht * Wt) + mask_offset) : (curr_row / Ht * Wt);

        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"blk", block_size}, {"num_rows", num_tile_rows_per_core}, {"tile_offset", tile_offset}, {"Wt", Wt}});
        if (has_mask) {
            AddRuntimeArgsForNode(
                reader_ra.runtime_arg_values,
                core,
                {{"pre_scale", scale_value}, {"Ht", Ht}, {"start_ht", curr_ht}, {"start_mask_id", mask_id}});
            if (attributes.is_causal_mask) {
                AddRuntimeArgsForNode(
                    reader_ra.runtime_arg_values,
                    core,
                    {{"mask_start_ht", mask_curr_ht}, {"mask_offset", mask_offset}});
            }
        }

        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values,
            core,
            {{"num_tiles", num_tile_rows_per_core * Wt},
             {"tile_offset", tile_offset},
             {"blk", block_size},
             {"mask_padded_data", static_cast<std::uint32_t>(mask_padded_data)},
             {"Wt", Wt}});

        if (use_large_kernel) {
            AddRuntimeArgsForNode(
                compute_ra.runtime_arg_values,
                core,
                {{"num_rows", num_tile_rows_per_core},
                 {"Wt", Wt},
                 {"blk", block_size},
                 {"mask_padded_data", static_cast<std::uint32_t>(mask_padded_data)},
                 {"dfb_length", dfb_length}});
        } else {
            AddRuntimeArgsForNode(
                compute_ra.runtime_arg_values,
                core,
                {{"num_rows", num_tile_rows_per_core},
                 {"Ht", Ht},
                 {"Wt", Wt},
                 {"blk", block_size},
                 {"start_ht", curr_ht}});
        }

        curr_row += num_tile_rows_per_core;
    }

    run_args.kernel_run_args = {std::move(reader_ra), std::move(writer_ra), std::move(compute_ra)};
    run_args.tensor_args.emplace(SRC, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(DST, output_tensor.mesh_tensor());
    if (has_mask) {
        run_args.tensor_args.emplace(MASK, tensor_args.mask.value().mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
