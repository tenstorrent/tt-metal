// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation.hpp"

#include <bit>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// Work distribution shared between create_program_artifacts (cache miss) and
// override_runtime_arguments (cache hit).  Keeping this in one place guarantees the cache-hit path
// targets exactly the same cores/kernel arg slots the miss path built, so the re-applied decode
// offsets can't drift from the program layout.  `Wt` is the per-row tile count (1 on the single-tile
// path).
struct RotaryWorkSplit {
    bool row_major = true;
    uint32_t num_cores = 0;
    uint32_t num_cores_x = 0;
    uint32_t num_cores_y = 0;
    uint32_t num_rows_per_core_group_1 = 0;
    uint32_t num_rows_per_core_group_2 = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    bool in_sharded = false;
    bool out_sharded = false;
    uint32_t num_input_tiles = 0;
    uint32_t num_output_tiles = 0;
};

RotaryWorkSplit compute_rotary_work_split(const Tensor& input, const Tensor& output, uint32_t Wt) {
    RotaryWorkSplit w;
    w.in_sharded = input.shard_spec().has_value();
    w.out_sharded = output.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = w.in_sharded ? input.shard_spec() : output.shard_spec();

    uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT;

    auto compute_with_storage_grid_size = input.device()->compute_with_storage_grid_size();
    w.num_cores_x = compute_with_storage_grid_size.x;
    w.num_cores_y = compute_with_storage_grid_size.y;

    if (shard_spec.has_value()) {
        w.row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        w.all_cores = shard_spec.value().grid;
        w.num_cores = w.all_cores.num_cores();
        w.core_group_1 = w.all_cores;
        w.core_group_2 = CoreRangeSet();
        w.num_rows_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        w.num_rows_per_core_group_2 = 0;
        w.num_input_tiles = w.in_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        w.num_output_tiles =
            w.out_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        auto bbox = w.all_cores.bounding_box();
        w.num_cores_x = bbox.end_coord.x + 1;
        w.num_cores_y = bbox.end_coord.y + 1;
    } else {
        w.row_major = true;
        std::tie(
            w.num_cores,
            w.all_cores,
            w.core_group_1,
            w.core_group_2,
            w.num_rows_per_core_group_1,
            w.num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, w.row_major);
        w.num_input_tiles = 2 * Wt;
        w.num_output_tiles = w.num_input_tiles;
    }
    return w;
}

// Single-tile (Wt == 1) path. The Wt >= 2 path implements HF rotate_half via
// inter-tile half-swap + scalar negation, which collapses when Wt == 1 (half_Wt
// == 0). Here we instead use matmul_tiles(input, trans_mat) with an in-L1
// transformation matrix that encodes [[0, I], [-I, 0]].
ttnn::device_operation::ProgramArtifacts create_single_tile_artifacts(
    const RotaryEmbeddingParams& operation_attributes,
    const RotaryEmbeddingInputs& tensor_args,
    Tensor& tensor_return_value) {
    // Spec resource names (function-local to avoid unity-build collisions).
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_group_1"};
    const KernelSpecName COMPUTE_G2{"compute_group_2"};
    const DFBSpecName INPUT_DFB{"input"};                            // legacy c_0
    const DFBSpecName TRANS_MAT_DFB{"trans_mat"};                    // legacy c_1
    const DFBSpecName COS_DFB{"cos"};                                // legacy c_2
    const DFBSpecName SIN_DFB{"sin"};                                // legacy c_3
    const DFBSpecName ROTATED_IN_INTERM_DFB{"rotated_in_interm"};    // legacy c_24
    const DFBSpecName COS_INTERM_DFB{"cos_interm"};                  // legacy c_25
    const DFBSpecName SIN_INTERM_DFB{"sin_interm"};                  // legacy c_26
    const DFBSpecName OUT_DFB{"out"};                                // legacy c_16
    const DFBSpecName UNTILIZED_COS_DFB{"untilized_cos"};            // legacy c_27 (aliases c_5's L1)
    const DFBSpecName UNTILIZED_COS_SYNC_DFB{"untilized_cos_sync"};  // legacy c_5 (aliases c_27's L1)
    const DFBSpecName UNTILIZED_SIN_DFB{"untilized_sin"};            // legacy c_28 (aliases c_6's L1)
    const DFBSpecName UNTILIZED_SIN_SYNC_DFB{"untilized_sin_sync"};  // legacy c_6 (aliases c_28's L1)
    const DFBSpecName RETILIZED_COS_DFB{"retilized_cos"};            // legacy c_29
    const DFBSpecName RETILIZED_SIN_DFB{"retilized_sin"};            // legacy c_30
    const TensorParamName SRC{"src"};
    const TensorParamName COS{"cos"};
    const TensorParamName SIN{"sin"};
    const TensorParamName DST{"dst"};

    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    auto& output = tensor_return_value;
    const auto& token_idx = operation_attributes.token_idx;

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_data_format);

    tt::DataFormat cos_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt::tile_size(cos_data_format);

    tt::DataFormat sin_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt::tile_size(sin_data_format);

    // trans_mat is constructed in L1 by the reader and is always bf16.
    tt::DataFormat trans_mat_data_format =
        (input_data_format == tt::DataFormat::Bfp8_b) ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_data_format);

    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    tt::DataFormat scalar_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt::tile_size(scalar_data_format);

    constexpr uint32_t Wt = 1;
    uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;
    uint32_t Wbytes = input.padded_shape()[-1] * sizeof(bfloat16);

    tt::tt_metal::IDevice* device = input.device();

    auto work = compute_rotary_work_split(input, output, Wt);
    bool row_major = work.row_major;
    uint32_t num_cores = work.num_cores;
    uint32_t num_cores_x = work.num_cores_x;
    uint32_t num_cores_y = work.num_cores_y;
    uint32_t num_rows_per_core_group_1 = work.num_rows_per_core_group_1;
    uint32_t num_rows_per_core_group_2 = work.num_rows_per_core_group_2;
    CoreRangeSet core_group_1 = work.core_group_1;
    CoreRangeSet core_group_2 = work.core_group_2;
    bool in_sharded = work.in_sharded;
    bool out_sharded = work.out_sharded;
    uint32_t num_input_tiles = work.num_input_tiles;
    uint32_t num_output_tiles = work.num_output_tiles;

    const bool decode = token_idx.has_value();
    uint32_t num_cos_sin_tiles = decode ? Wt : 2 * Wt;
    uint32_t num_interm_tiles = 1;

    // ---- Dataflow buffers (1:1 with the legacy CBs; sharded io borrows the tensor's memory,
    // replacing the legacy globally-allocated circular buffer + UpdateDynamicCircularBufferAddress pair) ----
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_data_format,
        .borrowed_from = in_sharded ? std::optional<TensorParamName>(SRC) : std::nullopt,
    });
    // trans_mat DFB at the slot the Wt>=2 path uses for "rotated input".
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = TRANS_MAT_DFB,
        .entry_size = trans_mat_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = trans_mat_data_format,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = COS_DFB,
        .entry_size = cos_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = cos_data_format,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SIN_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = sin_data_format,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = ROTATED_IN_INTERM_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = input_data_format,
    });
    // Keep sin/cos intermediates at input format regardless of sincos format.
    // The packer format stays stable across matmul / mul / add, avoiding
    // fragile pack_reconfig sequences after the matmul init for mixed precision.
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = COS_INTERM_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = input_data_format,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SIN_INTERM_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = input_data_format,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_data_format,
        .borrowed_from = out_sharded ? std::optional<TensorParamName>(DST) : std::nullopt,
    });
    if (decode) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RETILIZED_COS_DFB,
            .entry_size = cos_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = cos_data_format,
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RETILIZED_SIN_DFB,
            .entry_size = sin_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = sin_data_format,
        });
        // The untilized cos/sin "data" and "sync" indices shared one legacy circular-buffer descriptor's L1 region
        // (one allocation, two format descriptors): compute untilizes into the data index, the writer
        // waits on it, row-shuffles the data in place via a local NoC copy, then pushes the sync index
        // so compute's tilize (which waits on sync, reads/pops data) sees the shuffled rows. Modeled
        // as two aliased DFBs per pair. The data DFB genuinely has two locked consumers (compute's
        // tilize pops it, the writer wait_fronts it) on every node, which cannot fit the 1P+1C
        // invariant -- hence allow_instance_multi_binding; the legacy in-place ping-pong is preserved
        // byte-for-byte, sequenced by the aliased sync index rather than FIFO sync alone.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UNTILIZED_COS_DFB,
            .entry_size = scalar_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = scalar_data_format,
            .advanced_options = {.alias_with = {UNTILIZED_COS_SYNC_DFB}, .allow_instance_multi_binding = true},
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UNTILIZED_COS_SYNC_DFB,
            .entry_size = scalar_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = scalar_data_format,
            .advanced_options = {.alias_with = {UNTILIZED_COS_DFB}},
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UNTILIZED_SIN_DFB,
            .entry_size = scalar_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = scalar_data_format,
            .advanced_options = {.alias_with = {UNTILIZED_SIN_SYNC_DFB}, .allow_instance_multi_binding = true},
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UNTILIZED_SIN_SYNC_DFB,
            .entry_size = scalar_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = scalar_data_format,
            .advanced_options = {.alias_with = {UNTILIZED_SIN_DFB}},
        });
    }

    // ---- Tensor parameters (src/dst exist in every config: interleaved configs bind them from the
    // kernels, sharded configs use them as the borrowed DFBs' backing memory) ----
    const TensorParameter src_param{.unique_id = SRC, .spec = input.tensor_spec()};
    const TensorParameter cos_param{.unique_id = COS, .spec = cos.tensor_spec()};
    const TensorParameter sin_param{.unique_id = SIN, .spec = sin.tensor_spec()};
    const TensorParameter dst_param{.unique_id = DST, .spec = output.tensor_spec()};

    // ---- Defines (program structure is gated on these; the decode-only DFBs, bindings and args stay
    // conditional in step) ----
    KernelSpec::CompilerOptions::Defines reader_defines, writer_defines, compute_defines;
    if (decode) {
        reader_defines.emplace("DECODE_MODE", "1");
        writer_defines.emplace("DECODE_MODE", "1");
        compute_defines.emplace("DECODE_MODE", "1");
    }
    if (out_sharded) {
        writer_defines.emplace("OUT_SHARDED", "1");
    }

    // ---- Reader ----
    // The sharded reader has no src accessor (the input arrives on the borrowed input DFB), and
    // start_row_id is unread in decode mode (schema omits it in step with the kernel's #ifndef).
    Group<TensorBinding> reader_tensor_bindings;
    if (!in_sharded) {
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"});
    }
    reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = COS, .accessor_name = "cos"});
    reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = SIN, .accessor_name = "sin"});

    Group<std::string> reader_rta_names = {"num_rows"};
    if (!in_sharded) {
        reader_rta_names.push_back("start_id");
    }
    if (!decode) {
        reader_rta_names.push_back("start_row_id");
    }
    reader_rta_names.push_back("cos_sin_start_id");

    const KernelSpec reader{
        .unique_id = READER,
        .source = in_sharded
                      ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
                        "reader_rotary_embedding_single_tile_interleaved_start_id_sharded.cpp"
                      : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
                        "reader_rotary_embedding_single_tile_interleaved_start_id.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TRANS_MAT_DFB,
                 .accessor_name = "trans_mat",
                 .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args = {{"Ht", Ht}, {"HtWt", HtWt}},
        .runtime_arg_schema = {.runtime_arg_names = reader_rta_names},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ---- Writer ----
    // dst is bound (and its runtime args read) only on the interleaved-output path; under OUT_SHARDED
    // the writer drains nothing (out is borrowed on the output shard) and only runs the decode
    // shuffle + wait.
    Group<TensorBinding> writer_tensor_bindings;
    if (!out_sharded) {
        writer_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"});
    }

    Group<DFBBinding> writer_dfb_bindings = {
        DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}};
    if (decode) {
        // The writer touches each untilized data DFB twice: it wait_fronts it (a locked consume) and
        // raw-writes the shuffled rows back in place (a role-free write, bound here as its PRODUCER
        // side). Both bindings are required: compute self-loops these DFBs (untilize pushes, tilize
        // pops), and once any kernel self-loops a DFB the validator requires the producer and
        // consumer kernel sets to be equal — every same-side binding must come from a kernel that
        // appears on both sides. allow_instance_multi_binding (on the DFB spec) admits the resulting
        // 2P+2C census; on Gen1 the DFB lowers to a plain shared circular buffer, so the labels
        // carry no hardware semantics.
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_COS_DFB,
            .accessor_name = "untilized_cos",
            .endpoint_type = DFBEndpointType::PRODUCER});
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_COS_DFB,
            .accessor_name = "untilized_cos",
            .endpoint_type = DFBEndpointType::CONSUMER});
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_COS_SYNC_DFB,
            .accessor_name = "untilized_cos_sync",
            .endpoint_type = DFBEndpointType::PRODUCER});
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_SIN_DFB,
            .accessor_name = "untilized_sin",
            .endpoint_type = DFBEndpointType::PRODUCER});
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_SIN_DFB,
            .accessor_name = "untilized_sin",
            .endpoint_type = DFBEndpointType::CONSUMER});
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_SIN_SYNC_DFB,
            .accessor_name = "untilized_sin_sync",
            .endpoint_type = DFBEndpointType::PRODUCER});
    }

    Group<std::string> writer_rta_names = {"num_tiles"};
    if (!out_sharded) {
        writer_rta_names.push_back("start_id");
    }
    if (decode) {
        writer_rta_names.push_back("cos_sin_offset");
        writer_rta_names.push_back("Wt");
        writer_rta_names.push_back("Wbytes");
    }

    const KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
            "writer_rotary_embedding_interleaved_start_id.cpp",
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = writer_dfb_bindings,
        .tensor_bindings = writer_tensor_bindings,
        .runtime_arg_schema = {.runtime_arg_names = writer_rta_names},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- Compute (one KernelSpec per core group; the per-group row count stays a CTA) ----
    // Legacy resolved the TTNN compute config but copied only math_fidelity / fp32_dest_acc_en onto
    // its ComputeConfigDescriptor, so the resolved math_approx_mode / dst_full_sync_en were dropped
    // and the op ran at the descriptor defaults regardless of the caller's config. Reproduce that:
    // translate the resolved config, then reset the dropped fields to the descriptor-default results.
    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    sfpu_precision_mode(compute_hw) = Precision::Precise;  // dropped math_approx_mode (descriptor default false)
    double_buffer_dest(compute_hw) = true;                 // dropped dst_full_sync_en (descriptor default false)
    // Metal 2.0 requires an explicit unpack_modes entry for every consumed Float32 DFB when
    // enable_32_bit_dest (== fp32_dest_acc_en) is set, where legacy silently defaulted. The legacy op
    // set no unpack_to_dest_mode at all (all Default), so mirror that as UnpackToSrc. trans_mat and
    // the untilized/sync DFBs are never Float32 (bf16/bfp8 only); out is produced, not consumed.
    if (enable_32_bit_dest(compute_hw)) {
        auto& um = unpack_modes(compute_hw);
        if (input_data_format == tt::DataFormat::Float32) {
            um.emplace(INPUT_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            um.emplace(ROTATED_IN_INTERM_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            um.emplace(COS_INTERM_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            um.emplace(SIN_INTERM_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        if (cos_data_format == tt::DataFormat::Float32) {
            um.emplace(COS_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            if (decode) {
                um.emplace(RETILIZED_COS_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            }
        }
        if (sin_data_format == tt::DataFormat::Float32) {
            um.emplace(SIN_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            if (decode) {
                um.emplace(RETILIZED_SIN_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            }
        }
    }

    const auto make_compute = [&](const KernelSpecName& id, uint32_t num_rows_per_core_group) {
        // The compute-internal intermediates are self-loops (single toucher fills and drains them);
        // the decode untilized data DFBs bind compute as their genuine locked producer AND consumer.
        Group<DFBBinding> compute_dfb_bindings = {
            DFBBinding{.dfb_spec_name = INPUT_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = TRANS_MAT_DFB,
                .accessor_name = "trans_mat",
                .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = ROTATED_IN_INTERM_DFB,
                .accessor_name = "rotated_in_interm",
                .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = ROTATED_IN_INTERM_DFB,
                .accessor_name = "rotated_in_interm",
                .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = COS_INTERM_DFB,
                .accessor_name = "cos_interm",
                .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = COS_INTERM_DFB,
                .accessor_name = "cos_interm",
                .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = SIN_INTERM_DFB,
                .accessor_name = "sin_interm",
                .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = SIN_INTERM_DFB,
                .accessor_name = "sin_interm",
                .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
        };
        if (decode) {
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_COS_DFB,
                .accessor_name = "untilized_cos",
                .endpoint_type = DFBEndpointType::PRODUCER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_COS_DFB,
                .accessor_name = "untilized_cos",
                .endpoint_type = DFBEndpointType::CONSUMER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_COS_SYNC_DFB,
                .accessor_name = "untilized_cos_sync",
                .endpoint_type = DFBEndpointType::CONSUMER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_SIN_DFB,
                .accessor_name = "untilized_sin",
                .endpoint_type = DFBEndpointType::PRODUCER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_SIN_DFB,
                .accessor_name = "untilized_sin",
                .endpoint_type = DFBEndpointType::CONSUMER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_SIN_SYNC_DFB,
                .accessor_name = "untilized_sin_sync",
                .endpoint_type = DFBEndpointType::CONSUMER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RETILIZED_COS_DFB,
                .accessor_name = "retilized_cos",
                .endpoint_type = DFBEndpointType::PRODUCER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RETILIZED_COS_DFB,
                .accessor_name = "retilized_cos",
                .endpoint_type = DFBEndpointType::CONSUMER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RETILIZED_SIN_DFB,
                .accessor_name = "retilized_sin",
                .endpoint_type = DFBEndpointType::PRODUCER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RETILIZED_SIN_DFB,
                .accessor_name = "retilized_sin",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        return KernelSpec{
            .unique_id = id,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
                "rotary_embedding_single_tile_metal2.cpp",
            // Legacy compute defaulted opt_level to O3; Metal 2.0 defaults to O2, so set it explicitly.
            .compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(compute_dfb_bindings),
            .compile_time_args = {{"num_rows", num_rows_per_core_group}},
            .hw_config = compute_hw,
        };
    };

    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    kernels.push_back(make_compute(COMPUTE_G1, num_rows_per_core_group_1));
    work_units.push_back(
        WorkUnitSpec{.name = "rotary_group_1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});
    if (!core_group_2.ranges().empty()) {
        kernels.push_back(make_compute(COMPUTE_G2, num_rows_per_core_group_2));
        work_units.push_back(WorkUnitSpec{
            .name = "rotary_group_2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    // ---- Per-core runtime args (name-first tables built from the legacy node-first loop) ----
    uint32_t cos_sin_offset = 0;
    uint32_t cos_sin_start_id = 0;
    if (decode) {
        cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
        cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
    }

    uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_rows_per_core = i < g1_numcores ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        if (!decode) {
            cos_sin_start_id = num_tiles_written % HtWt;
        }
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_rows", num_rows_per_core}, {"cos_sin_start_id", cos_sin_start_id}});
        if (!in_sharded) {
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"start_id", num_tiles_written}});
        }
        if (!decode) {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values, core, {{"start_row_id", num_tiles_written / Wt % Ht}});
        }

        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"num_tiles", num_rows_per_core * Wt}});
        if (!out_sharded) {
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"start_id", num_tiles_written}});
        }
        if (decode) {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"cos_sin_offset", cos_sin_offset}, {"Wt", Wt}, {"Wbytes", Wbytes}});
        }
        num_tiles_written += num_rows_per_core * Wt;
    }

    ProgramSpec spec{
        .name = "rotary_embedding_single_tile",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = {src_param, cos_param, sin_param, dst_param},
        .work_units = std::move(work_units),
    };

    // The compute kernels have no runtime args, so they need no KernelRunArgs entry.
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {SRC, input.mesh_tensor()},
        {COS, cos.mesh_tensor()},
        {SIN, sin.mesh_tensor()},
        {DST, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts create_multi_tile_artifacts(
    const RotaryEmbeddingParams& operation_attributes,
    const RotaryEmbeddingInputs& tensor_args,
    Tensor& tensor_return_value) {
    // Spec resource names (function-local to avoid unity-build collisions).
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_group_1"};
    const KernelSpecName COMPUTE_G2{"compute_group_2"};
    const DFBSpecName INPUT_DFB{"input"};                            // legacy c_0
    const DFBSpecName ROTATED_INPUT_DFB{"rotated_input"};            // legacy c_1
    const DFBSpecName COS_DFB{"cos"};                                // legacy c_2
    const DFBSpecName SIN_DFB{"sin"};                                // legacy c_3
    const DFBSpecName SCALAR_DFB{"scalar"};                          // legacy c_4
    const DFBSpecName ROTATED_IN_INTERM_DFB{"rotated_in_interm"};    // legacy c_24
    const DFBSpecName COS_INTERM_DFB{"cos_interm"};                  // legacy c_25
    const DFBSpecName SIN_INTERM_DFB{"sin_interm"};                  // legacy c_26
    const DFBSpecName OUT_DFB{"out"};                                // legacy c_16
    const DFBSpecName UNTILIZED_COS_DFB{"untilized_cos"};            // legacy c_27 (aliases c_5's L1)
    const DFBSpecName UNTILIZED_COS_SYNC_DFB{"untilized_cos_sync"};  // legacy c_5 (aliases c_27's L1)
    const DFBSpecName UNTILIZED_SIN_DFB{"untilized_sin"};            // legacy c_28 (aliases c_6's L1)
    const DFBSpecName UNTILIZED_SIN_SYNC_DFB{"untilized_sin_sync"};  // legacy c_6 (aliases c_28's L1)
    const DFBSpecName RETILIZED_COS_DFB{"retilized_cos"};            // legacy c_29
    const DFBSpecName RETILIZED_SIN_DFB{"retilized_sin"};            // legacy c_30
    const TensorParamName SRC{"src"};
    const TensorParamName COS{"cos"};
    const TensorParamName SIN{"sin"};
    const TensorParamName DST{"dst"};

    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    auto& output = tensor_return_value;
    const auto& token_idx = operation_attributes.token_idx;

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_data_format);

    tt::DataFormat cos_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt::tile_size(cos_data_format);

    tt::DataFormat sin_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt::tile_size(sin_data_format);

    tt::DataFormat scalar_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt::tile_size(scalar_data_format);

    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t Wt = input.padded_shape()[-1] / TILE_WIDTH;
    uint32_t half_Wt = Wt / 2;
    uint32_t HtWt = Ht * Wt;
    uint32_t Wbytes = input.padded_shape()[-1] * sizeof(bfloat16);

    tt::tt_metal::IDevice* device = input.device();

    auto work = compute_rotary_work_split(input, output, Wt);
    bool row_major = work.row_major;
    uint32_t num_cores = work.num_cores;
    uint32_t num_cores_x = work.num_cores_x;
    uint32_t num_cores_y = work.num_cores_y;
    uint32_t num_rows_per_core_group_1 = work.num_rows_per_core_group_1;
    uint32_t num_rows_per_core_group_2 = work.num_rows_per_core_group_2;
    CoreRangeSet core_group_1 = work.core_group_1;
    CoreRangeSet core_group_2 = work.core_group_2;
    bool in_sharded = work.in_sharded;
    bool out_sharded = work.out_sharded;
    uint32_t num_input_tiles = work.num_input_tiles;
    uint32_t num_output_tiles = work.num_output_tiles;

    const bool decode = token_idx.has_value();
    uint32_t num_rotated_input_tiles = 2 * Wt;
    uint32_t num_cos_sin_tiles = decode ? Wt : 2 * Wt;
    uint32_t num_scalar_tiles = 1;
    uint32_t num_interm_tiles = 1;

    // ---- Dataflow buffers (1:1 with the legacy CBs; sharded io borrows the tensor's memory,
    // replacing the legacy globally-allocated circular buffer + UpdateDynamicCircularBufferAddress pair) ----
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_data_format,
        .borrowed_from = in_sharded ? std::optional<TensorParamName>(SRC) : std::nullopt,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = ROTATED_INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_rotated_input_tiles,
        .data_format_metadata = input_data_format,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = COS_DFB,
        .entry_size = cos_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = cos_data_format,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SIN_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_cos_sin_tiles,
        .data_format_metadata = sin_data_format,
    });
    // Used for bcast scalar
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SCALAR_DFB,
        .entry_size = scalar_single_tile_size,
        .num_entries = num_scalar_tiles,
        .data_format_metadata = scalar_data_format,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = ROTATED_IN_INTERM_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = input_data_format,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = COS_INTERM_DFB,
        .entry_size = cos_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = cos_data_format,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = SIN_INTERM_DFB,
        .entry_size = sin_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = sin_data_format,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_data_format,
        .borrowed_from = out_sharded ? std::optional<TensorParamName>(DST) : std::nullopt,
    });
    if (decode) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RETILIZED_COS_DFB,
            .entry_size = cos_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = cos_data_format,
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RETILIZED_SIN_DFB,
            .entry_size = sin_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = sin_data_format,
        });
        // See the single-tile builder for the aliased data/sync ping-pong scheme these DFBs carry;
        // the shapes are identical here, only Wt differs.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UNTILIZED_COS_DFB,
            .entry_size = scalar_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = scalar_data_format,
            .advanced_options = {.alias_with = {UNTILIZED_COS_SYNC_DFB}, .allow_instance_multi_binding = true},
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UNTILIZED_COS_SYNC_DFB,
            .entry_size = scalar_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = scalar_data_format,
            .advanced_options = {.alias_with = {UNTILIZED_COS_DFB}},
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UNTILIZED_SIN_DFB,
            .entry_size = scalar_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = scalar_data_format,
            .advanced_options = {.alias_with = {UNTILIZED_SIN_SYNC_DFB}, .allow_instance_multi_binding = true},
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UNTILIZED_SIN_SYNC_DFB,
            .entry_size = scalar_single_tile_size,
            .num_entries = Wt,
            .data_format_metadata = scalar_data_format,
            .advanced_options = {.alias_with = {UNTILIZED_SIN_DFB}},
        });
    }

    // ---- Tensor parameters ----
    const TensorParameter src_param{.unique_id = SRC, .spec = input.tensor_spec()};
    const TensorParameter cos_param{.unique_id = COS, .spec = cos.tensor_spec()};
    const TensorParameter sin_param{.unique_id = SIN, .spec = sin.tensor_spec()};
    const TensorParameter dst_param{.unique_id = DST, .spec = output.tensor_spec()};

    const uint16_t bfloat16_scalar = std::bit_cast<uint16_t>(bfloat16(-1.0f));

    // ---- Defines ----
    KernelSpec::CompilerOptions::Defines reader_defines, writer_defines, compute_defines;
    if (decode) {
        reader_defines.emplace("DECODE_MODE", "1");
        writer_defines.emplace("DECODE_MODE", "1");
        compute_defines.emplace("DECODE_MODE", "1");
    }
    if (out_sharded) {
        writer_defines.emplace("OUT_SHARDED", "1");
    }

    // ---- Reader ----
    Group<TensorBinding> reader_tensor_bindings;
    if (!in_sharded) {
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"});
    }
    reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = COS, .accessor_name = "cos"});
    reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = SIN, .accessor_name = "sin"});

    // The interleaved reader takes half_Wt in tiles; the sharded reader takes it in bytes
    // (half_Wt_size) for its in-L1 half-swap copies.
    KernelSpec::CompileTimeArgs reader_ctas = {
        {"scalar_value", bfloat16_scalar}, {"Ht", Ht}, {"Wt", Wt}, {"HtWt", HtWt}};
    if (in_sharded) {
        reader_ctas.emplace("half_Wt_size", half_Wt * input_single_tile_size);
    } else {
        reader_ctas.emplace("half_Wt", half_Wt);
    }

    Group<std::string> reader_rta_names = {"num_rows"};
    if (!in_sharded) {
        reader_rta_names.push_back("start_id");
    }
    if (!decode) {
        reader_rta_names.push_back("start_row_id");
    }
    reader_rta_names.push_back("cos_sin_start_id");

    const KernelSpec reader{
        .unique_id = READER,
        .source = in_sharded
                      ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
                        "reader_rotary_embedding_interleaved_start_id_sharded.cpp"
                      : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
                        "reader_rotary_embedding_interleaved_start_id.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = ROTATED_INPUT_DFB,
                 .accessor_name = "rotated_input",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = SCALAR_DFB, .accessor_name = "scalar", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args = reader_ctas,
        .runtime_arg_schema = {.runtime_arg_names = reader_rta_names},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ---- Writer (same source and schema shape as the single-tile variant) ----
    Group<TensorBinding> writer_tensor_bindings;
    if (!out_sharded) {
        writer_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"});
    }

    Group<DFBBinding> writer_dfb_bindings = {
        DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}};
    if (decode) {
        // The writer touches each untilized data DFB twice: it wait_fronts it (a locked consume) and
        // raw-writes the shuffled rows back in place (a role-free write, bound here as its PRODUCER
        // side). Both bindings are required: compute self-loops these DFBs (untilize pushes, tilize
        // pops), and once any kernel self-loops a DFB the validator requires the producer and
        // consumer kernel sets to be equal — every same-side binding must come from a kernel that
        // appears on both sides. allow_instance_multi_binding (on the DFB spec) admits the resulting
        // 2P+2C census; on Gen1 the DFB lowers to a plain shared circular buffer, so the labels
        // carry no hardware semantics.
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_COS_DFB,
            .accessor_name = "untilized_cos",
            .endpoint_type = DFBEndpointType::PRODUCER});
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_COS_DFB,
            .accessor_name = "untilized_cos",
            .endpoint_type = DFBEndpointType::CONSUMER});
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_COS_SYNC_DFB,
            .accessor_name = "untilized_cos_sync",
            .endpoint_type = DFBEndpointType::PRODUCER});
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_SIN_DFB,
            .accessor_name = "untilized_sin",
            .endpoint_type = DFBEndpointType::PRODUCER});
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_SIN_DFB,
            .accessor_name = "untilized_sin",
            .endpoint_type = DFBEndpointType::CONSUMER});
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = UNTILIZED_SIN_SYNC_DFB,
            .accessor_name = "untilized_sin_sync",
            .endpoint_type = DFBEndpointType::PRODUCER});
    }

    Group<std::string> writer_rta_names = {"num_tiles"};
    if (!out_sharded) {
        writer_rta_names.push_back("start_id");
    }
    if (decode) {
        writer_rta_names.push_back("cos_sin_offset");
        writer_rta_names.push_back("Wt");
        writer_rta_names.push_back("Wbytes");
    }

    const KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
            "writer_rotary_embedding_interleaved_start_id.cpp",
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = writer_dfb_bindings,
        .tensor_bindings = writer_tensor_bindings,
        .runtime_arg_schema = {.runtime_arg_names = writer_rta_names},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- Compute (one KernelSpec per core group) ----
    // NOTE: legacy create() left math_fidelity/fp32_dest_acc_en unset for the g1
    // ComputeConfig in the multi-tile path; preserve those defaults here (group 1 runs the
    // default-constructed Gen1 config while group 2 carries the caller's resolved knobs).
    ComputeHardwareConfig compute_hw_g1{ComputeGen1Config{}};
    auto compute_hw_g2 = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    // Legacy copied only math_fidelity / fp32_dest_acc_en onto the g2 descriptor, dropping the
    // resolved math_approx_mode / dst_full_sync_en -- reset those to the descriptor-default results.
    sfpu_precision_mode(compute_hw_g2) = Precision::Precise;  // dropped math_approx_mode (default false)
    double_buffer_dest(compute_hw_g2) = true;                 // dropped dst_full_sync_en (default false)
    // Metal 2.0 requires an explicit unpack_modes entry for every consumed Float32 DFB when
    // enable_32_bit_dest (== fp32_dest_acc_en) is set, where legacy silently defaulted (all entries
    // Default -> UnpackToSrc). Only g2 can have enable_32_bit_dest set (g1 runs the default config).
    // scalar and the untilized/sync DFBs are always Float16_b; out is produced, not consumed.
    if (enable_32_bit_dest(compute_hw_g2)) {
        auto& um = unpack_modes(compute_hw_g2);
        if (input_data_format == tt::DataFormat::Float32) {
            um.emplace(INPUT_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            um.emplace(ROTATED_INPUT_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            um.emplace(ROTATED_IN_INTERM_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        if (cos_data_format == tt::DataFormat::Float32) {
            um.emplace(COS_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            um.emplace(COS_INTERM_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            if (decode) {
                um.emplace(RETILIZED_COS_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            }
        }
        if (sin_data_format == tt::DataFormat::Float32) {
            um.emplace(SIN_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            um.emplace(SIN_INTERM_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            if (decode) {
                um.emplace(RETILIZED_SIN_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            }
        }
    }

    const auto make_compute = [&](const KernelSpecName& id,
                                  uint32_t num_rows_per_core_group,
                                  const ComputeHardwareConfig& compute_hw) {
        Group<DFBBinding> compute_dfb_bindings = {
            DFBBinding{.dfb_spec_name = INPUT_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = ROTATED_INPUT_DFB,
                .accessor_name = "rotated_in",
                .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = COS_DFB, .accessor_name = "cos", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = SIN_DFB, .accessor_name = "sin", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = SCALAR_DFB, .accessor_name = "scalar", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = ROTATED_IN_INTERM_DFB,
                .accessor_name = "rotated_in_interm",
                .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = ROTATED_IN_INTERM_DFB,
                .accessor_name = "rotated_in_interm",
                .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = COS_INTERM_DFB,
                .accessor_name = "cos_interm",
                .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = COS_INTERM_DFB,
                .accessor_name = "cos_interm",
                .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = SIN_INTERM_DFB,
                .accessor_name = "sin_interm",
                .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = SIN_INTERM_DFB,
                .accessor_name = "sin_interm",
                .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
        };
        if (decode) {
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_COS_DFB,
                .accessor_name = "untilized_cos",
                .endpoint_type = DFBEndpointType::PRODUCER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_COS_DFB,
                .accessor_name = "untilized_cos",
                .endpoint_type = DFBEndpointType::CONSUMER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_COS_SYNC_DFB,
                .accessor_name = "untilized_cos_sync",
                .endpoint_type = DFBEndpointType::CONSUMER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_SIN_DFB,
                .accessor_name = "untilized_sin",
                .endpoint_type = DFBEndpointType::PRODUCER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_SIN_DFB,
                .accessor_name = "untilized_sin",
                .endpoint_type = DFBEndpointType::CONSUMER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = UNTILIZED_SIN_SYNC_DFB,
                .accessor_name = "untilized_sin_sync",
                .endpoint_type = DFBEndpointType::CONSUMER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RETILIZED_COS_DFB,
                .accessor_name = "retilized_cos",
                .endpoint_type = DFBEndpointType::PRODUCER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RETILIZED_COS_DFB,
                .accessor_name = "retilized_cos",
                .endpoint_type = DFBEndpointType::CONSUMER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RETILIZED_SIN_DFB,
                .accessor_name = "retilized_sin",
                .endpoint_type = DFBEndpointType::PRODUCER});
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RETILIZED_SIN_DFB,
                .accessor_name = "retilized_sin",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        return KernelSpec{
            .unique_id = id,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
                "rotary_embedding.cpp",
            // Legacy compute defaulted opt_level to O3; Metal 2.0 defaults to O2, so set it explicitly.
            .compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(compute_dfb_bindings),
            .compile_time_args = {{"num_rows", num_rows_per_core_group}, {"Wt", Wt}, {"half_Wt", half_Wt}},
            .hw_config = compute_hw,
        };
    };

    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    kernels.push_back(make_compute(COMPUTE_G1, num_rows_per_core_group_1, compute_hw_g1));
    work_units.push_back(
        WorkUnitSpec{.name = "rotary_group_1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});
    if (!core_group_2.ranges().empty()) {
        kernels.push_back(make_compute(COMPUTE_G2, num_rows_per_core_group_2, compute_hw_g2));
        work_units.push_back(WorkUnitSpec{
            .name = "rotary_group_2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    // ---- Per-core runtime args (name-first tables built from the legacy node-first loop) ----
    uint32_t cos_sin_offset = 0;
    uint32_t cos_sin_start_id = 0;
    if (decode) {
        cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
        cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
    }

    uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_rows_per_core = i < g1_numcores ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        if (!decode) {
            cos_sin_start_id = num_tiles_written % HtWt;
        }
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_rows", num_rows_per_core}, {"cos_sin_start_id", cos_sin_start_id}});
        if (!in_sharded) {
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"start_id", num_tiles_written}});
        }
        if (!decode) {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values, core, {{"start_row_id", num_tiles_written / Wt % Ht}});
        }

        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"num_tiles", num_rows_per_core * Wt}});
        if (!out_sharded) {
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"start_id", num_tiles_written}});
        }
        if (decode) {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"cos_sin_offset", cos_sin_offset}, {"Wt", Wt}, {"Wbytes", Wbytes}});
        }
        num_tiles_written += num_rows_per_core * Wt;
    }

    ProgramSpec spec{
        .name = "rotary_embedding_multi_tile",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = {src_param, cos_param, sin_param, dst_param},
        .work_units = std::move(work_units),
    };

    // The compute kernels have no runtime args, so they need no KernelRunArgs entry.
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {SRC, input.mesh_tensor()},
        {COS, cos.mesh_tensor()},
        {SIN, sin.mesh_tensor()},
        {DST, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace

ttnn::device_operation::ProgramArtifacts RotaryEmbeddingProgramFactory::create_program_artifacts(
    const RotaryEmbeddingParams& operation_attributes,
    const RotaryEmbeddingInputs& tensor_args,
    Tensor& tensor_return_value) {
    if (tensor_args.input.padded_shape()[-1] / TILE_WIDTH == 1) {
        return create_single_tile_artifacts(operation_attributes, tensor_args, tensor_return_value);
    }
    return create_multi_tile_artifacts(operation_attributes, tensor_args, tensor_return_value);
}

tt::tt_metal::experimental::ProgramRunArgs RotaryEmbeddingProgramFactory::override_runtime_arguments(
    const RotaryEmbeddingParams& operation_attributes,
    const RotaryEmbeddingInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Spec resource names -- must match create_program_artifacts (function-local, per-factory). The
    // reader/writer ids and the two decode arg names are identical across both descriptor variants,
    // so one override serves both.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const TensorParamName SRC{"src"};
    const TensorParamName COS{"cos"};
    const TensorParamName SIN{"sin"};
    const TensorParamName DST{"dst"};

    // Runs on every program-cache hit. Only buffer addresses (never hashed) and the token_idx-derived
    // decode scalars (deliberately hash-excluded, so successive decode positions cache-hit one
    // program) can change across a hit. Everything else is a function of the shapes / memory configs /
    // seq_len that compute_program_hash keys on, so it is identical by construction.
    //
    // On this concept the framework refreshes nothing on its own, so re-bind every io tensor: the
    // interleaved arg-slot addresses AND the sharded borrowed-DFB backing addresses (the legacy
    // UpdateDynamicCircularBufferAddress block) all refresh through these TensorArguments.
    ProgramRunArgs params;
    params.tensor_args = {
        {SRC, tensor_args.input.mesh_tensor()},
        {COS, tensor_args.cos.mesh_tensor()},
        {SIN, tensor_args.sin.mesh_tensor()},
        {DST, tensor_return_value.mesh_tensor()},
    };

    // Decode mode re-applies the two token-derived scalars on every core. Prefill instead derives
    // cos_sin_start_id from the hashed work split (num_tiles_written % HtWt) and leaves
    // cos_sin_offset at 0, so neither arg is touched there. Both are core-invariant.
    const auto& token_idx = operation_attributes.token_idx;
    if (token_idx.has_value()) {
        const auto& input = tensor_args.input;
        // Wt == 1 on the single-tile path (X == TILE_WIDTH); this expression yields 1 there too.
        const uint32_t Wt = input.padded_shape()[-1] / TILE_WIDTH;
        const uint32_t Wbytes = input.padded_shape()[-1] * sizeof(bfloat16);
        const uint32_t cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
        const uint32_t cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;

        // Recompute the same work split / core order the miss path built, so the values land on
        // exactly the cores the cached program runs on.
        const auto work = compute_rotary_work_split(input, tensor_return_value, Wt);
        const auto cores = grid_to_cores(work.num_cores, work.num_cores_x, work.num_cores_y, work.row_major);

        KernelRunArgs reader_run_args{.kernel = READER};
        KernelRunArgs writer_run_args{.kernel = WRITER};
        for (const auto& core : cores) {
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"cos_sin_start_id", cos_sin_start_id}});
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"cos_sin_offset", cos_sin_offset}});
        }
        params.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    }

    return params;
}

}  // namespace ttnn::experimental::prim
