// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_rm_program_factory.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Nested so the names cannot collide with the sibling factories' in the unity build (this op's
// factories share one translation unit).
namespace wh_sharded_rm {
const DFBSpecName IN0{"in0"};
const DFBSpecName OUT{"out"};
const DFBSpecName IN{"in"};
const DFBSpecName TILIZE{"tilize"};
const DFBSpecName OUT_STAGING{"out_staging"};
const TensorParamName INPUT{"input"};
const TensorParamName OUTPUT{"output"};
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE{"compute"};
}  // namespace wh_sharded_rm

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeWHShardedRMProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    using namespace wh_sharded_rm;

    const auto& input_tensor = tensor_args.input;
    const auto& input = input_tensor.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.is_allocated(), "Operand to transpose_wh needs to be allocated in a buffer on device!");

    tt::DataFormat src0_dfb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_dfb_data_format);
    tt::DataFormat dst_dfb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_dfb_data_format);

    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t stick_size_bytes = W * input_tensor.element_size();
    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    uint32_t output_page_size, pack_num_pages, pack_num_pages_last_col, pack_num_pages_last_row,
        pack_num_pages_last_row_col;
    if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (W % TILE_WIDTH) * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        auto output_page_size_last_col = TILE_WIDTH * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages_last_col = dst_single_tile_size / output_page_size_last_col;
        auto output_page_size_last_row = TILE_HEIGHT * (W % TILE_WIDTH) * output_tensor.element_size();
        pack_num_pages_last_row = dst_single_tile_size / output_page_size_last_row;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) == 0) {
        output_page_size = (W % TILE_WIDTH) * (TILE_HEIGHT)*output_tensor.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        pack_num_pages_last_col = pack_num_pages;
        pack_num_pages_last_row = 1;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) == 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (TILE_WIDTH) * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        pack_num_pages_last_col = 1;
        pack_num_pages_last_row = pack_num_pages;
        pack_num_pages_last_row_col = 1;
    } else {
        output_page_size = dst_single_tile_size;
        pack_num_pages = 1;
        pack_num_pages_last_col = 1;
        pack_num_pages_last_row = 1;
        pack_num_pages_last_row_col = 1;
    }

    log_debug(tt::LogOp, "output_page_size: {}", output_page_size);
    log_debug(tt::LogOp, "pack_num_pages: {}", pack_num_pages);
    log_debug(tt::LogOp, "pack_num_pages_last_col: {}", pack_num_pages_last_col);
    log_debug(tt::LogOp, "pack_num_pages_last_row: {}", pack_num_pages_last_row);
    log_debug(tt::LogOp, "pack_num_pages_last_row_col: {}", pack_num_pages_last_row_col);

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t num_hw_blocks_per_core = shard_height / H;

    log_debug(tt::LogOp, "shard_height: {}", shard_height);
    log_debug(tt::LogOp, "dst_single_tile_size: {}", dst_single_tile_size);

    bool fp32_dest_acc_en = src0_dfb_data_format == tt::DataFormat::Float32;

    auto& all_cores = shard_spec.grid;
    [[maybe_unused]] uint32_t num_cores = shard_spec.num_cores();

    log_debug(tt::LogOp, "all_cores: {}", all_cores);
    log_debug(tt::LogOp, "num_cores: {}", num_cores);

    // Temporary fix until pack_untilize is fully fixed: for taller blocks the packer untilizes into
    // a staging buffer and the writer copies each column into the output shard, instead of the
    // packer writing the shard directly. The compute and writer kernels select that path on the
    // OUT_STAGING define, which is emitted exactly when the staging buffer is bound.
    const bool use_out_staging = ht > 8;

    ProgramSpec spec{.name = "transpose_wh_sharded_rm"};

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()});

    // Both io buffers are built on the tensors' own sharded L1 memory rather than on
    // Program-lifetime storage; override_runtime_arguments re-points them on a cache hit.
    // in0: the reader only peeks its base pointer and reads the shard back through the NoC.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = stick_size_bytes,
        .num_entries = shard_height,
        .data_format_metadata = src0_dfb_data_format,
        .borrowed_from = INPUT,
    });

    // out: paged by output_page_size over the whole output shard. The buffer's page count is derived
    // from its total size, which must therefore be a whole number of pages (the legacy circular
    // buffer configuration enforced the same divisibility).
    const uint32_t output_total_bytes = stick_size_bytes * shard_height;
    TT_FATAL(
        output_total_bytes % output_page_size == 0,
        "transpose_wh sharded RM: output shard size {} B must be divisible by output page size {} B",
        output_total_bytes,
        output_page_size);
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT,
        .entry_size = output_page_size,
        .num_entries = output_total_bytes / output_page_size,
        .data_format_metadata = dst_dfb_data_format,
        .borrowed_from = OUTPUT,
    });

    // in (double-buffered intermediate): reader-filled sticks the compute kernel tilizes.
    uint32_t num_in_tiles = wt * 2;  // double buffer
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN,
        .entry_size = src0_single_tile_size,
        .num_entries = num_in_tiles,
        .data_format_metadata = src0_dfb_data_format,
    });

    // tilize intermediate: the compute kernel both fills it (tilize) and drains it (transpose), so
    // it is self-looped onto that one kernel.
    uint32_t num_im_tiles = ht * wt;
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = TILIZE,
        .entry_size = src0_single_tile_size,
        .num_entries = num_im_tiles,
        .data_format_metadata = src0_dfb_data_format,
    });

    // Output staging buffer (only when ht > 8): compute packs each transposed column here and the
    // writer copies it into the output shard. The legacy factory also allocated an "untilize"
    // intermediate (c_26) on this path that no kernel referenced; it is not re-created here.
    if (use_out_staging) {
        uint32_t num_out_tiles = ht * 2;  // double buffer
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT_STAGING,
            .entry_size = dst_single_tile_size,
            .num_entries = num_out_tiles,
            .data_format_metadata = dst_dfb_data_format,
        });
    }

    KernelSpec::CompilerOptions::Defines out_staging_defines;
    if (use_out_staging) {
        out_staging_defines.insert({"OUT_STAGING", "1"});
    }

    // ---- reader ----
    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                  "reader_unary_transpose_wh_sharded_rm.cpp",
        .dfb_bindings =
            {// Self-loop: the reader is the only kernel that touches the input shard (base-pointer
             // peek, no FIFO ops).
             DFBBinding{
                 .dfb_spec_name = IN0,
                 .accessor_name = "in0",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = IN0,
                 .accessor_name = "in0",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = IN,
                 .accessor_name = "in",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .compile_time_args =
            {{"num_hw_blocks_per_core", num_hw_blocks_per_core},
             {"Ht", ht},
             {"H_per_tile", H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT},
             {"H_per_tile_last", H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT},
             {"Wt", wt},
             {"W_size_bytes", stick_size_bytes},
             {"l1_write_offset_bytes", wt * input_tensor.element_size() * TILE_WIDTH}},
        .hw_config = create_reader_datamovement_config(),
    });

    // ---- writer ----
    // The writer always constructs the output-shard handle (and peeks its base pointer); it drains
    // the staging buffer into it only on the OUT_STAGING path. With staging, nothing else touches the
    // output shard, so it is self-looped onto the writer; without, the compute kernel packs into it
    // directly and the writer is its (role-free) consumer.
    Group<DFBBinding> writer_dfb_bindings;
    if (use_out_staging) {
        writer_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = OUT,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = OUT,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = OUT_STAGING,
                .accessor_name = "out_staging",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
        };
    } else {
        writer_dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }};
    }
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                  "writer_unary_transpose_wh_sharded_rm.cpp",
        .compiler_options = {.defines = out_staging_defines},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .compile_time_args =
            {{"num_hw_blocks_per_core", num_hw_blocks_per_core},
             {"Ht", ht},
             {"Wt", wt},
             {"W_per_tile", W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH},
             {"W_per_tile_last", W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH},
             {"H_size_bytes", H * output_tensor.element_size()},
             {"l1_read_offset_bytes", ht * output_tensor.element_size() * TILE_HEIGHT}},
        .hw_config = create_writer_datamovement_config(),
    });

    // ---- compute ----
    KernelSpec::CompilerOptions::Defines compute_defines{{"SHARDED", "1"}};
    for (const auto& define : out_staging_defines) {
        compute_defines.insert(define);
    }

    // Legacy built a ComputeConfigDescriptor directly (no TTNN ComputeKernelConfig feeding it),
    // setting only fp32_dest_acc_en and unpack_to_dest_mode; every other field kept its Metal
    // default, which ComputeHardwareConfig's defaults reproduce.
    ComputeHardwareConfig compute_hw{.enable_32_bit_dest = fp32_dest_acc_en};
    if (src0_dfb_data_format == tt::DataFormat::Float32) {
        // Keep both the tilize input and its output (which feeds the transpose) in full Float32 on
        // the unpack-to-dest path; otherwise the unpacker falls back to tf32 and drops the low
        // mantissa bits.
        compute_hw.unpack_modes.insert({IN, UnpackMode::UnpackToDest});
        compute_hw.unpack_modes.insert({TILIZE, UnpackMode::UnpackToDest});
    }

    // The compute kernel is shared with TransposeWHProgramFactory; this factory compiles it with
    // SHARDED.
    Group<DFBBinding> compute_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = IN,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        // Self-loop: this kernel tilizes into the intermediate and transposes back out of it.
        DFBBinding{
            .dfb_spec_name = TILIZE,
            .accessor_name = "tilize",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = TILIZE,
            .accessor_name = "tilize",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };
    if (use_out_staging) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_STAGING,
            .accessor_name = "out_staging",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    } else {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm_metal2.cpp",
        // Legacy left opt_level unset on a ComputeConfigDescriptor, which resolves to O3;
        // a Metal 2.0 KernelSpec defaults to O2, so the level is restated explicitly here.
        .compiler_options = {.defines = std::move(compute_defines), .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {{"Ht", ht},
             {"Wt", wt},
             {"HtWt", ht * wt},
             {"num_hw_blocks_per_core", num_hw_blocks_per_core},
             {"last_output_row_num_datums", H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT},
             {"pack_num_pages", pack_num_pages},
             {"pack_num_pages_last_col", pack_num_pages_last_col},
             {"pack_num_pages_last_row", pack_num_pages_last_row},
             {"pack_num_pages_last_row_col", pack_num_pages_last_row_col}},
        .hw_config = compute_hw,
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_cores,
    });

    // ---- run args ----
    // All three kernels take compile-time args only, so the buffer addresses are the whole
    // per-dispatch state; they ride on the two borrowed buffers.
    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(INPUT, input);
    run_args.tensor_args.emplace(OUTPUT, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ProgramRunArgs TransposeWHShardedRMProgramFactory::override_runtime_arguments(
    const TransposeParams& /*operation_attributes*/,
    const TransposeInputs& tensor_args,
    Tensor& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    using namespace wh_sharded_rm;
    // All three kernels take compile-time args only, so the buffer addresses are the whole per-dispatch
    // state; both ride on the borrowed buffers (in0, out), which re-resolve their L1 address from these
    // tensor arguments.
    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(INPUT, tensor_args.input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output_tensor.mesh_tensor());
    return run_args;
}

}  // namespace ttnn::prim
