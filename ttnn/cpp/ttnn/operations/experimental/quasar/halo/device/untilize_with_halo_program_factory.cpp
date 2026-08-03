// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/halo/device/untilize_with_halo_program_factory.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <cmath>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

// TODO: Look into increasing this to tradeoff some L1 for performance (#19980)
constexpr int UNTILIZE_BLOCK_SIZE = 32;

constexpr bool ENABLE_UNTILIZE_DOUBLE_BUFFERING = true;

namespace {

// ---------------------------------------------------------------------------
// Metal 2.0 named resources for the halo (untilize_with_halo) program.
//
// Tensor parameters (Case-1 bindings; never bake addresses into RTAs):
//   IN          - input tensor.  On the tiled (untilize) path it backs the
//                 borrowed SRC dataflow buffer the compute kernel consumes; on
//                 the ROW_MAJOR (skip_untilize) path both readers read it
//                 directly via TensorAccessor(tensor::in).get_bank_base_address().
//   OUT         - output tensor.  BOTH readers NoC-scatter-write disjoint
//                 regions into it via TensorAccessor(tensor::out).get_bank_base_address().
//                 (This binding on both readers is legal — a TensorParameter
//                 has no endpoint-exclusivity constraint, unlike a DFB.)
//   PAD_CONFIG0/1, GATHER_CONFIG0/1 - op-owned sliding-window config tensors.
//                 Pure address sources read by base pointer (L1 path) or by
//                 page (DRAM path) from the respective reader.
//
// Dataflow buffers (genuine FIFOs only):
//   SRC_DFB           - borrowed from IN.  reader0 fake-pushes the resident
//                       input shard; compute consumes it.  (tiled path only)
//   UNTILIZE_OUT0/1   - real scratch FIFOs: compute produces, reader0/reader1
//                       consume (one each, SPSC-clean).  (tiled path only)
//   PAD0/1            - per-reader pad-immediate scratch (only when pad_val != 0).
//                       Cross-reader bound: reader0 produces PAD0 / consumes PAD1,
//                       reader1 produces PAD1 / consumes PAD0 (the pad value is the
//                       same constant on both, both run on the same core), so neither
//                       DM kernel self-loops a DFB.  See make_reader.
// ---------------------------------------------------------------------------
const TensorParamName IN{"in"};
const TensorParamName OUT{"out"};
const TensorParamName PAD_CONFIG0{"pad_config0"};
const TensorParamName PAD_CONFIG1{"pad_config1"};
const TensorParamName GATHER_CONFIG0{"gather_config0"};
const TensorParamName GATHER_CONFIG1{"gather_config1"};

const DFBSpecName SRC_DFB{"src"};
const DFBSpecName UNTILIZE_OUT0{"untilize_out0"};
const DFBSpecName UNTILIZE_OUT1{"untilize_out1"};
const DFBSpecName PAD0{"pad0"};
const DFBSpecName PAD1{"pad1"};
// DRAM-config landing scratch (only used when config_tensors_in_dram): each reader
// async-reads its config page from DRAM into a private L1 scratch, then reads it by
// base pointer.  Self-loop bound (one reader fills + reads -> validator-satisfying).
const DFBSpecName GATHER_SCRATCH0{"gather_scratch0"};
const DFBSpecName GATHER_SCRATCH1{"gather_scratch1"};
const DFBSpecName PAD_SCRATCH0{"pad_config_scratch0"};
const DFBSpecName PAD_SCRATCH1{"pad_config_scratch1"};

const KernelSpecName READER0{"reader0"};
const KernelSpecName READER1{"reader1"};
const KernelSpecName COMPUTE{"compute"};

constexpr const char* kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/quasar/halo/device/kernels/dataflow/halo_gather.cpp";
constexpr const char* kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/quasar/halo/device/kernels/compute/pack_untilize.cpp";

// Move one sliding-window config (host) tensor onto the device and return it as
// a sole-owner MeshTensor for ProgramArtifacts::op_owned_tensors (the framework
// keeps it alive at a stable address for the cached Program; see #44565).
MeshTensor build_config_mesh_tensor(
    const std::vector<std::vector<uint16_t>>& config,
    const ttnn::operations::sliding_window::ParallelConfig& parallel_config,
    bool is_block_sharded,
    MeshDevice* device,
    bool config_tensors_in_dram) {
    using namespace ttnn::operations;
    const auto host_tensor =
        sliding_window::construct_on_host_config_tensor(config, parallel_config, config_tensors_in_dram);
    Tensor device_tensor = sliding_window::move_config_tensor_to_device(
        host_tensor, parallel_config, is_block_sharded, device, config_tensors_in_dram);
    TT_ASSERT(device_tensor.dtype() == DataType::UINT16);
    return device_tensor.device_storage().release_mesh_tensor();
}

}  // namespace

ttnn::device_operation::ProgramArtifacts UntilizeWithHaloProgramFactory::create_program_artifacts(
    const HaloParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor) {
    const auto& pad_val = operation_attributes.pad_val;
    const int block_size = UNTILIZE_BLOCK_SIZE;
    const uint32_t ncores_nhw = operation_attributes.config.num_cores_nhw;
    const uint32_t max_out_nsticks_per_core = operation_attributes.max_out_nsticks_per_core;
    const bool config_tensors_in_dram = operation_attributes.config_tensors_in_dram;
    const bool remote_read = operation_attributes.remote_read;
    const bool transpose_mcast = operation_attributes.transpose_mcast;

    auto* device = input_tensor.device();

    const bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_height_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool is_width_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    TT_ASSERT(output_tensor.buffer() != nullptr, "Output buffer should be allocated on device!");

    const bool skip_untilize = input_tensor.layout() == Layout::ROW_MAJOR;
    const bool is_in_tiled = input_tensor.layout() == Layout::TILE;

    const auto& input_shape = input_tensor.padded_shape();

    const tt::DataFormat in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    const uint32_t out_nbytes = datum_size(out_df);

    const CoreRangeSet all_cores = output_tensor.shard_spec().value().grid;

    const ShardOrientation shard_orientation = output_tensor.shard_spec().value().orientation;
    const auto input_shard_shape = input_tensor.shard_spec().value().shape;
    const auto output_shard_shape = output_tensor.shard_spec().value().shape;
    TT_ASSERT(input_shard_shape[1] == output_shard_shape[1], "Expected input and output shard widths to match");

    const uint32_t input_nhw_height = input_shape[0] * input_shape[1] * input_shape[2];
    const uint32_t remapped_input_shard_shape_for_output_grid = tt::div_up(input_nhw_height, ncores_nhw);

    const uint32_t ntiles_per_block = tt::div_up(input_shard_shape[1], TILE_WIDTH);
    const uint32_t input_nblocks_per_core = tt::div_up(remapped_input_shard_shape_for_output_grid, TILE_HEIGHT);
    uint32_t input_npages = ntiles_per_block * input_nblocks_per_core;

    uint32_t in_page_size = tt::tile_size(in_df);

    // Calculate aligned stick size - used for both input and output since channels don't change
    const uint32_t stick_nbytes = output_shard_shape[1] * out_nbytes;
    uint32_t aligned_stick_nbytes = stick_nbytes;
    if (stick_nbytes % input_tensor.buffer()->alignment() != 0) {
        aligned_stick_nbytes = tt::round_up(stick_nbytes, input_tensor.buffer()->alignment());
    }
    const uint32_t out_tile_size = tt::tile_size(out_df);

    // For ROW_MAJOR input the kernel reads with aligned_stick_nbytes stride
    // across the full input shard.
    if (skip_untilize) {
        in_page_size = aligned_stick_nbytes;
        input_npages = input_shard_shape[0];
    }

    // We need to clamp in the case that the block size is larger than the nhw input size
    TT_FATAL(block_size % TILE_HEIGHT == 0, "Block size must be a multiple of tile height (was {})", block_size);
    const uint32_t clamped_block_size_height =
        std::min(static_cast<uint32_t>(block_size), input_nblocks_per_core * TILE_HEIGHT);
    TT_FATAL(
        clamped_block_size_height % TILE_HEIGHT == 0,
        "Block size must be a multiple of tile height (was {})",
        clamped_block_size_height);

    const uint32_t out_cb_pagesize = aligned_stick_nbytes;
    const uint32_t out_cb_npages = max_out_nsticks_per_core;
    (void)out_cb_pagesize;  // OUT is now a TensorParameter (no borrowed DFB); kept for parity / clarity.
    (void)out_cb_npages;

    const bool is_rm_orientation = shard_orientation == ShardOrientation::ROW_MAJOR;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, is_rm_orientation);

    // ------------------------------------------------------------------------
    // Build op-owned sliding-window config tensors.
    // ------------------------------------------------------------------------
    using namespace ttnn::operations;
    auto pad_metadata = sliding_window::generate_pad_metadata(operation_attributes.config);
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata(operation_attributes.config);
    auto shard_boundaries = sliding_window::generate_shard_boundaries(operation_attributes.config);
    const uint32_t input_shard_height = input_tensor.memory_config().shard_spec()->shape[0];
    auto tensor_metadata =
        sliding_window::generate_tensor_metadata(pad_metadata, operation_attributes.config, input_shard_height);

    const uint32_t num_cores_x = input_tensor.memory_config().shard_spec()->grid.bounding_box().grid_size().x;

    auto kernel_config = sliding_window::generate_halo_kernel_config_tensors(
        tensor_metadata,
        shard_boundaries,
        is_block_sharded,
        transpose_mcast,
        remote_read,
        device,
        num_cores_x,
        is_in_tiled,
        UNTILIZE_BLOCK_SIZE);

    // op_owned_tensors[0..3] = pad_config0, pad_config1, gather_config0, gather_config1.
    std::vector<MeshTensor> op_owned_tensors;
    op_owned_tensors.reserve(4);
    op_owned_tensors.push_back(build_config_mesh_tensor(
        kernel_config.pad_config0,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        config_tensors_in_dram));
    op_owned_tensors.push_back(build_config_mesh_tensor(
        kernel_config.pad_config1,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        config_tensors_in_dram));
    op_owned_tensors.push_back(build_config_mesh_tensor(
        kernel_config.gather_config0,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        config_tensors_in_dram));
    op_owned_tensors.push_back(build_config_mesh_tensor(
        kernel_config.gather_config1,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        config_tensors_in_dram));
    const MeshTensor& pad_config0_t = op_owned_tensors[0];
    const MeshTensor& pad_config1_t = op_owned_tensors[1];
    const MeshTensor& gather_config0_t = op_owned_tensors[2];
    const MeshTensor& gather_config1_t = op_owned_tensors[3];

    const auto number_of_blocks_per_core = sliding_window::remap_nhw_scalar_argument_across_full_grid(
        kernel_config.number_of_blocks_per_core, operation_attributes.parallel_config);

    // padding-config buffers carry a 4-byte sentinel page when there is no padding work.
    const uint32_t EMPTY_PADDING_CONFIG_BUFFER_SIZE = 4;
    const bool enable_padding =
        config_tensors_in_dram ||
        pad_config0_t.tensor_spec().compute_page_size_bytes() != EMPTY_PADDING_CONFIG_BUFFER_SIZE ||
        pad_config1_t.tensor_spec().compute_page_size_bytes() != EMPTY_PADDING_CONFIG_BUFFER_SIZE;

    // ------------------------------------------------------------------------
    // TensorParameters.
    // ------------------------------------------------------------------------
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = IN, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUT, .spec = output_tensor.tensor_spec()},
        TensorParameter{.unique_id = GATHER_CONFIG0, .spec = gather_config0_t.tensor_spec()},
        TensorParameter{.unique_id = GATHER_CONFIG1, .spec = gather_config1_t.tensor_spec()},
    };
    // pad_config* are bound to the readers only when enable_padding (see binding below); define
    // them (and their tensor_args) only then so the spec completeness check (every TensorParameter
    // must be bound by a kernel) holds.
    if (enable_padding) {
        tensor_parameters.push_back(TensorParameter{.unique_id = PAD_CONFIG0, .spec = pad_config0_t.tensor_spec()});
        tensor_parameters.push_back(TensorParameter{.unique_id = PAD_CONFIG1, .spec = pad_config1_t.tensor_spec()});
    }

    // ------------------------------------------------------------------------
    // Dataflow buffers (genuine FIFOs only).
    // ------------------------------------------------------------------------
    Group<DataflowBufferSpec> dataflow_buffers;

    if (!skip_untilize) {
        // SRC: borrowed from the resident (tiled) input shard. reader0 fake-pushes,
        // compute consumes. SPSC clean.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = SRC_DFB,
            .entry_size = in_page_size,
            .num_entries = input_npages,
            .data_format_metadata = in_df,
            .borrowed_from = IN,
        });

        const uint32_t output_ntiles = (clamped_block_size_height / TILE_HEIGHT) * ntiles_per_block;
        const uint32_t untilize_out_cb_num_pages = ENABLE_UNTILIZE_DOUBLE_BUFFERING ? 2 * output_ntiles : output_ntiles;
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UNTILIZE_OUT0,
            .entry_size = out_tile_size,
            .num_entries = untilize_out_cb_num_pages,
            .data_format_metadata = out_df,
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = UNTILIZE_OUT1,
            .entry_size = out_tile_size,
            .num_entries = untilize_out_cb_num_pages,
            .data_format_metadata = out_df,
        });
    }

    // Per-reader pad scratch. Always allocate it when padding is enabled: Quasar has no static
    // MEM_ZEROS L1 region (WH/BH-only) for the zero-pad case to copy from, so the kernel always
    // sources padding from this scratch DFB -- filled via noc.async_write_zeros for pad_val==0, or
    // the immediate value otherwise.
    const bool use_pad_scratch = enable_padding;
    const uint32_t pad_cb_pagesize = aligned_stick_nbytes;
    if (use_pad_scratch) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = PAD0,
            .entry_size = pad_cb_pagesize,
            .num_entries = 1,
            .data_format_metadata = out_df,
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = PAD1,
            .entry_size = pad_cb_pagesize,
            .num_entries = 1,
            .data_format_metadata = out_df,
        });
    }

    const tt::DataFormat kernel_config_df = tt::DataFormat::RawUInt16;  // UInt16 unsupported for CB types
    if (config_tensors_in_dram) {
        const uint32_t gather_page0 = gather_config0_t.tensor_spec().compute_page_size_bytes();
        const uint32_t gather_page1 = gather_config1_t.tensor_spec().compute_page_size_bytes();
        const uint32_t pad_page0 = pad_config0_t.tensor_spec().compute_page_size_bytes();
        const uint32_t pad_page1 = pad_config1_t.tensor_spec().compute_page_size_bytes();
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = GATHER_SCRATCH0,
            .entry_size = gather_page0,
            .num_entries = 1,
            .data_format_metadata = kernel_config_df});
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = GATHER_SCRATCH1,
            .entry_size = gather_page1,
            .num_entries = 1,
            .data_format_metadata = kernel_config_df});
        if (enable_padding) {
            dataflow_buffers.push_back(DataflowBufferSpec{
                .unique_id = PAD_SCRATCH0,
                .entry_size = pad_page0,
                .num_entries = 1,
                .data_format_metadata = kernel_config_df});
            dataflow_buffers.push_back(DataflowBufferSpec{
                .unique_id = PAD_SCRATCH1,
                .entry_size = pad_page1,
                .num_entries = 1,
                .data_format_metadata = kernel_config_df});
        }
    }

    // ------------------------------------------------------------------------
    // Compute kernel (tiled / non-skip path only).
    // ------------------------------------------------------------------------
    const uint32_t block_stride = 2;  // Skip every 2nd block because of split reader

    Group<KernelSpec> kernels;

    if (!skip_untilize) {
        KernelSpec compute{
            .unique_id = COMPUTE,
            .source = std::filesystem::path(kComputeKernelPath),
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = SRC_DFB, .accessor_name = "src", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = UNTILIZE_OUT0,
                     .accessor_name = "untilize_out0",
                     .endpoint_type = DFBEndpointType::PRODUCER},
                 DFBBinding{
                     .dfb_spec_name = UNTILIZE_OUT1,
                     .accessor_name = "untilize_out1",
                     .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"tiles_per_row", ntiles_per_block}, {"block_size", clamped_block_size_height / TILE_HEIGHT}},
            .runtime_arg_schema = {.runtime_arg_names = {"total_blocks"}},
            .hw_config = ttnn::to_compute_hardware_config(
                input_tensor.device()->arch(), operation_attributes.compute_kernel_config),
        };
        kernels.push_back(std::move(compute));
    }

    // ------------------------------------------------------------------------
    // Reader kernels (RISCV_0 / RISCV_1) — split readers scatter-write disjoint
    // output regions.  Common CTA layout; per-reader overrides follow.
    // ------------------------------------------------------------------------
    const auto make_reader = [&](const KernelSpecName& name,
                                 const DFBSpecName& untilize_out_dfb,
                                 const DFBSpecName& pad_dfb,
                                 const DFBSpecName& pad_other_dfb,
                                 const DFBSpecName& gather_scratch_dfb,
                                 const DFBSpecName& pad_scratch_dfb,
                                 const TensorParamName& gather_cfg,
                                 const TensorParamName& pad_cfg,
                                 uint32_t reader_block_start_offset,
                                 DataMovementProcessor processor,
                                 NOC noc) {
        DataMovementHardwareConfig reader_hw;
        if (device->arch() == tt::ARCH::QUASAR) {
            // QSR: this reader fills/drains DFBs with many sub-tile (per-row stick) NOC reads/writes (gather
            // scatter-writes, pad replication, partial-page DRAM config reads); that sub-tile pattern stalls the
            // DFB implicit-sync credit accounting. Opt out so explicit push_back/pop_front stay authoritative
            // (mirrors tilize default / transpose HC-sharded).
            reader_hw = DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
        } else {
            reader_hw = DataMovementGen1Config{.processor = processor, .noc = noc};
        }
        KernelSpec reader{
            .unique_id = name,
            .source = std::filesystem::path(kReaderKernelPath),
            .hw_config = std::move(reader_hw),
        };

        // Tensor bindings: OUT always (scatter-write target).  IN directly only on
        // the skip path; on the tiled path IN is reached via the borrowed SRC DFB.
        reader.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = OUT, .accessor_name = "out"});
        if (skip_untilize) {
            reader.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = IN, .accessor_name = "in"});
        }
        reader.tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = gather_cfg, .accessor_name = "gather_config"});
        if (enable_padding) {
            reader.tensor_bindings.push_back(
                TensorBinding{.tensor_parameter_name = pad_cfg, .accessor_name = "padding_config"});
        }

        // DFB bindings: untilize_out (consumer) on the tiled path; pad scratch
        // (cross-reader, see below) when used.
        if (!skip_untilize) {
            reader.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = untilize_out_dfb,
                .accessor_name = "untilize_out",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        // SRC is borrowed from the resident (tiled) input shard.  The compute consumes
        // it (wait_front), so it needs a producer that advances the FIFO front pointer
        // without moving data (the data is already resident via borrowed_from = IN).
        // Only reader0 (block_start_offset == 0) fake-pushes it -> single producer, SPSC.
        if (!skip_untilize && reader_block_start_offset == 0) {
            reader.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = SRC_DFB, .accessor_name = "src", .endpoint_type = DFBEndpointType::PRODUCER});
        }
        if (use_pad_scratch) {
            // Cross-reader pad scratch (the pad value is the SAME constant on both readers, and both
            // readers run on the same core, so PAD0/PAD1 are interchangeable L1 sources). Each reader
            // PRODUCES its own pad DFB (pad_fill) and CONSUMES the peer reader's identical pad DFB
            // (pad_read). Neither reader binds one DFB as both producer and consumer -> no self-loop,
            // and each DFB has exactly one producer + one consumer (different kernels) -> SPSC clean.
            reader.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = pad_dfb, .accessor_name = "pad_fill", .endpoint_type = DFBEndpointType::PRODUCER});
            reader.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = pad_other_dfb,
                .accessor_name = "pad_read",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        if (config_tensors_in_dram) {
            // DRAM config landing scratch (self-loop: reader fills via NoC read, then reads it).
            reader.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = gather_scratch_dfb,
                .accessor_name = "gather_config_scratch",
                .endpoint_type = DFBEndpointType::PRODUCER});
            reader.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = gather_scratch_dfb,
                .accessor_name = "gather_config_scratch",
                .endpoint_type = DFBEndpointType::CONSUMER});
            if (enable_padding) {
                reader.dfb_bindings.push_back(DFBBinding{
                    .dfb_spec_name = pad_scratch_dfb,
                    .accessor_name = "padding_config_scratch",
                    .endpoint_type = DFBEndpointType::PRODUCER});
                reader.dfb_bindings.push_back(DFBBinding{
                    .dfb_spec_name = pad_scratch_dfb,
                    .accessor_name = "padding_config_scratch",
                    .endpoint_type = DFBEndpointType::CONSUMER});
            }
        }

        reader.compile_time_args = {
            {"pad_val", pad_val},
            {"aligned_stick_nbytes", aligned_stick_nbytes},
            {"is_block_sharded", static_cast<uint32_t>(is_block_sharded)},
            {"remote_read", static_cast<uint32_t>(remote_read)},
            {"is_col_major", static_cast<uint32_t>(transpose_mcast ? 1 : 0)},
            {"is_width_sharded", static_cast<uint32_t>(is_width_sharded)},
            {"skip_untilize", static_cast<uint32_t>(skip_untilize)},
            {"block_size_height", clamped_block_size_height},
            {"block_size_width_tiles", ntiles_per_block},
            {"block_start_offset", reader_block_start_offset},
            {"block_stride", block_stride},
            {"enable_padding", static_cast<uint32_t>(enable_padding)},
            {"input_npages", input_npages},
        };

        // DRAM config path reads config pages by page_id; the page index is a
        // per-core RTA.  L1 config path reads the local shard via base pointer.
        if (config_tensors_in_dram) {
            reader.runtime_arg_schema = {.runtime_arg_names = {"config_read_index"}};
        }

        // Preprocessor gates for conditionally-bound resources.  kernel_main() is a
        // non-template function, so `if constexpr` discarded branches still perform
        // name lookup on dfb::/tensor:: tokens — references to resources not bound on
        // a path must be #ifdef-gated out entirely (see metal2_port_patterns.md).
        KernelSpec::CompilerOptions::Defines defines;
        if (config_tensors_in_dram) {
            defines.emplace("CONFIG_TENSOR_IN_DRAM", "1");
        }
        if (skip_untilize) {
            defines.emplace("SKIP_UNTILIZE", "1");
        }
        // Only reader0 references dfb::src (the fake-push producer); gate the token out
        // of reader1's build entirely (kernel_main is non-template -> unbound names in
        // discarded branches still fail lookup).
        if (!skip_untilize && reader_block_start_offset == 0) {
            defines.emplace("SRC_PRODUCER", "1");
        }
        if (enable_padding) {
            defines.emplace("ENABLE_PADDING", "1");
        }
        if (use_pad_scratch) {
            defines.emplace("USE_PAD_SCRATCH", "1");
        }
        reader.compiler_options.defines = std::move(defines);
        return reader;
    };

    KernelSpec reader0 = make_reader(
        READER0,
        UNTILIZE_OUT0,
        /*pad_dfb (own)=*/PAD0,
        /*pad_other_dfb (peer)=*/PAD1,
        GATHER_SCRATCH0,
        PAD_SCRATCH0,
        GATHER_CONFIG0,
        PAD_CONFIG0,
        /*block_start_offset=*/0,
        DataMovementProcessor::RISCV_0,
        NOC::RISCV_0_default);
    KernelSpec reader1 = make_reader(
        READER1,
        UNTILIZE_OUT1,
        /*pad_dfb (own)=*/PAD1,
        /*pad_other_dfb (peer)=*/PAD0,
        GATHER_SCRATCH1,
        PAD_SCRATCH1,
        GATHER_CONFIG1,
        PAD_CONFIG1,
        /*block_start_offset=*/1,
        DataMovementProcessor::RISCV_1,
        NOC::RISCV_1_default);

    kernels.push_back(std::move(reader0));
    kernels.push_back(std::move(reader1));

    // ------------------------------------------------------------------------
    // Work unit + program spec.
    // ------------------------------------------------------------------------
    Group<KernelSpecName> wu_kernels;
    if (!skip_untilize) {
        wu_kernels.push_back(COMPUTE);
    }
    wu_kernels.push_back(READER0);
    wu_kernels.push_back(READER1);

    WorkUnitSpec wu{
        .name = "untilize_with_halo",
        .kernels = std::move(wu_kernels),
        .target_nodes = all_cores,
    };

    ProgramSpec spec{
        .name = "untilize_with_halo",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {wu},
    };

    // ------------------------------------------------------------------------
    // ProgramRunArgs.
    // ------------------------------------------------------------------------
    ProgramRunArgs run_args;

    if (!skip_untilize) {
        KernelRunArgs compute_args{.kernel = COMPUTE};
        for (size_t core_id = 0; core_id < cores.size(); ++core_id) {
            compute_args.runtime_arg_values["total_blocks"][NodeCoord{cores[core_id].x, cores[core_id].y}] =
                static_cast<uint32_t>(number_of_blocks_per_core[core_id]);
        }
        run_args.kernel_run_args.push_back(std::move(compute_args));
    }

    KernelRunArgs reader0_args{.kernel = READER0};
    KernelRunArgs reader1_args{.kernel = READER1};
    if (config_tensors_in_dram) {
        for (uint32_t core_index = 0; core_index < cores.size(); ++core_index) {
            const auto& core = cores[core_index];
            uint32_t read_index = 0;
            if (is_height_sharded) {
                read_index = core_index;
            } else if (is_width_sharded) {
                read_index = 0;
            } else if (is_block_sharded) {
                read_index = is_rm_orientation ? core.y : core.x;
            }
            const NodeCoord node{core.x, core.y};
            reader0_args.runtime_arg_values["config_read_index"][node] = read_index;
            reader1_args.runtime_arg_values["config_read_index"][node] = read_index;
        }
    }
    run_args.kernel_run_args.push_back(std::move(reader0_args));
    run_args.kernel_run_args.push_back(std::move(reader1_args));

    run_args.tensor_args = {
        {IN, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUT, TensorArgument{std::cref(output_tensor.mesh_tensor())}},
        {GATHER_CONFIG0, TensorArgument{std::cref(gather_config0_t)}},
        {GATHER_CONFIG1, TensorArgument{std::cref(gather_config1_t)}},
    };
    if (enable_padding) {
        run_args.tensor_args.insert({PAD_CONFIG0, TensorArgument{std::cref(pad_config0_t)}});
        run_args.tensor_args.insert({PAD_CONFIG1, TensorArgument{std::cref(pad_config1_t)}});
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
        .op_owned_tensors = std::move(op_owned_tensors),
    };
}

}  // namespace ttnn::prim::qsr
