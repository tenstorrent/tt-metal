// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"

#include <algorithm>
#include <utility>

#include "tt-metalium/work_split.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"

using tt::tt_metal::MeshTensor;

using tt::tt_metal::KernelBuildOptLevel;
using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DataMovementGen1Config;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::SemaphoreBinding;
using tt::tt_metal::experimental::SemaphoreSpec;
using tt::tt_metal::experimental::SemaphoreSpecName;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::unpack_modes;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts SparseMatmulMultiCoreReuseMcast1DProgramFactory::create_program_artifacts(
    const ttnn::prim::SparseMatmulParams& operation_attributes,
    const ttnn::prim::SparseMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    using namespace tt;
    using namespace operations::matmul::utilities;

    // from create_mesh-workload
    auto matmul_attributes = ttnn::prim::MatmulParams{
        operation_attributes.program_config,
        /*bcast_batch=*/std::nullopt,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.compute_kernel_config,
        /*untilize_out=*/false,
        operation_attributes.user_core_coord,
        /*user_fused_activation=*/std::nullopt,
        /*user_run_batched=*/false,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        operation_attributes.output_tile,
        operation_attributes.global_cb,
        operation_attributes.sub_device_id};

    auto chosen_program_config = operations::matmul::get_program_config(
        tensor_args.input_tensors.at(0),
        tensor_args.input_tensors.at(1),
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*bias_single_tile_size=*/0,
        matmul_attributes);
    operations::matmul::normalize_program_config(
        chosen_program_config, tensor_args.input_tensors.at(0).device()->compute_with_storage_grid_size());

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& sparsity = tensor_args.input_tensors.at(2);
    const auto& output_tensor = tensor_return_value.at(0);
    auto program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config);
    auto compute_with_storage_grid_size = program_config.allowed_worker_cores.value().bounding_box().grid_size();
    auto in0_block_w = program_config.in0_block_w;
    auto out_subblock_h = program_config.out_subblock_h;
    auto out_subblock_w = program_config.out_subblock_w;
    auto out_block_h = program_config.out_block_h;
    auto out_block_w = program_config.out_block_w;
    auto per_core_M = program_config.per_core_M;
    auto per_core_N = program_config.per_core_N;
    auto mcast_in0 = program_config.mcast_in0;

    auto nnz = operation_attributes.nnz;
    auto is_input_a_sparse = operation_attributes.is_input_a_sparse;

    // Indexed/gather mode: an optional `indices` operand (optional_input_tensors[0]) holds the
    // compacted list of active sparse-group ids. When present, the reader/sender kernels iterate only
    // the num_active selected groups (bB = indices[i]) instead of scanning all batchB sparsity slots,
    // and the output group axis is compact (length num_active). This maps onto the existing
    // get_batch_from_reader=false semantics: every iterated batch is processed (none skipped), so
    // compute and the in0 receiver simply loop num_batch_compute = num_active.
    //
    // The mode is signalled to the kernels purely by the "num_active" named compile-time arg below
    // (0 = off), so no preprocessor defines and no compile-time arg layout changes are needed in the
    // shared reader kernels. Both readers are shared with the dense matmul factories, which pass
    // {"num_active", 0} for the same reason they already bind an unused sparsity slot.
    const bool use_indices = operation_attributes.use_indices && !tensor_args.optional_input_tensors.empty() &&
                             tensor_args.optional_input_tensors.at(0).has_value();
    uint32_t num_active = 0;
    if (use_indices) {
        num_active = tensor_args.optional_input_tensors.at(0)->logical_volume();
    }
    // In indexed mode the readers never broadcast per-slot validity (every iterated batch is valid),
    // so get_batch_from_reader is forced false regardless of nnz.
    const bool get_batch_from_reader = use_indices ? false : !nnz.has_value();

    const auto& ashape = get_matmul_tensor_padded_shape(a, /*transpose=*/false);
    const auto& bshape = get_matmul_tensor_padded_shape(b, /*transpose=*/false);
    const auto in0_tile = get_matmul_tile(a, /*transpose=*/false);
    const auto in1_tile = get_matmul_tile(b, /*transpose=*/false);
    // cannot use the output tensor tile directly as that might be changed by user override
    const auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    // The Metalium memory objects the spec is built against. A ProgramSpec declares TensorParameters
    // from a MeshTensor's TensorSpec and the paired ProgramRunArgs references the very same
    // MeshTensor -- the framework matches a TensorArgument back to its input by MeshTensor identity,
    // so these references, not copies, are what get bound.
    const MeshTensor& in0_mesh_tensor = a.mesh_tensor();
    const MeshTensor& in1_mesh_tensor = b.mesh_tensor();
    const MeshTensor& sparsity_mesh_tensor = sparsity.mesh_tensor();
    const MeshTensor& out_mesh_tensor = output_tensor.mesh_tensor();

    // DFB dataformats
    const auto in0_data_format = tt_metal::datatype_to_dataformat_converter(in0_mesh_tensor.dtype());
    const auto in1_data_format = tt_metal::datatype_to_dataformat_converter(in1_mesh_tensor.dtype());
    const auto output_data_format = tt_metal::datatype_to_dataformat_converter(out_mesh_tensor.dtype());

    auto* const device = a.device();

    const auto in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    const auto in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    // Tiles whose size is not a multiple of the DRAM alignment (e.g. bfp8 32x16 = 544B on Blackhole's
    // 64B alignment) are padded to it in DRAM. in0/in1 are interleaved here, so the reader copies
    // tiles at the padded stride; the buffers must hold entries at the aligned stride and the unpacker
    // walks tiles at the same stride. No-op when already aligned. Replaces the staging-buffer workaround.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const uint32_t in0_aligned_tile_size = tt::align(in0_single_tile_size, dram_alignment);
    const uint32_t in1_aligned_tile_size = tt::align(in1_single_tile_size, dram_alignment);
    const auto output_single_tile_size = output_tile.get_tile_size(output_data_format);

    // The in1 sender/writer's own "sparsity" slot (its tensor binding, page size and dataflow buffer)
    // carries the active-group id list in indexed/gather mode -- that kernel never reads the sparsity
    // mask there, so reusing the slot avoids adding an operand to a kernel shared with the dense
    // matmul factories. The in0 sender ignores its own slot entirely in this mode, so it keeps
    // pointing at the real sparsity tensor.
    const Tensor& in1_sparsity_tensor = use_indices ? tensor_args.optional_input_tensors.at(0).value() : sparsity;
    const MeshTensor& in1_sparsity_mesh_tensor = in1_sparsity_tensor.mesh_tensor();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto batchB = get_batch_size(bshape);

    // When input A and input B are sparse, the batch dims are same.
    // We pick batchB and set batchA to 1.
    // When input A is sparse but B is not, both in0 and in1 need to loop over the "additional"
    // batch dims in A that are not in B. So we divide by batchB and set that.
    // In the default case (only input B is sparse), we set batchA to the batch dims of A.
    uint32_t batchA;
    if (operation_attributes.is_input_a_sparse && operation_attributes.is_input_b_sparse) {
        batchA = 1;
    } else if (operation_attributes.is_input_a_sparse) {
        batchA = get_batch_size(ashape) / batchB;
    } else {
        batchA = get_batch_size(ashape);
    }

    const auto Mt = get_M_dim(ashape, in0_tile, /*fuse_batch=*/false);
    const auto Kt = get_K_dim(ashape, in0_tile);
    const auto Nt = get_N_dim(bshape, in1_tile);

    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);

    // This should allocate a DRAM buffer on the device
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_available = num_cores_x * num_cores_y;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    uint32_t num_blocks_y = ((Mt - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((Nt - 1) / per_core_N) + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;

    TT_FATAL(
        num_blocks_total <= num_cores_available,
        "Number of blocks exceeds number of cores available: {} blocks > {} cores",
        num_blocks_total,
        num_cores_available);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    // Only support mcast_in0 for now
    TT_FATAL(mcast_in0, "Only mcast_in0 is supported for sparse matmul");

    using tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids;

    uint32_t num_blocks = Kt / in0_block_w;
    // Only enable packer l1 accumulation when there are spills, otherwise
    // unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    const auto interm0_data_format = packer_l1_acc_en
                                         ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                         : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);
    // interm0 entry size follows interm0_data_format, not the output dtype.
    const auto interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_dfb_tiles = in0_block_tiles;
    if (batchA * batchB * num_blocks > 1) {
        in0_dfb_tiles *= ttnn::operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in0_dfb_size = in0_dfb_tiles * in0_aligned_tile_size;

    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_dfb_tiles = in1_block_tiles;
    if (batchA * batchB * num_blocks > 1) {
        in1_dfb_tiles *= ttnn::operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }

    uint32_t in1_dfb_size = in1_dfb_tiles * in1_aligned_tile_size;

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_dfb_tiles = out_block_tiles;  // No double buffer

    uint32_t out_dfb_size = out_dfb_tiles * output_single_tile_size;
    uint32_t interm0_dfb_tiles = out_block_tiles;  // No double buffer
    uint32_t interm0_dfb_size = interm0_dfb_tiles * interm0_single_tile_size;

    CoreCoord start_core = {0, 0};

    // The matmul region is the rectangle of size `compute_with_storage_grid_size`
    // anchored at `start_core`. The sparse 1D matmul path does not yet anchor at a sub-device
    // start, but keeping the rectangle expression here keeps the API uniform with the dense 1D
    // path and is safe (matmul_core_rect == full compute grid when start_core == (0, 0)).
    CoreRangeSet matmul_core_rect(CoreRange(
        start_core,
        CoreCoord(
            start_core.x + compute_with_storage_grid_size.x - 1, start_core.y + compute_with_storage_grid_size.y - 1)));

    uint32_t num_cores_with_work = num_blocks_total;

    uint32_t in0_sender_num_cores = 1;
    uint32_t num_cores = num_cores_with_work;

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores, matmul_core_rect, row_major);

    CoreRangeSet in0_mcast_sender_cores =
        num_cores_to_corerangeset_in_subcoregrids(start_core, in0_sender_num_cores, matmul_core_rect, row_major);

    CoreRangeSet all_cores_with_work =
        num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores_with_work, matmul_core_rect, row_major);
    CoreRange in0_mcast_receiver_cores_bounding_box = all_cores_with_work.bounding_box();
    uint32_t in0_mcast_receiver_num_cores = in0_mcast_receiver_cores_bounding_box.size();  // always mcast to full grid

    // There should not be any cores without work in the receiver grid. If a grid is
    // not rectangular, then there will be some cores without work in the receiver grid.
    // For example, if there are 12 blocks of work, it should be put into a 3x4 grid.
    // If its laid out in row major with 8 cores in first row and 4 cores in second row,
    // then there will be 4 cores without work in the receiver grid, causing a hang.
    // We check for this below and error out.
    TT_FATAL(
        num_cores_with_work == in0_mcast_receiver_num_cores,
        "num_cores_with_work ({}) must be equal to in0_mcast_receiver_num_cores ({}), please adjust the core grid to "
        "make it rectangular.",
        num_cores_with_work,
        in0_mcast_receiver_num_cores);

    CoreRangeSet in0_mcast_receivers;

    if (in0_mcast_receiver_num_cores > 1) {
        // Check against the actual rectangle width (instead of bare grid_size.x-1) so that
        // sub-devices anchored away from (0, 0) would wrap correctly.
        auto receiver_start_core = compute_with_storage_grid_size.x > 1 ? CoreCoord{start_core.x + 1, start_core.y}
                                                                        : CoreCoord{start_core.x, start_core.y + 1};
        in0_mcast_receivers =
            num_cores_to_corerangeset_in_subcoregrids(receiver_start_core, num_cores - 1, matmul_core_rect, row_major);
    }

    CoreCoord top_left_core = in0_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in0_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    uint32_t num_batch_compute = use_indices ? num_active : nnz.value_or(sparsity.logical_volume());
    // Compact output packs only the `nnz` active batch pairs in scan order. Detect it exactly as the
    // device op (device/sparse/sparse_matmul_device_operation.cpp): [1, nnz, M, N]. Shape matching,
    // rather than volume matching, prevents a same-volume tensor with incompatible geometry from
    // selecting compact writer indexing.
    // (Orthogonal to indexed/gather mode, which is already compact by construction and rejects nnz:
    // the writer's skip path -- the only thing this flag guards -- is never reached there.)
    const bool compact_output =
        nnz.has_value() &&
        output_tensor.logical_shape() == ttnn::Shape{1U, nnz.value(), a.logical_shape()[-2], b.logical_shape()[-1]};

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    const auto& a_shape_logical = get_matmul_tensor_logical_shape(a, /*transpose=*/false);
    const auto in0_last_ktile_w = a_shape_logical[-1] % in0_tile.get_width();

    // We don't support transpose for this program configuration. However, we retain the logic here
    // to keep the code consistent with the other program configurations.
    const auto transpose_a = false;
    const auto transpose_b = false;
    const auto in0_tensor_stride_w = transpose_a ? Mt : 1;
    const auto in0_tensor_stride_h = transpose_a ? 1 : Kt;
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;

    const auto in1_tensor_stride_w = transpose_b ? Kt : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : Nt;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
    const auto in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;

    // The sparsity page sizes the readers stride DRAM at. These are host-side size queries on the
    // operand buffers, not address deliveries -- the addresses themselves travel as tensor bindings.
    const uint32_t sparsity_pagesize = static_cast<uint32_t>(sparsity.buffer()->aligned_page_size());
    const uint32_t in1_sparsity_pagesize = static_cast<uint32_t>(in1_sparsity_tensor.buffer()->aligned_page_size());

    ////////////////////////////////////////////////////////////////////////////
    //                      Spec-scope resource names
    ////////////////////////////////////////////////////////////////////////////
    // Declared function-local rather than at file scope: the matmul factory .cpp files share one
    // unity-build target, so file-scope constants with these names would collide with the dense
    // factories' equivalents.
    const KernelSpecName IN0_SENDER{"in0_sender"};
    const KernelSpecName IN0_RECEIVER{"in0_receiver"};
    const KernelSpecName IN1_SENDER_WRITER{"in1_sender_writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName SPARSITY_DFB{"sparsity"};
    const DFBSpecName IN1_SPARSITY_DFB{"in1_sparsity"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName INTERM0_DFB{"intermed0"};

    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName SPARSITY{"sparsity"};
    const TensorParamName INDICES{"indices"};
    const TensorParamName OUTPUT{"output"};

    const SemaphoreSpecName IN0_SENDER_SEM{"in0_mcast_sender"};
    const SemaphoreSpecName IN0_RECEIVER_SEM{"in0_mcast_receiver"};

    // In indexed/gather mode the in1 sender/writer's sparsity slot carries the *indices* tensor, a
    // different tensor from the mask the in0 sender reads, so it needs its own TensorParameter.
    // Outside that mode both readers bind the one sparsity parameter.
    const TensorParamName& IN1_SPARSITY_PARAM = use_indices ? INDICES : SPARSITY;

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    // The output and partials buffers share one L1 region when their formats match -- legacy
    // expressed that as a single buffer descriptor carrying both buffer indices. Every member of an alias
    // group must have the same total backing size, so the partials buffer is sized against whichever
    // region it actually sits in.
    const bool separate_out_and_interm0 = interm0_data_format != output_data_format;
    const uint32_t interm0_total_size = separate_out_and_interm0 ? interm0_dfb_size : out_dfb_size;
    // A two-member alias group has to be a strict clique, so each side names the other; when the
    // regions are separate neither aliases anything.
    Group<DFBSpecName> out_aliases;
    Group<DFBSpecName> interm0_aliases;
    if (!separate_out_and_interm0) {
        out_aliases.push_back(INTERM0_DFB);
        interm0_aliases.push_back(OUT_DFB);
    }

    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = IN0_DFB,
            .entry_size = in0_aligned_tile_size,
            .num_entries = in0_dfb_size / in0_aligned_tile_size,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
        },
        DataflowBufferSpec{
            .unique_id = IN1_DFB,
            .entry_size = in1_aligned_tile_size,
            .num_entries = in1_dfb_size / in1_aligned_tile_size,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = in1_tile,
        },
        // One page of the sparsity mask, staged by the in0 sender and read back by it through a raw
        // pointer. Legacy carried no tile descriptor here: the page is a run of bf16 validity flags,
        // not tiles.
        DataflowBufferSpec{
            .unique_id = SPARSITY_DFB,
            .entry_size = sparsity_pagesize,
            .num_entries = 1,
            .data_format_metadata = tt::tt_metal::datatype_to_dataformat_converter(sparsity_mesh_tensor.dtype()),
        },
        // The in1 sender/writer's own slot; in indexed/gather mode it holds the active-group id list
        // instead of a sparsity page, so it is sized and typed from whichever tensor that kernel reads.
        DataflowBufferSpec{
            .unique_id = IN1_SPARSITY_DFB,
            .entry_size = in1_sparsity_pagesize,
            .num_entries = 1,
            .data_format_metadata = tt::tt_metal::datatype_to_dataformat_converter(in1_sparsity_mesh_tensor.dtype()),
        },
        DataflowBufferSpec{
            .unique_id = OUT_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = out_dfb_size / output_single_tile_size,
            .data_format_metadata = output_data_format,
            .tile_format_metadata = output_tile,
            .advanced_options = {.alias_with = out_aliases},
        },
        DataflowBufferSpec{
            .unique_id = INTERM0_DFB,
            .entry_size = interm0_single_tile_size,
            .num_entries = interm0_total_size / interm0_single_tile_size,
            .data_format_metadata = interm0_data_format,
            .tile_format_metadata = output_tile,
            .advanced_options = {.alias_with = interm0_aliases},
        },
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Semaphores
    ////////////////////////////////////////////////////////////////////////////
    Group<SemaphoreSpec> semaphores = {
        SemaphoreSpec{
            .unique_id = IN0_SENDER_SEM,
            .target_nodes = all_cores,
        },
        SemaphoreSpec{
            .unique_id = IN0_RECEIVER_SEM,
            .target_nodes = all_cores,
        },
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Defines
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;

    mm_kernel_defines["FUSE_ACTIVATION"] = "0";
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(),
        num_cores,
        mm_kernel_defines,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config));

    if (in0_mcast_receiver_num_cores == 1) {
        mm_kernel_in0_sender_writer_defines["SKIP_MCAST"] = "1";
    }

    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";

    // Both readers take the sparsity operand, and each one's dataflow buffer and tensor accessor are
    // bound only because this define is set: a binding the host does not declare produces no dfb::/
    // tensor:: token at all, so the kernel's references must not reach C++ name lookup. The sparsity
    // operand is mandatory for this op and batchB >= 1 always, so the condition is constant here --
    // but the define is still required, because the shared kernels default to sparsity-off.
    mm_kernel_in0_sender_writer_defines["SPARSITY"] = "1";
    mm_kernel_in1_sender_writer_defines["SPARSITY"] = "1";

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    // Resolved verbatim from the legacy DataMovementConfigDescriptors. Note this op puts the in0
    // sender on RISCV_0 and the in1 sender/writer on RISCV_1 -- the opposite of the dense 1D
    // factory. The two triples happen to coincide with the writer and reader defaults respectively,
    // but the arch-parameterised noc expressions are kept so the values cannot drift from the legacy
    // ones if a future arch changes the preferred NOCs.
    const auto in0_hw_config =
        DataMovementGen1Config{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in0_noc};
    const auto in1_hw_config =
        DataMovementGen1Config{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in1_noc};

    const bool has_in0_receiver_kernel = in0_mcast_receivers.num_cores() > 0;

    const Group<SemaphoreBinding> in0_mcast_sem_bindings = {
        SemaphoreBinding{.semaphore_spec_name = IN0_SENDER_SEM, .accessor_name = "in0_mcast_sender"},
        SemaphoreBinding{.semaphore_spec_name = IN0_RECEIVER_SEM, .accessor_name = "in0_mcast_receiver"},
    };

    Group<KernelSpec> kernels;

    // ---- in0 sender ------------------------------------------------------
    kernels.push_back(KernelSpec{
        .unique_id = IN0_SENDER,
        .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                  "reader_bmm_tile_layout_in0_sender_padding_metal2.cpp",
        .compiler_options =
            {
                .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_in0_sender_writer_defines),
            },
        .dfb_bindings =
            {
                // in0 is filled here (the payload arrives by NoC) and drained by the compute kernel.
                DFBBinding{
                    .dfb_spec_name = IN0_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // The sparsity page is staged and consumed by this one kernel, so it carries both
                // endpoints itself.
                DFBBinding{
                    .dfb_spec_name = SPARSITY_DFB,
                    .accessor_name = "sparsity",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SPARSITY_DFB,
                    .accessor_name = "sparsity",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .semaphore_bindings = in0_mcast_sem_bindings,
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = IN0, .accessor_name = "in0"},
                TensorBinding{.tensor_parameter_name = SPARSITY, .accessor_name = "sparsity"},
            },
        .compile_time_args =
            {
                // in0 tensor args
                {"in0_tensor_stride_w", static_cast<uint32_t>(in0_tensor_stride_w)},
                {"in0_tensor_stride_h", static_cast<uint32_t>(in0_tensor_stride_h)},
                {"in0_tensor_next_inner_dim_block_stride", static_cast<uint32_t>(in0_tensor_next_block_stride)},
                {"in0_tensor_next_h_dim_block_stride", static_cast<uint32_t>(in0_tensor_next_h_dim_block_stride)},
                // in0 block args
                {"in0_block_w", in0_block_w},
                {"in0_block_h", in0_block_h},
                {"in0_block_num_tiles", in0_block_num_tiles},
                {"in0_last_ktile_w", static_cast<uint32_t>(in0_last_ktile_w)},
                // transpose is not supported for sparse, so the h-direction last tile never pads
                {"in0_last_ktile_h", 0u},
                // shard args (read only on the sharded path, which this factory never takes)
                {"shard_width_in_tiles", 0u},
                {"shard_height_in_tiles", 0u},
                // in0/in1 common args
                {"num_blocks_inner_dim", num_blocks},
                {"num_blocks_w_dim", out_num_blocks_x},
                {"num_blocks_h_dim", out_num_blocks_y},
                // in0 mcast args
                {"in0_mcast_num_dests", num_cores - 1},
                {"in0_mcast_num_cores", in0_mcast_receiver_num_cores - 1},
                // batch args
                {"MtKt", Mt * Kt},
                {"in0_B", batchA},
                {"in1_B", batchA},
                {"in0_reuse_in_dfb", 0u},
                // sparsity args
                {"batchB", batchB},
                {"sparsity_pagesize", sparsity_pagesize},
                {"bcast_A", static_cast<uint32_t>(!is_input_a_sparse)},
                {"get_batch_from_reader", static_cast<uint32_t>(get_batch_from_reader)},
                // num_batch_compute (== nnz when supplied). The sender uses this to validate,
                // on-device, that count_nonzero(sparsity) matches the loop count baked into the
                // receiver/compute kernels, failing loudly instead of deadlocking.
                // See https://github.com/tenstorrent/tt-metal/issues/45943.
                {"num_batch_compute", num_batch_compute},
                // indexed/gather mode loop count (0 = not indexed)
                {"num_active", num_active},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"in0_tensor_start_tile_id",
                     "in0_mcast_dest_noc_start_x",
                     "in0_mcast_dest_noc_start_y",
                     "in0_mcast_dest_noc_end_x",
                     "in0_mcast_dest_noc_end_y",
                     "last_block_h"},
            },
        .hw_config = in0_hw_config,
    });

    // ---- in0 receiver ----------------------------------------------------
    if (has_in0_receiver_kernel) {
        kernels.push_back(KernelSpec{
            .unique_id = IN0_RECEIVER,
            .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                      "reader_bmm_tile_layout_in0_receiver_metal2.cpp",
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN0_DFB,
                        .accessor_name = "in0",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            .semaphore_bindings = in0_mcast_sem_bindings,
            .compile_time_args =
                {
                    {"in0_block_num_tiles", in0_block_num_tiles},
                    {"num_blocks_inner_dim", num_blocks},
                    {"num_blocks_w_dim", out_num_blocks_x},
                    {"num_blocks_h_dim", out_num_blocks_y},
                    {"batch", num_batch_compute},
                    {"get_batch_from_reader", static_cast<uint32_t>(get_batch_from_reader)},
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"in0_mcast_sender_noc_x", "in0_mcast_sender_noc_y"},
                },
            .hw_config = in0_hw_config,
        });
    }

    // ---- in1 sender / output writer --------------------------------------
    // The in1 multicast is skipped entirely (SKIP_MCAST), so the kernel's semaphore objects are
    // constructed but never used; they are still bound because the kernel constructs them
    // unconditionally. Legacy passed literal ids 0/0 into the same slots.
    kernels.push_back(KernelSpec{
        .unique_id = IN1_SENDER_WRITER,
        .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                  "reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp",
        .compiler_options =
            {
                .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_in1_sender_writer_defines),
            },
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN1_DFB,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // This kernel's own sparsity slot -- the active-group id list in indexed/gather mode,
                // the sparsity mask otherwise. Staged and read back here alone, so both endpoints.
                DFBBinding{
                    .dfb_spec_name = IN1_SPARSITY_DFB,
                    .accessor_name = "sparsity",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = IN1_SPARSITY_DFB,
                    .accessor_name = "sparsity",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .semaphore_bindings =
            {
                SemaphoreBinding{.semaphore_spec_name = IN0_SENDER_SEM, .accessor_name = "in1_mcast_sender"},
                SemaphoreBinding{.semaphore_spec_name = IN0_RECEIVER_SEM, .accessor_name = "in1_mcast_receiver"},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = IN1, .accessor_name = "in1"},
                TensorBinding{.tensor_parameter_name = IN1_SPARSITY_PARAM, .accessor_name = "sparsity"},
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"},
            },
        .compile_time_args =
            {
                // READER
                // in1 tensor args
                {"in1_tensor_stride_w", static_cast<uint32_t>(in1_tensor_stride_w)},
                {"in1_tensor_stride_h", static_cast<uint32_t>(in1_tensor_stride_h)},
                {"in1_tensor_next_block_stride", static_cast<uint32_t>(in1_tensor_next_block_stride)},
                {"in1_tensor_next_w_dim_block_stride", static_cast<uint32_t>(in1_tensor_next_w_dim_block_stride)},
                // in1 block args
                {"in1_block_w", in1_block_w},
                {"in1_block_h", in0_block_w},
                {"in1_block_num_tiles", in1_block_w * in0_block_w},
                // in0/in1 common args
                {"num_blocks_inner_dim", num_blocks},
                {"num_blocks_w_dim", out_num_blocks_x},
                {"num_blocks_h_dim", out_num_blocks_y},
                // in1 mcast args
                {"in1_mcast_num_dests", 0u},
                {"in1_mcast_num_cores", 0u},
                // batch args
                {"KtNt", Kt * Nt},
                {"batch", batchA},
                {"bcast_B", 1u},
                // sparsity args (in indexed/gather mode this slot carries the active-group id list)
                {"batchB", batchB},
                {"sparsity_pagesize", in1_sparsity_pagesize},
                // WRITER
                // out tensor args
                {"out_tensor_stride_w", 1u},
                {"out_tensor_stride_h", Nt},
                {"out_tensor_next_subblock_stride_w", out_subblock_w},
                {"out_tensor_next_subblock_stride_h", out_subblock_h * Nt},
                {"out_tensor_next_w_dim_block_stride", out_block_w},
                {"out_tensor_next_h_dim_block_stride", out_block_h * Nt},
                // out subblock args
                {"out_subblock_w", out_subblock_w},
                {"out_subblock_h", out_subblock_h},
                {"out_subblock_tile_count", out_subblock_w * out_subblock_h},
                // batch args
                {"MtNt", Mt * Nt},
                {"compact_output", static_cast<uint32_t>(compact_output)},
                // indexed/gather mode loop count (0 = not indexed)
                {"num_active", num_active},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"in1_tensor_start_tile_id",
                     "in1_mcast_dest_noc_start_x",
                     "in1_mcast_dest_noc_start_y",
                     "in1_mcast_dest_noc_end_x",
                     "in1_mcast_dest_noc_end_y",
                     "out_tensor_start_tile_id",
                     // padding args (READER)
                     "last_block_w",
                     // padding args (WRITER)
                     "out_num_nonzero_subblocks_h",
                     "out_last_subblock_h",
                     "padded_block_tiles_h_skip",
                     "out_num_nonzero_subblocks_w",
                     "out_last_num_nonzero_subblocks_w",
                     "out_last_subblock_w",
                     "padded_subblock_tiles_addr_skip",
                     "padded_block_tiles_w_skip",
                     "last_num_blocks_w_dim"},
            },
        .hw_config = in1_hw_config,
    });

    // ---- compute ----------------------------------------------------------
    {
        uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

        uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
        uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
        uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

        uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

        // The op resolves a TTNN ComputeKernelConfig and passes every field it resolves on to the
        // kernel, so translating the resolved config is faithful with nothing to pin back.
        auto compute_hw =
            ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config.value());

        // When accumulating in fp32 with the K reduction split across blocks, the partials buffer
        // holds Float32 and is reloaded into DEST between blocks. Unless that reload's view is marked
        // UnpackToDest, it is routed through SrcA and rounded to TF32 (10 mantissa bits), so the fp32
        // partial loses precision on every block boundary. Legacy expressed this as
        // unpack_to_dest_mode = UnpackToDestFp32 on the partials buffer, Default everywhere else.
        //
        // Metal 2.0 also *requires* an explicit entry for every Float32 buffer a compute kernel
        // consumes with enable_32_bit_dest on -- omitting one is a hard error at program build, where
        // legacy defaulted silently. So reproduce legacy's choice on the partials buffer and spell
        // out UnpackToSrc (== legacy Default) for the other Float32 buffers this kernel consumes.
        // The output buffer is produced, never consumed, so it takes no entry.
        if (fp32_dest_acc_en) {
            const bool mark_interm0 = interm0_data_format == tt::DataFormat::Float32;
            auto add_if_float32 = [&](const DFBSpecName& name, tt::DataFormat fmt) {
                if (fmt != tt::DataFormat::Float32) {
                    return;
                }
                unpack_modes(compute_hw)
                    .emplace(
                        name,
                        (mark_interm0 && name == INTERM0_DFB) ? tt::tt_metal::UnpackMode::UnpackToDest
                                                              : tt::tt_metal::UnpackMode::UnpackToSrc);
            };
            add_if_float32(IN0_DFB, in0_data_format);
            add_if_float32(IN1_DFB, in1_data_format);
            add_if_float32(INTERM0_DFB, interm0_data_format);
        }

        kernels.push_back(KernelSpec{
            .unique_id = COMPUTE,
            .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
                      "bmm_large_block_zm_fused_bias_activation_metal2.cpp",
            .compiler_options =
                {
                    .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_defines),
                    // A legacy ComputeConfigDescriptor with no opt_level resolves to O3; Metal 2.0's
                    // CompilerOptions defaults to O2, so it has to be stated.
                    .opt_level = KernelBuildOptLevel::O3,
                },
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN0_DFB,
                        .accessor_name = "in0",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = IN1_DFB,
                        .accessor_name = "in1",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    // The partials buffer is filled and re-read by this kernel alone.
                    DFBBinding{
                        .dfb_spec_name = INTERM0_DFB,
                        .accessor_name = "intermed0",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = INTERM0_DFB,
                        .accessor_name = "intermed0",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .compile_time_args =
                {
                    {"in0_block_w", in0_block_w},
                    {"in0_num_subblocks", in0_num_subblocks},
                    {"in0_block_num_tiles", in0_block_num_tiles},
                    {"in0_subblock_num_tiles", in0_subblock_num_tiles},
                    {"in1_num_subblocks", in1_num_subblocks},
                    {"in1_block_num_tiles", in1_block_num_tiles},
                    {"in1_block_w", in1_per_core_w},
                    {"num_blocks_inner_dim", num_blocks},
                    {"num_blocks_w_dim", out_num_blocks_x},
                    {"num_blocks_h_dim", out_num_blocks_y},
                    {"out_subblock_h", out_subblock_h},
                    {"out_subblock_w", out_subblock_w},
                    {"out_subblock_num_tiles", out_subblock_num_tiles},
                    {"batch", num_batch_compute},
                    {"out_block_num_tiles", out_block_tiles},
                    {"untilize_out", 0u},
                    {"get_batch_from_reader", static_cast<uint32_t>(get_batch_from_reader)},
                },
            .hw_config = compute_hw,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Work units
    ////////////////////////////////////////////////////////////////////////////
    // The in0 sender and the in0 receivers cover disjoint node sets whose union is
    // all_cores_with_work, so listing the in1 sender/writer and compute in both work units
    // reproduces their legacy placement over all_cores_with_work exactly.
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{
            .name = "sparse_mm_in0_sender",
            .kernels = {IN0_SENDER, IN1_SENDER_WRITER, COMPUTE},
            .target_nodes = in0_mcast_sender_cores,
        },
    };
    if (has_in0_receiver_kernel) {
        work_units.push_back(WorkUnitSpec{
            .name = "sparse_mm_in0_receivers",
            .kernels = {IN0_RECEIVER, IN1_SENDER_WRITER, COMPUTE},
            .target_nodes = in0_mcast_receivers,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = IN0, .spec = in0_mesh_tensor.tensor_spec()},
        TensorParameter{.unique_id = IN1, .spec = in1_mesh_tensor.tensor_spec()},
        TensorParameter{.unique_id = SPARSITY, .spec = sparsity_mesh_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = out_mesh_tensor.tensor_spec()},
    };
    if (use_indices) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = INDICES, .spec = in1_sparsity_mesh_tensor.tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime args (per-core loop)
    ////////////////////////////////////////////////////////////////////////////
    // Parameters for last row, col, or block, no need to re-calc h-dim since there's no split on height
    uint32_t last_per_core_N = Nt % per_core_N == 0 ? per_core_N : Nt % per_core_N;
    uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    uint32_t last_out_num_blocks_w = ((last_per_core_N - 1) / out_block_w) + 1;
    uint32_t last_block_num_nonzero_subblocks_w = ((last_out_block_w - 1) / out_subblock_w) + 1;
    uint32_t last_subblock_of_last_block_w =
        last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    KernelRunArgs in0_sender_run_args{.kernel = IN0_SENDER};
    KernelRunArgs in0_receiver_run_args{.kernel = IN0_RECEIVER};
    KernelRunArgs in1_sender_run_args{.kernel = IN1_SENDER_WRITER};

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i % num_blocks_x;
        uint32_t output_idx_y = i / num_blocks_x;

        // in0 sender and in1 sender
        if (core == start_core) {
            AddRuntimeArgsForNode(
                in0_sender_run_args.runtime_arg_values,
                core,
                {// in0 tensor args
                 {"in0_tensor_start_tile_id", Kt * per_core_M * output_idx_y},
                 // in0 mcast args
                 {"in0_mcast_dest_noc_start_x", static_cast<uint32_t>(start_core_noc.x)},
                 {"in0_mcast_dest_noc_start_y", static_cast<uint32_t>(start_core_noc.y)},
                 {"in0_mcast_dest_noc_end_x", static_cast<uint32_t>(end_core_noc.x)},
                 {"in0_mcast_dest_noc_end_y", static_cast<uint32_t>(end_core_noc.y)},
                 // padding args
                 {"last_block_h", out_block_h}});
        }
        // in0 receiver and in 1 sender
        else {
            AddRuntimeArgsForNode(
                in0_receiver_run_args.runtime_arg_values,
                core,
                {// in0 mcast args
                 {"in0_mcast_sender_noc_x", static_cast<uint32_t>(top_left_core_physical.x)},
                 {"in0_mcast_sender_noc_y", static_cast<uint32_t>(top_left_core_physical.y)}});
        }
        if (i < num_cores_with_work) {
            const bool last_x = (output_idx_x == num_blocks_x - 1);
            AddRuntimeArgsForNode(
                in1_sender_run_args.runtime_arg_values,
                core,
                {// READER
                 // in1 tensor args
                 {"in1_tensor_start_tile_id", per_core_N * output_idx_x},
                 // in1 mcast args
                 {"in1_mcast_dest_noc_start_x", 0u},
                 {"in1_mcast_dest_noc_start_y", 0u},
                 {"in1_mcast_dest_noc_end_x", 0u},
                 {"in1_mcast_dest_noc_end_y", 0u},
                 // WRITER
                 // out tensor args
                 {"out_tensor_start_tile_id", (output_idx_x * per_core_N) + (output_idx_y * per_core_M * Nt)},
                 // padding args (READER)
                 {"last_block_w", last_x ? last_out_block_w : out_block_w},
                 // padding args (WRITER)
                 {"out_num_nonzero_subblocks_h", out_block_h / out_subblock_h},
                 {"out_last_subblock_h", out_subblock_h},
                 {"padded_block_tiles_h_skip", 0u},
                 {"out_num_nonzero_subblocks_w", out_block_w / out_subblock_w},
                 {"out_last_num_nonzero_subblocks_w",
                  last_x ? last_block_num_nonzero_subblocks_w : out_block_w / out_subblock_w},
                 {"out_last_subblock_w", last_x ? last_subblock_of_last_block_w : out_subblock_w},
                 {"padded_subblock_tiles_addr_skip", last_x ? last_block_padded_subblock_tiles_addr_skip : 0u},
                 {"padded_block_tiles_w_skip", last_x ? last_block_padded_block_tiles_w_skip : 0u},
                 {"last_num_blocks_w_dim", last_x ? last_out_num_blocks_w : out_num_blocks_x}});
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble
    ////////////////////////////////////////////////////////////////////////////
    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(in0_sender_run_args));
    if (has_in0_receiver_kernel) {
        run_args.kernel_run_args.push_back(std::move(in0_receiver_run_args));
    }
    run_args.kernel_run_args.push_back(std::move(in1_sender_run_args));

    run_args.tensor_args = {
        {IN0, in0_mesh_tensor},
        {IN1, in1_mesh_tensor},
        {SPARSITY, sparsity_mesh_tensor},
        {OUTPUT, out_mesh_tensor},
    };
    if (use_indices) {
        run_args.tensor_args.emplace(INDICES, in1_sparsity_mesh_tensor);
    }

    ProgramSpec spec{
        .name = "sparse_matmul_multi_core_reuse_mcast_1d",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
