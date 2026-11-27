// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include "tt-metalium/utils/work_split.hpp"
#include "grid_sample_utils.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"

#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/discovery/hal.hpp>
#include <tt-metalium/utils/math.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/quasar/grid_sample/device/grid_sample_device_operation.hpp"

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;

namespace {

// The legacy factory gave its input / scalar / output buffers an unpack face geometry
// ({face_r_dim, num_faces}). The kernel build lowered that geometry to the equivalent tile shape and
// took every unpack/pack tile parameter (face rows, face count, tile dims, tile size) from it, so
// carrying that tile as the DFB's tile metadata reproduces the same kernel metadata. Same mapping as
// the build's own conversion: tall faces stack two rows of faces, a single face is FACE_WIDTH wide.
Tile tile_for_face_geometry(uint32_t face_r_dim, uint32_t num_faces) {
    const uint32_t tile_height =
        face_r_dim * (num_faces > 2 ? tt::constants::TILE_HEIGHT / tt::constants::FACE_HEIGHT : 1);
    const uint32_t tile_width = num_faces == 1 ? tt::constants::FACE_WIDTH : tt::constants::TILE_WIDTH;
    return Tile({tile_height, tile_width});
}

}  // namespace

ttnn::device_operation::ProgramArtifacts GridSampleBilinearProgramFactory::create_program_artifacts(
    const GridSampleParams& operation_attributes, const GridSampleInputs& tensor_args, Tensor& output_tensor) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& grid_tensor = tensor_args.grid;
    bool use_precomputed_grid = operation_attributes.use_precomputed_grid;
    bool batch_output_channels = operation_attributes.batch_output_channels;

    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& grid_mesh_tensor = grid_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();
    const bool is_sharded = grid_tensor.is_sharded();

    // Data formats and device
    const auto [input_dfb_data_format, grid_dfb_data_format, output_dfb_data_format] = std::make_tuple(
        tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(grid_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype()));
    tt::tt_metal::distributed::MeshDevice* const device = output_tensor.device();

    // Shape and dimensions
    const auto& [input_shape, grid_shape, output_shape] =
        std::tie(input_tensor.padded_shape(), grid_tensor.padded_shape(), output_tensor.padded_shape());
    const uint32_t input_batch = input_shape[0], input_height = input_shape[1], input_width = input_shape[2];
    const uint32_t grid_height = grid_shape[1], grid_width = grid_shape[2];
    const uint32_t grid_hw = grid_height * grid_width;
    const uint32_t grid_batching_factor = get_grid_batching_factor(grid_tensor, use_precomputed_grid);
    const bool enable_split_reader =
        should_use_split_reader(input_tensor, grid_tensor, use_precomputed_grid, "bilinear");

    tt::tt_metal::CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores, grid_nsticks_per_core, output_nsticks_per_core = 0;
    uint32_t num_sticks_per_core_group_1 = 0, num_sticks_per_core_group_2 = 0;
    std::vector<CoreCoord> logical_cores;
    // Calculate total work and determine actual cores needed
    const uint32_t total_grid_nsticks = grid_tensor.physical_volume() / grid_shape[-1];

    if (is_sharded) {
        const auto grid_shard_spec = grid_tensor.shard_spec().value();
        grid_nsticks_per_core = grid_shard_spec.shape[0];
        output_nsticks_per_core = output_tensor.shard_spec().value().shape[0];

        num_cores = grid_shard_spec.num_cores();
        all_cores = grid_shard_spec.grid;
        logical_cores = corerange_to_cores(
            all_cores, num_cores, grid_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    } else {
        const auto compute_grid_size = device->compute_with_storage_grid_size();
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, total_grid_nsticks);

        std::tie(num_cores, all_cores, core_group_1, core_group_2) =
            std::make_tuple(num_cores_used, all_cores_range, core_group_1_range, core_group_2_range);
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
        grid_nsticks_per_core = num_sticks_1;

        logical_cores = corerange_to_cores(all_cores, num_cores, true);
    }

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName GRID{"grid"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName GRID_DFB{"grid"};
    const DFBSpecName IN_0{"in_0"};          // reader 0 -> compute: the 4 corner sticks of a grid point
    const DFBSpecName IN_1{"in_1"};          // reader 1 -> compute (sharded split reader only)
    const DFBSpecName SCALAR_0{"scalar_0"};  // reader 0 -> compute: the bilinear weights of a grid point
    const DFBSpecName SCALAR_1{"scalar_1"};  // reader 1 -> compute (sharded split reader only)
    const DFBSpecName OUT_DFB{"out"};
    const KernelSpecName READER_0{"reader_0"};
    const KernelSpecName READER_1{"reader_1"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};                  // sharded
    const KernelSpecName COMPUTE_GROUP_1{"compute_group_1"};  // interleaved, core group 1
    const KernelSpecName COMPUTE_GROUP_2{"compute_group_2"};  // interleaved, core group 2

    const bool has_second_reader = is_sharded && enable_split_reader;

    // Create DFBs
    const uint32_t input_face_r_dim = REDUCTION_SIZE, input_num_faces = 2;
    const uint32_t scalar_face_r_dim = 1, scalar_num_faces = 2;
    const bool last_output_tile_is_partial = input_shape[-1] % tt::constants::TILE_WIDTH != 0;
    const bool single_partial_output_fits_in_face =
        last_output_tile_is_partial && input_shape[-1] <= tt::constants::FACE_WIDTH;
    const uint32_t output_face_r_dim = 1, output_num_faces = single_partial_output_fits_in_face ? 1U : 2U;

    // Sharded grid rows are stored at the buffer's aligned page size, so stride by the aligned stick size too.
    const uint32_t grid_stick_size = get_aligned_stick_size(grid_shape, grid_tensor);
    const uint32_t grid_dfb_num_pages = is_sharded ? grid_nsticks_per_core : 1;
    std::vector<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = GRID_DFB,
        .entry_size = grid_stick_size,
        .num_entries = grid_dfb_num_pages,
        .data_format_metadata = grid_dfb_data_format,
        .borrowed_from = is_sharded ? std::optional<TensorParamName>{GRID} : std::nullopt,
    });

    // Resolve compute kernel config early: under fp32_dest_acc_en + half-sync, DEST holds only 4
    // fp32 tiles, so each chunk must shrink to <= 4 tiles. When the user explicitly enables
    // dst_full_sync_en, full-sync DEST holds 8 fp32 tiles and the 4-tile clamp would be a
    // gratuitous slowdown — so we keep the full 8-tile chunk in that case. The compute kernel
    // reads the same flag (force_max_tiles_per_reduction_4) and recomputes its own MAX_TILES_PER_REDUCTION
    // accordingly.
    const auto [resolved_math_fidelity, resolved_math_approx, resolved_fp32_acc, resolved_l1_acc, user_dst_full_sync] =
        ttnn::get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
    (void)resolved_l1_acc;  // not consumed in this factory
    const bool force_4_tile_chunk = resolved_fp32_acc && !user_dst_full_sync;
    const uint32_t effective_max_tiles_per_reduction = force_4_tile_chunk ? 4U : MAX_TILES_PER_REDUCTION;

    const uint32_t in_ntiles_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(input_shape[-1]) / tt::constants::TILE_WIDTH));
    // When in_ntiles_c exceeds effective_max_tiles_per_reduction the channel dimension is processed
    // in chunks; the input DFB only needs to hold one chunk at a time, keeping the per-core L1
    // footprint bounded regardless of how wide the input stick is.
    const uint32_t max_tiles_per_iter = std::min<uint32_t>(in_ntiles_c, effective_max_tiles_per_reduction);
    const uint32_t in_nblocks_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(in_ntiles_c) / effective_max_tiles_per_reduction));
    const uint32_t input_chunk_nbytes =
        effective_max_tiles_per_reduction * tt::constants::TILE_WIDTH * input_tensor.element_size();
    const uint32_t input_dfb_page_size = max_tiles_per_iter * tt::constants::TILE_HW * input_tensor.element_size();
    // When the last channel chunk holds fewer than effective_max_tiles_per_reduction tiles, the
    // reader must pack it with a tight write stride (== chunk_bytes) so the unpacker's
    // tiles_to_reduce matches compute_pool_2d's `tilize_reconfig` path. We require
    // !last_tile_is_partial — currently guaranteed by the padded-width % TILE_WIDTH == 0 check in
    // grid_sample_device_operation.cpp — and window_size_hw (REDUCTION_SIZE=4) <= FACE_HEIGHT,
    // which is a static property of bilinear.
    const bool last_tile_is_partial = (input_shape[-1] % tt::constants::TILE_WIDTH) != 0;
    const bool last_chunk_partial =
        (in_nblocks_c > 1) && (in_ntiles_c % effective_max_tiles_per_reduction != 0) && !last_tile_is_partial;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = IN_0,
        .entry_size = input_dfb_page_size,
        .num_entries = BUFFERING_FACTOR,
        .data_format_metadata = input_dfb_data_format,
        .tile_format_metadata = tile_for_face_geometry(input_face_r_dim, input_num_faces),
    });

    if (has_second_reader) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = IN_1,
            .entry_size = input_dfb_page_size,
            .num_entries = BUFFERING_FACTOR,
            .data_format_metadata = input_dfb_data_format,
            .tile_format_metadata = tile_for_face_geometry(input_face_r_dim, input_num_faces),
        });
    }

    const uint32_t scalar_dfb_page_size = tt::tile_size(input_dfb_data_format);
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = SCALAR_0,
        .entry_size = scalar_dfb_page_size,
        .num_entries = BUFFERING_FACTOR,
        .data_format_metadata = input_dfb_data_format,
        .tile_format_metadata = tile_for_face_geometry(scalar_face_r_dim, scalar_num_faces),
    });

    if (has_second_reader) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = SCALAR_1,
            .entry_size = scalar_dfb_page_size,
            .num_entries = BUFFERING_FACTOR,
            .data_format_metadata = input_dfb_data_format,
            .tile_format_metadata = tile_for_face_geometry(scalar_face_r_dim, scalar_num_faces),
        });
    }

    const uint32_t out_ntiles_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(output_shape[-1]) / tt::constants::FACE_WIDTH));
    const uint32_t output_dfb_page_size = tt::constants::FACE_WIDTH * output_tensor.element_size();
    const uint32_t output_dfb_pages =
        is_sharded ? output_nsticks_per_core * out_ntiles_c : out_ntiles_c * BUFFERING_FACTOR;
    // The legacy output buffer also carried an explicit {1, FACE_WIDTH} tile when a single partial output
    // fits in one face; that is the same tile its face geometry ({1, 1}) lowers to.
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = output_dfb_page_size,
        .num_entries = output_dfb_pages,
        .data_format_metadata = output_dfb_data_format,
        .tile_format_metadata = tile_for_face_geometry(output_face_r_dim, output_num_faces),
        .borrowed_from = is_sharded ? std::optional<TensorParamName>{OUTPUT} : std::nullopt,
    });

    // Prepare stick size arguments with proper names
    const uint32_t input_stick_size = get_aligned_stick_size(input_shape, input_tensor);

    // Reader compile-time arguments. The interleaved and sharded readers are different sources and
    // name a few of the shared values differently, so each gets its own table.
    KernelSpec::CompileTimeArgs reader_compile_time_args = {
        {"input_stick_nbytes", input_stick_size},
        {"grid_stick_nbytes", grid_stick_size},
        {"input_height", input_height},
        {"input_width", input_width},
        {"grid_dtype", static_cast<uint32_t>(grid_tensor.dtype())},
        {"use_precomputed_grid", use_precomputed_grid ? 1U : 0U},
        {"align_corners", operation_attributes.align_corners ? 1U : 0U},
        {"in_nblocks_c", in_nblocks_c},
        {"input_chunk_nbytes", input_chunk_nbytes},
        {"last_chunk_partial", last_chunk_partial ? 1U : 0U},
    };
    if (is_sharded) {
        reader_compile_time_args.emplace("input_batch", input_batch);
        reader_compile_time_args.emplace("grid_batching_factor", grid_batching_factor);
        reader_compile_time_args.emplace("grid_hw", grid_hw);
        reader_compile_time_args.emplace("split_reader", enable_split_reader ? 1U : 0U);
        reader_compile_time_args.emplace("reader_id", 0U);  // set per reader below
        reader_compile_time_args.emplace("grid_nsticks_per_core", grid_nsticks_per_core);
    } else {
        reader_compile_time_args.emplace("grid_batches", grid_batching_factor);
        reader_compile_time_args.emplace("output_hw_size", grid_hw);
    }

    const std::string reader_kernel_path =
        is_sharded ? "ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample/device/kernels/dataflow/"
                     "reader_grid_sample_sharded.cpp"
                   : "ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample/device/kernels/dataflow/"
                     "reader_grid_sample_interleaved_start_id.cpp";

    std::vector<KernelSpec> kernels;

    // Reader 0. The grid DFB is reader 0's alone (a self-loop) unless a second reader also reads the
    // sharded grid in place, in which case the two readers split it as producer / consumer.
    {
        Group<DFBBinding> reader0_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = GRID_DFB,
                .accessor_name = "grid",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = IN_0,
                .accessor_name = "in",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = SCALAR_0,
                .accessor_name = "scalar",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
        };
        if (!has_second_reader) {
            reader0_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = GRID_DFB,
                .accessor_name = "grid",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        Group<TensorBinding> reader0_tensor_bindings = {
            TensorBinding{
                .tensor_parameter_name = INPUT,
                .accessor_name = "input",
            },
        };
        if (!is_sharded) {
            reader0_tensor_bindings.push_back(TensorBinding{
                .tensor_parameter_name = GRID,
                .accessor_name = "grid",
            });
        }
        auto reader0_compile_time_args = reader_compile_time_args;
        if (is_sharded) {
            reader0_compile_time_args["reader_id"] = 0U;
        }
        kernels.push_back(KernelSpec{
            .unique_id = READER_0,
            .source = reader_kernel_path,
            .dfb_bindings = std::move(reader0_dfb_bindings),
            .tensor_bindings = std::move(reader0_tensor_bindings),
            .compile_time_args = std::move(reader0_compile_time_args),
            .runtime_arg_schema =
                {
                    .runtime_arg_names = is_sharded ? Group<std::string>{"global_grid_stick_start"}
                                                    : Group<std::string>{"num_pages", "start_page_id"},
                },
            // Sharded: BRISC on NOC 0 (neither the reader nor the writer default). Interleaved: the
            // legacy ReaderConfigDescriptor, i.e. the reader default.
            .hw_config = is_sharded ? DataMovementHardwareConfig{.config_1xx =
                                                                     DataMovementHardwareConfig::DataMovement1XXConfig{
                                                                         .processor = DataMovementProcessor::RISCV_0,
                                                                         .noc = NOC::RISCV_0_default,
                                                                     }}
                                    : ttnn::create_reader_datamovement_config(),
        });
    }

    if (has_second_reader) {
        auto reader1_compile_time_args = reader_compile_time_args;
        reader1_compile_time_args["reader_id"] = 1U;
        kernels.push_back(KernelSpec{
            .unique_id = READER_1,
            .source = reader_kernel_path,
            // Reader 1 runs the same source against the second input / scalar DFBs, under the same
            // kernel-side names reader 0 uses for the first pair.
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = GRID_DFB,
                        .accessor_name = "grid",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = IN_1,
                        .accessor_name = "in",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = SCALAR_1,
                        .accessor_name = "scalar",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            .tensor_bindings =
                {
                    TensorBinding{
                        .tensor_parameter_name = INPUT,
                        .accessor_name = "input",
                    },
                },
            .compile_time_args = std::move(reader1_compile_time_args),
            .runtime_arg_schema = {.runtime_arg_names = {"global_grid_stick_start"}},
            // NCRISC on NOC 1 (neither the reader nor the writer default).
            .hw_config =
                DataMovementHardwareConfig{
                    .config_1xx =
                        DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = DataMovementProcessor::RISCV_1,
                            .noc = NOC::RISCV_1_default,
                        },
                },
        });
    }

    const bool is_output_tiled = false;
    const bool is_output_block_format = false;
    // No pre-tilize staging DFB: grid sample never produces tiled output.

    // Compute kernels
    const uint32_t channels_per_shard = input_shape[-1];

    auto pool_defines_map = ttnn::operations::pool::get_defines(ttnn::operations::pool::Pool2DType::AVG_POOL2D);

    // Compute settings carried over from the legacy ComputeConfigDescriptor, which set exactly these four
    // from the resolved compute kernel config (packer_l1_acc has no Metal 2.0 counterpart and was unused).
    // Metal 2.0 requires an explicit unpack mode for every Float32 DFB a compute kernel consumes under a
    // 32-bit Dest; the legacy op left them at the default, which is UnpackToSrc.
    ComputeHardwareConfig::ComputeUnpackModes unpack_modes;
    if (resolved_fp32_acc) {
        if (input_dfb_data_format == tt::DataFormat::Float32) {
            unpack_modes.emplace(IN_0, UnpackMode::UnpackToSrc);
            unpack_modes.emplace(SCALAR_0, UnpackMode::UnpackToSrc);
            if (has_second_reader) {
                unpack_modes.emplace(IN_1, UnpackMode::UnpackToSrc);
                unpack_modes.emplace(SCALAR_1, UnpackMode::UnpackToSrc);
            }
        }
        // Sharded: compute is the only kernel that uses the output DFB (self-loop), so it is also its consumer.
        if (is_sharded && output_dfb_data_format == tt::DataFormat::Float32) {
            unpack_modes.emplace(OUT_DFB, UnpackMode::UnpackToSrc);
        }
    }
    const ComputeHardwareConfig compute_hw_config{
        .fpu_math_fidelity = resolved_math_fidelity,
        .sfpu_precision_mode = resolved_math_approx ? Precision::Approximate : Precision::Precise,
        .enable_32_bit_dest = resolved_fp32_acc,
        .double_buffer_dest = !user_dst_full_sync,
        .unpack_modes = unpack_modes,
        .config_1xx = ComputeHardwareConfig::Compute1XXConfig{},
    };

    auto make_compute_spec = [&](const KernelSpecName& name, uint32_t total_interpolations) {
        KernelSpec::CompilerOptions::Defines compute_defines(pool_defines_map);
        Group<DFBBinding> compute_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = IN_0,
                .accessor_name = "in_0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = SCALAR_0,
                .accessor_name = "scalar_0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
        };
        if (has_second_reader) {
            compute_defines.emplace("SPLIT_READER", "1");
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = IN_1,
                .accessor_name = "in_1",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = SCALAR_1,
                .accessor_name = "scalar_1",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        if (is_sharded) {
            // The packed sticks land straight in the resident output shard and nothing drains them.
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        return KernelSpec{
            .unique_id = name,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample/device/kernels/compute/"
                "compute_pool_2d.cpp",
            .compiler_options =
                {
                    .defines = std::move(compute_defines),
                    .opt_level = KernelBuildOptLevel::O3,
                },
            .dfb_bindings = std::move(compute_dfb_bindings),
            .compile_time_args =
                {
                    {"in_ntiles_c", in_ntiles_c},
                    {"window_size_hw", REDUCTION_SIZE},
                    {"split_reader", enable_split_reader ? 1U : 0U},
                    {"max_out_sticks_per_core", total_interpolations},
                    {"in_c", channels_per_shard},
                    {"in_nblocks_c", in_nblocks_c},
                    {"max_sticks_for_reduction", MAX_ROWS_FOR_REDUCTION},
                    {"one_scalar_per_core", ONE_SCALAR_PER_CORE ? 1U : 0U},
                    {"is_output_tiled", is_output_tiled ? 1U : 0U},
                    {"is_output_block_format", is_output_block_format ? 1U : 0U},
                    {"force_max_tiles_per_reduction_4", force_4_tile_chunk ? 1U : 0U},
                },
            .hw_config = compute_hw_config,
        };
    };

    std::vector<WorkUnitSpec> work_units;
    if (is_sharded) {
        kernels.push_back(make_compute_spec(COMPUTE, grid_batching_factor * grid_nsticks_per_core));
        Group<KernelSpecName> wu_kernels = {READER_0, COMPUTE};
        if (has_second_reader) {
            wu_kernels.push_back(READER_1);
        }
        work_units.push_back(WorkUnitSpec{
            .name = "grid_sample_bilinear",
            .kernels = std::move(wu_kernels),
            .target_nodes = all_cores,
        });
    } else {
        if (core_group_1.num_cores() > 0) {
            kernels.push_back(make_compute_spec(COMPUTE_GROUP_1, grid_batching_factor * num_sticks_per_core_group_1));
            work_units.push_back(WorkUnitSpec{
                .name = "grid_sample_bilinear_group_1",
                .kernels = {READER_0, COMPUTE_GROUP_1, WRITER},
                .target_nodes = core_group_1,
            });
        }
        if (core_group_2.num_cores() > 0) {
            kernels.push_back(make_compute_spec(COMPUTE_GROUP_2, grid_batching_factor * num_sticks_per_core_group_2));
            work_units.push_back(WorkUnitSpec{
                .name = "grid_sample_bilinear_group_2",
                .kernels = {READER_0, COMPUTE_GROUP_2, WRITER},
                .target_nodes = core_group_2,
            });
        }
    }

    // Writer kernel (interleaved only)
    bool has_writer = !is_sharded;
    if (has_writer) {
        kernels.push_back(KernelSpec{
            .unique_id = WRITER,
            .source = "ttnn/cpp/ttnn/operations/experimental/quasar/grid_sample/device/kernels/dataflow/"
                      "writer_grid_sample_interleaved.cpp",
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .tensor_bindings =
                {
                    TensorBinding{
                        .tensor_parameter_name = OUTPUT,
                        .accessor_name = "output",
                    },
                },
            .compile_time_args =
                {
                    {"output_stick_size", get_aligned_stick_size(output_shape, output_tensor)},
                    {"ntiles_c", out_ntiles_c},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"num_sticks_to_write", "start_stick_id"}},
            // The legacy WriterConfigDescriptor, i.e. the writer default.
            .hw_config = ttnn::create_writer_datamovement_config(),
        });
    }

    // Set runtime arguments
    KernelRunArgs reader0_run_args{.kernel = READER_0};
    KernelRunArgs reader1_run_args{.kernel = READER_1};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    if (is_sharded) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];

            // Runtime arguments for sharded reader
            AddRuntimeArgsForNode(
                reader0_run_args.runtime_arg_values, core, {{"global_grid_stick_start", i * grid_nsticks_per_core}});
            if (enable_split_reader) {
                AddRuntimeArgsForNode(
                    reader1_run_args.runtime_arg_values,
                    core,
                    {{"global_grid_stick_start", i * grid_nsticks_per_core}});
            }
        }
    } else {
        uint32_t grid_processed = 0;
        uint32_t output_processed = 0;

        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t grid_sticks =
                core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;
            const uint32_t output_sticks = batch_output_channels ? grid_sticks : grid_sticks * grid_batching_factor;

            // Runtime arguments for interleaved reader
            AddRuntimeArgsForNode(
                reader0_run_args.runtime_arg_values,
                core,
                {{"num_pages", grid_sticks}, {"start_page_id", grid_processed}});

            // Runtime arguments for interleaved writer
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"num_sticks_to_write", output_sticks}, {"start_stick_id", output_processed}});

            grid_processed += grid_sticks;
            output_processed += output_sticks;
        }
    }

    // The compute kernels have no runtime args, so they need no KernelRunArgs entry.
    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader0_run_args));
    if (has_second_reader) {
        run_args.kernel_run_args.push_back(std::move(reader1_run_args));
    }
    if (has_writer) {
        run_args.kernel_run_args.push_back(std::move(writer_run_args));
    }
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {GRID, grid_mesh_tensor},
        {OUTPUT, output_mesh_tensor},
    };

    ProgramSpec spec{
        .name = "grid_sample_bilinear",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = GRID, .spec = grid_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
