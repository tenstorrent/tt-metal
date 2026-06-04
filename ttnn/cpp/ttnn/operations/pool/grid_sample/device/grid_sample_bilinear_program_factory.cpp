// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"
#include "grid_sample_utils.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/pool/grid_sample/device/grid_sample_device_operation.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor GridSampleBilinearProgramFactory::create_descriptor(
    const GridSampleParams& operation_attributes, const GridSampleInputs& tensor_args, Tensor& output_tensor) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& grid_tensor = tensor_args.grid;
    bool use_precomputed_grid = operation_attributes.use_precomputed_grid;
    bool batch_output_channels = operation_attributes.batch_output_channels;

    const bool is_sharded = grid_tensor.is_sharded();
    ProgramDescriptor desc;

    // Data formats and device
    const auto [input_cb_data_format, grid_cb_data_format, output_cb_data_format] = std::make_tuple(
        tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(grid_tensor.dtype()),
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype()));
    tt::tt_metal::IDevice* const device = output_tensor.device();

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

    uint32_t cb_idx = tt::CBIndex::c_0;

    // Create CBs
    const auto input_face_geometry = FaceGeometry{.face_r_dim = REDUCTION_SIZE, .num_faces = 2};
    const auto scalar_face_geometry = FaceGeometry{.face_r_dim = 1, .num_faces = 2};
    const bool last_output_tile_is_partial = input_shape[-1] % tt::constants::TILE_WIDTH != 0;
    const bool single_partial_output_fits_in_face =
        last_output_tile_is_partial && input_shape[-1] <= tt::constants::FACE_WIDTH;
    const auto output_face_geometry =
        FaceGeometry{.face_r_dim = 1, .num_faces = single_partial_output_fits_in_face ? 1U : 2U};
    const std::optional<TileDescriptor> output_tile =
        single_partial_output_fits_in_face ? std::optional{TileDescriptor{1, tt::constants::FACE_WIDTH, false}}
                                           : std::nullopt;

    const uint32_t grid_stick_size =
        is_sharded ? grid_shape[-1] * grid_tensor.element_size() : get_aligned_stick_size(grid_shape, grid_tensor);
    const uint32_t grid_cb_index = cb_idx++;
    const uint32_t grid_cb_num_pages = is_sharded ? grid_nsticks_per_core : 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = grid_cb_num_pages * grid_stick_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(grid_cb_index),
            .data_format = grid_cb_data_format,
            .page_size = grid_stick_size,
        }}},
        .buffer = is_sharded ? grid_tensor.buffer() : nullptr,
    });

    // Resolve compute kernel config early: under fp32_dest_acc_en + half-sync, DEST holds only 4
    // fp32 tiles, so each chunk must shrink to <= 4 tiles. When the user explicitly enables
    // dst_full_sync_en, full-sync DEST holds 8 fp32 tiles and the 4-tile clamp would be a
    // gratuitous slowdown — so we keep the full 8-tile chunk in that case. The compute kernel
    // reads the same flag (ct_arg[16]) and recomputes its own MAX_TILES_PER_REDUCTION accordingly.
    const auto [resolved_math_fidelity, resolved_math_approx, resolved_fp32_acc, resolved_l1_acc, user_dst_full_sync] =
        ttnn::get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
    (void)resolved_l1_acc;  // not consumed in this factory
    const bool force_4_tile_chunk = resolved_fp32_acc && !user_dst_full_sync;
    const uint32_t effective_max_tiles_per_reduction = force_4_tile_chunk ? 4U : MAX_TILES_PER_REDUCTION;

    const uint32_t in_ntiles_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(input_shape[-1]) / tt::constants::TILE_WIDTH));
    // When in_ntiles_c exceeds effective_max_tiles_per_reduction the channel dimension is processed
    // in chunks; the input CB only needs to hold one chunk at a time, keeping the per-core L1
    // footprint bounded regardless of how wide the input stick is.
    const uint32_t max_tiles_per_iter = std::min<uint32_t>(in_ntiles_c, effective_max_tiles_per_reduction);
    const uint32_t in_nblocks_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(in_ntiles_c) / effective_max_tiles_per_reduction));
    const uint32_t input_chunk_nbytes =
        effective_max_tiles_per_reduction * tt::constants::TILE_WIDTH * input_tensor.element_size();
    const uint32_t input_cb_page_size = max_tiles_per_iter * tt::constants::TILE_HW * input_tensor.element_size();
    // When the last channel chunk holds fewer than effective_max_tiles_per_reduction tiles, the
    // reader must pack it with a tight write stride (== chunk_bytes) so the unpacker's
    // tiles_to_reduce matches compute_pool_2d's `tilize_reconfig` path. We require
    // !last_tile_is_partial — currently guaranteed by the padded-width % TILE_WIDTH == 0 check in
    // grid_sample_device_operation.cpp — and window_size_hw (REDUCTION_SIZE=4) <= FACE_HEIGHT,
    // which is a static property of bilinear.
    const bool last_tile_is_partial = (input_shape[-1] % tt::constants::TILE_WIDTH) != 0;
    const bool last_chunk_partial =
        (in_nblocks_c > 1) && (in_ntiles_c % effective_max_tiles_per_reduction != 0) && !last_tile_is_partial;
    const uint32_t input_cb_index_0 = cb_idx++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = BUFFERING_FACTOR * input_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_cb_index_0),
            .data_format = input_cb_data_format,
            .page_size = input_cb_page_size,
            .face_geometry = input_face_geometry,
        }}},
    });

    uint32_t input_cb_index_1 = DUMMY_CB_ID;
    if (is_sharded && enable_split_reader) {
        input_cb_index_1 = cb_idx++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = BUFFERING_FACTOR * input_cb_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(input_cb_index_1),
                .data_format = input_cb_data_format,
                .page_size = input_cb_page_size,
                .face_geometry = input_face_geometry,
            }}},
        });
    }

    const uint32_t scalar_cb_page_size = tt::tile_size(input_cb_data_format);
    const uint32_t scalar_cb_index_0 = cb_idx++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = BUFFERING_FACTOR * scalar_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(scalar_cb_index_0),
            .data_format = input_cb_data_format,
            .page_size = scalar_cb_page_size,
            .face_geometry = scalar_face_geometry,
        }}},
    });

    uint32_t scalar_cb_index_1 = DUMMY_CB_ID;
    if (is_sharded && enable_split_reader) {
        scalar_cb_index_1 = cb_idx++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = BUFFERING_FACTOR * scalar_cb_page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(scalar_cb_index_1),
                .data_format = input_cb_data_format,
                .page_size = scalar_cb_page_size,
                .face_geometry = scalar_face_geometry,
            }}},
        });
    }

    const uint32_t out_ntiles_c =
        static_cast<uint32_t>(std::ceil(static_cast<float>(output_shape[-1]) / tt::constants::FACE_WIDTH));
    const uint32_t output_cb_page_size = tt::constants::FACE_WIDTH * output_tensor.element_size();
    const uint32_t output_cb_pages =
        is_sharded ? output_nsticks_per_core * out_ntiles_c : out_ntiles_c * BUFFERING_FACTOR;
    const uint32_t output_cb_index = cb_idx++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_pages * output_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_cb_page_size,
            .tile = output_tile,
            .face_geometry = output_face_geometry,
        }}},
        .buffer = is_sharded ? output_tensor.buffer() : nullptr,
    });

    // Prepare stick size arguments with proper names
    const uint32_t input_stick_size = get_aligned_stick_size(input_shape, input_tensor);
    const uint32_t grid_stick_size_arg =
        is_sharded ? grid_shape[-1] * grid_tensor.element_size() : get_aligned_stick_size(grid_shape, grid_tensor);

    // Reader compile-time arguments - shared arguments first, then specific ones
    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index_0,                              // ct_arg[0]: input_cb_index_0
        grid_cb_index,                                 // ct_arg[1]: grid_cb_index
        scalar_cb_index_0,                             // ct_arg[2]: scalar_cb_index_0
        input_stick_size,                              // ct_arg[3]: input_stick_size
        grid_stick_size_arg,                           // ct_arg[4]: grid_stick_size
        input_batch,                                   // ct_arg[5]: input_batch
        input_height,                                  // ct_arg[6]: input_height
        input_width,                                   // ct_arg[7]: input_width
        grid_batching_factor,                          // ct_arg[8]: grid_batching_factor (shared)
        static_cast<uint32_t>(grid_tensor.dtype()),    // ct_arg[9]: grid_dtype (shared)
        grid_hw,                                       // ct_arg[10]: grid_hw (shared)
        use_precomputed_grid ? 1U : 0U,                // ct_arg[11]: use_precomputed_grid (shared)
        operation_attributes.align_corners ? 1U : 0U,  // ct_arg[12]: align_corners (shared)
        in_nblocks_c,                                  // ct_arg[13]: in_nblocks_c (shared)
        input_chunk_nbytes,                            // ct_arg[14]: input_chunk_nbytes (shared)
        last_chunk_partial ? 1U : 0U                   // ct_arg[15]: last_chunk_partial (shared)
    };

    if (is_sharded) {
        reader_compile_time_args.push_back(enable_split_reader ? 1U : 0U);  // ct_arg[16]: enable_split_reader
        reader_compile_time_args.push_back(0U);  // ct_arg[17]: reader_id (will be set later per reader)
        reader_compile_time_args.push_back(grid_nsticks_per_core);  // ct_arg[18]: grid_nsticks_per_core
    }

    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);
    if (!is_sharded) {
        tt::tt_metal::TensorAccessorArgs(*grid_tensor.buffer()).append_to(reader_compile_time_args);
    }

    const std::string reader_kernel_path =
        is_sharded ? "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_sharded.cpp"
                   : "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/"
                     "reader_grid_sample_interleaved_start_id.cpp";

    KernelDescriptor reader0_desc;
    KernelDescriptor reader1_desc;
    if (is_sharded) {
        auto reader0_compile_time_args = reader_compile_time_args;
        reader0_compile_time_args[17] = 0;  // ct_arg[17]: reader_id = 0

        reader0_desc.kernel_source = reader_kernel_path;
        reader0_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader0_desc.core_ranges = all_cores;
        reader0_desc.compile_time_args = std::move(reader0_compile_time_args);
        reader0_desc.config = DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
        };

        if (enable_split_reader) {
            auto reader1_compile_time_args = reader_compile_time_args;
            reader1_compile_time_args[0] = input_cb_index_1;   // ct_arg[0]: input_cb_index_1
            reader1_compile_time_args[2] = scalar_cb_index_1;  // ct_arg[2]: scalar_cb_index_1
            reader1_compile_time_args[17] = 1;                 // ct_arg[17]: reader_id = 1

            reader1_desc.kernel_source = reader_kernel_path;
            reader1_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            reader1_desc.core_ranges = all_cores;
            reader1_desc.compile_time_args = std::move(reader1_compile_time_args);
            reader1_desc.config = DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
            };
        }
    } else {
        reader0_desc.kernel_source = reader_kernel_path;
        reader0_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader0_desc.core_ranges = all_cores;
        reader0_desc.compile_time_args = std::move(reader_compile_time_args);
        // Use ReaderConfigDescriptor so the framework preserves
        // ReaderDataMovementConfig defaults (preferred_noc_for_dram_read(arch)).
        reader0_desc.config = ReaderConfigDescriptor{};
    }

    const bool is_output_tiled = false;
    const bool is_output_block_format = false;
    const uint32_t pre_tilize_cb_id =
        32;  // Unused CB for pool compute kernel for grid sample, we don't have tiled output in gridsample

    // Compute kernels
    const uint32_t channels_per_shard = input_shape[-1];

    auto pool_defines_map = ttnn::operations::pool::get_defines(ttnn::operations::pool::Pool2DType::AVG_POOL2D);

    auto make_compute_desc = [&](tt::tt_metal::CoreRangeSet cores, uint32_t total_interpolations) {
        // Compute kernel compile-time arguments
        std::vector<uint32_t> compute_compile_time_args = {
            in_ntiles_c,                       // ct_arg[0]: in_ntiles_c
            REDUCTION_SIZE,                    // ct_arg[1]: REDUCTION_SIZE
            enable_split_reader,               // ct_arg[2]: enable_split_reader
            total_interpolations,              // ct_arg[3]: total_interpolations
            channels_per_shard,                // ct_arg[4]: channels_per_shard
            in_nblocks_c,                      // ct_arg[5]: in_nblocks_c
            MAX_ROWS_FOR_REDUCTION,            // ct_arg[6]: MAX_ROWS_FOR_REDUCTION
            input_cb_index_0,                  // ct_arg[7]: input_cb_index_0
            input_cb_index_1,                  // ct_arg[8]: input_cb_index_1
            scalar_cb_index_0,                 // ct_arg[9]: scalar_cb_index_0
            scalar_cb_index_1,                 // ct_arg[10]: scalar_cb_index_1
            output_cb_index,                   // ct_arg[11]: output_cb_index
            ONE_SCALAR_PER_CORE,               // ct_arg[12]: ONE_SCALAR_PER_CORE
            pre_tilize_cb_id,                  // ct_arg[13]: pre_tilize_cb_id
            is_output_tiled ? 1U : 0U,         // ct_arg[14]: is_output_tiled
            is_output_block_format ? 1U : 0U,  // ct_arg[15]: is_output_block_format
            force_4_tile_chunk ? 1U : 0U,      // ct_arg[16]: force_max_tiles_per_reduction_4
            DUMMY_CB_ID,                       // ct_arg[17]: unused (in_idx_cb_id for mpwi)
            DUMMY_CB_ID,                       // ct_arg[18]: unused (pack_tmp_cb_id for mpwi)
            DUMMY_CB_ID,                       // ct_arg[19]: unused (pack_idx_tmp_cb_id for mpwi)
            DUMMY_CB_ID,                       // ct_arg[20]: unused (right_inc_cb_id for mpwi)
            DUMMY_CB_ID,                       // ct_arg[21]: unused (down_left_wrap_inc_cb_id for mpwi)
            DUMMY_CB_ID,                       // ct_arg[22]: unused (up_left_wrap_inc_cb_id for mpwi)
            DUMMY_CB_ID,                       // ct_arg[23]: unused (out_idx_cb_id for mpwi)
            1,                                 // ct_arg[24]: stride_h (unused by grid_sample)
            1,                                 // ct_arg[25]: stride_w (unused by grid_sample)
            1,                                 // ct_arg[26]: in_h_padded (unused by grid_sample)
            1,                                 // ct_arg[27]: in_w_padded (unused by grid_sample)
            1,                                 // ct_arg[28]: eff_kernel_h (unused by grid_sample)
            1,                                 // ct_arg[29]: eff_kernel_w (unused by grid_sample)
            1,                                 // ct_arg[30]: pad_l (unused by grid_sample)
            DUMMY_CB_ID,                       // ct_arg[31]: intra_kernel_right_inc_cb_id (unused)
            DUMMY_CB_ID,                       // ct_arg[32]: intra_kernel_down_left_wrap_inc_cb_id (unused)
            DUMMY_CB_ID,                       // ct_arg[33]: compute_tmp_idx_cb_id (unused)
            DUMMY_CB_ID,                       // ct_arg[34]: clear_value_cb_id (unused)
            1,                                 // ct_arg[35]: kernel_h (unused by grid_sample)
            1,                                 // ct_arg[36]: kernel_w (unused by grid_sample)
            0,                                 // ct_arg[37]: indexes_32_bit (unused by grid_sample)
            DUMMY_CB_ID,                       // ct_arg[38]: fast_tilize_cb_id (tiled-output only)
        };

        KernelDescriptor compute_desc;
        compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = std::move(cores);
        compute_desc.compile_time_args = std::move(compute_compile_time_args);
        compute_desc.defines = {pool_defines_map.begin(), pool_defines_map.end()};
        compute_desc.config = ComputeConfigDescriptor{
            .math_fidelity = resolved_math_fidelity,
            .fp32_dest_acc_en = resolved_fp32_acc,
            .dst_full_sync_en = user_dst_full_sync,
            .math_approx_mode = resolved_math_approx,
        };
        return compute_desc;
    };

    std::vector<KernelDescriptor> compute_descs;
    if (is_sharded || core_group_1.num_cores() > 0) {
        compute_descs.push_back(make_compute_desc(
            is_sharded ? all_cores : core_group_1,
            grid_batching_factor * (is_sharded ? grid_nsticks_per_core : num_sticks_per_core_group_1)));
    }
    if (!is_sharded && core_group_2.num_cores() > 0) {
        compute_descs.push_back(make_compute_desc(core_group_2, grid_batching_factor * num_sticks_per_core_group_2));
    }

    // Writer kernel (interleaved only)
    KernelDescriptor writer_desc;
    bool has_writer = !is_sharded;
    if (has_writer) {
        // Writer compile-time arguments - expanded row by row for readability
        std::vector<uint32_t> writer_compile_time_args = {
            output_cb_index,                                      // ct_arg[0]: output_cb_index
            get_aligned_stick_size(output_shape, output_tensor),  // ct_arg[1]: output_stick_size
            out_ntiles_c                                          // ct_arg[2]: out_ntiles_c
        };
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = all_cores;
        writer_desc.compile_time_args = std::move(writer_compile_time_args);
        // Use WriterConfigDescriptor so the framework preserves
        // WriterDataMovementConfig defaults (preferred_noc_for_dram_write(arch)).
        writer_desc.config = WriterConfigDescriptor{};
    }

    // Set runtime arguments
    if (is_sharded) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];

            // Runtime arguments for sharded reader
            KernelDescriptor::CoreRuntimeArgs reader_runtime_args = {
                input_tensor.buffer()->address(),  // rt_arg[0]: input_buffer_address
                i * grid_nsticks_per_core          // rt_arg[1]: grid_stick_offset
            };

            reader0_desc.runtime_args.emplace_back(core, reader_runtime_args);
            if (enable_split_reader) {
                reader1_desc.runtime_args.emplace_back(core, reader_runtime_args);
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

            // Runtime arguments for interleaved reader - expanded row by row
            KernelDescriptor::CoreRuntimeArgs reader_runtime_args = {
                input_tensor.buffer()->address(),  // rt_arg[0]: input_buffer_address
                grid_tensor.buffer()->address(),   // rt_arg[1]: grid_buffer_address
                grid_sticks,                       // rt_arg[2]: grid_sticks
                grid_processed                     // rt_arg[3]: grid_processed
            };

            // Runtime arguments for interleaved writer - expanded row by row
            KernelDescriptor::CoreRuntimeArgs writer_runtime_args = {
                output_tensor.buffer()->address(),  // rt_arg[0]: output_buffer_address
                output_sticks,                      // rt_arg[1]: output_sticks
                output_processed                    // rt_arg[2]: output_processed
            };

            reader0_desc.runtime_args.emplace_back(core, std::move(reader_runtime_args));
            writer_desc.runtime_args.emplace_back(core, std::move(writer_runtime_args));

            grid_processed += grid_sticks;
            output_processed += output_sticks;
        }
    }

    desc.kernels.push_back(std::move(reader0_desc));
    if (is_sharded && enable_split_reader) {
        desc.kernels.push_back(std::move(reader1_desc));
    }
    for (auto& cd : compute_descs) {
        desc.kernels.push_back(std::move(cd));
    }
    if (has_writer) {
        desc.kernels.push_back(std::move(writer_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
