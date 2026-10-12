// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <cstdint>
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"
#include "grid_sample_utils.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/pool/grid_sample/device/grid_sample_device_operation.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor GridSampleNearestProgramFactory::create_descriptor(
    const GridSampleParams& operation_attributes, const GridSampleInputs& tensor_args, Tensor& output_tensor) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& grid_tensor = tensor_args.grid;
    const bool use_precomputed_grid = operation_attributes.use_precomputed_grid;
    const bool align_corners = operation_attributes.align_corners;

    const bool is_sharded = grid_tensor.is_sharded();
    // Rank-5 (N, H_out, W_out, K, 2) interleaved grid: each core gathers its sticks from the grid's 4-byte DRAM
    // pages into an L1 CB at kernel start (12 bulk reads, one per DRAM bank, then a word compaction) and then
    // runs exactly like the sharded-grid path. Replaces the 286 µs row-major ReshapeView + InterleavedToSharded.
    const bool gather_grid = is_rank5_grid(grid_tensor) && !is_sharded;
    const bool grid_in_l1 = is_sharded || gather_grid;
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
    const uint32_t input_height = input_shape[1], input_width = input_shape[2];
    const uint32_t grid_height = grid_shape[1], grid_width = grid_shape[2];
    const uint32_t grid_hw = grid_height * grid_width;
    const uint32_t grid_batching_factor = get_grid_batching_factor(grid_tensor, use_precomputed_grid, "nearest");
    const uint32_t grid_last_dim = grid_last_dim_equiv(grid_tensor);
    const bool enable_split_reader =
        gather_grid || should_use_split_reader(input_tensor, grid_tensor, use_precomputed_grid, "nearest");
    tt::tt_metal::CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores, grid_nsticks_per_core, output_nsticks_per_core = 0;
    uint32_t num_sticks_per_core_group_1 = 0, num_sticks_per_core_group_2 = 0;
    std::vector<CoreCoord> logical_cores;

    if (is_sharded) {
        const auto grid_shard_spec = grid_tensor.shard_spec().value();
        all_cores = grid_shard_spec.grid;
        num_cores = grid_shard_spec.num_cores();
        grid_nsticks_per_core = grid_shard_spec.shape[0];
        output_nsticks_per_core = output_tensor.shard_spec().value().shape[0];
        logical_cores = corerange_to_cores(
            all_cores, num_cores, grid_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    } else if (gather_grid) {
        // the output's height sharding defines the work split; each core gathers that many grid sticks
        const auto out_shard_spec = output_tensor.shard_spec().value();
        all_cores = out_shard_spec.grid;
        num_cores = out_shard_spec.num_cores();
        grid_nsticks_per_core = out_shard_spec.shape[0];
        output_nsticks_per_core = out_shard_spec.shape[0];
        logical_cores = corerange_to_cores(
            all_cores, num_cores, out_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    } else {
        const auto compute_grid_size = device->compute_with_storage_grid_size();
        uint32_t grid_nsticks = grid_tensor.physical_volume() / grid_shape[-1];
        if (output_tensor.shard_spec().has_value()) {
            // output rows per core = grid sticks per core (channel-batched) or sticks x K (width-extended)
            grid_nsticks = output_tensor.shard_spec().value().shape[0] * output_tensor.shard_spec().value().num_cores() /
                           (operation_attributes.batch_output_channels ? 1 : grid_batching_factor);
        } else {
            grid_nsticks = tt::round_up(grid_nsticks, compute_grid_size.x * compute_grid_size.y);
        }
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, grid_nsticks);

        std::tie(num_cores, all_cores, core_group_1, core_group_2) =
            std::make_tuple(num_cores_used, all_cores_range, core_group_1_range, core_group_2_range);
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
        grid_nsticks_per_core = num_sticks_1;
        output_nsticks_per_core = num_sticks_1;
        logical_cores = corerange_to_cores(all_cores, num_cores, true);
    }

    uint32_t cb_idx = tt::CBIndex::c_0;

    // Create CBs
    const uint32_t grid_stick_size =
        grid_in_l1 ? grid_last_dim * grid_tensor.element_size() : get_aligned_stick_size(grid_shape, grid_tensor);
    const uint32_t grid_cb_index0 = cb_idx++;
    const uint32_t grid_cb_num_pages0 = grid_in_l1 ? grid_nsticks_per_core : 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = grid_cb_num_pages0 * grid_stick_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(grid_cb_index0),
            .data_format = grid_cb_data_format,
            .page_size = grid_stick_size,
        }}},
        .buffer = is_sharded ? grid_tensor.buffer() : nullptr,
    });

    uint32_t grid_cb_index1 = DUMMY_CB_ID;
    // Nearest mode has support for DRAM interleaved grid and sharded output
    if (enable_split_reader && !grid_in_l1) {
        grid_cb_index1 = cb_idx++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = grid_stick_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(grid_cb_index1),
                .data_format = grid_cb_data_format,
                .page_size = grid_stick_size,
            }}},
        });
    }

    const uint32_t output_cb_page_size =
        static_cast<uint32_t>(static_cast<float>(output_shape[-1]) * output_tensor.element_size());
    const uint32_t output_cb_pages = output_nsticks_per_core;

    // Fill CB - single page to hold pre-filled stick
    const uint32_t fill_cb_index = cb_idx++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(fill_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_cb_page_size,
        }}},
    });

    const uint32_t output_cb_index = cb_idx++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_pages * output_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_cb_page_size,
        }}},
        .buffer = output_tensor.buffer(),
    });

    // Sharded bf16 grid without precompute: the compute RISCs turn the grid shard into int16 (h, w) indices
    // (grid_sample_nearest_index.cpp) so the data-movement RISCs do no coordinate math.
    const bool indices_from_cb = grid_in_l1 && !use_precomputed_grid && grid_tensor.dtype() == DataType::BFLOAT16;
    uint32_t idx_cb_index = DUMMY_CB_ID;
    uint32_t prog_cb_index[3] = {DUMMY_CB_ID, DUMMY_CB_ID, DUMMY_CB_ID};
    if (indices_from_cb) {
        idx_cb_index = cb_idx++;
        const uint32_t idx_bytes = grid_nsticks_per_core * grid_batching_factor * 4;
        desc.cbs.push_back(CBDescriptor{
            .total_size = idx_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx_cb_index),
                .data_format = tt::DataFormat::UInt32,
                .page_size = idx_bytes,
            }}},
        });
        for (uint32_t t = 0; t < 3; t++) {
            prog_cb_index[t] = cb_idx++;
            desc.cbs.push_back(CBDescriptor{
                .total_size = 32,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(prog_cb_index[t]),
                    .data_format = tt::DataFormat::UInt32,
                    .page_size = 32,
                }}},
            });
        }
    }

    // Gather mode: a staging CB with one contiguous run of pages per DRAM bank, and a 2-page CB whose received
    // counter tells the compute threads (and the other reader) that the grid shard is in L1.
    uint32_t stage_cb_index = DUMMY_CB_ID;
    uint32_t grid_ready_cb_index[2] = {DUMMY_CB_ID, DUMMY_CB_ID};  // one per reader: a shared counter would be a racy read-modify-write
    uint32_t grid_page_aligned_bytes = 0;
    uint32_t num_grid_banks = 1;
    uint32_t stage_run_bytes = 0;
    if (gather_grid) {
        grid_page_aligned_bytes = grid_tensor.buffer()->aligned_page_size();
        num_grid_banks = device->allocator()->get_num_banks(grid_tensor.buffer()->buffer_type());
        const uint32_t pages_per_core = grid_nsticks_per_core * grid_batching_factor;
        stage_run_bytes = (pages_per_core / num_grid_banks + 2) * grid_page_aligned_bytes;
        stage_cb_index = cb_idx++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = stage_run_bytes * num_grid_banks,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(stage_cb_index),
                .data_format = tt::DataFormat::UInt32,
                .page_size = stage_run_bytes * num_grid_banks,
            }}},
        });
        for (uint32_t r = 0; r < 2; r++) {
            grid_ready_cb_index[r] = cb_idx++;
            desc.cbs.push_back(CBDescriptor{
                .total_size = 32,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(grid_ready_cb_index[r]),
                    .data_format = tt::DataFormat::UInt32,
                    .page_size = 32,
                }}},
            });
        }
    }

    // Read de-duplication table of the writers (see writer_grid_sample_nearest_sharded.cpp): one direct-mapped
    // table and one deferred-copy list per reader.
    uint32_t dedup_cb_index = DUMMY_CB_ID;
    static const bool dedup_disabled = std::getenv("TT_GRID_SAMPLE_NO_DEDUP") != nullptr;  // A/B switch
    if (indices_from_cb && output_tensor.is_sharded() && !dedup_disabled) {
        // six tables (three compute threads x two readers) followed by the two readers' deferred-copy lists;
        // must match DEDUP_ENTRIES / DEDUP_CHUNK of the kernels
        constexpr uint32_t dedup_entries = 256, dedup_chunk = 256;
        const uint32_t dedup_bytes = 6 * dedup_entries * 8 + 2 * dedup_chunk * 8;
        dedup_cb_index = cb_idx++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = dedup_bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(dedup_cb_index),
                .data_format = tt::DataFormat::UInt32,
                .page_size = dedup_bytes,
            }}},
        });
    }

    // Prepare stick size arguments with proper names
    const uint32_t input_stick_size = get_aligned_stick_size(input_shape, input_tensor);
    const uint32_t grid_stick_size_arg =
        grid_in_l1 ? grid_last_dim * grid_tensor.element_size() : get_aligned_stick_size(grid_shape, grid_tensor);

    // Writer for nearest mode with sharded grid - matches sharded reader argument structure
    std::vector<uint32_t> writer_compile_time_args = {
        grid_cb_index0,                              // ct_arg[0]: grid_cb_index
        output_cb_index,                             // ct_arg[1]: output_cb_index (replaces scalar_cb_index)
        input_stick_size,                            // ct_arg[2]: input_stick_size
        grid_stick_size_arg,                         // ct_arg[3]: grid_stick_size
        input_height,                                // ct_arg[4]: input_height
        input_width,                                 // ct_arg[5]: input_width
        grid_batching_factor,                        // ct_arg[6]: grid_batching_factor
        static_cast<uint32_t>(grid_tensor.dtype()),  // ct_arg[7]: grid_dtype
        grid_hw,                                     // ct_arg[8]: grid_hw
        use_precomputed_grid ? 1U : 0U,              // ct_arg[9]: use_precomputed_grid
        align_corners ? 1U : 0U,                     // ct_arg[10]: align_corners
        enable_split_reader ? 1U : 0U,               // ct_arg[11]: split_reader (same as reader)
        0U,                                          // ct_arg[12]: reader_id (will be set per core)
        grid_nsticks_per_core,                       // ct_arg[13]: grid_nsticks_per_core
        grid_in_l1 ? 1U : 0U,                        // ct_arg[14]: grid in L1 (sharded grid, or gathered there)
        fill_cb_index,                               // ct_arg[15]: fill_cb_index
        input_shape[0],                              // ct_arg[16]: batch_size
        output_tensor.is_sharded() ? 1U : 0U,       // ct_arg[17]: output_is_sharded
        idx_cb_index,                                // ct_arg[18]: idx_cb (indices from the compute threads)
        prog_cb_index[0],                            // ct_arg[19]: prog_cb0
        prog_cb_index[1],                            // ct_arg[20]: prog_cb1
        prog_cb_index[2],                            // ct_arg[21]: prog_cb2
        indices_from_cb ? 1U : 0U,                   // ct_arg[22]: indices_from_cb
        gather_grid ? 1U : 0U,                       // ct_arg[23]: gather the rank-5 grid from DRAM at start
        stage_cb_index,                              // ct_arg[24]: staging CB (one run per DRAM bank)
        grid_ready_cb_index[0],                      // ct_arg[25]: "my half of the grid shard is in L1" counter CB of reader 0
        grid_page_aligned_bytes,                     // ct_arg[26]: aligned page size of the grid buffer
        num_grid_banks,                              // ct_arg[27]: DRAM banks the grid is interleaved over
        stage_run_bytes,                             // ct_arg[28]: bytes per bank run in the staging CB
        grid_ready_cb_index[1],                      // ct_arg[29]: counter CB of reader 1
        gather_grid ? static_cast<uint32_t>(grid_tensor.physical_volume() / grid_shape[-1]) : 0U,  // ct_arg[30]: total grid pages (partial last shard)
        dedup_cb_index                               // ct_arg[31]: read de-duplication scratch CB
    };

    // Add tensor accessor args for input tensor (32 compile time args offset)
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(writer_compile_time_args);
    if (!is_sharded) {
        tt::tt_metal::TensorAccessorArgs(*grid_tensor.buffer()).append_to(writer_compile_time_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(writer_compile_time_args);
    }

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_nearest_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_0_default,
    };

    KernelDescriptor writer1_desc;
    if (enable_split_reader) {
        auto writer1_compile_time_args = writer_compile_time_args;
        writer1_compile_time_args[0] = grid_in_l1 ? grid_cb_index0 : grid_cb_index1;  // ct_arg[0]: grid_cb_index1
        writer1_compile_time_args[12] = 1;                                            // ct_arg[12]: reader_id = 1

        writer1_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_nearest_sharded.cpp";
        writer1_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer1_desc.core_ranges = all_cores;
        writer1_desc.compile_time_args = std::move(writer1_compile_time_args);
        writer1_desc.config = DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
        };
    }

    // Set runtime arguments
    if (grid_in_l1) {
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];

            // Runtime arguments for sharded reader (+ the grid buffer address when it is gathered from DRAM)
            writer_desc.emplace_runtime_args(
                core,
                {
                    input_tensor.buffer(),      // rt_arg[0]: input_buffer_address
                    i * grid_nsticks_per_core,  // rt_arg[1]: grid_stick_offset
                    grid_tensor.buffer()        // rt_arg[2]: grid_buffer_address (gather mode)
                });
            if (enable_split_reader) {
                writer1_desc.emplace_runtime_args(
                    core,
                    {
                        input_tensor.buffer(),      // rt_arg[0]: input_buffer_address
                        i * grid_nsticks_per_core,  // rt_arg[1]: grid_stick_offset
                        grid_tensor.buffer()        // rt_arg[2]: grid_buffer_address (gather mode)
                    });
            }
        }
    } else {
        uint32_t grid_processed = 0;

        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t grid_sticks =
                core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

            // Runtime arguments for interleaved reader - expanded row by row
            writer_desc.emplace_runtime_args(
                core,
                {
                    input_tensor.buffer(),  // rt_arg[0]: input_buffer_address
                    grid_tensor.buffer(),   // rt_arg[1]: grid_buffer_address
                    grid_sticks,            // rt_arg[2]: grid_sticks
                    grid_processed          // rt_arg[3]: grid_processed
                });
            if (enable_split_reader) {
                // For split reader in nearest mode, second writer needs same runtime args
                // The kernel logic handles the split internally via reader_id
                writer1_desc.emplace_runtime_args(
                    core,
                    {
                        input_tensor.buffer(),  // rt_arg[0]: input_buffer_address
                        grid_tensor.buffer(),   // rt_arg[1]: grid_buffer_address
                        grid_sticks,            // rt_arg[2]: grid_sticks
                        grid_processed          // rt_arg[3]: grid_processed
                    });
            }

            grid_processed += grid_sticks;
        }
    }

    desc.kernels.push_back(std::move(writer_desc));
    if (enable_split_reader) {
        desc.kernels.push_back(std::move(writer1_desc));
    }
    if (indices_from_cb) {
        KernelDescriptor index_desc;
        index_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/compute/grid_sample_nearest_index.cpp";
        index_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        index_desc.core_ranges = all_cores;
        index_desc.compile_time_args = {
            grid_cb_index0,           // 0: grid shard
            idx_cb_index,             // 1: indices out
            prog_cb_index[0],         // 2..4: progress counters of the three threads
            prog_cb_index[1],
            prog_cb_index[2],
            grid_nsticks_per_core,    // 5
            grid_batching_factor,     // 6: K
            input_height,             // 7
            input_width,              // 8
            align_corners ? 1U : 0U,  // 9
            grid_stick_size_arg,      // 10
            gather_grid ? 1U : 0U,    // 11: wait for the readers to gather the grid shard first
            grid_ready_cb_index[0],   // 12
            grid_ready_cb_index[1],   // 13
            stage_cb_index,           // 14: the staged bank runs (gather mode: coordinates are read from here)
            grid_page_aligned_bytes,  // 15
            num_grid_banks,           // 16
            stage_run_bytes,          // 17
            dedup_cb_index,           // 18: read de-duplication tables
            dedup_cb_index != DUMMY_CB_ID ? 1U : 0U,  // 19: publish stick indices / copy-from-sample words
            grid_hw,                  // 20
            enable_split_reader ? 1U : 0U  // 21: a stick's reader is its parity when the readers split the sticks
        };
        index_desc.config = ComputeConfigDescriptor{};
        for (uint32_t i = 0; i < num_cores; i++) {
            index_desc.emplace_runtime_args(logical_cores[i], {i * grid_nsticks_per_core});
        }
        desc.kernels.push_back(std::move(index_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
