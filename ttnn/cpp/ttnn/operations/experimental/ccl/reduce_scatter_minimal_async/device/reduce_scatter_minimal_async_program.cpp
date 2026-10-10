// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>
#include <array>
#include <bitset>
#include <type_traits>
#include <variant>

#include <tt-metalium/allocator.hpp>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_program_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_line_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/overloaded.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"

#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

using namespace tt::constants;
using namespace tt::tt_metal;

// Import types from the new pattern
using ttnn::experimental::prim::ReduceScatterProgramArtifacts;

namespace ttnn {

namespace operations::experimental::ccl::detail {

std::unordered_map<std::string, uint32_t> get_ring_reader_named_compile_args(
    const uint32_t ring_index,
    const uint32_t ring_size,
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t intermediate_2_cb_index,
    const uint32_t reader_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t page_size,
    const uint32_t output_tensor_num_pages,
    const uint32_t input_batch_num_pages,
    const uint32_t output_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t output_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t slice_Ht,
    const uint32_t slice_Wt,
    const bool fuse_op,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {
            {"my_chip_id", ring_index},
            {"ring_size", ring_size},
            {"cb_input_id", input_cb_index},
            {"cb_interm_id", intermediate_cb_index},
            {"cb_reader_output_id", reader_output_cb_index},
            {"tile_granularity", tile_granularity},
            {"page_size", page_size},
            {"output_num_pages", output_tensor_num_pages},
            {"batch_num_pages", input_batch_num_pages},
            {"slice_B", slice_B},
        };
    }
    return {
        {"my_chip_id", ring_index},
        {"ring_size", ring_size},
        {"cb_input_id", input_cb_index},
        {"cb_interm_id", intermediate_cb_index},
        {"cb_interm2_id", intermediate_2_cb_index},
        {"cb_reader_output_id", reader_output_cb_index},
        {"tile_granularity", tile_granularity},
        {"page_size", page_size},
        {"input_batch_num_pages", input_batch_num_pages},
        {"output_batch_num_pages", output_batch_num_pages},
        {"input_channel_num_pages", input_channel_num_pages},
        {"output_channel_num_pages", output_channel_num_pages},
        {"input_tensor_B", input_tensor_B},
        {"input_tensor_Wt", input_tensor_Wt},
        {"slice_C", slice_C},
        {"slice_Ht", slice_Ht},
        {"slice_Wt", slice_Wt},
        {"fuse_op", fuse_op},
        {"dim", normalized_dim},
    };
}

std::unordered_map<std::string, uint32_t> get_ring_writer_named_compile_args(
    const uint32_t ring_index,
    const uint32_t ring_size,
    const uint32_t compute_output_cb_index,
    const uint32_t reader_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t page_size,
    const uint32_t num_tiles_to_write_per_packet,
    const uint32_t output_tensor_num_pages,
    const uint32_t output_batch_num_pages,
    const uint32_t input_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t output_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t slice_Ht,
    const uint32_t slice_Wt,
    const uint32_t normalized_dim,
    const bool fuse_op) {
    if (normalized_dim == 0) {
        return {
            {"my_chip_id", ring_index},
            {"ring_size", ring_size},
            {"cb_compute_output_id", compute_output_cb_index},
            {"cb_reader_output_id", reader_output_cb_index},
            {"tile_granularity", tile_granularity},
            {"page_size", page_size},
            {"num_tiles_to_write_per_packet", num_tiles_to_write_per_packet},
            {"output_num_pages", output_tensor_num_pages},
            {"batch_num_pages", input_batch_num_pages},
            {"slice_B", slice_B},
        };
    }
    return {
        {"my_chip_id", ring_index},
        {"ring_size", ring_size},
        {"cb_compute_output_id", compute_output_cb_index},
        {"cb_reader_output_id", reader_output_cb_index},
        {"tile_granularity", tile_granularity},
        {"page_size", page_size},
        {"num_tiles_to_write_per_packet", num_tiles_to_write_per_packet},
        {"output_batch_num_pages", output_batch_num_pages},
        // Batch stride of the tiled (input-shaped) intermediate; the writer needs it to give each
        // batch its own staging region on that layout, as the chunk-paged one already does.
        {"input_batch_num_pages", input_batch_num_pages},
        {"input_channel_num_pages", input_channel_num_pages},
        {"output_channel_num_pages", output_channel_num_pages},
        {"input_tensor_B", input_tensor_B},
        {"input_tensor_Wt", input_tensor_Wt},
        {"slice_C", slice_C},
        {"slice_Ht", slice_Ht},
        {"slice_Wt", slice_Wt},
        {"dim", normalized_dim},
        {"fuse_op", fuse_op},
    };
}

std::unordered_map<std::string, uint32_t> get_ring_compute_named_compile_args(
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t intermediate_2_cb_index,
    const uint32_t compute_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t ring_size,
    const uint32_t input_tensor_B,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t normalized_dim,
    const bool fuse_op) {
    if (normalized_dim == 0) {
        return {
            {"cb_input_id", input_cb_index},
            {"cb_interm_id", intermediate_cb_index},
            {"cb_compute_output_id", compute_output_cb_index},
            {"tile_granularity", tile_granularity},
            {"ring_size", ring_size},
            {"slice_B", slice_B},
        };
    }
    return {
        {"cb_input_id", input_cb_index},
        {"cb_interm_id", intermediate_cb_index},
        {"cb_interm2_id", intermediate_2_cb_index},
        {"cb_compute_output_id", compute_output_cb_index},
        {"tile_granularity", tile_granularity},
        {"ring_size", ring_size},
        {"input_tensor_B", input_tensor_B},
        {"slice_C", slice_C},
        {"fuse_op", fuse_op},
    };
}

std::vector<uint32_t> get_line_reader_compile_args(
    const uint32_t ring_index,
    const uint32_t ring_size,
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t reader_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t page_size,
    const uint32_t input_tensor_num_pages,
    const uint32_t output_tensor_num_pages,
    const uint32_t input_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t output_batch_num_pages,
    const uint32_t output_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t slice_Ht,
    const uint32_t slice_Wt,
    const bool fuse_op,
    const uint32_t sync_with_other_direction,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {
            ring_index,                // my_chip_id
            ring_size,                 // ring_size
            input_cb_index,            // cb_input_id
            intermediate_cb_index,     // cb_intermediate_id
            reader_output_cb_index,    // cb_reader_output_id
            tile_granularity,          // tile_granularity
            page_size,                 // page_size
            input_tensor_num_pages,    // input_num_pages
            output_tensor_num_pages,   // output_num_pages
            input_batch_num_pages,     // batch_num_pages
            slice_B,                   // slice_B
            sync_with_other_direction  // sync_with_other_direction
        };
    }
    return {
        ring_index,                 // my_chip_id
        ring_size,                  // ring_size
        input_cb_index,             // cb_input_id
        intermediate_cb_index,      // cb_intermediate_id
        reader_output_cb_index,     // cb_reader_output_id
        tile_granularity,           // tile_granularity
        page_size,                  // page_size
        input_tensor_num_pages,     // input_num_pages
        input_batch_num_pages,      // input_batch_num_pages
        input_channel_num_pages,    // input_channel_num_pages
        output_batch_num_pages,     // output_batch_num_pages
        output_channel_num_pages,   // output_channel_num_pages
        input_tensor_B,             // input_tensor_B
        input_tensor_Wt,            //         input_tensor_Wt
        slice_C,                    // slice_C
        slice_Ht,                   // slice_Ht
        slice_Wt,                   // slice_Wt
        fuse_op,                    //         fuse_op
        sync_with_other_direction,  // sync_with_other_direction
        normalized_dim,             // dim
    };
}

std::vector<uint32_t> get_line_writer_compile_args(
    const uint32_t ring_size,
    const uint32_t compute_output_cb_index,
    const uint32_t reader_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t page_size,
    const uint32_t tiles_to_write_per_packet,
    const uint32_t input_tensor_num_pages,
    const uint32_t output_tensor_num_pages,
    const uint32_t input_batch_num_pages,
    const uint32_t input_channel_num_pages,
    const uint32_t output_batch_num_pages,
    const uint32_t output_channel_num_pages,
    const uint32_t input_tensor_B,
    const uint32_t input_tensor_Wt,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t slice_Ht,
    const uint32_t slice_Wt,
    const uint32_t normalized_dim,
    const uint32_t sync_with_other_direction) {
    if (normalized_dim == 0) {
        return {
            ring_size,                  // ring_size
            compute_output_cb_index,    // cb_compute_output_id
            reader_output_cb_index,     // cb_reader_output_id
            tile_granularity,           // tile_granularity
            page_size,                  // page_size
            tiles_to_write_per_packet,  // contig_pages_advanced
            input_tensor_num_pages,     // input_num_pages
            output_tensor_num_pages,    // output_num_pages
            input_batch_num_pages,      // batch_num_pages
            slice_B,                    // slice_B
            sync_with_other_direction,  // sync_with_other_direction
        };
    }
    return {
        ring_size,                  // ring_size
        compute_output_cb_index,    // cb_compute_output_id
        reader_output_cb_index,     // cb_reader_output_id
        tile_granularity,           // tile_granularity
        page_size,                  // page_size
        tiles_to_write_per_packet,  //         contig_pages_advanced
        input_tensor_num_pages,     // input_num_pages
        input_batch_num_pages,      // input_batch_num_pages
        input_channel_num_pages,    // input_channel_num_pages
        output_batch_num_pages,     // output_batch_num_pages
        output_channel_num_pages,   // output_channel_num_pages
        input_tensor_B,             //         input_tensor_b
        input_tensor_Wt,            // input_tensor_Wt
        slice_C,                    // slice_C
        slice_Ht,                   // slice_Ht
        slice_Wt,                   //         slice_Wt
        normalized_dim,             // dim
        sync_with_other_direction   // sync_with_other_direction
    };
}

std::vector<uint32_t> get_line_reduce_compile_args(
    const uint32_t input_cb_index,
    const uint32_t intermediate_cb_index,
    const uint32_t compute_output_cb_index,
    const uint32_t tile_granularity,
    const uint32_t input_tensor_B,
    const uint32_t slice_B,
    const uint32_t slice_C,
    const uint32_t normalized_dim) {
    if (normalized_dim == 0) {
        return {input_cb_index, intermediate_cb_index, compute_output_cb_index, tile_granularity, slice_B};
    }
    return {input_cb_index, intermediate_cb_index, compute_output_cb_index, tile_granularity, input_tensor_B, slice_C};
}

}  // namespace operations::experimental::ccl::detail

using namespace ccl;
using ttnn::experimental::ccl::append_fabric_mux_connection_ct_args;
using ttnn::experimental::ccl::append_fabric_mux_connection_rt_args;

namespace {

template <typename ProgramOrDesc>
inline constexpr bool is_legacy_program_v = std::is_same_v<std::decay_t<ProgramOrDesc>, tt::tt_metal::Program>;

tt::tt_metal::CoreRangeSet as_core_range_set(const tt::tt_metal::CoreRangeSet& cores) { return cores; }

template <typename ProgramOrDesc>
void add_cb(
    ProgramOrDesc& target, const tt::tt_metal::CoreRangeSet& cores, const tt::tt_metal::CircularBufferConfig& config) {
    if constexpr (is_legacy_program_v<ProgramOrDesc>) {
        tt::tt_metal::CreateCircularBuffer(target, cores, config);
    } else {
        tt::tt_metal::CBDescriptor::FormatDescriptors format_descriptors;
        const auto& data_formats = config.data_formats();
        const auto& page_sizes = config.page_sizes();
        for (uint8_t buffer_index : config.local_buffer_indices()) {
            format_descriptors.push_back(tt::tt_metal::CBFormatDescriptor{
                .buffer_index = buffer_index,
                .data_format = data_formats.at(buffer_index).value(),
                .page_size = page_sizes.at(buffer_index).value(),
            });
        }
        target.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = config.total_size(),
            .core_ranges = cores,
            .format_descriptors = std::move(format_descriptors),
        });
    }
}

template <typename ProgramOrDesc>
uint32_t add_semaphore(
    ProgramOrDesc& target,
    const std::variant<tt::tt_metal::CoreRange, tt::tt_metal::CoreRangeSet>& core_spec,
    uint32_t initial_value) {
    if constexpr (is_legacy_program_v<ProgramOrDesc>) {
        return tt::tt_metal::CreateSemaphore(target, core_spec, initial_value);
    } else {
        const tt::tt_metal::CoreRangeSet cores = std::visit(
            ttsl::overloaded{
                [](const tt::tt_metal::CoreRange& core_range) { return tt::tt_metal::CoreRangeSet(core_range); },
                [](const tt::tt_metal::CoreRangeSet& core_range_set) { return core_range_set.merge_ranges(); },
            },
            core_spec);
        TT_FATAL(!cores.ranges().empty(), "Expecting a non-empty CoreRangeSet");
        // Mirrors tt::tt_metal::NUM_SEMAPHORES (tt_metal/impl/buffers/semaphore.hpp).
        constexpr uint32_t kSemaphoresPerCore = 16;
        std::bitset<kSemaphoresPerCore> used_semaphore_ids;
        for (const auto& core_range : cores.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                    const tt::tt_metal::CoreCoord core(x, y);
                    for (const auto& semaphore : target.semaphores) {
                        if (semaphore.core_type == tt::CoreType::WORKER && semaphore.core_ranges.contains(core)) {
                            used_semaphore_ids.set(semaphore.id);
                        }
                    }
                }
            }
        }
        std::optional<uint32_t> semaphore_id;
        for (uint32_t candidate = 0; candidate < kSemaphoresPerCore; ++candidate) {
            if (!used_semaphore_ids.test(candidate)) {
                semaphore_id = candidate;
                break;
            }
        }
        TT_FATAL(semaphore_id.has_value(), "No available semaphore ID");
        target.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = semaphore_id.value(),
            .core_type = tt::CoreType::WORKER,
            .core_ranges = cores,
            .initial_value = initial_value,
        });
        return semaphore_id.value();
    }
}

template <typename ProgramOrDesc, typename CoreSpec, typename Config>
tt::tt_metal::KernelHandle add_kernel(
    ProgramOrDesc& target, const std::string& kernel_path, const CoreSpec& cores, const Config& config) {
    if constexpr (is_legacy_program_v<ProgramOrDesc>) {
        return tt::tt_metal::CreateKernel(target, kernel_path, cores, config);
    } else {
        tt::tt_metal::KernelDescriptor kernel;
        kernel.kernel_source = kernel_path;
        kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = as_core_range_set(cores);
        kernel.compile_time_args = config.compile_args;
        kernel.defines.reserve(config.defines.size());
        for (const auto& [name, value] : config.defines) {
            kernel.defines.emplace_back(name, value);
        }
        kernel.named_compile_time_args.reserve(config.named_compile_args.size());
        for (const auto& [name, value] : config.named_compile_args) {
            kernel.named_compile_time_args.emplace_back(name, value);
        }
        kernel.compiler_include_paths = config.compiler_include_paths;
        if constexpr (std::is_same_v<std::decay_t<Config>, tt::tt_metal::ComputeConfig>) {
            kernel.opt_level = config.opt_level;
            kernel.config = tt::tt_metal::ComputeConfigDescriptor{
                .math_fidelity = config.math_fidelity,
                .fp32_dest_acc_en = config.fp32_dest_acc_en,
                .dst_full_sync_en = config.dst_full_sync_en,
                .unpack_to_dest_mode = config.unpack_to_dest_mode,
                .bfp8_pack_precise = config.bfp8_pack_precise,
                .math_approx_mode = config.math_approx_mode,
                .enable_trisc2_rvv = config.enable_trisc2_rvv,
            };
        } else if constexpr (std::is_base_of_v<tt::tt_metal::DataMovementConfig, std::decay_t<Config>>) {
            kernel.opt_level = config.opt_level;
            if constexpr (std::is_same_v<std::decay_t<Config>, tt::tt_metal::ReaderDataMovementConfig>) {
                kernel.config = tt::tt_metal::ReaderConfigDescriptor{};
            } else if constexpr (std::is_same_v<std::decay_t<Config>, tt::tt_metal::WriterDataMovementConfig>) {
                kernel.config = tt::tt_metal::WriterConfigDescriptor{};
            } else {
                kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = config.processor,
                    .noc = config.noc,
                    .noc_mode = config.noc_mode,
                };
            }
        }
        const tt::tt_metal::KernelHandle kernel_id = static_cast<tt::tt_metal::KernelHandle>(target.kernels.size());
        target.kernels.push_back(std::move(kernel));
        return kernel_id;
    }
}

template <typename ProgramOrDesc>
void set_runtime_args(
    ProgramOrDesc& target,
    tt::tt_metal::KernelHandle kernel_id,
    std::initializer_list<tt::tt_metal::CoreCoord> cores,
    const std::vector<uint32_t>& runtime_args) {
    TT_FATAL(cores.size() == 1, "reduce_scatter runtime args are set on one core at a time");
    const tt::tt_metal::CoreCoord core = *cores.begin();
    if constexpr (is_legacy_program_v<ProgramOrDesc>) {
        tt::tt_metal::SetRuntimeArgs(target, kernel_id, {core}, runtime_args);
    } else {
        tt::tt_metal::KernelDescriptor::RTArgList args;
        args.append(runtime_args);
        target.kernels[kernel_id].emplace_runtime_args(core, args);
    }
}

void append_reduce_scatter_common_args(
    tt::tt_metal::KernelDescriptor::RTArgList& common_args,
    bool is_ring,
    const std::optional<GlobalSemaphore>& barrier,
    const std::vector<GlobalSemaphore>& semaphores,
    const Tensor& input,
    const Tensor& intermediate,
    const Tensor& output,
    const std::optional<Tensor>& penult) {
    TT_FATAL(input.buffer() != nullptr, "reduce_scatter input buffer is null");
    TT_FATAL(intermediate.buffer() != nullptr, "reduce_scatter intermediate buffer is null");
    TT_FATAL(output.buffer() != nullptr, "reduce_scatter output buffer is null");
    common_args.push_back(input.buffer());
    common_args.push_back(intermediate.buffer());
    common_args.push_back(output.buffer());
    if (penult.has_value()) {
        common_args.push_back(penult->buffer());
    } else {
        common_args.push_back(0u);
    }
    if (barrier.has_value()) {
        common_args.push_back(
            barrier->address());  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
    } else {
        common_args.push_back(0u);
    }
    common_args.push_back(
        semaphores.at(0).address());  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
    if (is_ring) {
        common_args.push_back(
            semaphores.at(1)
                .address());  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
        common_args.push_back(
            semaphores.at(2)
                .address());  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
    } else {
        common_args.push_back(0u);
        common_args.push_back(0u);
    }
}

}  // namespace

template <typename ProgramOrDesc>
auto build_ring_reduce_scatter_program_impl(
    ProgramOrDesc& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const std::optional<Tensor>& penult_intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto* mesh_device = input_tensor.device();
    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;

    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device_coord,
        is_first_chip,
        is_last_chip);

    TT_FATAL(ring_size % 2 == 0, "reduce_scatter_minimal_async ring implementation doesn't support odd ring size");

    bool fuse_op = fused_op_signaler.has_value();
    // Fused with a batched producer, the ring kernels make one traversal per batch so each batch's
    // reduce-scatter overlaps the matmul producing the next; every worker then has to hold a share of
    // every batch, which the page-major split gives and the unit-major split does not.

    // op hyperparams
    // Get worker cores
    // 2 senders per direction (2: forward, backward) per link (num_links)
    // Each sender is reader + compute + writer
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    uint64_t input_data_size_bytes = input_tensor.buffer()->size();
    uint32_t num_workers_per_direction =
        num_workers_per_direction_opt.value_or(ttnn::experimental::ccl::reduce_scatter_default_workers(
            *mesh_device,
            sub_device_id,
            topology,
            input_data_size_bytes,
            num_links,
            ring_size,
            num_directions_per_link,
            num_mux_cores_per_direction_per_link,
            core_grid_offset));
    if (num_workers_per_direction == 1) {
        num_mux_cores_per_direction_per_link = 0;
    }
    uint32_t num_buffers_full_size_channels = num_buffers_per_channel.value_or(1);
    log_trace(tt::LogOp, "DEBUG: num_workers_per_direction: {}", num_workers_per_direction);

    uint32_t num_cores_per_link = ttnn::experimental::ccl::reduce_scatter_core_count_per_link(
        num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [unicast_forward_args, unicast_backward_args] = ccl::get_forward_backward_line_unicast_configuration(
        sender_device_coord, forward_coord, backward_coord, mesh_device);
    auto [mcast_forward_args, mcast_backward_args] = ccl::get_forward_backward_line_mcast_configuration(
        sender_device_coord, forward_coord, backward_coord, ring_size - 1, ring_size - 1, mesh_device);

    const auto [all_core_range, all_cores] =
        choose_worker_cores(num_links, num_cores_per_link, mesh_device, sub_device_id, core_grid_offset);

    const auto mux_connection_valid = [&backward_coord, &forward_coord](const uint32_t dir) {
        return (!dir && backward_coord.has_value()) || (dir && forward_coord.has_value());
    };

    std::vector<CoreRange> sender_worker_core_ranges;
    sender_worker_core_ranges.reserve(num_links * num_directions_per_link * num_workers_per_direction);
    std::vector<CoreRange> mux_core_ranges;
    mux_core_ranges.reserve(num_links * num_directions_per_link);
    if (num_mux_cores_per_direction_per_link) {
        uint32_t core_id = 0;
        for (uint32_t link = 0; link < num_links; link++) {
            for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
                const auto& mux_core = all_cores[core_id++];
                if (mux_connection_valid(dir)) {
                    mux_core_ranges.emplace_back(mux_core);
                }
                for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                    const auto& worker_core = all_cores[core_id++];
                    sender_worker_core_ranges.emplace_back(worker_core);
                }
            }
        }
    } else {
        uint32_t core_id = 0;
        for (uint32_t link = 0; link < num_links; link++) {
            for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
                for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                    const auto& worker_core = all_cores[core_id++];
                    sender_worker_core_ranges.emplace_back(worker_core);
                }
            }
        }
    }

    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet mux_core_range_set = CoreRangeSet(mux_core_ranges);

    // Tensor Info
    const auto& input_tensor_shape = input_tensor.padded_shape();
    TT_FATAL(
        !(input_tensor_shape[-2] % tt::constants::TILE_HEIGHT),
        "Input tensor height ({}) must be divisible by tile height ({}).",
        input_tensor_shape[-2],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        !(input_tensor_shape[-1] % tt::constants::TILE_WIDTH),
        "Input tensor width ({}) must be divisible by tile width ({}).",
        input_tensor_shape[-1],
        tt::constants::TILE_WIDTH);

    const auto [normalized_dim, input_tensor_C, input_tensor_B] =
        (input_tensor_shape.rank() == 2)
            ? ttnn::experimental::ccl::reduce_scatter_map_2d_to_4d(dim)
            : ttnn::experimental::ccl::reduce_scatter_map_nd_to_4d(input_tensor_shape, dim);
    const uint32_t input_tensor_Ht = input_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t input_tensor_Wt = input_tensor_shape[-1] / tt::constants::TILE_WIDTH;

    uint32_t slice_B = input_tensor_B;
    uint32_t slice_C = input_tensor_C;
    uint32_t slice_Ht = input_tensor_Ht;
    uint32_t slice_Wt = input_tensor_Wt;
    if (normalized_dim == 0) {
        slice_B /= ring_size;
    } else if (normalized_dim == 1) {
        slice_C /= ring_size;
    } else if (normalized_dim == 2) {
        slice_Ht /= ring_size;
    } else if (normalized_dim == 3) {
        slice_Wt /= ring_size;
    } else {
        TT_FATAL(
            false, "reduce_scatter_minimal_async ring implementation only supports scattering on dim 0, 1, 2, or 3");
    }

    TT_FATAL(
        !(fuse_op && normalized_dim == 0),
        "reduce_scatter_minimal_async ring implementation can't be fused with matmul when scattering on dim 0");

    const uint32_t input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const uint32_t output_tensor_num_pages = input_tensor_num_pages / ring_size;
    const uint32_t input_batch_num_pages = input_tensor_num_pages / input_tensor_B;
    const bool per_batch_traversals = fuse_op && input_tensor_B > 1;
    const uint32_t output_batch_num_pages = output_tensor_num_pages / slice_B;
    const uint32_t input_channel_num_pages = input_batch_num_pages / input_tensor_C;
    const uint32_t output_channel_num_pages = output_batch_num_pages / slice_C;

    // Extract compute kernel config parameters
    const bool fp32_dest_acc_en = ttnn::get_fp32_dest_acc_en(compute_kernel_config);
    const tt::tt_metal::MathFidelity math_fidelity = compute_kernel_config.has_value()
                                                         ? ttnn::get_math_fidelity(compute_kernel_config)
                                                         : tt::tt_metal::MathFidelity::HiFi4;
    // Hardware constraint: FP32 destination accumulator can only hold 4 tiles vs 8 for FP16
    const uint32_t max_dst_size = fp32_dest_acc_en ? 4 : 8;

    // scatter-write currently only supports 4 distinct noc addresses
    uint32_t max_target_noc_addresses_per_packet = 4;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
    uint32_t tile_granularity = std::min(4 * num_tiles_to_write_per_packet, max_dst_size);
    uint32_t cb_num_pages = 2 * tile_granularity;  // double buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    // Contiguous fast path: when enabled, the intermediate is a chunk-paged row-major staging tensor
    // and the writer sends whole chunks with fused-unicast writes instead of scatter. The sizing must
    // match compute_output_specs exactly, so both derive it from the same helper. Which layout applies
    // is read off the intermediate tensor actually in use (either caller-provided or the one
    // compute_output_specs sized), so this cannot disagree with what was allocated. See
    // rs-contiguous-interm-design.
    const auto staging = ttnn::experimental::ccl::reduce_scatter_ring_interm_staging_params(
        input_tensor, topology, dim, ring_size, fp32_dest_acc_en);
    const bool use_contiguous_interm = ttnn::experimental::ccl::reduce_scatter_use_contiguous_interm(
        input_tensor, intermediate_tensor, topology, dim, ring_size, fp32_dest_acc_en);
    if (use_contiguous_interm) {
        TT_FATAL(
            staging.tile_granularity == tile_granularity,
            "contiguous-interm tile_granularity mismatch: {} vs {}",
            staging.tile_granularity,
            tile_granularity);
        const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
        TT_FATAL(
            staging.page_bytes % dram_alignment == 0,
            "contiguous-interm page bytes ({}) must be a multiple of DRAM alignment ({}); otherwise the "
            "device page stride (aligned_page_size) would not match chunk_id*page_bytes addressing",
            staging.page_bytes,
            dram_alignment);
    } else if (input_tensor_B > 1) {
        // The tiled staging layout gives every batch its own region at b * input_batch_num_pages, and all
        // batches are in flight within one ring step, so the intermediate has to hold the whole input. The
        // shared-region layout this replaced only needed one batch's worth, and a caller-provided buffer
        // sized that way (the fused matmul + reduce-scatter test did this) is otherwise overrun silently
        // into whatever follows it in DRAM. The public op already rejects such a buffer in validate; this
        // covers callers that reach the builder directly, such as matmul_reduce_scatter_async.
        TT_FATAL(
            intermediate_tensor.buffer()->num_pages() >= input_tensor_num_pages,
            "reduce_scatter_minimal_async: the tiled intermediate must hold the whole input ({} pages) so that "
            "each of the {} batches can stage into its own region; got {} pages. Allocate it with the input "
            "tensor's shape.",
            input_tensor_num_pages,
            input_tensor_B,
            intermediate_tensor.buffer()->num_pages());
    }

    // input_tensor from reader -> compute
    uint32_t input_cb_index = tt::CB::c_in0;
    add_cb(
        program,
        sender_worker_core_range_set,
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes));
    // interm_tensor from reader -> compute
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    add_cb(
        program,
        sender_worker_core_range_set,
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes));
    // output_tensor from reader -> compute
    uint32_t intermediate_2_cb_index = tt::CB::c_in2;
    add_cb(
        program,
        sender_worker_core_range_set,
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_2_cb_index, df}})
            .set_page_size(intermediate_2_cb_index, l1_scratch_cb_page_size_bytes));
    // input_tensor from reader -> writer
    uint32_t reader_output_cb_index = tt::CB::c_in3;
    add_cb(
        program,
        sender_worker_core_range_set,
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes));
    // reduced tensor from compute -> writer
    uint32_t compute_output_cb_index = tt::CB::c_in4;
    add_cb(
        program,
        sender_worker_core_range_set,
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes));

    std::map<std::string, std::string> writer_defines;
    if (num_mux_cores_per_direction_per_link) {
        writer_defines["USE_WORKER_MUX"] = "1";
    }

    // KERNEL CREATION
    if (fuse_op) {
        if constexpr (is_legacy_program_v<ProgramOrDesc>) {
            fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
        } else {
            TT_FATAL(false, "fused reduce-scatter is not supported on the descriptor factory");
        }
    }

    // Kernel Runtime Args
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    const auto buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    // Transient Fabric Mux V2: one mux core per (link, direction), serving num_workers_per_direction
    // client channels (one channel per sender worker). The V2 mux auto-terminates once all of its
    // clients close their connections, so no explicit termination signalling is required. The mux
    // kernels themselves are created per-core in the loop below via add_fabric_mux_v2_to_program (each
    // needs the src/dst fabric node ids + link for the direction it forwards to).
    // The mux stays below the floor of the L1_SMALL region, where carried semaphores live (#56769).
    const size_t mux_l1_small_floor_address = ttnn::ccl::l1_small_floor_address(*mesh_device);
    tt::tt_fabric::FabricMuxV2Config mux_config(
        *mesh_device,
        static_cast<uint8_t>(num_workers_per_direction),
        static_cast<uint8_t>(num_buffers_full_size_channels),
        buffer_size_bytes_full_size_channel,
        mux_base_l1_address,
        mux_l1_small_floor_address);

    auto reader_named_compile_args = operations::experimental::ccl::detail::get_ring_reader_named_compile_args(
        ring_index,
        ring_size,
        input_cb_index,
        intermediate_cb_index,
        intermediate_2_cb_index,
        reader_output_cb_index,
        tile_granularity,
        page_size,
        output_tensor_num_pages,
        input_batch_num_pages,
        output_batch_num_pages,
        input_channel_num_pages,
        output_channel_num_pages,
        input_tensor_B,
        input_tensor_Wt,
        slice_B,
        slice_C,
        slice_Ht,
        slice_Wt,
        fuse_op,
        normalized_dim);
    if (normalized_dim != 0) {
        // Staging-layout switch consumed by the unified ring reader. chunks_per_channel is only read
        // by the chunk-paged branch, but the named arg must always be present for the kernel to
        // compile, so it is passed unconditionally.
        reader_named_compile_args["contiguous_interm"] = use_contiguous_interm ? 1 : 0;
        reader_named_compile_args["chunks_per_channel"] = staging.chunks_per_channel;
    }

    // Positional args: TensorAccessorArgs
    std::vector<uint32_t> reader_compile_args;
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);
    if (normalized_dim != 0) {
        TT_FATAL(
            !use_contiguous_interm || penult_intermediate_tensor.has_value(),
            "contiguous-interm path requires a penult intermediate staging tensor");
        // Penult intermediate staging accessor. Only the chunk-paged branch reads it; the tiled branch gets the
        // output tensor's args as a placeholder (paired with a null address below) so that both
        // branches share one compile-time/runtime arg layout.
        const auto& penult_intermediate_accessor_source =
            use_contiguous_interm ? *penult_intermediate_tensor : output_tensor;
        tt::tt_metal::TensorAccessorArgs(penult_intermediate_accessor_source.buffer()).append_to(reader_compile_args);
    }

    std::string reader_kernel_path = normalized_dim == 0
                                         ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                           "device/kernels/dim_zero_ring_reduce_scatter_minimal_async_reader.cpp"
                                         : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                           "device/kernels/ring_reduce_scatter_minimal_async_reader.cpp";

    auto reader_kernel_id = add_kernel(
        program,
        reader_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args, {}, reader_named_compile_args));

    // Writer
    auto writer_named_compile_args = operations::experimental::ccl::detail::get_ring_writer_named_compile_args(
        ring_index,
        ring_size,
        compute_output_cb_index,
        reader_output_cb_index,
        tile_granularity,
        page_size,
        num_tiles_to_write_per_packet,
        output_tensor_num_pages,
        output_batch_num_pages,
        input_batch_num_pages,
        input_channel_num_pages,
        output_channel_num_pages,
        input_tensor_B,
        input_tensor_Wt,
        slice_B,
        slice_C,
        slice_Ht,
        slice_Wt,
        normalized_dim,
        fuse_op);
    if (normalized_dim != 0) {
        // Staging-layout switch consumed by the unified ring writer. The chunk-paged sizing args are
        // only read by that branch, but must always be present for the kernel to compile.
        writer_named_compile_args["contiguous_interm"] = use_contiguous_interm ? 1 : 0;
        writer_named_compile_args["chunks_per_channel"] = staging.chunks_per_channel;
        writer_named_compile_args["interm_tiles_per_packet"] = staging.interm_tiles_per_packet;
    }

    // Positional args: routing info, TensorAccessorArgs. The V2 fabric mux client needs no worker-side
    // compile-time args (the device-side FabricMuxV2Sender is built entirely from runtime args).
    std::vector<uint32_t> writer_compile_args;
    // Each TensorAccessorArgs always emits args_config and aligned_page_size; a sharded accessor reserves the
    // extra space for its own shape/bank words when appending.
    constexpr uint32_t min_args_per_tensor_accessor = 2;
    writer_compile_args.reserve(
        unicast_forward_args.size() + mcast_forward_args.size() + unicast_backward_args.size() +
        mcast_backward_args.size() + 2 * min_args_per_tensor_accessor);

    writer_compile_args.insert(writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());
    tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);
    if (normalized_dim != 0) {
        // See the reader above: placeholder accessor args keep one arg layout for both branches.
        const auto& penult_intermediate_accessor_source =
            use_contiguous_interm ? *penult_intermediate_tensor : output_tensor;
        tt::tt_metal::TensorAccessorArgs(penult_intermediate_accessor_source.buffer()).append_to(writer_compile_args);
    }

    std::string writer_kernel_path = normalized_dim == 0
                                         ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                           "device/kernels/dim_zero_ring_reduce_scatter_minimal_async_writer.cpp"
                                         : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                           "device/kernels/ring_reduce_scatter_minimal_async_writer.cpp";

    auto writer_kernel_id = add_kernel(
        program,
        writer_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args, writer_defines, writer_named_compile_args));

    // Compute kernel
    auto compute_kernel_config_ = tt::tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .named_compile_args = operations::experimental::ccl::detail::get_ring_compute_named_compile_args(
            input_cb_index,
            intermediate_cb_index,
            intermediate_2_cb_index,
            compute_output_cb_index,
            tile_granularity,
            ring_size,
            input_tensor_B,
            slice_B,
            slice_C,
            normalized_dim,
            fuse_op)};

    std::string compute_kernel_path = normalized_dim == 0
                                          ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                            "device/kernels/dim_zero_ring_reduction.cpp"
                                          : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                                            "device/kernels/ring_reduction.cpp";

    auto compute_kernel_id =
        add_kernel(program, compute_kernel_path, sender_worker_core_range_set, compute_kernel_config_);

    auto worker_core_iter = sender_worker_core_range_set.ranges().cbegin();
    auto mux_core_iter = mux_core_range_set.ranges().cbegin();
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            CoreCoord mux_virtual_core = CoreCoord{0, 0};
            if (mux_connection_valid(dir) && num_mux_cores_per_direction_per_link) {
                auto mux_logical_core = *((mux_core_iter++)->begin());
                mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);

                // Create the V2 mux kernel on this core, forwarding toward this direction's neighbor
                // (forward_coord when dir==1, backward_coord when dir==0).
                const auto src_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
                const auto dst_node_id =
                    mesh_device->get_fabric_node_id(dir ? forward_coord.value() : backward_coord.value());
                tt::tt_fabric::add_fabric_mux_v2_to_program(
                    program, mux_config, mux_logical_core, src_node_id, dst_node_id, link);
            }

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                auto core = *((worker_core_iter++)->begin());
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

                // FWD core needs BWD core coordinate (and vice-versa) to send sem incrs in 2nd last iter
                uint32_t opposite_mux_core_offset =
                    (link * num_cores_per_link) +
                    ((1 - dir) * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
                uint32_t opposite_core_idx = opposite_mux_core_offset + num_mux_cores_per_direction_per_link + worker;
                auto opposite_core = all_cores[opposite_core_idx];
                auto opposite_core_coord = mesh_device->worker_core_from_logical_core(opposite_core);

                uint32_t worker_id = (link * num_workers_per_direction) + worker;
                uint32_t num_workers = num_links * num_workers_per_direction;

                const auto
                    [unit_start,
                     unit_end,
                     start_tiles_read,
                     start_tiles_to_read,
                     start_pages_read_in_row,
                     start_row_offset] =
                        ttnn::experimental::ccl::reduce_scatter_get_worker_split(
                            worker_id,
                            num_workers,
                            input_tensor_B,
                            slice_C,
                            /*allow_unit_major=*/!per_batch_traversals,
                            output_batch_num_pages,
                            output_channel_num_pages,
                            slice_Wt,
                            input_tensor_Wt,
                            normalized_dim);

                // for dim 0 scatters we process each slice in batches
                // for all other dims we process each slice in the (batch, channel) units this worker owns,
                // all of them inside every ring step -- or, fused with a batched producer, one batch's units
                // per traversal
                uint32_t tiles_per_worker_per_repeat = start_tiles_to_read - start_tiles_read;
                uint32_t num_repeats =
                    (normalized_dim == 0) ? slice_B : (per_batch_traversals ? slice_C : (unit_end - unit_start));
                uint32_t chunks_per_sync_val =
                    chunks_per_sync.value_or(ttnn::experimental::ccl::reduce_scatter_default_chunks_per_sync(
                        topology, tiles_per_worker_per_repeat, num_repeats, tile_granularity));
                if (!chunks_per_sync.has_value() && normalized_dim != 0 && !fuse_op) {
                    // The dims 1-3 kernels carry the worker's whole share of the slice per step; see the
                    // constants' comments for why their default interval is capped, why a short step syncs
                    // on every chunk, and why dim 0 and the fused path are exempt.
                    const uint32_t chunks_per_step = ttnn::experimental::ccl::reduce_scatter_chunks_per_step(
                        tiles_per_worker_per_repeat, num_repeats, tile_granularity);
                    chunks_per_sync_val =
                        chunks_per_step <= ttnn::experimental::ccl::RING_UNIT_STEP_SHORT_STEP_CHUNKS
                            ? 1
                            : std::min(
                                  chunks_per_sync_val, ttnn::experimental::ccl::RING_UNIT_STEP_MAX_CHUNKS_PER_SYNC);
                }
                log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

                std::vector<uint32_t> reader_rt_args;
                if (normalized_dim == 0) {
                    reader_rt_args = {
                        dir,                  // direction
                        chunks_per_sync_val,  // chunks_per_sync
                        start_tiles_read,     // start_tiles_read
                        start_tiles_to_read,  // start_tiles_to_read
                    };
                } else {
                    reader_rt_args = {
                        dir,                      // direction
                        chunks_per_sync_val,      // chunks_per_sync
                        start_tiles_read,         // start_tiles_read
                        start_tiles_to_read,      // start_tiles_to_read
                        start_pages_read_in_row,  // start_pages_read_in_row
                        start_row_offset,         // start_row_offset
                        unit_start,               // unit_start
                        unit_end,                 // unit_end
                    };
                }
                if (fuse_op) {
                    fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }

                set_runtime_args(program, reader_kernel_id, {core}, reader_rt_args);

                // Writer RT args
                std::vector<uint32_t> writer_rt_args;
                if (normalized_dim == 0) {
                    writer_rt_args = {
                        virtual_core.x,                                              // this core.x
                        virtual_core.y,                                              // this core.y
                        opposite_core_coord.x,                                       // opposite direction core.x
                        opposite_core_coord.y,                                       // opposite direction core.y
                        barrier_semaphore.has_value() && !using_persistent_buffers,  // use_barrier_sem
                        dir,                                                         // direction
                        chunks_per_sync_val,                                         // chunks_per_sync
                        start_tiles_read,                                            // start_tiles_read
                        start_tiles_to_read,                                         // tiles_to_read
                    };
                } else {
                    writer_rt_args = {
                        virtual_core.x,                                              // this core.x
                        virtual_core.y,                                              // this core.y
                        opposite_core_coord.x,                                       // opposite direction core.x
                        opposite_core_coord.y,                                       // opposite direction core.y
                        barrier_semaphore.has_value() && !using_persistent_buffers,  // use_barrier_sem
                        dir,                                                         // direction
                        chunks_per_sync_val,                                         // chunks_per_sync
                        start_pages_read_in_row,                                     // start_pages_read_in_row
                        start_row_offset,                                            // start_row_offset
                        start_tiles_read,                                            // start_tiles_read
                        start_tiles_to_read,                                         // tiles_to_read
                        unit_start,                                                  // unit_start
                        unit_end,                                                    // unit_end
                    };
                }
                if (num_mux_cores_per_direction_per_link) {
                    // V2 fabric mux client connection: this worker is channel `worker` on its
                    // direction's mux. Two per-worker semaphores (flow control + teardown) back the
                    // connection; append_client_connection_rt_args serializes exactly the client args
                    // the device-side FabricMuxV2Sender::build_from_args expects.
                    const auto flow_control_sem_id = add_semaphore(program, core, 0);
                    const auto teardown_sem_id = add_semaphore(program, core, 0);
                    mux_config.append_client_connection_rt_args(
                        mux_virtual_core,
                        static_cast<uint8_t>(worker),
                        tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{
                            .flow_control_sem_id = flow_control_sem_id,
                            .teardown_sem_id = teardown_sem_id,
                        },
                        writer_rt_args);
                }
                if (!num_mux_cores_per_direction_per_link) {
                    if (dir) {  // forward
                        writer_rt_args.push_back(forward_coord.has_value());
                        if (forward_coord.has_value()) {
                            const auto src_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
                            const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                            tt::tt_fabric::append_fabric_connection_rt_args(
                                src_fabric_node_id, dst_fabric_node_id, link, program, {core}, writer_rt_args);
                        }
                        writer_rt_args.push_back(false);
                    } else {
                        writer_rt_args.push_back(false);
                        writer_rt_args.push_back(backward_coord.has_value());
                        if (backward_coord.has_value()) {
                            const auto src_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
                            const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                            tt::tt_fabric::append_fabric_connection_rt_args(
                                src_fabric_node_id, dst_fabric_node_id, link, program, {core}, writer_rt_args);
                        }
                    }
                }
                set_runtime_args(program, writer_kernel_id, {core}, writer_rt_args);

                // Shared by both compute kernels. dim_zero_ring_reduction.cpp has no unit loop and
                // stops reading after dir, leaving the two trailing values unread; the split helper
                // still reports the full span for dim 0, so nothing depends on them there.
                std::vector<uint32_t> compute_rt_args = {
                    start_tiles_read,     // start_tiles_read
                    start_tiles_to_read,  // start_tiles_to_read
                    dir,                  // dir
                    unit_start,           // unit_start
                    unit_end};            // unit_end
                set_runtime_args(program, compute_kernel_id, {core}, compute_rt_args);
            }
        }
    }

    // Common bindings: input, intermediate, output, penult, barrier, sem0, sem1, ack.
    if constexpr (is_legacy_program_v<ProgramOrDesc>) {
        const auto common_args = ReduceScatterProgramArtifacts::collect_runtime_args(
            true,
            barrier_semaphore,
            semaphore,
            input_tensor,
            intermediate_tensor,
            output_tensor,
            penult_intermediate_tensor);
        tt::tt_metal::SetCommonRuntimeArgs(program, reader_kernel_id, common_args);
        tt::tt_metal::SetCommonRuntimeArgs(program, writer_kernel_id, common_args);
        return ReduceScatterProgramArtifacts{
            GetCommonRuntimeArgs(program, reader_kernel_id), GetCommonRuntimeArgs(program, writer_kernel_id)};
    } else {
        tt::tt_metal::KernelDescriptor::RTArgList common_args;
        append_reduce_scatter_common_args(
            common_args,
            true,
            barrier_semaphore,
            semaphore,
            input_tensor,
            intermediate_tensor,
            output_tensor,
            penult_intermediate_tensor);
        program.kernels[reader_kernel_id].emplace_common_runtime_args(common_args);
        program.kernels[writer_kernel_id].emplace_common_runtime_args(common_args);
        return;
    }
}

ReduceScatterProgramArtifacts build_ring_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const std::optional<Tensor>& penult_intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return build_ring_reduce_scatter_program_impl(
        program,
        input_tensor,
        intermediate_tensor,
        penult_intermediate_tensor,
        sender_device_coord,
        forward_coord,
        backward_coord,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        fused_op_signaler,
        chunks_per_sync,
        num_workers_per_direction_opt,
        num_buffers_per_channel,
        core_grid_offset,
        compute_kernel_config);
}

void build_ring_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::ProgramDescriptor& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const std::optional<Tensor>& penult_intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    build_ring_reduce_scatter_program_impl(
        program,
        input_tensor,
        intermediate_tensor,
        penult_intermediate_tensor,
        sender_device_coord,
        forward_coord,
        backward_coord,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        fused_op_signaler,
        chunks_per_sync,
        num_workers_per_direction_opt,
        num_buffers_per_channel,
        core_grid_offset,
        compute_kernel_config);
}

template <typename ProgramOrDesc>
auto build_line_reduce_scatter_program_impl(
    ProgramOrDesc& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    // Unused (Line never takes the Ring-only contiguous penult intermediate path); present only for call-signature
    // parity with build_ring_reduce_scatter_minimal_async_program_artifacts.
    [[maybe_unused]] const std::optional<Tensor>& penult_intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    /**
     * Line Reduce Scatter
     *
     *   IN 0     IN 1     IN 2     IN 3            OUT 0    OUT 1    OUT 2    OUT 3
     *   C0       C1       C2       C3              C0       C1       C2       C3
     *  ┌────┐   ┌────┐   ┌────┐   ┌────┐          ┌────┐   ......   ......   ......
     *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
     *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
     *  │    │   │    │   │    │   │    │          │////│   .    .   .    .   .    .
     *  ├────┤   ├────┤   ├────┤   ├────┤          └────┘   ┌────┐   ......   ......
     *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
     *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
     *  │    │   │    │   │    │   │    │          .    .   │////│   .    .   .    .
     *  ├────┤   ├────┤   ├────┤   ├────┤  ────►   ......   └────┘   ┌────┐   ......
     *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
     *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
     *  │    │   │    │   │    │   │    │          .    .   .    .   │////│   .    .
     *  ├────┤   ├────┤   ├────┤   ├────┤          ......   ......   └────┘   ┌────┐
     *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
     *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
     *  │    │   │    │   │    │   │    │          .    .   .    .   .    .   │////│
     *  └────┘   └────┘   └────┘   └────┘          ......   ......   ......   └────┘
     *
     *
     * There are (ring_size - 1) algorithmic steps in Line Reduce Scatter.
     * Each device must send (num_forward_targets) partials forward and
     * (num_backward_targets) partials backward.
     *
     * On each step, a device will:
     * - if first device in a direction, send a slice in that direction
     * - otherwise, receive a slice, locally reduce it, and send the result in that direction
     *
     */
    auto* mesh_device = input_tensor.device();
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;

    // op hyperparams
    // Get worker cores
    // 2 senders (reader + core + writer) per direction (forward, backward) per link
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    uint64_t input_data_size_bytes = input_tensor.buffer()->size();
    uint32_t num_workers_per_direction =
        num_workers_per_direction_opt.value_or(ttnn::experimental::ccl::reduce_scatter_default_workers(
            *mesh_device,
            sub_device_id,
            topology,
            input_data_size_bytes,
            num_links,
            ring_size,
            num_directions_per_link,
            num_mux_cores_per_direction_per_link,
            core_grid_offset));
    log_trace(tt::LogOp, "DEBUG: num_workers_per_direction: {}", num_workers_per_direction);
    uint32_t num_buffers_full_size_channels = num_buffers_per_channel.value_or(1);

    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device_coord,
        is_first_chip,
        is_last_chip);

    bool fuse_op = fused_op_signaler.has_value();

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    auto [unicast_forward_args, unicast_backward_args] = ccl::get_forward_backward_line_unicast_configuration(
        sender_device_coord, forward_coord, backward_coord, mesh_device);
    auto [num_targets_forward, num_targets_backward] =
        ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, true);
    const bool use_fabric_2d_neighbor_barrier =
        topology == ccl::Topology::Linear && tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig());
    // A logical line embedded in a 2D fabric can turn at an intermediate device. A single line
    // multicast range follows the physical direction selected by its first hop, so it cannot
    // represent such a turn. The reduce-scatter data path already relays through immediate logical
    // neighbors; use the same one-hop connectivity for its startup barrier on Fabric2D.
    const uint32_t barrier_targets_forward =
        use_fabric_2d_neighbor_barrier ? std::min(num_targets_forward, 1u) : num_targets_forward;
    const uint32_t barrier_targets_backward =
        use_fabric_2d_neighbor_barrier ? std::min(num_targets_backward, 1u) : num_targets_backward;
    auto [mcast_forward_args, mcast_backward_args] = ccl::get_forward_backward_line_mcast_configuration(
        sender_device_coord,
        forward_coord,
        backward_coord,
        barrier_targets_forward,
        barrier_targets_backward,
        mesh_device);

    uint32_t num_cores_per_link = ttnn::experimental::ccl::reduce_scatter_core_count_per_link(
        num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    const auto [all_core_range, all_cores] =
        choose_worker_cores(num_links, num_cores_per_link, mesh_device, sub_device_id, core_grid_offset);

    const auto mux_connection_valid = [&backward_coord, &forward_coord](const uint32_t dir) {
        return (!dir && backward_coord.has_value()) || (dir && forward_coord.has_value());
    };

    std::vector<CoreRange> sender_worker_core_ranges;
    sender_worker_core_ranges.reserve(num_links * num_directions_per_link * num_workers_per_direction);
    std::vector<CoreRange> mux_core_ranges;
    mux_core_ranges.reserve(num_links * num_directions_per_link);
    std::vector<CoreRange> termination_master_core_ranges;
    termination_master_core_ranges.reserve(num_links * num_directions_per_link);
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            if (mux_connection_valid(dir)) {
                mux_core_ranges.emplace_back(mux_core);
            }
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];
                sender_worker_core_ranges.emplace_back(worker_core);
                if (worker == 0) {
                    termination_master_core_ranges.emplace_back(worker_core);
                }
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet mux_core_range_set = CoreRangeSet(mux_core_ranges);

    // Extract compute kernel config parameters
    const bool fp32_dest_acc_en = ttnn::get_fp32_dest_acc_en(compute_kernel_config);
    const tt::tt_metal::MathFidelity math_fidelity = compute_kernel_config.has_value()
                                                         ? ttnn::get_math_fidelity(compute_kernel_config)
                                                         : tt::tt_metal::MathFidelity::HiFi4;
    // Hardware constraint: FP32 destination accumulator can only hold 4 tiles vs 8 for FP16
    const uint32_t max_dst_size = fp32_dest_acc_en ? 4 : 8;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t max_scatter_write_pages = 2;
    uint32_t tiles_to_write_per_packet = std::min(num_pages_per_packet, max_scatter_write_pages);
    uint32_t tile_granularity = std::min(4 * num_pages_per_packet, max_dst_size);
    uint32_t cb_num_pages = 3 * tile_granularity;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes);
    add_cb(program, sender_worker_core_range_set, cb_input_config);
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes);
    add_cb(program, sender_worker_core_range_set, cb_intermediate_config);
    uint32_t reader_output_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_reader_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes);
    add_cb(program, sender_worker_core_range_set, cb_reader_output_config);
    uint32_t compute_output_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes);
    add_cb(program, sender_worker_core_range_set, cb_compute_output_config);

    // Tensor Info
    const auto& input_tensor_shape = input_tensor.padded_shape();
    TT_FATAL(
        !(input_tensor_shape[-2] % tt::constants::TILE_HEIGHT),
        "Input tensor height ({}) must be divisible by tile height ({}).",
        input_tensor_shape[-2],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        !(input_tensor_shape[-1] % tt::constants::TILE_WIDTH),
        "Input tensor width ({}) must be divisible by tile width ({}).",
        input_tensor_shape[-1],
        tt::constants::TILE_WIDTH);

    const auto [normalized_dim, input_tensor_C, input_tensor_B] =
        (input_tensor_shape.rank() == 2)
            ? ttnn::experimental::ccl::reduce_scatter_map_2d_to_4d(dim)
            : ttnn::experimental::ccl::reduce_scatter_map_nd_to_4d(input_tensor_shape, dim);
    const uint32_t input_tensor_Ht = input_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t input_tensor_Wt = input_tensor_shape[-1] / tt::constants::TILE_WIDTH;

    uint32_t slice_B = input_tensor_B;
    uint32_t slice_C = input_tensor_C;
    uint32_t slice_Ht = input_tensor_Ht;
    uint32_t slice_Wt = input_tensor_Wt;
    if (normalized_dim == 0) {
        slice_B /= ring_size;
    } else if (normalized_dim == 1) {
        slice_C /= ring_size;
    } else if (normalized_dim == 2) {
        slice_Ht /= ring_size;
    } else if (normalized_dim == 3) {
        slice_Wt /= ring_size;
    } else {
        TT_FATAL(
            false, "reduce_scatter_minimal_async line implementation only supports scattering on dim 0, 1, 2, or 3");
    }

    TT_FATAL(
        !(fuse_op && normalized_dim == 0),
        "reduce_scatter_minimal_async line implementation can't be fused with matmul when scattering on dim 0");

    const uint32_t input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const uint32_t output_tensor_num_pages = input_tensor_num_pages / ring_size;
    const uint32_t input_batch_num_pages = input_tensor_num_pages / input_tensor_B;
    const uint32_t output_batch_num_pages = output_tensor_num_pages / slice_B;
    const uint32_t input_channel_num_pages = input_batch_num_pages / input_tensor_C;
    const uint32_t output_channel_num_pages = output_batch_num_pages / slice_C;

    bool input_is_sharded = input_tensor.is_sharded();
    bool intermediate_is_sharded = intermediate_tensor.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;

    if (input_is_sharded) {
        reader_compute_defines["INPUT_IS_SHARDED"] = "1";
    }
    if (intermediate_is_sharded) {
        reader_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
        writer_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        reader_compute_defines["OUTPUT_IS_SHARDED"] = "1";
        writer_compute_defines["OUTPUT_IS_SHARDED"] = "1";
    }
    // KERNEL CREATION
    if (fuse_op) {
        if constexpr (is_legacy_program_v<ProgramOrDesc>) {
            fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
        } else {
            TT_FATAL(false, "fused reduce-scatter is not supported on the descriptor factory");
        }
    }

    // Common Kernel Runtime Args
    const bool sync_with_other_direction = !(is_first_chip || is_last_chip);
    uint32_t fwd_bwd_semaphore_address = add_semaphore(program, sender_worker_core_range_set, 0);
    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;

    const auto num_full_size_channels = num_workers_per_direction;
    const auto num_header_only_channels = 0;
    const auto buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    // Fabric mux kernel
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        0,
        buffer_size_bytes_full_size_channel,
        mux_base_l1_address);

    // mux kernel
    auto mux_kernel_id = add_kernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    // Reader
    std::vector<uint32_t> sender_reader_compile_args =
        operations::experimental::ccl::detail::get_line_reader_compile_args(
            ring_index,
            ring_size,
            input_cb_index,
            intermediate_cb_index,
            reader_output_cb_index,
            tile_granularity,
            page_size,
            input_tensor_num_pages,
            output_tensor_num_pages,
            input_batch_num_pages,
            input_channel_num_pages,
            output_batch_num_pages,
            output_channel_num_pages,
            input_tensor_B,
            input_tensor_Wt,
            slice_B,
            slice_C,
            slice_Ht,
            slice_Wt,
            fuse_op,
            sync_with_other_direction,
            normalized_dim);

    if (input_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(input_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
    }
    if (intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_reader_compile_args);
    }
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_reader_compile_args);
    }

    std::string sender_reader_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_line_reduce_scatter_minimal_async_reader.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/line_reduce_scatter_minimal_async_reader.cpp";

    auto reader_kernel_id = add_kernel(
        program,
        sender_reader_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args, reader_compute_defines));

    // Writer
    std::vector<uint32_t> sender_writer_compile_args =
        operations::experimental::ccl::detail::get_line_writer_compile_args(
            ring_size,
            compute_output_cb_index,
            reader_output_cb_index,
            tile_granularity,
            page_size,
            tiles_to_write_per_packet,
            input_tensor_num_pages,
            output_tensor_num_pages,
            input_batch_num_pages,
            input_channel_num_pages,
            output_batch_num_pages,
            output_channel_num_pages,
            input_tensor_B,
            input_tensor_Wt,
            slice_B,
            slice_C,
            slice_Ht,
            slice_Wt,
            normalized_dim,
            sync_with_other_direction);

    append_fabric_mux_connection_ct_args(
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        num_workers_per_direction,
        sender_writer_compile_args);

    sender_writer_compile_args.push_back(
        use_fabric_2d_neighbor_barrier ? static_cast<uint32_t>((num_targets_forward != 0) + (num_targets_backward != 0))
                                       : ring_size - 1);  // barrier_target_count

    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
    sender_writer_compile_args.insert(
        sender_writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());

    if (intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_writer_compile_args);
    }
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_writer_compile_args);
    }

    std::string sender_writer_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_line_reduce_scatter_minimal_async_writer.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/line_reduce_scatter_minimal_async_writer.cpp";

    auto writer_kernel_id = add_kernel(
        program,
        sender_writer_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));

    // Reduce kernel
    auto sender_reduce_kernel_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .compile_args = operations::experimental::ccl::detail::get_line_reduce_compile_args(
            input_cb_index,
            intermediate_cb_index,
            compute_output_cb_index,
            tile_granularity,
            input_tensor_B,
            slice_B,
            slice_C,
            normalized_dim)};

    std::string sender_reduce_kernel_path =
        normalized_dim == 0 ? "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/dim_zero_line_reduction.cpp"
                            : "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/"
                              "device/kernels/line_reduction.cpp";

    auto reduce_kernel_id =
        add_kernel(program, sender_reduce_kernel_path, sender_worker_core_range_set, sender_reduce_kernel_config);

    auto worker_core_iter = sender_worker_core_range_set.ranges().cbegin();
    auto mux_core_iter = mux_core_range_set.ranges().cbegin();
    auto termination_master_core_iter = termination_master_core_ranges.cbegin();
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const bool is_forward = dir;
            CoreCoord mux_virtual_core = {0, 0};
            if (mux_connection_valid(dir)) {
                auto mux_logical_core = *((mux_core_iter++)->begin());
                mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);
                std::vector<uint32_t> mux_rt_args = {};
                const auto src_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
                if (dir) {  // forward
                    const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                } else {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                }
                set_runtime_args(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
            }

            auto termination_master_logical_core = *((termination_master_core_iter++)->begin());
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                auto core = *((worker_core_iter++)->begin());
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

                // FWD core needs BWD core coordinate for fwd/bwd final reduction sync.
                // For final synchronization, each core needs to know the coordinate of the opposite direction's core.
                uint32_t opposite_mux_core_offset =
                    (link * num_cores_per_link) +
                    ((1 - dir) * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
                uint32_t opposite_core_idx = opposite_mux_core_offset + num_mux_cores_per_direction_per_link + worker;
                auto opposite_core = all_cores[opposite_core_idx];
                auto opposite_core_coord = mesh_device->worker_core_from_logical_core(opposite_core);

                /**
                 * Every chip has a final reduction step. On the ends, there is only one input to reduce.
                 * In the middle, you must reduce from both input directions.
                 * FWD/BWD readers need to synchronize in order to avoid race conditions.
                 *
                 * We'll say that FWD always leads, then signals BWD to follow.
                 */
                const bool is_first_device_in_direction = is_forward ? is_first_chip : is_last_chip;
                const int num_targets_in_direction = is_forward ? num_targets_forward : num_targets_backward;
                // The number of reduction steps is 0 for chips on the ends, or num_targets_in_direction for chips in
                // the middle
                const int num_intermediate_reduction_steps =
                    is_first_device_in_direction ? 0 : num_targets_in_direction;
                const bool do_final_reduction = !is_first_device_in_direction;
                const int num_total_reduction_steps = num_intermediate_reduction_steps + (do_final_reduction ? 1 : 0);

                const uint32_t worker_id = (link * num_workers_per_direction) + worker;
                const uint32_t num_workers = num_links * num_workers_per_direction;
                const auto [start_tiles_read, start_tiles_to_read, start_pages_read_in_row, start_row_offset] =
                    ttnn::experimental::ccl::reduce_scatter_get_tile_offsets(
                        worker_id,
                        num_workers,
                        output_batch_num_pages,
                        output_channel_num_pages,
                        slice_Wt,
                        input_tensor_Wt,
                        normalized_dim);

                // for dim 0 scatters we process each slice in batches
                // for all other dims we process each slice in channels
                uint32_t tiles_per_worker_per_repeat = start_tiles_to_read - start_tiles_read;
                uint32_t num_repeats = (normalized_dim == 0) ? slice_B : slice_C;
                uint32_t chunks_per_sync_val =
                    chunks_per_sync.value_or(ttnn::experimental::ccl::reduce_scatter_default_chunks_per_sync(
                        topology, tiles_per_worker_per_repeat, num_repeats, tile_granularity));
                log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

                // Reader RT args
                std::vector<uint32_t> reader_rt_args = {
                    fwd_bwd_semaphore_address,
                    is_forward,                    // is_forward
                    is_first_device_in_direction,  // is_first_device_in_direction
                    num_targets_in_direction,      // num_targets_in_direction
                    do_final_reduction,            // do_final_reduction
                    chunks_per_sync_val,           // chunks_per_sync
                    start_tiles_read,              // start_tiles_read
                    start_tiles_to_read,           // start_tiles_to_read
                    start_pages_read_in_row,       // start_pages_read_in_row (unused by dim0 kernel)
                    start_row_offset,              // start_row_offset (unused by dim0 kernel)
                };

                if (input_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(input_tensor, reader_rt_args);
                }
                if (intermediate_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(intermediate_tensor, reader_rt_args);
                }
                if (output_is_sharded) {
                    shard_builder::extend_sharding_run_time_args(output_tensor, reader_rt_args);
                }
                if (fuse_op) {
                    fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }
                set_runtime_args(program, reader_kernel_id, {core}, reader_rt_args);

                CoreCoord termination_master_virtual_core =
                    mesh_device->worker_core_from_logical_core(termination_master_logical_core);

                std::vector<uint32_t> writer_rt_args = {
                    virtual_core.x,  // out_ready_sem_noc0_x
                    virtual_core.y,  // out_ready_sem_noc0_y
                    fwd_bwd_semaphore_address,
                    opposite_core_coord.x,
                    opposite_core_coord.y,
                    barrier_semaphore.has_value() && !using_persistent_buffers,  // use_barrier_sem
                    is_forward,                                                  // is_forward
                    is_first_device_in_direction,                                // is_first_device_in_direction
                    num_targets_in_direction,                                    // num_targets_in_direction
                    do_final_reduction,                                          // do_final_reduction
                    chunks_per_sync_val,                                         // chunks_per_sync
                    start_pages_read_in_row,                                     // start_pages_read_in_row
                    start_row_offset,                                            // start_row_offset
                    start_tiles_read,                                            // start_tiles_read
                    start_tiles_to_read,                                         // start_tiles_to_read
                };
                if constexpr (is_legacy_program_v<ProgramOrDesc>) {
                    append_fabric_mux_connection_rt_args(
                        mux_connection_valid(dir),
                        mux_virtual_core,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_kernel_config,
                        core,
                        worker,
                        worker == 0,
                        termination_master_virtual_core,
                        program,
                        writer_rt_args);
                    if (intermediate_is_sharded) {
                        shard_builder::extend_sharding_run_time_args(intermediate_tensor, writer_rt_args);
                    }
                    if (output_is_sharded) {
                        shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
                    }
                    set_runtime_args(program, writer_kernel_id, {core}, writer_rt_args);
                } else {
                    tt::tt_metal::KernelDescriptor::RTArgList writer_args;
                    writer_args.append(writer_rt_args);
                    append_fabric_mux_connection_rt_args(
                        mux_connection_valid(dir),
                        mux_virtual_core,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_kernel_config,
                        core,
                        worker,
                        worker == 0,
                        termination_master_virtual_core,
                        program,
                        writer_args);
                    if (intermediate_is_sharded || output_is_sharded) {
                        std::vector<uint32_t> sharding_rt_args;
                        if (intermediate_is_sharded) {
                            shard_builder::extend_sharding_run_time_args(intermediate_tensor, sharding_rt_args);
                        }
                        if (output_is_sharded) {
                            shard_builder::extend_sharding_run_time_args(output_tensor, sharding_rt_args);
                        }
                        writer_args.append(sharding_rt_args);
                    }
                    program.kernels[writer_kernel_id].emplace_runtime_args(core, writer_args);
                }

                std::vector<uint32_t> reduce_rt_args = {
                    num_total_reduction_steps,
                    start_tiles_read,
                    start_tiles_to_read,
                };
                set_runtime_args(program, reduce_kernel_id, {core}, reduce_rt_args);
            }
        }
    }

    // Common bindings: input, intermediate, output, penult, barrier, sem0, sem1, ack.
    if constexpr (is_legacy_program_v<ProgramOrDesc>) {
        const auto common_args = ReduceScatterProgramArtifacts::collect_runtime_args(
            false, barrier_semaphore, semaphore, input_tensor, intermediate_tensor, output_tensor);
        tt::tt_metal::SetCommonRuntimeArgs(program, reader_kernel_id, common_args);
        tt::tt_metal::SetCommonRuntimeArgs(program, writer_kernel_id, common_args);
        return ReduceScatterProgramArtifacts{
            GetCommonRuntimeArgs(program, reader_kernel_id), GetCommonRuntimeArgs(program, writer_kernel_id)};
    } else {
        tt::tt_metal::KernelDescriptor::RTArgList common_args;
        append_reduce_scatter_common_args(
            common_args,
            false,
            barrier_semaphore,
            semaphore,
            input_tensor,
            intermediate_tensor,
            output_tensor,
            std::nullopt);
        program.kernels[reader_kernel_id].emplace_common_runtime_args(common_args);
        program.kernels[writer_kernel_id].emplace_common_runtime_args(common_args);
        return;
    }
}

ReduceScatterProgramArtifacts build_line_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const std::optional<Tensor>& penult_intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return build_line_reduce_scatter_program_impl(
        program,
        input_tensor,
        intermediate_tensor,
        penult_intermediate_tensor,
        sender_device_coord,
        forward_coord,
        backward_coord,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        fused_op_signaler,
        chunks_per_sync,
        num_workers_per_direction_opt,
        num_buffers_per_channel,
        core_grid_offset,
        compute_kernel_config);
}

void build_line_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::ProgramDescriptor& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const std::optional<Tensor>& penult_intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    const CoreCoord core_grid_offset,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    build_line_reduce_scatter_program_impl(
        program,
        input_tensor,
        intermediate_tensor,
        penult_intermediate_tensor,
        sender_device_coord,
        forward_coord,
        backward_coord,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        fused_op_signaler,
        chunks_per_sync,
        num_workers_per_direction_opt,
        num_buffers_per_channel,
        core_grid_offset,
        compute_kernel_config);
}

}  // namespace ttnn

// Implementations for the prim namespace - wrappers to ttnn namespace functions
namespace ttnn::experimental::prim {

ReduceScatterProgramArtifacts::RuntimeArgs ReduceScatterProgramArtifacts::collect_runtime_args(
    bool is_ring,
    const std::optional<GlobalSemaphore>& barrier,
    const std::vector<GlobalSemaphore>& semaphores,
    const Tensor& input,
    const Tensor& intermediate,
    const Tensor& output,
    const std::optional<Tensor>& penult) {
    using Args = ttnn::ccl::ReduceScatterCommonArgs;
    RuntimeArgs args{};
    args[Args::input] = input.buffer()->address();
    args[Args::intermediate] = intermediate.buffer()->address();
    args[Args::output] = output.buffer()->address();
    args[Args::penult] = penult ? penult->buffer()->address() : 0;
    args[Args::barrier] = barrier ? barrier->address() : 0;
    args[Args::semaphore_0] = semaphores.at(0).address();
    args[Args::semaphore_1] = is_ring ? semaphores.at(1).address() : 0;
    args[Args::ack] = is_ring ? semaphores.at(2).address() : 0;
    return args;
}

ReduceScatterProgramArtifacts build_ring_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const std::optional<Tensor>& penult_intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ::ttnn::build_ring_reduce_scatter_minimal_async_program_artifacts(
        program,
        input_tensor,
        intermediate_tensor,
        penult_intermediate_tensor,
        sender_device_coord,
        forward_coord,
        backward_coord,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        fused_op_signaler,
        chunks_per_sync,
        num_workers_per_direction_opt,
        num_buffers_per_channel,
        core_grid_offset,
        compute_kernel_config);
}

ReduceScatterProgramArtifacts build_line_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const std::optional<Tensor>& penult_intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ::ttnn::build_line_reduce_scatter_minimal_async_program_artifacts(
        program,
        input_tensor,
        intermediate_tensor,
        penult_intermediate_tensor,
        sender_device_coord,
        forward_coord,
        backward_coord,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        fused_op_signaler,
        chunks_per_sync,
        num_workers_per_direction_opt,
        num_buffers_per_channel,
        core_grid_offset,
        compute_kernel_config);
}

void build_ring_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::ProgramDescriptor& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const std::optional<Tensor>& penult_intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    ::ttnn::build_ring_reduce_scatter_minimal_async_program_artifacts(
        program,
        input_tensor,
        intermediate_tensor,
        penult_intermediate_tensor,
        sender_device_coord,
        forward_coord,
        backward_coord,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        fused_op_signaler,
        chunks_per_sync,
        num_workers_per_direction_opt,
        num_buffers_per_channel,
        core_grid_offset,
        compute_kernel_config);
}

void build_line_reduce_scatter_minimal_async_program_artifacts(
    tt::tt_metal::ProgramDescriptor& program,
    const Tensor& input_tensor,
    const Tensor& intermediate_tensor,
    const std::optional<Tensor>& penult_intermediate_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    CoreCoord core_grid_offset,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    ::ttnn::build_line_reduce_scatter_minimal_async_program_artifacts(
        program,
        input_tensor,
        intermediate_tensor,
        penult_intermediate_tensor,
        sender_device_coord,
        forward_coord,
        backward_coord,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        fused_op_signaler,
        chunks_per_sync,
        num_workers_per_direction_opt,
        num_buffers_per_channel,
        core_grid_offset,
        compute_kernel_config);
}

// Mesh Workload Factory implementations
RingReduceScatterMeshWorkloadFactory::cached_mesh_workload_t RingReduceScatterMeshWorkloadFactory::create_mesh_workload(
    const ReduceScatterMinimalAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ReduceScatterMinimalAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), cached_program.shared_variables);
    }

    return {std::move(mesh_workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<RingReduceScatterMeshWorkloadFactory::shared_variables_t>
RingReduceScatterMeshWorkloadFactory::create_at(
    const ReduceScatterMinimalAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const ReduceScatterMinimalAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto& intermediate_tensor = tensor_return_value.at(0);
    auto& output_tensor = tensor_return_value.at(1);
    // Contiguous staging layout only: the penult intermediate the 2nd-last iteration stages one
    // direction's contribution into, instead of scatter-writing it directly into the tiled output
    // tensor. compute_output_specs declares it as a third output on exactly the configurations that use
    // it (and create_output_tensors either allocates it or adopts the caller's persistent buffer), so
    // its presence here is the signal that the contiguous layout is in play — the factory never
    // allocates it. Absent on the tiled layout (Linear, Ring with scatter dim 0, or a caller-provided
    // input-shaped intermediate). See rs-contiguous-interm-design.
    const std::optional<Tensor> penult_intermediate_tensor =
        tensor_return_value.size() > 2 ? std::optional<Tensor>(tensor_return_value.at(2)) : std::nullopt;

    const auto forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    const auto backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "forward_coord or backward_coord is null");

    const uint32_t ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, mesh_coordinate, operation_attributes.cluster_axis);

    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> fused_op_signaler = std::nullopt;
    tt::tt_metal::Program program{};
    return {
        std::move(program),
        ::ttnn::build_ring_reduce_scatter_minimal_async_program_artifacts(
            program,
            input_tensor,
            intermediate_tensor,
            penult_intermediate_tensor,
            mesh_coordinate,
            forward_coord,
            backward_coord,
            output_tensor,
            operation_attributes.dim,
            operation_attributes.num_links,
            operation_attributes.ring_size,
            ring_index,
            operation_attributes.topology,
            operation_attributes.semaphore,
            operation_attributes.barrier_semaphore,
            operation_attributes.using_persistent_buffers,
            operation_attributes.sub_device_id,
            fused_op_signaler,
            operation_attributes.chunks_per_sync,
            operation_attributes.num_workers_per_link,
            operation_attributes.num_buffers_per_channel,
            CoreCoord(0, 0),
            operation_attributes.compute_kernel_config)};
}

void RingReduceScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ReduceScatterMinimalAsyncParams& operation_attributes,
    const ReduceScatterMinimalAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& intermediate = tensor_return_value.at(0);
    const auto& output = tensor_return_value.at(1);
    // See create_at: index 2 exists exactly on the contiguous staging layout. It is reallocated per
    // invocation like the intermediate, so its address has to be re-published to the kernels here.
    const std::optional<Tensor> penult_intermediate =
        tensor_return_value.size() > 2 ? std::optional<Tensor>(tensor_return_value.at(2)) : std::nullopt;

    const auto args = ReduceScatterProgramArtifacts::collect_runtime_args(
        true,
        operation_attributes.barrier_semaphore,
        operation_attributes.semaphore,
        input,
        intermediate,
        output,
        penult_intermediate);
    for (const auto& [coordinate_range, shared_vars] : cached_workload.shared_variables) {
        shared_vars.override_runtime_arguments(args);
    }
}

LineReduceScatterMeshWorkloadFactory::cached_mesh_workload_t LineReduceScatterMeshWorkloadFactory::create_mesh_workload(
    const ReduceScatterMinimalAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ReduceScatterMinimalAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), cached_program.shared_variables);
    }

    return {std::move(mesh_workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<LineReduceScatterMeshWorkloadFactory::shared_variables_t>
LineReduceScatterMeshWorkloadFactory::create_at(
    const ReduceScatterMinimalAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const ReduceScatterMinimalAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto& intermediate_tensor = tensor_return_value.at(0);
    auto& output_tensor = tensor_return_value.at(1);

    const auto forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    const auto backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "forward_coord or backward_coord is null");

    const uint32_t ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, mesh_coordinate, operation_attributes.cluster_axis);

    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> fused_op_signaler = std::nullopt;
    tt::tt_metal::Program program{};
    return {
        std::move(program),
        ::ttnn::build_line_reduce_scatter_minimal_async_program_artifacts(
            program,
            input_tensor,
            intermediate_tensor,
            /*penult_intermediate_tensor=*/std::nullopt,
            mesh_coordinate,
            forward_coord,
            backward_coord,
            output_tensor,
            operation_attributes.dim,
            operation_attributes.num_links,
            operation_attributes.ring_size,
            ring_index,
            operation_attributes.topology,
            operation_attributes.semaphore,
            operation_attributes.barrier_semaphore,
            operation_attributes.using_persistent_buffers,
            operation_attributes.sub_device_id,
            fused_op_signaler,
            operation_attributes.chunks_per_sync,
            operation_attributes.num_workers_per_link,
            operation_attributes.num_buffers_per_channel,
            CoreCoord(0, 0),
            operation_attributes.compute_kernel_config)};
}

void LineReduceScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ReduceScatterMinimalAsyncParams& operation_attributes,
    const ReduceScatterMinimalAsyncInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& intermediate = tensor_return_value.at(0);
    const auto& output = tensor_return_value.at(1);

    TT_FATAL(
        operation_attributes.topology == ttnn::ccl::Topology::Linear,
        "LineReduceScatterMeshWorkloadFactory expects Linear topology");
    const auto args = ReduceScatterProgramArtifacts::collect_runtime_args(
        false, operation_attributes.barrier_semaphore, operation_attributes.semaphore, input, intermediate, output);
    for (const auto& [coordinate_range, shared_vars] : cached_workload.shared_variables) {
        shared_vars.override_runtime_arguments(args);
    }
}

}  // namespace ttnn::experimental::prim
