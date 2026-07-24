// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/device/reshape_row_major_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>

#define MASK_64 0xFFFFFFFFFFFFFFC0
#define MASK_16 0xFFFFFFFFFFFFFFF0

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {
constexpr uint32_t kSmallDestWriteSlots = 8;

// Kept local (not a shared header): Quasar's factory is an intentional mirror of this
// file; a cross-op helper would only dedupe ~30 lines and add CMake/packaging coupling.
// Non-clean dest staging uses a multi-slot L1 ring. Cap slots by per-core L1 budget so
// wide odd destinations (e.g. bf16 width 100001 from #50191) still fit.
uint32_t choose_num_dest_write_slots(
    IDevice* device,
    bool pages_16b_aligned,
    bool can_use_dual_kernel,
    uint32_t cb_size0,
    uint32_t dest_slot_size_bytes) {
    if (pages_16b_aligned) {
        return 1u;
    }

    const uint32_t l1_reserved = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t l1_size = device->l1_size_per_core();
    TT_FATAL(l1_size > l1_reserved, "L1 size ({}) must exceed reserved base ({})", l1_size, l1_reserved);
    const uint32_t l1_available = l1_size - l1_reserved;

    const uint32_t num_kernel_copies = can_use_dual_kernel ? 2u : 1u;
    const uint32_t source_cb_bytes = cb_size0 * 2u * num_kernel_copies;
    const uint32_t min_dest_cb_bytes = dest_slot_size_bytes * num_kernel_copies;
    TT_FATAL(
        l1_available >= source_cb_bytes + min_dest_cb_bytes,
        "RM reshape dest staging does not fit in L1: need at least {} B dest + {} B source, have {} B",
        min_dest_cb_bytes,
        source_cb_bytes,
        l1_available);

    const uint32_t max_slots = (l1_available - source_cb_bytes) / (dest_slot_size_bytes * num_kernel_copies);
    return std::max(1u, std::min(kSmallDestWriteSlots, max_slots));
}
}  // namespace

ProgramDescriptor ReshapeViewRMProgramFactory::create_descriptor(
    const ReshapeViewParams& operation_attributes, const ReshapeViewInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& sub_core_grid = operation_attributes.sub_core_grid;

    // get datum size
    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t data_size = input.element_size();
    IDevice* device = input.device();
    // Multi device pre-computation
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange default_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    CoreRangeSet total_cores = sub_core_grid.has_value() ? sub_core_grid.value() : CoreRangeSet(default_cores);
    uint32_t num_cores_total = total_cores.num_cores();

    auto input_log_shape = input.logical_shape();
    auto output_log_shape = output.logical_shape();

    log_debug(tt::LogOp, "reshape_view: row major program factory");
    log_debug(tt::LogOp, "input shape: {}", input_log_shape);
    log_debug(tt::LogOp, "output shape: {}", output_log_shape);
    log_debug(tt::LogOp, "data size: {}", data_size);

    uint32_t source_page_size_bytes = input_log_shape[-1] * data_size;
    uint32_t dest_page_size_bytes = output_log_shape[-1] * data_size;
    uint32_t source_read_size_bytes = ((source_page_size_bytes - 1) & MASK_64) + 128;
    uint32_t read_start_page = 0;
    uint32_t write_start_page = 0;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Find how many input pages each core is responsible for so that we always start at the beginning of a read and
    // write page Since the logical volumes match, we are guaranteed that the very last page is aligned
    uint32_t responsibility = ((input_log_shape[-2] - 1) / num_cores_total) + 1;
    while ((responsibility * source_page_size_bytes) % dest_page_size_bytes != 0) {
        responsibility++;
    }
    const uint32_t cb_size0 = source_read_size_bytes;
    const uint32_t dest_slot_size_bytes = ((dest_page_size_bytes - 1) & MASK_64) + 80;

    const bool pages_16b_aligned = (source_page_size_bytes % 16 == 0) && (dest_page_size_bytes % 16 == 0);
    const bool pages_divisible =
        (source_page_size_bytes % dest_page_size_bytes == 0 || dest_page_size_bytes % source_page_size_bytes == 0);
    // Avoid dual-kernel on non-aligned DRAM dests (Blackhole SYS-1419 / #50191).
    const bool can_use_dual_kernel = pages_divisible && (pages_16b_aligned || !dst_buffer->is_dram());

    const uint32_t num_dest_write_slots =
        choose_num_dest_write_slots(device, pages_16b_aligned, can_use_dual_kernel, cb_size0, dest_slot_size_bytes);
    const uint32_t cb_size1 = dest_slot_size_bytes * num_dest_write_slots;

    const uint32_t write_alignment =
        dst_buffer->is_dram() ? tt::tt_metal::hal::get_dram_alignment() : tt::tt_metal::hal::get_l1_alignment();
    const uint32_t noc_write_align = std::min(write_alignment, tt::tt_metal::hal::get_l1_alignment());
    const uint32_t dest_write_size_bytes =
        pages_16b_aligned ? dest_page_size_bytes : tt::align(dest_page_size_bytes, noc_write_align);

    constexpr uint32_t src0_cb_index = 0;
    constexpr uint32_t src1_cb_index = 1;
    constexpr uint32_t src2_cb_index = 2;
    constexpr uint32_t src3_cb_index = 3;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_size0 * 2,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = cb_size0,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_size1,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = cb_data_format,
            .page_size = cb_size1,
        }}},
    });

    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t)(source_page_size_bytes % 64 == 0) ? 1 : 0,
        (std::uint32_t)(source_page_size_bytes % 16 == 0) ? 1 : 0,
        src0_cb_index,
        src1_cb_index,
        source_page_size_bytes,
        dest_page_size_bytes,
        num_dest_write_slots,
        dest_slot_size_bytes,
        dest_write_size_bytes};
    TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*dst_buffer).append_to(compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/rm_reshape_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = total_cores;
    reader_desc.compile_time_args = compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // Second kernel uses CBs 2/3 with otherwise identical compile-time args; build its
    // dedicated arg vector so we don't alias reader_desc.compile_time_args.
    std::vector<uint32_t> writer_compile_time_args;
    KernelDescriptor writer_desc;
    if (can_use_dual_kernel) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb_size0 * 2,
            .core_ranges = total_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = src2_cb_index,
                .data_format = cb_data_format,
                .page_size = cb_size0,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb_size1,
            .core_ranges = total_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = src3_cb_index,
                .data_format = cb_data_format,
                .page_size = cb_size1,
            }}},
        });

        writer_compile_time_args = compile_time_args;
        writer_compile_time_args[2] = src2_cb_index;
        writer_compile_time_args[3] = src3_cb_index;

        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/rm_reshape_interleaved.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = total_cores;
        writer_desc.compile_time_args = std::move(writer_compile_time_args);
        writer_desc.config = WriterConfigDescriptor{};
    }

    uint32_t done = 0;
    for (auto core : corerange_to_cores(total_cores, std::nullopt)) {
        if (done == 1) {
            // Idle core: skip BufferBinding registration by passing 0u for buffer slots —
            // the kernel short-circuits when the "done" flag (last arg) is 1 and never
            // dereferences the address.
            reader_desc.emplace_runtime_args(core, {0u, 0u, source_read_size_bytes, 0u, 0u, 0u, 0u, 1u});
            if (can_use_dual_kernel) {
                writer_desc.emplace_runtime_args(core, {0u, 0u, source_read_size_bytes, 0u, 0u, 0u, 0u, 1u});
            }
        } else {
            // Create the circular buffers

            // set the runtime args
            // set the compile time args
            const uint32_t start_of_read = read_start_page;
            uint32_t end_of_read = read_start_page + responsibility;
            end_of_read = end_of_read < input_log_shape[-2] ? end_of_read : input_log_shape[-2];
            uint32_t pages_for_this_core = end_of_read - start_of_read;
            uint32_t write_jump = (pages_for_this_core * source_page_size_bytes) / dest_page_size_bytes;

            if (can_use_dual_kernel) {
                // Split work in half - determine split point and second write position
                uint32_t mid_read, second_write_pos;
                if (source_page_size_bytes >= dest_page_size_bytes) {
                    // Split by input pages
                    uint32_t half_pages = pages_for_this_core / 2;
                    mid_read = start_of_read + half_pages;
                    second_write_pos = write_start_page + (half_pages * source_page_size_bytes / dest_page_size_bytes);
                } else {
                    // Split by output pages
                    uint32_t total_bytes_for_core = pages_for_this_core * source_page_size_bytes;
                    uint32_t total_output_pages_for_core = total_bytes_for_core / dest_page_size_bytes;
                    uint32_t half_output_pages = total_output_pages_for_core / 2;
                    mid_read = start_of_read + (half_output_pages * dest_page_size_bytes / source_page_size_bytes);
                    second_write_pos = write_start_page + half_output_pages;
                }

                reader_desc.emplace_runtime_args(
                    core,
                    {src_buffer,
                     dst_buffer,
                     source_read_size_bytes,
                     start_of_read,
                     mid_read,
                     write_start_page,
                     0u,
                     0u});

                writer_desc.emplace_runtime_args(
                    core,
                    {src_buffer, dst_buffer, source_read_size_bytes, mid_read, end_of_read, second_write_pos, 0u, 0u});
            } else {
                // Original single kernel approach
                reader_desc.emplace_runtime_args(
                    core,
                    {src_buffer,
                     dst_buffer,
                     source_read_size_bytes,
                     start_of_read,
                     end_of_read,
                     write_start_page,
                     0u,  // write_start_offset removed (always 0)
                     done});
            }
            write_start_page += write_jump;
            read_start_page = end_of_read;
            done = (end_of_read == input_log_shape[-2]) ? 1 : 0;
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    if (can_use_dual_kernel) {
        desc.kernels.push_back(std::move(writer_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
