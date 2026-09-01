// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/device/reshape_row_major_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#define MASK_64 0xFFFFFFFFFFFFFFC0
#define MASK_16 0xFFFFFFFFFFFFFFF0

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
// Page-size alignment the kernel needs to write straight from the source DFB; must stay in
// sync with MASK_16/OFFSET_16 and the can_be_clean predicate in rm_reshape_interleaved.cpp.
constexpr uint32_t noc_page_alignment_bytes = 16;

// Depth of the L1 staging ring used when pages are not NoC-aligned. Eight destination
// pages share one write barrier, which is enough to keep the tiny-page case (2 B pages for
// the [N, 1] reshape in #50191) off a per-page barrier without making the ring so deep that
// wide destinations stop fitting in L1.
constexpr uint32_t small_dest_write_slots = 8;

// Kept local (not a shared header): Quasar's factory is an intentional mirror of this
// file; a cross-op helper would only dedupe ~30 lines and add CMake/packaging coupling.
// Non-clean dest staging uses a multi-slot L1 ring. Cap slots by per-core L1 budget so
// wide odd destinations (e.g. bf16 width 100001 from #50191) still fit.
uint32_t choose_num_dest_write_slots(
    IDevice* device,
    bool pages_noc_aligned,
    bool can_use_dual_kernel,
    uint32_t dfb_size0,
    uint32_t dest_slot_size_bytes) {
    if (pages_noc_aligned) {
        return 1u;
    }

    // Budget the staging ring against live L1 occupancy so DFBs never collide with
    // tensors already allocated in L1 (vadv2 regression). DFBs grow upward from the
    // base; L1 tensors are allocated downward from the top. The ceiling is the lowest
    // occupied L1 address — or the full core size when nothing is live.
    const uint32_t l1_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const std::optional<DeviceAddr> lowest_occupied = device->lowest_occupied_compute_l1_address();
    const uint32_t l1_ceiling =
        lowest_occupied.has_value() ? static_cast<uint32_t>(lowest_occupied.value()) : device->l1_size_per_core();
    TT_FATAL(l1_ceiling > l1_base, "L1 ceiling ({}) must exceed base ({})", l1_ceiling, l1_base);
    const uint32_t l1_available = l1_ceiling - l1_base;

    const uint32_t num_kernel_copies = can_use_dual_kernel ? 2u : 1u;
    const uint32_t source_dfb_bytes = dfb_size0 * 2u * num_kernel_copies;
    const uint32_t min_dest_dfb_bytes = dest_slot_size_bytes * num_kernel_copies;
    TT_FATAL(
        l1_available >= source_dfb_bytes + min_dest_dfb_bytes,
        "RM reshape dest staging does not fit in L1: need at least {} B dest + {} B source, have {} B",
        min_dest_dfb_bytes,
        source_dfb_bytes,
        l1_available);

    const uint32_t max_slots = (l1_available - source_dfb_bytes) / (dest_slot_size_bytes * num_kernel_copies);
    return std::max(1u, std::min(small_dest_write_slots, max_slots));
}
}  // namespace

ttnn::device_operation::ProgramArtifacts ReshapeViewRMProgramFactory::create_program_artifacts(
    const ReshapeViewParams& operation_attributes, const ReshapeViewInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& input_mt = input.mesh_tensor();
    const auto& output_mt = output.mesh_tensor();
    const auto& sub_core_grid = operation_attributes.sub_core_grid;

    // get datum size
    tt::DataFormat dfb_data_format = datatype_to_dataformat_converter(input.dtype());
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
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Find how many input pages each core is responsible for so that we always start at the beginning of a read and
    // write page Since the logical volumes match, we are guaranteed that the very last page is aligned
    uint32_t responsibility = ((input_log_shape[-2] - 1) / num_cores_total) + 1;
    while ((responsibility * source_page_size_bytes) % dest_page_size_bytes != 0) {
        responsibility++;
    }
    const uint32_t dfb_size0 = source_read_size_bytes;
    const uint32_t dest_slot_size_bytes = ((dest_page_size_bytes - 1) & MASK_64) + 80;

    const bool pages_noc_aligned = (source_page_size_bytes % noc_page_alignment_bytes == 0) &&
                                   (dest_page_size_bytes % noc_page_alignment_bytes == 0);
    const bool dest_noc_aligned = (dest_page_size_bytes % noc_page_alignment_bytes == 0);
    const bool pages_divisible =
        (source_page_size_bytes % dest_page_size_bytes == 0 || dest_page_size_bytes % source_page_size_bytes == 0);
    // Avoid dual-kernel when dest writes are too small for DRAM (Blackhole SYS-1419 / #50191).
    // Only dest alignment matters: source alignment selects the clean-vs-staging read path
    // but doesn't affect the size of writes hitting DRAM.
    const bool can_use_dual_kernel = pages_divisible && (dest_noc_aligned || !dst_buffer->is_dram());

    const uint32_t num_dest_write_slots =
        choose_num_dest_write_slots(device, pages_noc_aligned, can_use_dual_kernel, dfb_size0, dest_slot_size_bytes);
    const uint32_t dfb_size1 = dest_slot_size_bytes * num_dest_write_slots;

    const uint32_t write_alignment =
        dst_buffer->is_dram() ? tt::tt_metal::hal::get_dram_alignment() : tt::tt_metal::hal::get_l1_alignment();
    const uint32_t noc_write_align = std::min(write_alignment, tt::tt_metal::hal::get_l1_alignment());
    const uint32_t dest_write_size_bytes =
        pages_noc_aligned ? dest_page_size_bytes : tt::align(dest_page_size_bytes, noc_write_align);

    // ---- Metal 2.0 spec construction ----
    // Resource names. The RM source is instantiated as two KernelSpecs over the SAME node set (a
    // dual-instance work-split): the reader-config instance touches src0/src1 as scratch, the
    // writer-config instance touches src2/src3. The two instances touch DISJOINT DFBs, so each DFB
    // has a single toucher and is self-looped (its owning kernel bound PRODUCER + CONSUMER).
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName SRC0{"src0"};
    const DFBSpecName SRC1{"src1"};
    const DFBSpecName SRC2{"src2"};
    const DFBSpecName SRC3{"src3"};
    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};

    // Named CTAs — identical for both instances. The per-instance difference is the DFB binding
    // (src0/src1 vs src2/src3), not a compile-time arg, so a single CTA table serves both.
    const KernelSpec::CompileTimeArgs cta = {
        {"src_aligned_to_64", (source_page_size_bytes % 64 == 0) ? 1u : 0u},
        {"src_aligned_to_16", (source_page_size_bytes % 16 == 0) ? 1u : 0u},
        {"source_page_size_bytes", source_page_size_bytes},
        {"dest_page_size_bytes", dest_page_size_bytes},
        {"num_dest_write_slots", num_dest_write_slots},
        {"dest_slot_size_bytes", dest_slot_size_bytes},
        {"dest_write_size_bytes", dest_write_size_bytes},
    };

    const KernelSpec::RuntimeArgSchema rta_schema = {
        .runtime_arg_names =
            {"source_read_size_bytes",
             "read_start_page",
             "read_end_page",
             "write_start_page",
             "write_start_offset",
             "nop"},
    };

    // Both instances read `tensor::src` and write `tensor::dst`; each self-loops its own scratch DFBs
    // (accessor_names in0/in1, mapped to distinct DFB specs per instance).
    auto make_rm_kernel = [&](const KernelSpecName& id,
                              DataMovementHardwareConfig hw,
                              const DFBSpecName& d0,
                              const DFBSpecName& d1) {
        return KernelSpec{
            .unique_id = id,
            .source = "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/rm_reshape_interleaved.cpp",
            .dfb_bindings =
                {
                    DFBBinding{.dfb_spec_name = d0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
                    DFBBinding{.dfb_spec_name = d0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                    DFBBinding{.dfb_spec_name = d1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
                    DFBBinding{.dfb_spec_name = d1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
                },
            .tensor_bindings =
                {
                    TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"},
                    TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"},
                },
            .compile_time_args = cta,
            .runtime_arg_schema = rta_schema,
            .hw_config = std::move(hw),
        };
    };

    auto make_scratch_dfb = [&](const DFBSpecName& id, uint32_t entry_size, uint32_t num_entries) {
        return DataflowBufferSpec{
            .unique_id = id,
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = dfb_data_format,
        };
    };

    ProgramSpec spec;
    spec.name = "reshape_view_rm";
    spec.kernels.push_back(make_rm_kernel(READER, create_reader_datamovement_config(), SRC0, SRC1));
    spec.dataflow_buffers.push_back(make_scratch_dfb(SRC0, dfb_size0, 2));
    spec.dataflow_buffers.push_back(make_scratch_dfb(SRC1, dfb_size1, 1));
    if (can_use_dual_kernel) {
        spec.kernels.push_back(make_rm_kernel(WRITER, create_writer_datamovement_config(), SRC2, SRC3));
        spec.dataflow_buffers.push_back(make_scratch_dfb(SRC2, dfb_size0, 2));
        spec.dataflow_buffers.push_back(make_scratch_dfb(SRC3, dfb_size1, 1));
    }
    spec.tensor_parameters = {
        TensorParameter{.unique_id = SRC, .spec = input_mt.tensor_spec()},
        TensorParameter{.unique_id = DST, .spec = output_mt.tensor_spec()},
    };
    WorkUnitSpec work_unit{.name = "main", .kernels = {READER}, .target_nodes = total_cores};
    if (can_use_dual_kernel) {
        work_unit.kernels.push_back(WRITER);
    }
    spec.work_units = {work_unit};

    // ---- Per-node runtime args. The run-args table is keyed name-first (name -> node -> value);
    // AddRuntimeArgsForNode builds that from the per-core loop below. ----
    KernelRunArgs reader_kra{.kernel = READER};
    KernelRunArgs writer_kra{.kernel = WRITER};

    uint32_t done = 0;
    for (auto core : corerange_to_cores(total_cores, std::nullopt)) {
        if (done == 1) {
            // Idle core: the kernel short-circuits on nop==1 before building any TensorAccessor, so
            // the framework-delivered src/dst base addresses are never used here — harmless.
            AddRuntimeArgsForNode(
                reader_kra.runtime_arg_values,
                core,
                {{"source_read_size_bytes", source_read_size_bytes},
                 {"read_start_page", 0u},
                 {"read_end_page", 0u},
                 {"write_start_page", 0u},
                 {"write_start_offset", 0u},
                 {"nop", 1u}});
            if (can_use_dual_kernel) {
                AddRuntimeArgsForNode(
                    writer_kra.runtime_arg_values,
                    core,
                    {{"source_read_size_bytes", source_read_size_bytes},
                     {"read_start_page", 0u},
                     {"read_end_page", 0u},
                     {"write_start_page", 0u},
                     {"write_start_offset", 0u},
                     {"nop", 1u}});
            }
        } else {
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

                AddRuntimeArgsForNode(
                    reader_kra.runtime_arg_values,
                    core,
                    {{"source_read_size_bytes", source_read_size_bytes},
                     {"read_start_page", start_of_read},
                     {"read_end_page", mid_read},
                     {"write_start_page", write_start_page},
                     {"write_start_offset", 0u},
                     {"nop", 0u}});
                AddRuntimeArgsForNode(
                    writer_kra.runtime_arg_values,
                    core,
                    {{"source_read_size_bytes", source_read_size_bytes},
                     {"read_start_page", mid_read},
                     {"read_end_page", end_of_read},
                     {"write_start_page", second_write_pos},
                     {"write_start_offset", 0u},
                     {"nop", 0u}});
            } else {
                // Original single kernel approach
                AddRuntimeArgsForNode(
                    reader_kra.runtime_arg_values,
                    core,
                    {{"source_read_size_bytes", source_read_size_bytes},
                     {"read_start_page", start_of_read},
                     {"read_end_page", end_of_read},
                     {"write_start_page", write_start_page},
                     {"write_start_offset", 0u},  // write_start_offset removed (always 0)
                     {"nop", done}});
            }
            write_start_page += write_jump;
            read_start_page = end_of_read;
            done = (end_of_read == input_log_shape[-2]) ? 1 : 0;
        }
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_kra));
    if (can_use_dual_kernel) {
        run_args.kernel_run_args.push_back(std::move(writer_kra));
    }
    run_args.tensor_args = {
        {SRC, TensorArgument{input_mt}},
        {DST, TensorArgument{output_mt}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
