// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_device_operation_types.hpp"
#include "move_overlap_program_factory.hpp"

#include <cmath>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

std::vector<CoreRange> get_multicast_regions(const CoreRangeSet& all_cores, const CoreCoord& logical_controller) {
    TT_ASSERT(!all_cores.ranges().empty() and all_cores.ranges().size() <= 2);
    const CoreCoord logical_zero = {0, 0};
    TT_ASSERT(logical_controller == logical_zero);

    std::vector<CoreRange> logical_core_ranges;
    auto split_core_range_containing_controller = [&](const CoreRange& controller_core_range) {
        TT_ASSERT(controller_core_range.start_coord == logical_controller);
        CoreRange right_block(
            CoreCoord(logical_controller.x + 1, logical_controller.y), controller_core_range.end_coord);
        CoreRange remaining_stick = CoreRange(
            CoreCoord(logical_controller.x, logical_controller.y + 1),
            CoreCoord(logical_controller.x, controller_core_range.end_coord.y));

        logical_core_ranges.push_back(right_block);
        logical_core_ranges.push_back(remaining_stick);
    };

    CoreRange range_0 = *all_cores.ranges().begin();
    if (all_cores.ranges().size() == 1) {
        split_core_range_containing_controller(range_0);
    } else {
        CoreRange range_1 = *all_cores.ranges().rbegin();
        if (range_0.start_coord == logical_controller) {
            split_core_range_containing_controller(range_0);
            logical_core_ranges.push_back(range_1);
        } else if (range_1.start_coord == logical_controller) {
            split_core_range_containing_controller(range_1);
            logical_core_ranges.push_back(range_0);
        } else {
            TT_THROW("Core {} is not included in set of core ranges!", logical_controller.str());
        }
    }

    TT_ASSERT(logical_core_ranges.size() == 2 or logical_core_ranges.size() == 3);
    return logical_core_ranges;
}

// Metal 2.0 resource names (ProgramSpec scope).
const m2::KernelSpecName READER{"reader"};
const m2::KernelSpecName WRITER{"writer"};
const m2::DFBSpecName SCRATCH{"scratch"};
const m2::SemaphoreSpecName SEM{"sem"};
const m2::TensorParamName INPUT{"input"};
const m2::TensorParamName OUTPUT{"output"};

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts MoveOverlapProgramFactory::create_program_artifacts(
    const MoveOperationAttributes& /*operation_attributes*/,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE;  // resolve the file-local ids/helpers below
    using namespace tt::constants;

    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& input_mt = input.mesh_tensor();
    const auto& output_mt = output.mesh_tensor();

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const bool tilized = input.layout() == Layout::TILE;
    const uint32_t page_size = input.buffer()->page_size();

    const uint32_t num_pages =
        tilized ? (output.physical_volume() / TILE_HW) : (output.physical_volume() / output.padded_shape()[-1]);
    const tt::tt_metal::IDevice* device = output.device();
    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);

    const auto num_l1_banks = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    const uint32_t size_per_l1_bank = tt::tt_metal::detail::calculate_bank_size_spread(
        output.buffer()->size(), output.buffer()->page_size(), num_l1_banks, tt::tt_metal::hal::get_l1_alignment());

    // Scratch CB is a temp L1 buffer to copy src data into before writing to dst.
    const uint32_t aligned_page_size = round_up_to_mul32(page_size);

    //
    // -------- Build the ProgramSpec --------
    //

    m2::ProgramSpec spec;
    spec.name = "move_overlap";

    // Scratch DFB: L1-allocated staging FIFO. The reader both fills it (src -> CB) and
    // drains it (CB -> dst), so it is bound as a self-loop (PRODUCER + CONSUMER) on the
    // one reader kernel. num_entries * entry_size reproduces the legacy total_size.
    spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = SCRATCH,
        .entry_size = aligned_page_size,
        .num_entries = size_per_l1_bank / aligned_page_size,
        .data_format_metadata = cb_data_format,
    });

    // Semaphore used by the controller core to coordinate multicast.
    spec.semaphores.push_back(m2::SemaphoreSpec{
        .unique_id = SEM,
        .target_nodes = all_cores,
    });

    // Tensor parameters (src / dst). Their base addresses reach the kernel through the
    // typed binding channel (refreshed on cache hit), replacing the legacy Buffer* RTAs
    // and TensorAccessorArgs CTAs.
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});

    const std::filesystem::path kernel_path =
        tilized ? "ttnn/cpp/ttnn/operations/experimental/quasar/move/device/kernels/dataflow/"
                  "move_interleaved_with_overlap.cpp"
                : "ttnn/cpp/ttnn/operations/experimental/quasar/move/device/kernels/dataflow/"
                  "move_stick_layout_interleaved_with_overlap.cpp";
    const std::filesystem::path writer_kernel_path =
        tilized ? "ttnn/cpp/ttnn/operations/experimental/quasar/move/device/kernels/dataflow/"
                  "move_interleaved_with_overlap_writer.cpp"
                : "ttnn/cpp/ttnn/operations/experimental/quasar/move/device/kernels/dataflow/"
                  "move_stick_layout_interleaved_with_overlap_writer.cpp";

    m2::KernelSpec reader{
        .unique_id = READER,
        .source = kernel_path,
        .dfb_bindings =
            {
                // Producer side of the cross-kernel scratch DFB (the writer kernel is the consumer);
                // splitting producer/consumer across two kernels avoids a forbidden DM self-loop.
                m2::DFBBinding{
                    .dfb_spec_name = SCRATCH,
                    .accessor_name = "scratch",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
            },
        .semaphore_bindings =
            {
                m2::SemaphoreBinding{.semaphore_spec_name = SEM, .accessor_name = "sem"},
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    // Named CTA: the stick-layout kernel needs the (unaligned) page size at compile time.
    if (!tilized) {
        reader.compile_time_args = {{"page_size", page_size}};
    }

    // Named RTA schema (per-node values supplied below). One named arg per legacy
    // positional slot; the legacy src/dst-address (slots 0,1) and semaphore-id (slot 4)
    // slots are gone — those are now bindings.
    m2::Group<std::string> rta_names = {
        "start_id",      "num_pages",           "controller_noc_x",    "controller_noc_y",  "control_value",
        "is_controller", "range_0_start_noc_x", "range_0_start_noc_y", "range_0_end_noc_x", "range_0_end_noc_y",
        "range_0_size",  "range_1_start_noc_x", "range_1_start_noc_y", "range_1_end_noc_x", "range_1_end_noc_y",
        "range_1_size",  "range_2_start_noc_x", "range_2_start_noc_y", "range_2_end_noc_x", "range_2_end_noc_y",
        "range_2_size",  "do_third_multicast",
    };
    if (!tilized) {
        rta_names.push_back("aligned_page_size");
    }
    reader.runtime_arg_schema.runtime_arg_names = std::move(rta_names);

    spec.kernels.push_back(reader);

    // Consumer side of the cross-kernel scratch DFB: drains CB -> dst. Runs on the WRITER processor
    // (the opposite RISC from the reader) on the same cores, sharing the scratch L1 buffer SPSC. Its
    // wait_front(scratch) cannot unblock until the reader's post-handshake push_back, so dst writes
    // only begin once every core has read src (the overlap-safety invariant).
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_path,
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = SCRATCH,
                    .accessor_name = "scratch",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    // The stick-layout writer needs the (unaligned) page size at compile time, like its reader.
    if (!tilized) {
        writer.compile_time_args = {{"page_size", page_size}};
    }

    // Writer RTAs are the subset the drain phase needs (no sem/multicast coordination).
    {
        m2::Group<std::string> writer_rta_names = {"start_id", "num_pages"};
        if (!tilized) {
            writer_rta_names.push_back("aligned_page_size");
        }
        writer.runtime_arg_schema.runtime_arg_names = std::move(writer_rta_names);
    }

    spec.kernels.push_back(writer);

    spec.work_units.push_back(m2::WorkUnitSpec{
        .name = "wu",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    });

    //
    // -------- Build the ProgramRunArgs --------
    //

    const CoreCoord logical_controller = CoreCoord{0, 0};
    const CoreCoord noc_controller = device->worker_core_from_logical_core(logical_controller);
    std::vector<CoreRange> logical_multicast_regions = get_multicast_regions(all_cores, logical_controller);

    std::vector<CoreRange> noc_multicast_regions;
    noc_multicast_regions.reserve(logical_multicast_regions.size());
    for (const auto& logical_cr : logical_multicast_regions) {
        const CoreRange noc_cr(
            device->worker_core_from_logical_core(logical_cr.start_coord),
            device->worker_core_from_logical_core(logical_cr.end_coord));
        noc_multicast_regions.push_back(noc_cr);
    }

    const CoreRange range_0_noc = noc_multicast_regions[0];
    const CoreRange range_1_noc = noc_multicast_regions[1];
    const bool do_third_multicast = (noc_multicast_regions.size() == 3);

    m2::ProgramRunArgs run_args;
    m2::KernelRunArgs reader_run_args;
    reader_run_args.kernel = READER;
    m2::KernelRunArgs writer_run_args;
    writer_run_args.kernel = WRITER;

    for (uint32_t i = 0, pages_handled_per_core = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_pages_per_core = 0;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        const bool is_controller = (i == 0);

        m2::KernelRunArgs::RuntimeArgValues vals = {
            {"start_id", pages_handled_per_core},
            {"num_pages", num_pages_per_core},
            {"control_value", num_cores - 1},
            {"controller_noc_x", static_cast<uint32_t>(noc_controller.x)},
            {"controller_noc_y", static_cast<uint32_t>(noc_controller.y)},
            {"is_controller", static_cast<uint32_t>(is_controller)},
            {"range_0_start_noc_x", static_cast<uint32_t>(range_0_noc.start_coord.x)},
            {"range_0_start_noc_y", static_cast<uint32_t>(range_0_noc.start_coord.y)},
            {"range_0_end_noc_x", static_cast<uint32_t>(range_0_noc.end_coord.x)},
            {"range_0_end_noc_y", static_cast<uint32_t>(range_0_noc.end_coord.y)},
            {"range_0_size", static_cast<uint32_t>(logical_multicast_regions[0].size())},
            {"range_1_start_noc_x", static_cast<uint32_t>(range_1_noc.start_coord.x)},
            {"range_1_start_noc_y", static_cast<uint32_t>(range_1_noc.start_coord.y)},
            {"range_1_end_noc_x", static_cast<uint32_t>(range_1_noc.end_coord.x)},
            {"range_1_end_noc_y", static_cast<uint32_t>(range_1_noc.end_coord.y)},
            {"range_1_size", static_cast<uint32_t>(logical_multicast_regions[1].size())},
            {"range_2_start_noc_x", static_cast<uint32_t>(noc_multicast_regions.back().start_coord.x)},
            {"range_2_start_noc_y", static_cast<uint32_t>(noc_multicast_regions.back().start_coord.y)},
            {"range_2_end_noc_x", static_cast<uint32_t>(noc_multicast_regions.back().end_coord.x)},
            {"range_2_end_noc_y", static_cast<uint32_t>(noc_multicast_regions.back().end_coord.y)},
            {"range_2_size", static_cast<uint32_t>(logical_multicast_regions.back().size())},
            {"do_third_multicast", static_cast<uint32_t>(do_third_multicast)},
        };
        if (!tilized) {
            vals.insert({"aligned_page_size", aligned_page_size});
        }

        reader_run_args.runtime_arg_values.push_back(
            m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = std::move(vals)});

        m2::KernelRunArgs::RuntimeArgValues writer_vals = {
            {"start_id", pages_handled_per_core},
            {"num_pages", num_pages_per_core},
        };
        if (!tilized) {
            writer_vals.insert({"aligned_page_size", aligned_page_size});
        }
        writer_run_args.runtime_arg_values.push_back(
            m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = std::move(writer_vals)});

        pages_handled_per_core += num_pages_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(INPUT, input_mt);
    run_args.tensor_args.emplace(OUTPUT, output_mt);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
