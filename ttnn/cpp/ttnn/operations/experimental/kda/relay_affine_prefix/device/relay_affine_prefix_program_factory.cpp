// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/relay_affine_prefix/device/relay_affine_prefix_device_operation.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/operations/experimental/kda/factory/kda_projection.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t cb_a = 0;
constexpr uint32_t cb_b = 1;
constexpr uint32_t cb_ein = 2;
constexpr uint32_t cb_eout = 3;
constexpr uint32_t cb_chron = 4;
constexpr uint32_t cb_stage = 5;
// Tile-column buffers per sender: later columns are read out of their head cores while earlier ones are on the link.
constexpr uint32_t sender_slots = 4;
// On the head cores: the sender releases a head once it has read the head's last column.
constexpr uint32_t ack_semaphore_id = 0;
// On the sender cores, ids 0, 1, ...: per head the sender serves, the head's finished columns.
constexpr uint32_t first_notify_semaphore_id = 0;
constexpr const char* kernel_dir = "ttnn/cpp/ttnn/operations/experimental/kda/relay_affine_prefix/device/kernels/";

tt::tt_metal::ProgramDescriptor build_program(
    const RelayAffinePrefixParams& attrs,
    const RelayAffinePrefixInputs& in,
    const std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinate& coord) {
    using tt::tt_metal::CoreCoord;
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh = in.packed.device();
    const auto& entry = outputs[0];
    const auto& final_state = outputs[1];
    const uint32_t axis = attrs.sequence_parallel_axis;
    const uint32_t sp_size = mesh->shape()[axis];
    const uint32_t sp_rank = coord[axis];
    auto successor = coord;
    successor[axis] = (sp_rank + 1) % sp_size;

    const uint32_t batch_heads = in.initial_state.logical_shape()[0];
    const uint32_t kt = in.initial_state.logical_shape()[1] / tt::constants::TILE_HEIGHT;
    const uint32_t state_tiles = kt * kt;
    const uint32_t state_tile = tt::constants::TILE_HW * sizeof(float);
    const uint32_t summary_tile = tt::constants::TILE_HW * sizeof(uint16_t);
    const uint32_t payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    // Fabric packets of whole FP32 tiles, up to three.
    const uint32_t chunk_bytes = std::min(3 * state_tile, payload / state_tile * state_tile);
    const uint32_t heads_per_link = (batch_heads + attrs.num_links - 1) / attrs.num_links;

    const std::vector<CoreCoord> head_cores(attrs.cores.begin(), attrs.cores.begin() + batch_heads);
    const std::vector<CoreCoord> sender_cores(
        attrs.cores.begin() + batch_heads, attrs.cores.begin() + batch_heads + attrs.num_links);
    const auto head_set = kda_factory_detail::merged_core_set(head_cores);
    const auto sender_set = kda_factory_detail::merged_core_set(sender_cores);

    desc.cbs = {
        kda_factory_detail::cb_descriptor(cb_a, state_tiles, summary_tile, tt::DataFormat::Float16_b, head_set),
        kda_factory_detail::cb_descriptor(cb_b, state_tiles, summary_tile, tt::DataFormat::Float16_b, head_set),
        kda_factory_detail::sharded_cb_descriptor(cb_ein, in.entry_landing),
        kda_factory_detail::sharded_cb_descriptor(cb_eout, in.state_staging),
        kda_factory_detail::cb_descriptor(cb_chron, 1, 64, tt::DataFormat::UInt32, head_set),
        // One page: the sender addresses its unit buffers itself.
        kda_factory_detail::cb_descriptor(
            cb_stage, 1, sender_slots * kt * state_tile, tt::DataFormat::Float32, sender_set),
    };
    desc.semaphores.push_back(
        tt::tt_metal::SemaphoreDescriptor{.id = ack_semaphore_id, .core_ranges = head_set, .initial_value = 0});
    for (uint32_t k = 0; k < heads_per_link; ++k) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = first_notify_semaphore_id + k, .core_ranges = sender_set, .initial_value = 0});
    }

    const std::vector<uint32_t> chronology = {sp_rank, sp_size, attrs.local_rows};
    std::vector<uint32_t> reader_ct = {cb_a, cb_b, cb_ein, cb_chron};
    reader_ct.insert(reader_ct.end(), chronology.begin(), chronology.end());
    reader_ct.push_back(kt);
    for (const auto* tensor : {&in.packed, &in.initial_state, &in.actual_start}) {
        tt::tt_metal::TensorAccessorArgs(tensor->buffer()).append_to(reader_ct);
    }
    std::vector<uint32_t> writer_ct = {cb_eout, cb_chron};
    writer_ct.insert(writer_ct.end(), chronology.begin(), chronology.end());
    writer_ct.push_back(kt);
    writer_ct.push_back(ack_semaphore_id);
    for (const auto* tensor : {&entry, &final_state}) {
        tt::tt_metal::TensorAccessorArgs(tensor->buffer()).append_to(writer_ct);
    }
    std::vector<uint32_t> sender_ct = {cb_stage};
    sender_ct.insert(sender_ct.end(), chronology.begin(), chronology.end());
    sender_ct.insert(sender_ct.end(), {kt * state_tile, chunk_bytes, kt, sender_slots, ack_semaphore_id});
    tt::tt_metal::TensorAccessorArgs(in.actual_start.buffer()).append_to(sender_ct);

    const auto arch = mesh->arch();
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        ttnn::get_compute_kernel_config_args(arch, attrs.compute_kernel_config);
    (void)packer_l1_acc;

    tt::tt_metal::KernelDescriptor reader{
        .kernel_source = std::string(kernel_dir) + "dataflow/reader_relay_affine_prefix.cpp",
        .core_ranges = head_set,
        .compile_time_args = std::move(reader_ct),
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };
    tt::tt_metal::KernelDescriptor writer{
        .kernel_source = std::string(kernel_dir) + "dataflow/writer_relay_affine_prefix.cpp",
        .core_ranges = head_set,
        .compile_time_args = std::move(writer_ct),
        .config = tt::tt_metal::WriterConfigDescriptor{},
    };
    tt::tt_metal::KernelDescriptor compute{
        .kernel_source = std::string(kernel_dir) + "compute/relay_affine_prefix.cpp",
        .core_ranges = head_set,
        .compile_time_args = {cb_a, cb_b, cb_ein, cb_eout, kt},
        // The matmul runs at the affine prefix's fidelity, accumulating in an FP32 DST.
        .config =
            tt::tt_metal::ComputeConfigDescriptor{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = dst_full_sync_en,
                .math_approx_mode = math_approx_mode},
    };
    tt::tt_metal::KernelDescriptor sender{
        .kernel_source = std::string(kernel_dir) + "dataflow/sender_relay_affine_prefix.cpp",
        .core_ranges = sender_set,
        .compile_time_args = std::move(sender_ct),
        // NoC 0, away from the head writers' DRAM copies (the writer default, NoC 1): the relay's hops are latency
        // bound.
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_0},
    };

    const auto [arrival_address, final_address] = attrs.semaphore_addresses;
    for (uint32_t h = 0; h < batch_heads; ++h) {
        const auto& core = head_cores[h];
        const auto sender = mesh->worker_core_from_logical_core(sender_cores[h % attrs.num_links]);
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt;
        reader_rt.push_back(in.packed.buffer());
        reader_rt.push_back(in.initial_state.buffer());
        reader_rt.push_back(in.actual_start.buffer());
        reader_rt.push_back(h);
        reader_rt.push_back(arrival_address);
        reader.emplace_runtime_args(core, reader_rt);
        tt::tt_metal::KernelDescriptor::RTArgList writer_rt;
        writer_rt.push_back(entry.buffer());
        writer_rt.push_back(final_state.buffer());
        writer_rt.push_back(h);
        writer_rt.push_back(in.entry_landing.buffer());
        writer_rt.push_back(in.final_landing.buffer());
        writer_rt.push_back(final_address);
        writer_rt.push_back(first_notify_semaphore_id + h / attrs.num_links);
        writer_rt.push_back(static_cast<uint32_t>(sender.x));
        writer_rt.push_back(static_cast<uint32_t>(sender.y));
        writer.emplace_runtime_args(core, writer_rt);
    }
    desc.kernels = {std::move(reader), std::move(writer), std::move(compute), std::move(sender)};
    constexpr uint32_t sender_kernel = 3;

    const auto source_node = mesh->get_fabric_node_id(coord);
    const auto destination_node = mesh->get_fabric_node_id(successor);
    const auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh, source_node, destination_node);
    TT_FATAL(
        links.size() >= attrs.num_links,
        "relay_affine_prefix uses {} links; the hop to the successor has {}",
        attrs.num_links,
        links.size());
    for (uint32_t link = 0; link < attrs.num_links; ++link) {
        const auto& core = sender_cores[link];
        std::vector<uint32_t> heads, notify;
        for (uint32_t h = link; h < batch_heads; h += attrs.num_links) {
            const auto head = mesh->worker_core_from_logical_core(head_cores[h]);
            heads.push_back(static_cast<uint32_t>(head.x) | (static_cast<uint32_t>(head.y) << 16));
            notify.push_back(first_notify_semaphore_id + h / attrs.num_links);
        }
        tt::tt_metal::KernelDescriptor::RTArgList sender_rt;
        sender_rt.push_back(in.actual_start.buffer());
        sender_rt.push_back(static_cast<uint32_t>(*destination_node.mesh_id));
        sender_rt.push_back(static_cast<uint32_t>(destination_node.chip_id));
        sender_rt.push_back(in.state_staging.buffer());
        sender_rt.push_back(in.entry_landing.buffer());
        sender_rt.push_back(in.final_landing.buffer());
        sender_rt.push_back(arrival_address);
        sender_rt.push_back(final_address);
        sender_rt.push_back(static_cast<uint32_t>(heads.size()));
        sender_rt.append(heads);
        sender_rt.append(notify);
        std::vector<uint32_t> fabric_args;
        tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
            source_node, destination_node, links[link], desc, core, fabric_args, tt::CoreType::WORKER);
        sender_rt.append(fabric_args);
        desc.kernels[sender_kernel].emplace_runtime_args(core, sender_rt);
    }
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor RelayAffinePrefixProgramFactory::create_workload_descriptor(
    const RelayAffinePrefixParams& attrs,
    const RelayAffinePrefixInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    // The output gate's projection (and the copy of its first columns), the same program on every device.
    std::optional<tt::tt_metal::ProgramDescriptor> gate;
    if (const auto& config = attrs.output_gate) {
        gate = kda_factory_detail::projection_program(
            *in.gate_input,
            *in.gate_weight,
            outputs[2],
            config->program_config,
            config->output_column,
            config->compute_kernel_config);
        if (config->output_column > 0) {
            gate = tt::tt_metal::merge_program_descriptors(
                {*gate,
                 kda_factory_detail::copy_tile_columns_program(
                     *in.gate_copy_source,
                     config->copy_source_column,
                     outputs[2],
                     config->output_column,
                     config->copy_cores)});
        }
    }
    for (const auto& coord : tensor_coords.coords()) {
        auto program = build_program(attrs, in, outputs, coord);
        if (gate) {
            program = tt::tt_metal::merge_program_descriptors({program, *gate});
        }
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(program)});
    }
    return workload;
}

}  // namespace ttnn::experimental::prim
