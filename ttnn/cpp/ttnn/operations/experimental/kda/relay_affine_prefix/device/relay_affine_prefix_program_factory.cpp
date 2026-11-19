// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/relay_affine_prefix/device/relay_affine_prefix_program_factory.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t cb_e = 0;
constexpr uint32_t cb_b = 1;
constexpr uint32_t cb_ein = 2;
constexpr uint32_t cb_eout = 3;
constexpr uint32_t cb_chron = 4;
constexpr uint32_t cb_stage = 5;
// cb_ein's landing slot again, which compute unpacks straight to DST to add the entry state exactly.
constexpr uint32_t cb_ein_exact = 6;
// One BF16 identity tile: compute adds B as I @ B in the matmul, exactly (B in srcA, 1.0 in srcB).
constexpr uint32_t cb_identity = 7;
// The final carry, on the head cores: the final transition's FP32 E and B, the relayed final state (the landing
// buffer's final slot) in its matmul and exact views, and the reader's flags for compute.
constexpr uint32_t cb_fe = 8;
constexpr uint32_t cb_fb = 9;
constexpr uint32_t cb_p = 10;
constexpr uint32_t cb_p_exact = 11;
constexpr uint32_t cb_ctl = 12;
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
    const auto& carry = outputs[2];
    // Without an end bound the reader's end accessor is never read.
    const auto& actual_end = in.actual_end.value_or(in.actual_start);
    const uint32_t axis = attrs.sequence_parallel_axis;
    const uint32_t sp_size = mesh->shape()[axis];
    const uint32_t sp_rank = coord[axis];
    auto successor = coord;
    successor[axis] = (sp_rank + 1) % sp_size;
    // A ring only where the fabric config wraps the axis and its closing link is wired.
    const bool ring = ttnn::ccl::get_axis_topology(in.packed, tt::tt_fabric::GetFabricConfig(*mesh), axis) ==
                      tt::tt_fabric::Topology::Ring;
    // The successor is the forward neighbor, one hop, except for the last rank of a line that does not wrap, which
    // reaches rank 0 backward, sp_size - 1 hops (as qkv_causal_conv1d_silu's exchange_route).
    const bool successor_forward = ring || sp_rank + 1 < sp_size;
    const uint32_t successor_hops = successor_forward ? 1 : sp_size - 1;
    // 2D routing addresses the successor itself; 1D routing leaves on the link to the neighbor it heads through.
    auto first_hop = coord;
    first_hop[axis] = successor_forward ? (sp_rank + 1) % sp_size : (sp_rank + sp_size - 1) % sp_size;
    const bool fabric_2d = tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig(*mesh));
    auto predecessor = coord;
    predecessor[axis] = (sp_rank + sp_size - 1) % sp_size;
    const auto source_node = mesh->get_fabric_node_id(coord);
    const auto destination_node = mesh->get_fabric_node_id(successor);
    const auto predecessor_node = mesh->get_fabric_node_id(predecessor);
    const auto connection_node = fabric_2d ? destination_node : mesh->get_fabric_node_id(first_hop);
    const auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh, source_node, connection_node);

    // The ranks this rank's final carry reaches each way: half the ring each way, else to either end of the line.
    const uint32_t ahead = ring ? sp_size / 2 : sp_size - 1 - sp_rank;
    const uint32_t behind = sp_size - 1 - ahead;
    const auto [batch_heads, key_dim] = relay_state_geometry(in);
    const uint32_t kt = key_dim / tt::constants::TILE_HEIGHT;
    const uint32_t state_tiles = kt * kt;
    const uint32_t state_tile = tt::constants::TILE_HW * sizeof(float);
    const uint32_t summary_tile = tt::constants::TILE_HW * sizeof(uint16_t);
    const uint32_t heads_per_link = (batch_heads + attrs.num_links - 1) / attrs.num_links;

    // A 2D multicast leaves every hop in one ETH direction. Where the SP axis turns on a 2D fabric, the carry goes to
    // each other rank as a 2D unicast instead, on a connection in the direction its route leaves: the relay's own
    // towards the successor, else one of the link's second sender (towards the predecessor, or across the mesh), else
    // one more on the relay's sender.
    const bool unicast_peers = fabric_2d && !ttnn::ccl::is_axis_straight(*mesh, axis);
    struct CarrySender {
        std::vector<tt::tt_fabric::FabricNodeId> connections;  // the node each connection is opened towards
        std::vector<uint32_t> peer_args;                       // per rank: its connection, chip and mesh
        uint32_t peers = 0;
    };
    std::array<CarrySender, 2> carry_senders;  // the relay's sender, then the second
    if (unicast_peers) {
        // A connection takes two of a sender core's semaphores, beside one per head the sender serves.
        const uint32_t max_connections = std::min(
            relay_affine_prefix_max_connections,
            (relay_affine_prefix_sender_semaphores - heads_per_link) / relay_affine_prefix_connection_semaphores);
        // Routes in one direction share its links, so a link's ETH core tells the direction.
        const auto same_direction = [&](const tt::tt_fabric::FabricNodeId& peer,
                                        const tt::tt_fabric::FabricNodeId& node) {
            const auto node_links = tt::tt_fabric::get_forwarding_link_indices(*mesh, source_node, node);
            const auto peer_links = tt::tt_fabric::get_forwarding_link_indices(*mesh, source_node, peer);
            return !node_links.empty() &&
                   std::find(peer_links.begin(), peer_links.end(), node_links.front()) != peer_links.end() &&
                   tt::tt_fabric::get_forwarding_eth_core(*mesh, source_node, peer, node_links.front()) ==
                       tt::tt_fabric::get_forwarding_eth_core(*mesh, source_node, node, node_links.front());
        };
        carry_senders[0].connections = {destination_node};
        for (uint32_t step = 1; step < sp_size; ++step) {
            auto peer_coord = coord;
            peer_coord[axis] = (sp_rank + step) % sp_size;
            const auto peer = mesh->get_fabric_node_id(peer_coord);
            std::optional<std::pair<uint32_t, uint32_t>> found;  // sender, connection
            for (uint32_t s = 0; s < carry_senders.size() && !found; ++s) {
                const auto& connections = carry_senders[s].connections;
                for (uint32_t c = 0; c < connections.size() && !found; ++c) {
                    if (same_direction(peer, connections[c])) {
                        found = {s, c};
                    }
                }
            }
            if (!found) {
                const uint32_t s = carry_senders[1].connections.size() < max_connections ? 1 : 0;
                TT_FATAL(
                    carry_senders[s].connections.size() < max_connections,
                    "relay_affine_prefix: the routes from SP rank {} to the other ranks leave in more directions than "
                    "its two carry senders have semaphores for ({} connections each at {} heads per link)",
                    sp_rank,
                    max_connections,
                    heads_per_link);
                carry_senders[s].connections.push_back(peer);
                found = {s, static_cast<uint32_t>(carry_senders[s].connections.size() - 1)};
            }
            auto& sender = carry_senders[found->first];
            sender.peer_args.insert(
                sender.peer_args.end(),
                {found->second, static_cast<uint32_t>(peer.chip_id), static_cast<uint32_t>(*peer.mesh_id)});
            ++sender.peers;
        }
    }
    // The relay's sender sends the carry along its connection; a second sender per link covers the ranks behind when
    // that connection heads forward (unicast: the ranks the relay's connection does not lead to).
    const uint32_t relay_range = unicast_peers ? carry_senders[0].peers : successor_forward ? ahead : behind;
    const bool back_sender = unicast_peers ? carry_senders[1].peers != 0 : successor_forward && behind != 0;
    const uint32_t back_range = unicast_peers ? carry_senders[1].peers : back_sender ? behind : 0;

    const std::vector<CoreCoord> head_cores(attrs.cores.begin(), attrs.cores.begin() + batch_heads);
    // The relay's sender per link, then (with ranks behind) the second carry sender per link, towards the predecessor.
    // Every chip places both (the programs of a range merge), and a second sender without ranks behind its rank idles.
    const std::vector<CoreCoord> sender_cores(
        attrs.cores.begin() + batch_heads, attrs.cores.begin() + batch_heads + 2 * attrs.num_links);
    const auto head_set = kda_factory_detail::merged_core_set(head_cores);
    const auto sender_set = kda_factory_detail::merged_core_set(sender_cores);
    // The next state stages in a program CB at one address on every head and sender core: a sender reads a head's
    // columns at its own cb_eout address.
    const auto eout_set = head_set.merge(sender_set);

    const uint32_t sender_slots = relay_affine_prefix_sender_slots(kt, mesh->arch());
    const uint32_t chron_slot_bytes =
        std::max(tt::tt_metal::hal::get_dram_alignment(), tt::tt_metal::hal::get_l1_alignment());
    // A K x K slot of the landing buffer on each head core, in place.
    const auto landing_slot_cb = [&](uint32_t index, uint32_t slot) {
        auto cb = kda_factory_detail::cb_descriptor(index, state_tiles, state_tile, tt::DataFormat::Float32, head_set);
        cb.buffer = in.landing.buffer();
        cb.address_offset = slot * state_tiles * state_tile;
        return cb;
    };
    desc.cbs = {
        kda_factory_detail::cb_descriptor(cb_e, state_tiles, summary_tile, tt::DataFormat::Float16_b, head_set),
        kda_factory_detail::cb_descriptor(cb_b, state_tiles, summary_tile, tt::DataFormat::Float16_b, head_set),
        landing_slot_cb(cb_ein, 0),
        landing_slot_cb(cb_ein_exact, 0),
        kda_factory_detail::cb_descriptor(cb_identity, 1, summary_tile, tt::DataFormat::Float16_b, head_set),
        kda_factory_detail::cb_descriptor(cb_eout, state_tiles, state_tile, tt::DataFormat::Float32, eout_set),
        // Two DRAM-aligned slots: actual_start, then the slab index.
        kda_factory_detail::cb_descriptor(cb_chron, 1, 2 * chron_slot_bytes, tt::DataFormat::UInt32, head_set),
        // One page: the sender addresses its slots itself.
        kda_factory_detail::cb_descriptor(
            cb_stage, 1, sender_slots * kt * state_tile, tt::DataFormat::Float32, sender_set),
        kda_factory_detail::cb_descriptor(cb_fe, state_tiles, state_tile, tt::DataFormat::Float32, head_set),
        kda_factory_detail::cb_descriptor(cb_fb, state_tiles, state_tile, tt::DataFormat::Float32, head_set),
        landing_slot_cb(cb_p, 1),
        landing_slot_cb(cb_p_exact, 1),
        kda_factory_detail::cb_descriptor(cb_ctl, 1, 64, tt::DataFormat::UInt32, head_set),
    };
    desc.semaphores.push_back(
        tt::tt_metal::SemaphoreDescriptor{.id = ack_semaphore_id, .core_ranges = head_set, .initial_value = 0});
    for (uint32_t k = 0; k < heads_per_link; ++k) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = first_notify_semaphore_id + k, .core_ranges = sender_set, .initial_value = 0});
    }

    // sp_rank and the per-chip routing are runtime args, so a range's per-chip programs merge into one.
    const std::vector<uint32_t> chronology = {sp_size, attrs.local_rows};
    std::vector<uint32_t> reader_ct = {cb_e, cb_b, cb_ein, cb_chron};
    reader_ct.insert(reader_ct.end(), chronology.begin(), chronology.end());
    reader_ct.push_back(kt);
    reader_ct.push_back(cb_ein_exact);
    reader_ct.push_back(cb_identity);
    // The slab (slab_kernel_args), its entry's tiles and the word the slab index lands at: the reader reads the
    // initial state from the entry, and the writer and senders write the final carry back into it.
    const bool slab = in.slab_index.has_value();
    std::vector<uint32_t> slab_ct;
    for (const uint32_t arg : kda_factory_detail::slab_kernel_args(
             slab ? &in.initial_state : nullptr, attrs.slab_index_scale, attrs.slab_index_offset)) {
        slab_ct.push_back(arg);
    }
    slab_ct.push_back(batch_heads * state_tiles);
    slab_ct.push_back(chron_slot_bytes / sizeof(uint32_t));
    reader_ct.insert(reader_ct.end(), slab_ct.begin(), slab_ct.end());
    const auto& slab_index = slab ? *in.slab_index : in.actual_start;
    for (const auto* tensor : {&in.packed, &in.initial_state, &in.actual_start, &slab_index}) {
        tt::tt_metal::TensorAccessorArgs(tensor->buffer()).append_to(reader_ct);
    }
    reader_ct.insert(reader_ct.end(), {cb_fe, cb_fb, cb_p, cb_p_exact, cb_ctl, uint32_t(in.actual_end.has_value())});
    for (const auto* tensor : {&in.final_e, &in.final_b, &actual_end}) {
        tt::tt_metal::TensorAccessorArgs(tensor->buffer()).append_to(reader_ct);
    }
    std::vector<uint32_t> writer_ct = {cb_eout, cb_chron};
    writer_ct.insert(writer_ct.end(), chronology.begin(), chronology.end());
    writer_ct.push_back(kt);
    writer_ct.push_back(ack_semaphore_id);
    for (const auto* tensor : {&entry, &final_state}) {
        tt::tt_metal::TensorAccessorArgs(tensor->buffer()).append_to(writer_ct);
    }
    tt::tt_metal::TensorAccessorArgs(carry.buffer()).append_to(writer_ct);
    writer_ct.insert(writer_ct.end(), slab_ct.begin(), slab_ct.end());
    std::vector<uint32_t> sender_ct = {cb_stage, cb_eout};
    sender_ct.insert(sender_ct.end(), chronology.begin(), chronology.end());
    sender_ct.insert(
        sender_ct.end(), {kt * state_tile, relay_affine_prefix_packet_bytes(), kt, sender_slots, ack_semaphore_id});
    tt::tt_metal::TensorAccessorArgs(in.actual_start.buffer()).append_to(sender_ct);
    tt::tt_metal::TensorAccessorArgs(carry.buffer()).append_to(sender_ct);
    sender_ct.insert(sender_ct.end(), slab_ct.begin(), slab_ct.end());
    tt::tt_metal::TensorAccessorArgs(slab_index.buffer()).append_to(sender_ct);
    sender_ct.push_back(uint32_t(unicast_peers));

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
    auto compute_config = kda_factory_detail::compute_config_descriptor(mesh->arch(), attrs.compute_kernel_config);
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    for (const uint32_t exact : {cb_ein_exact, cb_p_exact, cb_fb}) {
        compute_config.unpack_to_dest_mode[exact] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    tt::tt_metal::KernelDescriptor compute{
        .kernel_source = std::string(kernel_dir) + "compute/relay_affine_prefix.cpp",
        .core_ranges = head_set,
        .compile_time_args =
            {cb_e, cb_b, cb_ein, cb_eout, kt, cb_ein_exact, cb_identity, cb_fe, cb_fb, cb_p, cb_p_exact, cb_ctl},
        .config = std::move(compute_config),
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

    const auto [arrival_address, final_address, carry_address] = attrs.semaphore_addresses;
    // Every core reads the same buffers, so they are bound once per kernel.
    reader.emplace_common_runtime_args(
        {in.packed.buffer(),
         in.initial_state.buffer(),
         in.actual_start.buffer(),
         slab_index.buffer(),
         in.final_e.buffer(),
         in.final_b.buffer(),
         actual_end.buffer()});
    writer.emplace_common_runtime_args({entry.buffer(), final_state.buffer(), in.landing.buffer(), carry.buffer()});
    for (uint32_t h = 0; h < batch_heads; ++h) {
        const auto& core = head_cores[h];
        const auto sender_core = mesh->worker_core_from_logical_core(sender_cores[h % attrs.num_links]);
        const auto back_core = mesh->worker_core_from_logical_core(sender_cores[attrs.num_links + h % attrs.num_links]);
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt;
        reader_rt.push_back(h);
        reader_rt.push_back(arrival_address);
        reader_rt.push_back(final_address);
        reader_rt.push_back(sp_rank);
        reader.emplace_runtime_args(core, reader_rt);
        tt::tt_metal::KernelDescriptor::RTArgList writer_rt;
        writer_rt.push_back(h);
        writer_rt.push_back(first_notify_semaphore_id + h / attrs.num_links);
        writer_rt.push_back(static_cast<uint32_t>(sender_core.x));
        writer_rt.push_back(static_cast<uint32_t>(sender_core.y));
        writer_rt.push_back(uint32_t(back_sender));
        writer_rt.push_back(static_cast<uint32_t>(back_core.x));
        writer_rt.push_back(static_cast<uint32_t>(back_core.y));
        writer_rt.push_back(sp_rank);
        writer.emplace_runtime_args(core, writer_rt);
    }
    desc.kernels = {std::move(reader), std::move(writer), std::move(compute), std::move(sender)};
    tt::tt_metal::KernelHandle sender_kernel = 3;

    desc.kernels[sender_kernel].emplace_common_runtime_args(
        {in.actual_start.buffer(), in.landing.buffer(), carry.buffer(), slab_index.buffer()});
    // The second carry senders connect to the predecessor where they carry; an idle one is given the relay's
    // connection, which it never opens, so every chip's runtime args keep one layout.
    const auto back_node = !back_sender    ? connection_node
                           : unicast_peers ? carry_senders[1].connections.front()
                                           : predecessor_node;
    const auto back_links = tt::tt_fabric::get_forwarding_link_indices(*mesh, source_node, back_node);
    const auto drain_core = mesh->worker_core_from_logical_core(sender_cores.front());
    TT_FATAL(
        links.size() >= attrs.num_links && back_links.size() >= attrs.num_links,
        "relay_affine_prefix uses {} links; the hops towards the successor and predecessor have {} and {}",
        attrs.num_links,
        links.size(),
        back_links.size());
    for (uint32_t link = 0; link < attrs.num_links; ++link) {
        std::vector<uint32_t> heads, notify;
        for (uint32_t h = link; h < batch_heads; h += attrs.num_links) {
            const auto head = mesh->worker_core_from_logical_core(head_cores[h]);
            heads.push_back(static_cast<uint32_t>(head.x) | (static_cast<uint32_t>(head.y) << 16));
            notify.push_back(first_notify_semaphore_id + h / attrs.num_links);
        }
        for (uint32_t direction = 0; direction < 2; ++direction) {
            const bool backward = direction == 1;
            const auto& core = sender_cores[direction * attrs.num_links + link];
            const auto& peer = backward ? back_node : destination_node;
            tt::tt_metal::KernelDescriptor::RTArgList sender_rt;
            sender_rt.push_back(static_cast<uint32_t>(*peer.mesh_id));
            sender_rt.push_back(static_cast<uint32_t>(peer.chip_id));
            sender_rt.push_back(arrival_address);
            sender_rt.push_back(final_address);
            sender_rt.push_back(sp_rank);
            sender_rt.push_back(successor_hops);
            sender_rt.push_back(static_cast<uint32_t>(heads.size()));
            sender_rt.append(heads);
            sender_rt.append(notify);
            sender_rt.push_back(carry_address);
            sender_rt.push_back(static_cast<uint32_t>(drain_core.x));
            sender_rt.push_back(static_cast<uint32_t>(drain_core.y));
            sender_rt.push_back(attrs.num_links);
            sender_rt.push_back(link);
            sender_rt.push_back(uint32_t(!backward));
            sender_rt.push_back(backward ? back_range : relay_range);
            // An idle second sender is given the relay's connection, which it never opens.
            std::vector<tt::tt_fabric::FabricNodeId> nodes = {backward ? back_node : connection_node};
            std::vector<uint32_t> node_links = {backward ? back_links[link] : links[link]};
            const auto& carry_sender = carry_senders[direction];
            if (unicast_peers && (!backward || back_sender)) {
                // The further connections, each on this link's index among its direction's links.
                for (size_t c = 1; c < carry_sender.connections.size(); ++c) {
                    const auto& node = carry_sender.connections[c];
                    const auto extra_links = tt::tt_fabric::get_forwarding_link_indices(*mesh, source_node, node);
                    TT_FATAL(
                        extra_links.size() > link,
                        "relay_affine_prefix uses {} links; a carry route from SP rank {} leaves on a direction with "
                        "{}",
                        attrs.num_links,
                        sp_rank,
                        extra_links.size());
                    nodes.push_back(node);
                    node_links.push_back(extra_links[link]);
                }
            }
            sender_rt.push_back(!backward || back_sender ? static_cast<uint32_t>(nodes.size()) : 0u);
            std::vector<uint32_t> fabric_args;
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                *mesh, source_node, nodes, node_links, desc, sender_kernel, core, fabric_args);
            sender_rt.append(fabric_args);
            // Unicast: per rank this sender carries to, its connection, chip and mesh.
            sender_rt.append(carry_sender.peer_args);
            desc.kernels[sender_kernel].emplace_runtime_args(core, sender_rt);
        }
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
    for (const auto& coord : tensor_coords.coords()) {
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord), build_program(attrs, in, outputs, coord)});
    }
    return workload;
}

}  // namespace ttnn::experimental::prim
