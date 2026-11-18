// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/relay_affine_prefix/device/relay_affine_prefix_program_factory.hpp"

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
// The final carry, on the head cores: the final transition's FP32 E and B, and the relayed final state (the landing
// buffer's final slot) in its matmul and exact views.
constexpr uint32_t cb_fe = 8;
constexpr uint32_t cb_fb = 9;
constexpr uint32_t cb_p = 10;
constexpr uint32_t cb_p_exact = 11;
// The reader's flags and chunk counts for compute.
constexpr uint32_t cb_ctl = 12;
// The entry correction, on every core of a head: the seeds its chunks start from (the entry state, and P for a
// separated tail), the chunk terms F(0) and F(I), their differences and the chunk outputs.
constexpr uint32_t cb_seed = 13;
constexpr uint32_t cb_tail = 14;
constexpr uint32_t cb_terms = 15;
constexpr uint32_t cb_diff = 16;
constexpr uint32_t cb_out = 17;
// On the head cores: the sender releases a head once it has read the head's last column.
constexpr uint32_t ack_semaphore_id = 0;
// On every core of a head: the seeds the head's first core has written there.
constexpr uint32_t landed_semaphore_id = 1;
// On the sender cores, ids 0, 1, ...: head h's finished columns in id h / num_links of its link's senders.
constexpr const char* kernel_dir = "ttnn/cpp/ttnn/operations/experimental/kda/relay_affine_prefix/device/kernels/";

uint32_t xy(const tt::tt_metal::CoreCoord& noc) {
    return static_cast<uint32_t>(noc.x) | (static_cast<uint32_t>(noc.y) << 16);
}

tt::tt_metal::ProgramDescriptor build_program(
    const RelayAffinePrefixParams& attrs,
    const RelayAffinePrefixInputs& in,
    const std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinate& coord) {
    using tt::tt_metal::CoreCoord;
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh = in.packed.device();
    const auto& output = outputs[0];
    const auto& carry = outputs[1];
    // Without an end bound the reader's end accessor is never read.
    const auto& actual_end = in.actual_end.value_or(in.actual_start);
    const uint32_t axis = attrs.sequence_parallel_axis;
    const uint32_t sp_size = mesh->shape()[axis];
    const uint32_t sp_rank = coord[axis];
    auto successor = coord;
    successor[axis] = (sp_rank + 1) % sp_size;
    // A ring only where the fabric config wraps the axis and its closing link is wired.
    const bool ring = ttnn::ccl::get_axis_topology(in.packed, tt::tt_fabric::GetFabricConfig(), axis) ==
                      tt::tt_fabric::Topology::Ring;
    // The successor is the forward neighbor, one hop, except for the last rank of a line that does not wrap, which
    // reaches rank 0 backward, sp_size - 1 hops (as qkv_causal_conv1d_silu's exchange_route).
    const bool successor_forward = ring || sp_rank + 1 < sp_size;
    // The ranks this rank's final carry reaches each way: half the ring each way, else to either end of the line.
    const uint32_t ahead = ring ? sp_size / 2 : sp_size - 1 - sp_rank;
    const uint32_t behind = sp_size - 1 - ahead;
    // The relay's sender multicasts the carry along its connection; a second sender per link covers the ranks behind
    // when that connection heads forward.
    const uint32_t relay_range = successor_forward ? ahead : behind;
    const bool back_sender = successor_forward && behind != 0;
    const uint32_t successor_hops = successor_forward ? 1 : sp_size - 1;
    // 2D routing addresses the successor itself; 1D routing leaves on the link to the neighbor it heads through.
    auto first_hop = coord;
    first_hop[axis] = successor_forward ? (sp_rank + 1) % sp_size : (sp_rank + sp_size - 1) % sp_size;
    const bool fabric_2d = tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig());

    const auto [batch_heads, key_dim] = relay_state_geometry(in);
    const uint32_t kt = key_dim / tt::constants::TILE_HEIGHT;
    const uint32_t state_tiles = kt * kt;
    const uint32_t state_tile = tt::constants::TILE_HW * sizeof(float);
    const uint32_t summary_tile = tt::constants::TILE_HW * sizeof(uint16_t);
    const uint32_t heads_per_link = (batch_heads + attrs.num_links - 1) / attrs.num_links;
    const uint32_t num_chunks = attrs.local_rows / tt::constants::TILE_HEIGHT;
    const uint32_t cores_per_head =
        relay_affine_prefix_cores_per_head(attrs.cores.size(), batch_heads, attrs.num_links);
    const uint32_t core_chunks = num_chunks / cores_per_head;
    const uint32_t grid_x = mesh->compute_with_storage_grid_size().x;
    const uint32_t first_core = attrs.cores[0].y * grid_x + attrs.cores[0].x;
    // Sharded terms are read in place; interleaved ones stream a chunk row at a time.
    const bool streamed = !in.terms.memory_config().is_sharded();

    const std::vector<CoreCoord> correction_cores(
        attrs.cores.begin(), attrs.cores.begin() + batch_heads * cores_per_head);
    std::vector<CoreCoord> head_cores;
    for (uint32_t h = 0; h < batch_heads; ++h) {
        head_cores.push_back(correction_cores[h * cores_per_head]);
    }
    // The relay's sender per link, then (with ranks behind) the second carry sender per link, towards the predecessor.
    // Every chip places both (the programs of a range merge), and a second sender without ranks behind its rank idles.
    const std::vector<CoreCoord> sender_cores(
        attrs.cores.begin() + correction_cores.size(),
        attrs.cores.begin() + correction_cores.size() + 2 * attrs.num_links);
    const auto correction_set = kda_factory_detail::merged_core_set(correction_cores);
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
    // A chunk row of F(0) then of F(I). In place, the core's whole terms shard: its chunk rows of F(0), then of F(I).
    // Streamed, two chunks' rows.
    auto terms_cb = kda_factory_detail::cb_descriptor(
        cb_terms, (streamed ? 2 : core_chunks) * 2 * kt, state_tile, tt::DataFormat::Float32, correction_set);
    if (!streamed) {
        terms_cb.buffer = in.terms.buffer();
    }
    const uint32_t output_tile = tt::constants::TILE_HW * sizeof(uint16_t);
    desc.cbs = {
        // The seeds first: every core of a head takes them at the address its head's first core writes them to.
        kda_factory_detail::cb_descriptor(cb_seed, state_tiles, state_tile, tt::DataFormat::Float32, correction_set),
        kda_factory_detail::cb_descriptor(cb_tail, state_tiles, state_tile, tt::DataFormat::Float32, correction_set),
        std::move(terms_cb),
        kda_factory_detail::cb_descriptor(
            cb_diff, (streamed ? 2 : core_chunks) * kt, state_tile, tt::DataFormat::Float32, correction_set),
        // A block of chunk rows: all of a seed's chunks in place, two chunks streamed.
        kda_factory_detail::cb_descriptor(
            cb_out, (streamed ? 2 : core_chunks) * kt, output_tile, tt::DataFormat::Float16_b, correction_set),
        // Two DRAM-aligned slots: actual_start (then the writer's chronology words), then the slab index.
        kda_factory_detail::cb_descriptor(cb_chron, 1, 2 * chron_slot_bytes, tt::DataFormat::UInt32, correction_set),
        kda_factory_detail::cb_descriptor(cb_ctl, 1, 64, tt::DataFormat::UInt32, correction_set),
        kda_factory_detail::cb_descriptor(cb_e, state_tiles, summary_tile, tt::DataFormat::Float16_b, head_set),
        kda_factory_detail::cb_descriptor(cb_b, state_tiles, summary_tile, tt::DataFormat::Float16_b, head_set),
        landing_slot_cb(cb_ein, 0),
        landing_slot_cb(cb_ein_exact, 0),
        kda_factory_detail::cb_descriptor(cb_identity, 1, summary_tile, tt::DataFormat::Float16_b, head_set),
        kda_factory_detail::cb_descriptor(cb_eout, state_tiles, state_tile, tt::DataFormat::Float32, eout_set),
        // One page: the sender addresses its slots itself.
        kda_factory_detail::cb_descriptor(
            cb_stage, 1, sender_slots * kt * state_tile, tt::DataFormat::Float32, sender_set),
        kda_factory_detail::cb_descriptor(cb_fe, state_tiles, state_tile, tt::DataFormat::Float32, head_set),
        kda_factory_detail::cb_descriptor(cb_fb, state_tiles, state_tile, tt::DataFormat::Float32, head_set),
        landing_slot_cb(cb_p, 1),
        landing_slot_cb(cb_p_exact, 1),
    };
    desc.semaphores.push_back(
        tt::tt_metal::SemaphoreDescriptor{.id = ack_semaphore_id, .core_ranges = head_set, .initial_value = 0});
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = landed_semaphore_id, .core_ranges = correction_set, .initial_value = 0});
    for (uint32_t k = 0; k < heads_per_link; ++k) {
        desc.semaphores.push_back(
            tt::tt_metal::SemaphoreDescriptor{.id = k, .core_ranges = sender_set, .initial_value = 0});
    }

    // sp_rank and the per-chip routing are runtime args, so a range's per-chip programs merge into one.
    std::vector<uint32_t> reader_ct = {
        cb_e,
        cb_b,
        cb_ein,
        cb_chron,
        sp_size,
        attrs.local_rows,
        kt,
        cb_ein_exact,
        cb_identity,
        cb_fe,
        cb_fb,
        cb_p,
        cb_p_exact,
        cb_ctl,
        uint32_t(in.actual_end.has_value()),
        cb_seed,
        cb_tail,
        cb_terms,
        core_chunks,
        uint32_t(streamed),
        landed_semaphore_id,
        cores_per_head,
        grid_x,
        first_core};
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
    reader_ct.push_back(ack_semaphore_id);
    const auto& slab_index = slab ? *in.slab_index : in.actual_start;
    for (const auto* tensor :
         {&in.packed,
          &in.initial_state,
          &in.actual_start,
          &in.final_e,
          &in.final_b,
          &actual_end,
          &in.terms,
          &slab_index}) {
        tt::tt_metal::TensorAccessorArgs(tensor->buffer()).append_to(reader_ct);
    }
    std::vector<uint32_t> writer_ct = {
        cb_eout,
        cb_chron,
        cb_out,
        kt,
        num_chunks,
        ack_semaphore_id,
        cores_per_head,
        grid_x,
        core_chunks,
        attrs.num_links,
        first_core,
        uint32_t(streamed)};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct);
    tt::tt_metal::TensorAccessorArgs(carry.buffer()).append_to(writer_ct);
    writer_ct.insert(writer_ct.end(), slab_ct.begin(), slab_ct.end());
    std::vector<uint32_t> sender_ct = {cb_stage, cb_eout, sp_size, attrs.local_rows};
    sender_ct.insert(
        sender_ct.end(), {kt * state_tile, relay_affine_prefix_packet_bytes(), kt, sender_slots, ack_semaphore_id});
    tt::tt_metal::TensorAccessorArgs(in.actual_start.buffer()).append_to(sender_ct);
    tt::tt_metal::TensorAccessorArgs(carry.buffer()).append_to(sender_ct);
    sender_ct.insert(sender_ct.end(), slab_ct.begin(), slab_ct.end());
    tt::tt_metal::TensorAccessorArgs(slab_index.buffer()).append_to(sender_ct);

    tt::tt_metal::KernelDescriptor reader{
        .kernel_source = std::string(kernel_dir) + "dataflow/reader_relay_affine_prefix.cpp",
        .core_ranges = correction_set,
        .compile_time_args = std::move(reader_ct),
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };
    tt::tt_metal::KernelDescriptor writer{
        .kernel_source = std::string(kernel_dir) + "dataflow/writer_relay_affine_prefix.cpp",
        .core_ranges = correction_set,
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
        .core_ranges = correction_set,
        .compile_time_args =
            {cb_e,
             cb_b,
             cb_ein,
             cb_eout,
             kt,
             cb_ein_exact,
             cb_identity,
             cb_fe,
             cb_fb,
             cb_p,
             cb_p_exact,
             cb_ctl,
             cb_seed,
             cb_tail,
             cb_terms,
             cb_diff,
             cb_out,
             core_chunks,
             uint32_t(streamed),
             cores_per_head,
             grid_x,
             first_core},
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
    // The correction cores take no per-core args (the dispatch of 96 cores' own args costs the call more than their
    // work): each kernel finds its head and share from its place in the grid, the rest is common per device.
    tt::tt_metal::KernelDescriptor::RTArgList reader_args;
    for (auto* buffer :
         {in.packed.buffer(),
          in.initial_state.buffer(),
          in.actual_start.buffer(),
          in.final_e.buffer(),
          in.final_b.buffer(),
          actual_end.buffer(),
          in.terms.buffer(),
          slab_index.buffer(),
          in.landing.buffer()}) {
        reader_args.push_back(buffer);
    }
    reader_args.push_back(arrival_address);
    reader_args.push_back(final_address);
    reader_args.push_back(sp_rank);
    for (const auto& core : correction_cores) {
        reader_args.push_back(xy(mesh->worker_core_from_logical_core(core)));
    }
    reader.emplace_common_runtime_args(reader_args);
    tt::tt_metal::KernelDescriptor::RTArgList writer_args;
    writer_args.push_back(output.buffer());
    writer_args.push_back(carry.buffer());
    writer_args.push_back(uint32_t(back_sender));
    for (const auto& core : sender_cores) {
        writer_args.push_back(xy(mesh->worker_core_from_logical_core(core)));
    }
    writer.emplace_common_runtime_args(writer_args);
    desc.kernels = {std::move(reader), std::move(writer), std::move(compute), std::move(sender)};
    tt::tt_metal::KernelHandle sender_kernel = 3;

    const auto source_node = mesh->get_fabric_node_id(coord);
    const auto destination_node = mesh->get_fabric_node_id(successor);
    const auto connection_node = fabric_2d ? destination_node : mesh->get_fabric_node_id(first_hop);
    const auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh, source_node, connection_node);
    desc.kernels[sender_kernel].emplace_common_runtime_args(
        {in.actual_start.buffer(), in.landing.buffer(), carry.buffer(), slab_index.buffer()});
    // The second carry senders connect to the predecessor where they carry; an idle one is given the relay's
    // connection, which it never opens, so every chip's runtime args keep one layout.
    auto predecessor = coord;
    predecessor[axis] = (sp_rank + sp_size - 1) % sp_size;
    const auto back_node = back_sender ? mesh->get_fabric_node_id(predecessor) : connection_node;
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
            heads.push_back(xy(head));
            notify.push_back(h / attrs.num_links);
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
            sender_rt.push_back(backward ? (back_sender ? behind : 0) : relay_range);
            sender_rt.push_back(uint32_t(!backward || back_sender));
            std::vector<uint32_t> fabric_args;
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                *mesh,
                source_node,
                {backward ? back_node : connection_node},
                {backward ? back_links[link] : links[link]},
                desc,
                sender_kernel,
                core,
                fabric_args);
            sender_rt.append(fabric_args);
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
