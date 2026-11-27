// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/relay_chunk_scan/device/relay_chunk_scan_program_factory.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/discovery/hal.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/workload/legacy/tensor_accessor_args.hpp>
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
// The final carry, on the relay cores: the final transition's FP32 E and B, the relayed final states (the landing
// buffer's final slots) in their matmul and exact views, and the reader's flags for compute.
constexpr uint32_t cb_fe = 8;
constexpr uint32_t cb_fb = 9;
constexpr uint32_t cb_p = 10;
constexpr uint32_t cb_p_exact = 11;
constexpr uint32_t cb_ctl = 12;
// The entry correction, on every correction core of a head: the seeds its chunks start from (the entry state, and P for
// a separated tail), the chunk terms F(0) and F(I), their differences and the chunk outputs.
constexpr uint32_t cb_seed = 13;
constexpr uint32_t cb_tail = 14;
constexpr uint32_t cb_terms = 15;
constexpr uint32_t cb_diff = 16;
constexpr uint32_t cb_out = 17;
// The gated output norm, on every core of a head: the squares of a chunk row, the mean's scaler, 1 / rms, the gate
// rows, the weight and its rows broadcast down a tile, and the normalized rows the writer writes.
constexpr uint32_t cb_square = 18;
constexpr uint32_t cb_scaler = 19;
constexpr uint32_t cb_inv_rms = 20;
constexpr uint32_t cb_gate = 21;
constexpr uint32_t cb_weight = 22;
constexpr uint32_t cb_weight_rows = 23;
constexpr uint32_t cb_norm = 24;
constexpr uint32_t cb_gate_factor = 25;
// The output gate's operands, a K block each of x and of W, on the cores that form it.
constexpr uint32_t cb_gin0 = 26;
constexpr uint32_t cb_gin1 = 27;
// Each receiver's count of the K block slots it has taken, on its leader (gate_matmul_dataflow.hpp): x's words, then
// W's.
constexpr uint32_t cb_progress = 28;
constexpr uint32_t gate_progress_words = 32;
// On the relay cores, ids 0, 1, ...: per head slot, the senders' releases of the head.
constexpr uint32_t first_ack_semaphore_id = 0;
// On the sender cores, ids 0, 1, ...: per head the sender serves, the head's finished columns.
constexpr uint32_t first_notify_semaphore_id = 0;
// On the correction cores: the seeds the head's relay core has written there, then the output gate's x and W blocks
// their leaders have sent there (gate_matmul_dataflow.hpp).
constexpr uint32_t landed_semaphore_id = 0;
constexpr uint32_t x_valid_semaphore_id = 1;
constexpr uint32_t w_valid_semaphore_id = 2;
constexpr const char* kernel_dir = "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/";

uint32_t xy(const tt::tt_metal::CoreCoord& noc) {
    return static_cast<uint32_t>(noc.x) | (static_cast<uint32_t>(noc.y) << 16);
}

// Cores as runs along a row or a column, whichever gives fewer.
std::vector<std::vector<tt::tt_metal::CoreCoord>> core_runs(std::vector<tt::tt_metal::CoreCoord> cores) {
    using tt::tt_metal::CoreCoord;
    const auto runs_along = [&](bool vertical) {
        const auto key = [&](const CoreCoord& c) { return vertical ? std::pair{c.x, c.y} : std::pair{c.y, c.x}; };
        std::ranges::sort(cores, [&](const auto& a, const auto& b) { return key(a) < key(b); });
        std::vector<std::vector<CoreCoord>> runs;
        for (const auto& core : cores) {
            if (!runs.empty() && key(runs.back().back()).first == key(core).first &&
                key(runs.back().back()).second + 1 == key(core).second) {
                runs.back().push_back(core);
            } else {
                runs.push_back({core});
            }
        }
        return runs;
    };
    auto runs = runs_along(false);
    if (auto vertical = runs_along(true); vertical.size() < runs.size()) {
        runs = std::move(vertical);
    }
    return runs;
}

// A set of the output gate's sharing table (gate_matmul_dataflow.hpp): a leader and its receivers.
struct GateSet {
    tt::tt_metal::CoreCoord leader;
    std::vector<tt::tt_metal::CoreCoord> receivers;
};

std::vector<uint32_t> gate_set_table(
    const tt::tt_metal::distributed::MeshDevice& mesh, const std::vector<std::vector<GateSet>>& groups) {
    const auto noc = [&](const tt::tt_metal::CoreCoord& core) { return xy(mesh.worker_core_from_logical_core(core)); };
    std::vector<uint32_t> table(groups.size());
    for (size_t g = 0; g < groups.size(); ++g) {
        table[g] = table.size();
        table.push_back(groups[g].size());
        for (const auto& set : groups[g]) {
            const auto rectangles = core_runs(set.receivers);
            table.insert(table.end(), {noc(set.leader), uint32_t(set.receivers.size()), uint32_t(rectangles.size())});
            uint32_t nodes = 0;
            for (const auto& run : rectangles) {
                const uint32_t low = noc(run.front());
                const uint32_t high = noc(run.back());
                table.insert(table.end(), {low, high, uint32_t(run.size())});
                nodes += ((high & 0xFFFF) - (low & 0xFFFF) + 1) * ((high >> 16) - (low >> 16) + 1);
            }
            TT_FATAL(
                nodes <= gate_progress_words,
                "relay_chunk_scan: an output gate share spans {} NoC nodes, more than its {} progress words",
                nodes,
                gate_progress_words);
        }
    }
    return table;
}

tt::tt_metal::ProgramDescriptor build_program(
    const RelayChunkScanParams& attrs,
    const RelayChunkScanInputs& in,
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

    // The ranks this rank's final carry reaches each way: on a ring the whole ring both ways (a head's carry goes one
    // way, see carry_mask), else to either end of the line.
    const uint32_t ahead = ring ? sp_size - 1 : sp_size - 1 - sp_rank;
    const uint32_t behind = ring ? sp_size - 1 : sp_size - 1 - ahead;
    const auto [batch_heads, key_dim] = relay_state_geometry(in);
    const uint32_t kt = key_dim / tt::constants::TILE_HEIGHT;
    const uint32_t state_tiles = kt * kt;
    const uint32_t state_tile = tt::constants::TILE_HW * sizeof(float);
    const uint32_t summary_tile = tt::constants::TILE_HW * sizeof(uint16_t);
    const uint32_t heads_per_link = (batch_heads + attrs.num_links - 1) / attrs.num_links;
    const uint32_t num_chunks = attrs.local_rows / tt::constants::TILE_HEIGHT;
    const uint32_t cores_per_head = relay_chunk_scan_cores_per_head(attrs.cores.size(), batch_heads, attrs.num_links);
    const uint32_t core_chunks = num_chunks / cores_per_head;
    const uint32_t grid_y = mesh->compute_with_storage_grid_size().y;
    // The correction cores' layout (kernels/correction_core.hpp): head h's are a column from row first_row.
    const uint32_t first_row = attrs.cores[0].y;
    const uint32_t heads_per_column = relay_chunk_scan_heads_per_column(grid_y, first_row, cores_per_head);
    // Sharded terms are read in place; interleaved ones stream a chunk row at a time.
    const bool streamed = !in.terms.memory_config().is_sharded();

    // A 2D multicast leaves every hop in one ETH direction. Where the SP axis turns on a 2D fabric, the carry goes to
    // each other rank as a 2D unicast instead, on a connection in the direction its route leaves: the relay's own
    // towards the successor, else one of the link's second sender (towards the predecessor, or across the mesh), else
    // one more on the relay's sender.
    const bool unicast_peers = fabric_2d && !ttnn::ccl::is_axis_straight(*mesh, axis);
    struct CarrySender {
        std::vector<tt::tt_fabric::FabricNodeId> connections;  // the node each connection is opened towards
        // Per rank, the sender's route table (see `routes` below): its step, connection, chip, mesh and 1D hops.
        std::vector<uint32_t> routes;
        uint32_t peers = 0;
    };
    std::array<CarrySender, 2> carry_senders;  // the relay's sender, then the second
    if (unicast_peers) {
        // A connection takes two of a sender core's semaphores, beside one per head the sender serves.
        const uint32_t max_connections = std::min(
            relay_chunk_scan_max_connections,
            (relay_chunk_scan_sender_semaphores - heads_per_link) / relay_chunk_scan_connection_semaphores);
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
                    "relay_chunk_scan: the routes from SP rank {} to the other ranks leave in more directions than "
                    "its two carry senders have semaphores for ({} connections each at {} heads per link)",
                    sp_rank,
                    max_connections,
                    heads_per_link);
                carry_senders[s].connections.push_back(peer);
                found = {s, static_cast<uint32_t>(carry_senders[s].connections.size() - 1)};
            }
            auto& sender = carry_senders[found->first];
            sender.routes.insert(
                sender.routes.end(),
                {step, found->second, static_cast<uint32_t>(peer.chip_id), static_cast<uint32_t>(*peer.mesh_id), 1u});
            ++sender.peers;
        }
    }
    // The relay's sender sends the carry along its connection; a second sender per link covers the ranks behind when
    // that connection heads forward (unicast: the ranks the relay's connection does not lead to).
    const uint32_t relay_range = unicast_peers ? carry_senders[0].peers : successor_forward ? ahead : behind;
    const bool back_sender = unicast_peers ? carry_senders[1].peers != 0 : successor_forward && behind != 0;
    const uint32_t back_range = unicast_peers ? carry_senders[1].peers : back_sender ? behind : 0;
    // The senders that carry head h's final carry (bit 0 the relay's, bit 1 the second): on a ring each reaches every
    // rank, so they alternate a link's relay cores between them and each streams about half (a relay core's heads share
    // one: its writer counts both slots' columns in one semaphore, in slot order); else both, each to its own ranks.
    const bool alternate = !unicast_peers && ring && relay_range != 0 && back_range != 0;
    // The SP-sharded slab: head h's entry and carry live on SP rank h / stored_heads only. A rank sends its heads'
    // entry to the first rank, and the first rank each carry to its rank, as unicasts along the one route of a link's
    // senders that reaches the rank: on a ring the relay's sender reaches up to half the ring ahead and the second
    // sender the rest behind, on a line each the ranks on its side, with unicast_peers each its peers.
    const bool slab = in.slab_index.has_value();
    const uint32_t stored_heads = slab ? in.initial_state.logical_shape()[1] : batch_heads;
    // Per sender (the relay's, then the second), every rank it reaches as a unicast: its step (rank - sp_rank) mod
    // sp_size, connection, chip, mesh and 1D hops.
    std::array<std::vector<uint32_t>, 2> routes;
    const auto add_route = [&](uint32_t sender, uint32_t step, uint32_t hops) {
        auto peer_coord = coord;
        peer_coord[axis] = (sp_rank + step) % sp_size;
        const auto peer = mesh->get_fabric_node_id(peer_coord);
        routes[sender].insert(
            routes[sender].end(),
            {step, 0u, static_cast<uint32_t>(peer.chip_id), static_cast<uint32_t>(*peer.mesh_id), hops});
    };
    if (unicast_peers) {
        routes = {carry_senders[0].routes, carry_senders[1].routes};
    } else if (slab) {
        // The relay's connection: forward to the ranks ahead, or (a line's last rank) back to the rest.
        const uint32_t forward = ring ? sp_size / 2 : relay_range;
        for (uint32_t k = 1; k <= forward; ++k) {
            add_route(0, successor_forward ? k : sp_size - k, k);
        }
        for (uint32_t k = 1; back_sender && k < sp_size - forward; ++k) {
            add_route(1, sp_size - k, k);
        }
    }
    // The senders that reach the rank `step` ranks on (bit 0 the relay's, bit 1 the second).
    const auto reaching = [&](uint32_t step) {
        uint32_t mask = 0;
        for (uint32_t sender = 0; sender < routes.size(); ++sender) {
            for (size_t r = 0; r < routes[sender].size(); r += 5) {
                mask |= uint32_t(routes[sender][r] == step) << sender;
            }
        }
        return mask;
    };
    if (slab) {
        for (uint32_t step = 1; step < sp_size; ++step) {
            TT_FATAL(
                std::popcount(reaching(step)) == 1,
                "relay_chunk_scan: SP rank {}'s senders must reach the rank {} on exactly once",
                sp_rank,
                step);
        }
    }
    const auto carry_mask = [&](uint32_t h) {
        if (slab) {
            // Only the head's rank: the sender that reaches it, none for this rank's own heads.
            const uint32_t step = (h / stored_heads + sp_size - sp_rank) % sp_size;
            return step == 0 ? 0u : reaching(step);
        }
        const uint32_t link_cores =
            relay_chunk_scan_link_relay_cores(batch_heads, attrs.num_links, h % attrs.num_links);
        return alternate ? 1u << (h / attrs.num_links % link_cores % 2)
                         : uint32_t(relay_range != 0) | uint32_t(back_range != 0) << 1;
    };

    const std::vector<CoreCoord> correction_cores(
        attrs.cores.begin(), attrs.cores.begin() + batch_heads * cores_per_head);
    // The relay's sender per link, then (with ranks behind) the second carry sender per link, towards the predecessor.
    // Every chip places both (the programs of a range merge), and a second sender without ranks behind its rank idles.
    const std::vector<CoreCoord> sender_cores(
        attrs.cores.begin() + correction_cores.size(),
        attrs.cores.begin() + correction_cores.size() + 2 * attrs.num_links);
    // The relay cores run the hop, each for up to relay_chunk_scan_heads_per_relay_core heads of one link.
    constexpr uint32_t relay_heads = relay_chunk_scan_heads_per_relay_core;
    const std::vector<CoreCoord> relay_cores(
        attrs.cores.end() - relay_chunk_scan_relay_cores(batch_heads, attrs.num_links), attrs.cores.end());
    std::vector<std::vector<uint32_t>> relay_core_heads(relay_cores.size());
    for (uint32_t h = 0; h < batch_heads; ++h) {
        relay_core_heads[relay_chunk_scan_relay_slot(h, batch_heads, attrs.num_links).core].push_back(h);
    }
    // The output gate x @ W: every core of a head forms its chunks' rows of it. The cores at a place of a batch's heads
    // need the same rows of x: each run of them along a row or column reads x and multicasts it along the run, on the
    // writer's NoC 1 from its last core. A head's cores need its columns of W: its first core reads them and multicasts
    // them down the head's column to the others on the reader's NoC 0.
    // Buffers that are never in use on one core at the same time, at one address: each holds `bytes` in whole `units`
    // (what it reserves at once), so the size is a common multiple of the units.
    struct SharedPart {
        uint32_t index;
        tt::DataFormat format;
        uint32_t bytes;
        uint32_t unit;
    };
    const auto shared_cb = [](const tt::tt_metal::CoreRangeSet& cores, std::initializer_list<SharedPart> parts) {
        uint32_t unit = 1;
        uint32_t bytes = 0;
        for (const auto& part : parts) {
            unit = std::lcm(unit, part.unit);
            bytes = std::max(bytes, part.bytes);
        }
        tt::tt_metal::CBDescriptor cb{.total_size = (bytes + unit - 1) / unit * unit, .core_ranges = cores};
        for (const auto& part : parts) {
            cb.format_descriptors.push_back(tt::tt_metal::CBFormatDescriptor{
                .buffer_index = part.index, .data_format = part.format, .page_size = tt::tile_size(part.format)});
        }
        return cb;
    };
    const auto output_format = tt::tt_metal::datatype_to_dataformat_converter(attrs.output_dtype);
    const uint32_t factor_bytes = (streamed ? 1 : core_chunks) * kt * tt::tile_size(output_format);
    const auto x_format = tt::tt_metal::datatype_to_dataformat_converter(in.gate_input.dtype());
    const auto w_format = tt::tt_metal::datatype_to_dataformat_converter(in.gate_weight.dtype());
    const uint32_t gate_k = in.gate_input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t gate_bytes = core_chunks * kt * state_tile;
    // In the workspace, on the correction cores: K blocks of x (after the matmul, each chunk's sigmoid(gate) * weight),
    // then K blocks of W. x's and W's buffers hold `depth` K tiles in 3 and 4 slots, the largest depth that fits, in
    // the smallest K blocks whose slots have read transaction ids (gate_matmul_dataflow.hpp): small blocks keep reads
    // and multicasts close behind the matmul.
    const auto x_block = [&](uint32_t kb) { return core_chunks * kb * tt::tile_size(x_format); };
    const auto w_block = [&](uint32_t kb) { return kb * kt * tt::tile_size(w_format); };
    const auto gate_buffers = [&](uint32_t kb, uint32_t depth) {
        return std::array{
            shared_cb(
                {},
                {{cb_gin0, x_format, 3 * depth / kb * x_block(kb), x_block(kb)},
                 {cb_gate_factor, output_format, factor_bytes, factor_bytes}}),
            shared_cb({}, {{cb_gin1, w_format, 4 * depth / kb * w_block(kb), w_block(kb)}})};
    };
    const auto& workspace_shard = in.gate_workspace.memory_config().shard_spec()->shape;
    const uint32_t workspace_bytes = workspace_shard[0] * workspace_shard[1] * in.gate_workspace.element_size();
    // The buffers fit the workspace, and each slot of x's and W's (a buffer they share an address with can make more)
    // reads under its own transaction id, of which there are 15.
    const auto fits = [&](uint32_t kb, uint32_t depth) {
        const auto buffers = gate_buffers(kb, depth);
        return buffers[0].total_size + buffers[1].total_size <= workspace_bytes &&
               buffers[0].total_size / x_block(kb) <= 15 && buffers[1].total_size / w_block(kb) <= 15;
    };
    uint32_t gate_depth = 0;
    for (const uint32_t depth : {8u, 4u, 2u}) {
        if (gate_depth == 0 && gate_k % depth == 0 && fits(depth, depth)) {
            gate_depth = depth;
        }
    }
    TT_FATAL(
        gate_depth != 0,
        "relay_chunk_scan: the output gate's buffers do not fit gate_workspace's {} B per core",
        workspace_bytes);
    uint32_t gate_kb = gate_depth;
    while (gate_kb > 2 && fits(gate_kb / 2, gate_depth)) {
        gate_kb /= 2;
    }
    // x is shared among the cores at a place of a batch's heads.
    const uint32_t batches = batch_heads / attrs.num_heads;
    const uint32_t gate_x_groups = batches * cores_per_head;
    std::vector<std::vector<GateSet>> gate_groups;
    for (uint32_t b = 0; b < batches; ++b) {
        for (uint32_t place = 0; place < cores_per_head; ++place) {
            std::vector<CoreCoord> members;
            for (uint32_t h = b * attrs.num_heads; h < (b + 1) * attrs.num_heads; ++h) {
                members.push_back(correction_cores[h * cores_per_head + place]);
            }
            auto& sets = gate_groups.emplace_back();
            for (const auto& run : core_runs(members)) {
                auto& set = sets.emplace_back(GateSet{.leader = run.back()});
                set.receivers.assign(run.begin(), run.end() - 1);
            }
        }
    }
    for (uint32_t h = 0; h < batch_heads; ++h) {
        const auto first = correction_cores.begin() + h * cores_per_head;
        gate_groups.push_back({GateSet{*first, {first + 1, first + cores_per_head}}});
    }
    const auto gate_table = gate_set_table(*mesh, gate_groups);
    const uint32_t local_k = in.gate_local ? in.gate_local->padded_shape()[-1] / tt::constants::TILE_WIDTH : 0;
    const uint32_t gate_columns = in.gate_weight.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t tp_rank = mesh->shape().dims() == 2 ? coord[1 - axis] : 0;
    const auto& gate_local = in.gate_local.value_or(in.gate_input);

    const auto correction_set = kda_factory_detail::merged_core_set(correction_cores);
    const auto relay_set = kda_factory_detail::merged_core_set(relay_cores);
    const auto sender_set = kda_factory_detail::merged_core_set(sender_cores);
    // The next states stage in a program CB at one address on every relay and sender core: a sender reads a head's
    // columns at its own cb_eout address, in the head's slot.
    const auto eout_set = relay_set.merge(sender_set);

    const uint32_t sender_slots = relay_chunk_scan_sender_slots(kt, mesh->arch());
    const uint32_t chron_slot_bytes =
        std::max(tt::tt_metal::hal::get_dram_alignment(), tt::tt_metal::hal::get_l1_alignment());
    // The landing buffer on each relay core, in place: a K x K entry state per head slot, then a final state per slot.
    const auto landing_cb = [&](uint32_t index, uint32_t first_slot) {
        auto cb = kda_factory_detail::cb_descriptor(
            index, relay_heads * state_tiles, state_tile, tt::DataFormat::Float32, relay_set);
        cb.buffer = in.landing.buffer();
        cb.address_offset = first_slot * state_tiles * state_tile;
        return cb;
    };
    // A chunk row of F(0) then of F(I). In place, the core's whole terms shard: its chunk rows of F(0), then of F(I).
    // Streamed, two chunks' rows.
    auto terms_cb = kda_factory_detail::cb_descriptor(
        cb_terms, (streamed ? 2 : core_chunks) * 2 * kt, state_tile, tt::DataFormat::Float32, correction_set);
    if (!streamed) {
        terms_cb.buffer = in.terms.buffer();
    }
    const auto tile_cb = [&](uint32_t index, uint32_t tiles, tt::tt_metal::DataType dtype) {
        const auto format = tt::tt_metal::datatype_to_dataformat_converter(dtype);
        return kda_factory_detail::cb_descriptor(index, tiles, tt::tile_size(format), format, correction_set);
    };
    auto gate_cbs = gate_buffers(gate_kb, gate_depth);
    for (auto& cb : gate_cbs) {
        cb.core_ranges = correction_set;
        cb.buffer = in.gate_workspace.buffer();
    }
    gate_cbs[1].address_offset = gate_cbs[0].total_size;
    const auto control_set = correction_set.merge(relay_set);
    desc.cbs = {
        // First the buffers both roles take at one address: the seeds, which a head's relay core writes into its
        // correction cores at its own address, and the chronology. Each role's own buffers then start from there.
        kda_factory_detail::cb_descriptor(cb_seed, state_tiles, state_tile, tt::DataFormat::Float32, control_set),
        kda_factory_detail::cb_descriptor(cb_tail, state_tiles, state_tile, tt::DataFormat::Float32, control_set),
        // Two DRAM-aligned slots: actual_start (then the writer's chronology words), then the slab index.
        kda_factory_detail::cb_descriptor(cb_chron, 1, 2 * chron_slot_bytes, tt::DataFormat::UInt32, control_set),
        kda_factory_detail::cb_descriptor(cb_ctl, 1, 64, tt::DataFormat::UInt32, control_set),
        // The next states, at one address on the relay and sender cores, before either role's own buffers.
        kda_factory_detail::cb_descriptor(
            cb_eout, relay_heads * state_tiles, state_tile, tt::DataFormat::Float32, eout_set),
        // One page: the sender addresses its slots itself.
        kda_factory_detail::cb_descriptor(
            cb_stage, 1, sender_slots * kt * state_tile, tt::DataFormat::Float32, sender_set),
        std::move(terms_cb),
        // Every chunk's gate rows, FP32, which the matmul accumulates in place; in place, then the differences, which
        // compute forms once the gate rows have become factors (streamed, each chunk's difference comes before its
        // factors, so it has a row of its own).
        streamed ? kda_factory_detail::cb_descriptor(
                       cb_gate, core_chunks * kt, state_tile, tt::DataFormat::Float32, correction_set)
                 : shared_cb(
                       correction_set,
                       {{cb_gate, tt::DataFormat::Float32, gate_bytes, gate_bytes},
                        {cb_diff, tt::DataFormat::Float32, core_chunks * kt * state_tile, gate_bytes}}),
        std::move(gate_cbs[0]),
        std::move(gate_cbs[1]),
        kda_factory_detail::cb_descriptor(
            cb_progress, 1, 2 * gate_progress_words * sizeof(uint32_t), tt::DataFormat::UInt32, correction_set),
        // One FP32 chunk row, which compute normalizes itself.
        kda_factory_detail::cb_descriptor(cb_out, kt, state_tile, tt::DataFormat::Float32, correction_set),
        // The squares feed only the row means: in the output's precision.
        tile_cb(cb_square, kt, attrs.output_dtype),
        kda_factory_detail::cb_descriptor(cb_scaler, 1, summary_tile, tt::DataFormat::Float16_b, correction_set),
        kda_factory_detail::cb_descriptor(cb_inv_rms, 1, state_tile, tt::DataFormat::Float32, correction_set),
        tile_cb(cb_weight, kt, in.norm_weight.dtype()),
        tile_cb(cb_weight_rows, kt, in.norm_weight.dtype()),
        // One normalized row, single-buffered: L1 beside the in-place terms has no room for two.
        tile_cb(cb_norm, kt, attrs.output_dtype),
        // The relay core's heads' [E | B], one head after the other; on the first rank, once the hops are done, the
        // heads' final transitions (FP32): E in the entry states' landing slots, B in cb_b's storage.
        kda_factory_detail::cb_descriptor(
            cb_e, relay_heads * state_tiles, summary_tile, tt::DataFormat::Float16_b, relay_set),
        shared_cb(
            relay_set,
            {{cb_b, tt::DataFormat::Float16_b, relay_heads * state_tiles * summary_tile, state_tiles * summary_tile},
             {cb_fb, tt::DataFormat::Float32, relay_heads * state_tiles * state_tile, state_tiles * state_tile}}),
        landing_cb(cb_fe, 0),
        landing_cb(cb_ein, 0),
        landing_cb(cb_ein_exact, 0),
        kda_factory_detail::cb_descriptor(cb_identity, 1, summary_tile, tt::DataFormat::Float16_b, relay_set),
        landing_cb(cb_p, relay_heads),
        landing_cb(cb_p_exact, relay_heads),
    };
    if (streamed) {
        desc.cbs.push_back(
            kda_factory_detail::cb_descriptor(cb_diff, kt, state_tile, tt::DataFormat::Float32, correction_set));
    }
    for (uint32_t slot = 0; slot < relay_heads; ++slot) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = first_ack_semaphore_id + slot, .core_ranges = relay_set, .initial_value = 0});
    }
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = landed_semaphore_id, .core_ranges = correction_set, .initial_value = 0});
    for (const uint32_t id : {x_valid_semaphore_id, w_valid_semaphore_id}) {
        desc.semaphores.push_back(
            tt::tt_metal::SemaphoreDescriptor{.id = id, .core_ranges = correction_set, .initial_value = 0});
    }
    for (uint32_t k = 0; k < heads_per_link; ++k) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = first_notify_semaphore_id + k, .core_ranges = sender_set, .initial_value = 0});
    }

    // sp_rank and the per-chip routing are runtime args, so a range's per-chip programs merge into one.
    const std::vector<uint32_t> chronology = {sp_size, attrs.local_rows};
    // The output gate's operand args (kda_gate::Operands), the same in each kernel that moves them: compile-time, then
    // where its common args start, which append_gate_args appends, the group table last.
    const auto append_gate_ct = [&](std::vector<uint32_t>& ct, uint32_t common) {
        ct.insert(
            ct.end(),
            {cb_gin0,
             cb_gin1,
             cb_progress,
             core_chunks,
             kt,
             gate_kb,
             gate_k,
             local_k,
             gate_columns,
             attrs.num_heads,
             gate_x_groups,
             common,
             x_valid_semaphore_id,
             w_valid_semaphore_id,
             gate_progress_words});
        tt::tt_metal::TensorAccessorArgs(in.gate_input.buffer()).append_to(ct);
        tt::tt_metal::TensorAccessorArgs(gate_local.buffer()).append_to(ct);
        tt::tt_metal::TensorAccessorArgs(in.gate_weight.buffer()).append_to(ct);
    };
    // The chronology every reader derives: actual_start, the slab index and the optional valid end.
    std::vector<uint32_t> slab_ct;
    for (const uint32_t arg : kda_factory_detail::slab_kernel_args(
             slab ? &in.initial_state : nullptr, attrs.slab_index_scale, attrs.slab_index_offset)) {
        slab_ct.push_back(arg);
    }
    // An entry's tiles: this rank's heads.
    slab_ct.push_back(stored_heads * state_tiles);
    slab_ct.push_back(chron_slot_bytes / sizeof(uint32_t));
    const auto& slab_index = slab ? *in.slab_index : in.actual_start;
    const auto append_chronology_ct = [&](std::vector<uint32_t>& ct) {
        ct.insert(ct.end(), {cb_chron, cb_ctl, sp_size, attrs.local_rows, uint32_t(in.actual_end.has_value())});
        ct.insert(ct.end(), slab_ct.begin(), slab_ct.end());
        for (const auto* tensor : {&in.actual_start, &slab_index, &actual_end}) {
            tt::tt_metal::TensorAccessorArgs(tensor->buffer()).append_to(ct);
        }
    };
    std::vector<uint32_t> reader_ct;
    append_chronology_ct(reader_ct);
    reader_ct.insert(
        reader_ct.end(),
        {kt,
         cb_seed,
         cb_tail,
         cb_terms,
         core_chunks,
         uint32_t(streamed),
         landed_semaphore_id,
         cores_per_head,
         heads_per_column,
         first_row});
    tt::tt_metal::TensorAccessorArgs(in.terms.buffer()).append_to(reader_ct);
    reader_ct.insert(reader_ct.end(), {cb_scaler, cb_gate, cb_weight, attrs.num_heads});
    tt::tt_metal::TensorAccessorArgs(in.norm_weight.buffer()).append_to(reader_ct);
    // The output gate's operands, whose common args follow the norm weight.
    append_gate_ct(reader_ct, 6);
    std::vector<uint32_t> writer_ct = {
        cb_chron, cb_norm, kt, num_chunks, cores_per_head, heads_per_column, core_chunks, first_row, attrs.num_heads};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct);
    // The output gate's operands, whose common args follow the output.
    append_gate_ct(writer_ct, 1);
    std::vector<uint32_t> relay_reader_ct;
    append_chronology_ct(relay_reader_ct);
    relay_reader_ct.insert(
        relay_reader_ct.end(),
        {kt,
         cb_e,
         cb_b,
         cb_ein,
         cb_ein_exact,
         cb_identity,
         cb_fe,
         cb_fb,
         cb_p,
         cb_p_exact,
         cb_seed,
         cb_tail,
         landed_semaphore_id,
         first_ack_semaphore_id,
         relay_heads,
         cores_per_head,
         core_chunks});
    for (const auto* tensor : {&in.packed, &in.initial_state, &in.final_e, &in.final_b}) {
        tt::tt_metal::TensorAccessorArgs(tensor->buffer()).append_to(relay_reader_ct);
    }
    std::vector<uint32_t> relay_writer_ct = {cb_eout, cb_chron, kt, first_ack_semaphore_id, relay_heads};
    tt::tt_metal::TensorAccessorArgs(carry.buffer()).append_to(relay_writer_ct);
    relay_writer_ct.insert(relay_writer_ct.end(), slab_ct.begin(), slab_ct.end());
    std::vector<uint32_t> sender_ct = {cb_stage, cb_eout};
    sender_ct.insert(sender_ct.end(), chronology.begin(), chronology.end());
    sender_ct.insert(
        sender_ct.end(),
        {kt * state_tile, relay_chunk_scan_packet_bytes(), kt, sender_slots, first_ack_semaphore_id, relay_heads});
    tt::tt_metal::TensorAccessorArgs(in.actual_start.buffer()).append_to(sender_ct);
    sender_ct.push_back(uint32_t(unicast_peers));
    // The slab, its entry's tiles and the stage word its index lands at, the slab index and the slab itself: a rank's
    // heads of the entry state go to the first rank from its senders.
    sender_ct.insert(sender_ct.end(), slab_ct.begin(), slab_ct.end());
    tt::tt_metal::TensorAccessorArgs(slab_index.buffer()).append_to(sender_ct);
    tt::tt_metal::TensorAccessorArgs(in.initial_state.buffer()).append_to(sender_ct);

    tt::tt_metal::KernelDescriptor reader{
        .kernel_source = std::string(kernel_dir) + "dataflow/reader_relay_chunk_scan.cpp",
        .core_ranges = correction_set,
        .compile_time_args = std::move(reader_ct),
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };
    tt::tt_metal::KernelDescriptor writer{
        .kernel_source = std::string(kernel_dir) + "dataflow/writer_relay_chunk_scan.cpp",
        .core_ranges = correction_set,
        .compile_time_args = std::move(writer_ct),
        .config = tt::tt_metal::WriterConfigDescriptor{},
    };
    auto compute_config = kda_factory_detail::compute_config_descriptor(mesh->arch(), attrs.compute_kernel_config);
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    auto relay_compute_config = compute_config;
    for (const uint32_t exact : {cb_ein_exact, cb_p_exact, cb_fb}) {
        relay_compute_config.unpack_to_dest_mode[exact] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    // The FP32 gate rows, and an FP32 output's g, reach DST exactly.
    compute_config.unpack_to_dest_mode[cb_gate] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    if (attrs.output_dtype == tt::tt_metal::DataType::FLOAT32) {
        compute_config.unpack_to_dest_mode[cb_gate_factor] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    tt::tt_metal::KernelDescriptor compute{
        .kernel_source = std::string(kernel_dir) + "compute/relay_chunk_scan.cpp",
        .core_ranges = correction_set,
        .compile_time_args =
            {cb_ctl,
             kt,
             cb_seed,
             cb_tail,
             cb_terms,
             cb_diff,
             cb_out,
             core_chunks,
             uint32_t(streamed),
             cb_square,
             cb_scaler,
             cb_inv_rms,
             cb_gate,
             cb_weight,
             cb_weight_rows,
             cb_norm,
             cb_gate_factor,
             std::bit_cast<uint32_t>(attrs.norm_epsilon),
             uint32_t(attrs.output_dtype == tt::tt_metal::DataType::FLOAT32),
             cb_gin0,
             cb_gin1,
             gate_kb,
             gate_k / gate_kb,
             static_cast<uint32_t>(attrs.gate_math_fidelity)},
        .config = std::move(compute_config),
    };
    tt::tt_metal::KernelDescriptor relay_reader{
        .kernel_source = std::string(kernel_dir) + "dataflow/reader_relay_core.cpp",
        .core_ranges = relay_set,
        .compile_time_args = std::move(relay_reader_ct),
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };
    tt::tt_metal::KernelDescriptor relay_writer{
        .kernel_source = std::string(kernel_dir) + "dataflow/writer_relay_core.cpp",
        .core_ranges = relay_set,
        .compile_time_args = std::move(relay_writer_ct),
        .config = tt::tt_metal::WriterConfigDescriptor{},
    };
    tt::tt_metal::KernelDescriptor relay_compute{
        .kernel_source = std::string(kernel_dir) + "compute/relay_core.cpp",
        .core_ranges = relay_set,
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
             relay_heads},
        .config = std::move(relay_compute_config),
    };
    tt::tt_metal::KernelDescriptor sender{
        .kernel_source = std::string(kernel_dir) + "dataflow/sender_relay_chunk_scan.cpp",
        .core_ranges = sender_set,
        .compile_time_args = std::move(sender_ct),
        // NoC 0, away from the relay writers' DRAM copies (the writer default, NoC 1): the relay's hops are latency
        // bound.
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_0},
    };

    // The entry states' and final states' (and carries') arrivals.
    const auto [arrival_address, final_address] = attrs.semaphore_addresses;
    const auto append_gate_args = [&](tt::tt_metal::KernelDescriptor::RTArgList& args) {
        args.push_back(in.gate_input.buffer());
        args.push_back(gate_local.buffer());
        args.push_back(in.gate_weight.buffer());
        args.push_back(tp_rank);
        args.append(gate_table);
    };
    const auto append_chronology_args = [&](tt::tt_metal::KernelDescriptor::RTArgList& args) {
        args.push_back(in.actual_start.buffer());
        args.push_back(slab_index.buffer());
        args.push_back(actual_end.buffer());
        args.push_back(sp_rank);
    };
    // The correction cores take no per-core args (their dispatch would cost more than the work): each kernel finds its
    // head and share from its place in the grid, the rest is common per device.
    tt::tt_metal::KernelDescriptor::RTArgList reader_args;
    append_chronology_args(reader_args);
    reader_args.push_back(in.terms.buffer());
    reader_args.push_back(in.norm_weight.buffer());
    append_gate_args(reader_args);
    reader.emplace_common_runtime_args(reader_args);
    tt::tt_metal::KernelDescriptor::RTArgList writer_args;
    writer_args.push_back(output.buffer());
    append_gate_args(writer_args);
    writer.emplace_common_runtime_args(writer_args);
    // The relay cores: common, then per core its heads, each head's correction cores and its link's senders.
    tt::tt_metal::KernelDescriptor::RTArgList relay_reader_args;
    append_chronology_args(relay_reader_args);
    for (auto* buffer :
         {in.packed.buffer(),
          in.initial_state.buffer(),
          in.final_e.buffer(),
          in.final_b.buffer(),
          in.landing.buffer()}) {
        relay_reader_args.push_back(buffer);
    }
    relay_reader_args.push_back(arrival_address);
    relay_reader_args.push_back(final_address);
    relay_reader.emplace_common_runtime_args(relay_reader_args);
    relay_writer.emplace_common_runtime_args({carry.buffer(), in.landing.buffer(), final_address, sp_rank});
    for (uint32_t r = 0; r < relay_cores.size(); ++r) {
        const auto& heads = relay_core_heads[r];
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt, writer_rt;
        reader_rt.push_back(static_cast<uint32_t>(heads.size()));
        writer_rt.push_back(static_cast<uint32_t>(heads.size()));
        for (const uint32_t h : heads) {
            reader_rt.push_back(h);
            for (uint32_t place = 0; place < cores_per_head; ++place) {
                reader_rt.push_back(
                    xy(mesh->worker_core_from_logical_core(correction_cores[h * cores_per_head + place])));
            }
            const uint32_t link = h % attrs.num_links;
            writer_rt.push_back(h);
            writer_rt.push_back(first_notify_semaphore_id + h / attrs.num_links);
            writer_rt.push_back(xy(mesh->worker_core_from_logical_core(sender_cores[link])));
            writer_rt.push_back(xy(mesh->worker_core_from_logical_core(sender_cores[attrs.num_links + link])));
            writer_rt.push_back(carry_mask(h));
        }
        relay_reader.emplace_runtime_args(relay_cores[r], reader_rt);
        relay_writer.emplace_runtime_args(relay_cores[r], writer_rt);
    }
    desc.kernels = {
        std::move(reader),
        std::move(writer),
        std::move(compute),
        std::move(sender),
        std::move(relay_reader),
        std::move(relay_writer),
        std::move(relay_compute)};
    tt::tt_metal::KernelHandle sender_kernel = 3;

    desc.kernels[sender_kernel].emplace_common_runtime_args(
        {in.actual_start.buffer(), in.landing.buffer(), slab_index.buffer(), in.initial_state.buffer()});
    // The second carry senders connect to the predecessor where they carry; an idle one is given the relay's
    // connection, which it never opens, so every chip's runtime args keep one layout.
    const auto back_node = !back_sender    ? connection_node
                           : unicast_peers ? carry_senders[1].connections.front()
                                           : predecessor_node;
    const auto back_links = tt::tt_fabric::get_forwarding_link_indices(*mesh, source_node, back_node);
    TT_FATAL(
        links.size() >= attrs.num_links && back_links.size() >= attrs.num_links,
        "relay_chunk_scan uses {} links; the hops towards the successor and predecessor have {} and {}",
        attrs.num_links,
        links.size(),
        back_links.size());
    for (uint32_t link = 0; link < attrs.num_links; ++link) {
        std::vector<uint32_t> heads, notify, head_slots, masks, head_indices;
        for (uint32_t h = link; h < batch_heads; h += attrs.num_links) {
            const auto [relay_core, slot] = relay_chunk_scan_relay_slot(h, batch_heads, attrs.num_links);
            heads.push_back(xy(mesh->worker_core_from_logical_core(relay_cores[relay_core])));
            notify.push_back(first_notify_semaphore_id + h / attrs.num_links);
            head_slots.push_back(slot);
            masks.push_back(carry_mask(h));
            head_indices.push_back(h);
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
            sender_rt.append(head_slots);
            // Whether this sender carries each head's final carry.
            for (const uint32_t mask : masks) {
                sender_rt.push_back(mask >> direction & 1);
            }
            sender_rt.append(head_indices);
            sender_rt.push_back(uint32_t(!backward));
            sender_rt.push_back(backward ? back_range : relay_range);
            // An idle second sender is given the relay's connection, which it never opens.
            const bool connected = !backward || back_sender;
            std::vector<tt::tt_fabric::FabricNodeId> nodes = {backward ? back_node : connection_node};
            std::vector<uint32_t> node_links = {backward ? back_links[link] : links[link]};
            const auto& carry_sender = carry_senders[direction];
            if (unicast_peers && connected) {
                // The further connections, each on this link's index among its direction's links.
                for (size_t c = 1; c < carry_sender.connections.size(); ++c) {
                    const auto& node = carry_sender.connections[c];
                    const auto extra_links = tt::tt_fabric::get_forwarding_link_indices(*mesh, source_node, node);
                    TT_FATAL(
                        extra_links.size() > link,
                        "relay_chunk_scan uses {} links; a carry route from SP rank {} leaves on a direction with "
                        "{}",
                        attrs.num_links,
                        sp_rank,
                        extra_links.size());
                    nodes.push_back(node);
                    node_links.push_back(extra_links[link]);
                }
            }
            sender_rt.push_back(connected ? static_cast<uint32_t>(nodes.size()) : 0u);
            // The route table, before the connection args (which the kernel consumes only for the connections it
            // opens).
            const auto& sender_routes = connected ? routes[direction] : std::vector<uint32_t>{};
            sender_rt.push_back(static_cast<uint32_t>(sender_routes.size() / 5));
            sender_rt.append(sender_routes);
            std::vector<uint32_t> fabric_args;
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                *mesh, source_node, nodes, node_links, desc, sender_kernel, core, fabric_args);
            sender_rt.append(fabric_args);
            desc.kernels[sender_kernel].emplace_runtime_args(core, sender_rt);
        }
    }
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor RelayChunkScanProgramFactory::create_workload_descriptor(
    const RelayChunkScanParams& attrs,
    const RelayChunkScanInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& coord : tensor_coords.coords()) {
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord), build_program(attrs, in, outputs, coord)});
    }
    return workload;
}

}  // namespace ttnn::experimental::prim
