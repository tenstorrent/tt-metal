// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/relay_chunk_scan/device/relay_chunk_scan_program_factory.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <iterator>
#include <numeric>
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
// The entry correction, on every core of a head: the seeds its chunks start from (the entry state, and P for a
// separated tail), the chunk terms F(0) and F(I), their differences and the chunk outputs.
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
// On the head cores: the sender releases a head once it has read the head's last column.
constexpr uint32_t ack_semaphore_id = 0;
// On every core of a head: the seeds the head's first core has written there.
constexpr uint32_t landed_semaphore_id = 1;
// On the sender cores, ids 0, 1, ...: per head the sender serves, the head's finished columns.
constexpr uint32_t first_notify_semaphore_id = 0;
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

    // The ranks this rank's final carry reaches each way: half the ring each way, else to either end of the line.
    const uint32_t ahead = ring ? sp_size / 2 : sp_size - 1 - sp_rank;
    const uint32_t behind = sp_size - 1 - ahead;
    const auto [batch_heads, key_dim] = relay_state_geometry(in);
    const uint32_t kt = key_dim / tt::constants::TILE_HEIGHT;
    const uint32_t state_tiles = kt * kt;
    const uint32_t state_tile = tt::constants::TILE_HW * sizeof(float);
    const uint32_t summary_tile = tt::constants::TILE_HW * sizeof(uint16_t);
    const uint32_t heads_per_link = (batch_heads + attrs.num_links - 1) / attrs.num_links;
    const uint32_t num_chunks = attrs.local_rows / tt::constants::TILE_HEIGHT;
    const uint32_t cores_per_head =
        relay_chunk_scan_cores_per_head(attrs.cores.size(), batch_heads, attrs.num_links);
    const uint32_t core_chunks = num_chunks / cores_per_head;
    const uint32_t grid_x = mesh->compute_with_storage_grid_size().x;
    const uint32_t first_core = attrs.cores[0].y * grid_x + attrs.cores[0].x;
    // Sharded terms are read in place; interleaved ones stream a chunk row at a time.
    const bool streamed = !in.terms.memory_config().is_sharded();

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

    const std::vector<CoreCoord> correction_cores(
        attrs.cores.begin(), attrs.cores.begin() + batch_heads * cores_per_head);
    std::vector<CoreCoord> head_cores;
    head_cores.reserve(batch_heads);
    for (uint32_t h = 0; h < batch_heads; ++h) {
        head_cores.push_back(correction_cores[h * cores_per_head]);
    }
    // The relay's sender per link, then (with ranks behind) the second carry sender per link, towards the predecessor.
    // Every chip places both (the programs of a range merge), and a second sender without ranks behind its rank idles.
    const std::vector<CoreCoord> sender_cores(
        attrs.cores.begin() + correction_cores.size(),
        attrs.cores.begin() + correction_cores.size() + 2 * attrs.num_links);
    // A helper per head forms the output gate of the chunks of the head's first core.
    const std::vector<CoreCoord> helper_cores(attrs.cores.end() - batch_heads, attrs.cores.end());
    // The output gate x @ W: each head's other cores form their chunks' rows of it, the helpers the head cores'. The
    // cores at a place of a batch's heads (at place 0, the helpers) need the same rows of x: each run of them along a
    // row or column reads x and multicasts it along the run. A head's cores and helper need its columns of W: one of
    // them reads them and sends them to the others. Each multicast runs one way on its NoC (NoC 0 east and south,
    // NoC 1 west and north), so it starts from the end of its run that way. A head's W moves north on NoC 1 (the
    // writer's) from its last core when its helper is above it, else south on NoC 0 (the reader's) from its second
    // core; its cores' x moves on the other kernel. The helpers move x on NoC 1.
    const uint32_t batches = batch_heads / attrs.num_heads;
    const uint32_t gate_x_groups = batches * cores_per_head;
    std::vector<uint32_t> w_north((batch_heads + 31) / 32);
    const auto north = [&](uint32_t h) { return helper_cores[h].y < head_cores[h].y; };
    for (uint32_t h = 0; h < batch_heads; ++h) {
        w_north[h / 32] |= uint32_t{north(h)} << (h % 32);
    }
    std::vector<std::vector<GateSet>> gate_groups;
    for (uint32_t b = 0; b < batches; ++b) {
        for (uint32_t place = 0; place < cores_per_head; ++place) {
            std::vector<CoreCoord> members;
            for (uint32_t h = b * attrs.num_heads; h < (b + 1) * attrs.num_heads; ++h) {
                members.push_back(place == 0 ? helper_cores[h] : correction_cores[h * cores_per_head + place]);
            }
            const auto runs = core_runs(members);
            std::vector<CoreCoord> leaders;
            for (const auto& run : runs) {
                const auto head_of = [&](const CoreCoord& core) {
                    return static_cast<uint32_t>(std::ranges::find(correction_cores, core) - correction_cores.begin()) /
                           cores_per_head;
                };
                leaders.push_back(place == 0 || !north(head_of(run.front())) ? run.back() : run.front());
            }
            auto& sets = gate_groups.emplace_back();
            for (size_t r = 0; r < runs.size(); ++r) {
                auto& set = sets.emplace_back(GateSet{.leader = leaders[r]});
                std::ranges::copy_if(
                    runs[r], std::back_inserter(set.receivers), [&](const auto& core) { return core != leaders[r]; });
            }
        }
    }
    for (uint32_t h = 0; h < batch_heads; ++h) {
        auto& sets = gate_groups.emplace_back();
        const auto first = correction_cores.begin() + h * cores_per_head + 1;
        const auto last = correction_cores.begin() + (h + 1) * cores_per_head;
        if (first == last) {
            sets.push_back({helper_cores[h], {}});
            continue;
        }
        auto& set =
            sets.emplace_back(north(h) ? GateSet{*(last - 1), {first, last - 1}} : GateSet{*first, {first + 1, last}});
        set.receivers.push_back(helper_cores[h]);
    }
    const auto gate_table = gate_set_table(*mesh, gate_groups);
    const uint32_t gate_k = in.gate_input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t local_k = in.gate_local ? in.gate_local->padded_shape()[-1] / tt::constants::TILE_WIDTH : 0;
    const uint32_t gate_columns = in.gate_weight.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t tp_rank = mesh->shape().dims() == 2 ? coord[1 - axis] : 0;
    const auto& gate_local = in.gate_local.value_or(in.gate_input);

    const auto correction_set = kda_factory_detail::merged_core_set(correction_cores);
    const auto head_set = kda_factory_detail::merged_core_set(head_cores);
    const auto sender_set = kda_factory_detail::merged_core_set(sender_cores);
    const auto helper_set = kda_factory_detail::merged_core_set(helper_cores);
    // The next state stages in a program CB at one address on every head and sender core: a sender reads a head's
    // columns at its own cb_eout address.
    const auto eout_set = head_set.merge(sender_set);

    const uint32_t sender_slots = relay_chunk_scan_sender_slots(kt, mesh->arch());
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
    const auto tile_cb = [&](uint32_t index, uint32_t tiles, tt::tt_metal::DataType dtype) {
        const auto format = tt::tt_metal::datatype_to_dataformat_converter(dtype);
        return kda_factory_detail::cb_descriptor(index, tiles, tt::tile_size(format), format, correction_set);
    };
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
    const auto gate_set = correction_set.merge(helper_set);
    const auto output_format = tt::tt_metal::datatype_to_dataformat_converter(attrs.output_dtype);
    const uint32_t factor_bytes = (streamed ? 1 : core_chunks) * kt * tt::tile_size(output_format);
    const auto x_format = tt::tt_metal::datatype_to_dataformat_converter(in.gate_input.dtype());
    const auto w_format = tt::tt_metal::datatype_to_dataformat_converter(in.gate_weight.dtype());
    // In the workspace: three K blocks of x on the cores that form the gate (the final transition's E on the head
    // cores, and once both are used, each chunk's sigmoid(gate) * weight: in place all of the core's, streamed one),
    // then four K blocks of W (the final transition's B). The largest K block whose buffers fit: a block's handshakes
    // cost the same whatever its size.
    const auto gate_buffers = [&](uint32_t kb) {
        const uint32_t x_block = core_chunks * kb * tt::tile_size(x_format);
        const uint32_t w_block = kb * kt * tt::tile_size(w_format);
        return std::array{
            shared_cb(
                gate_set,
                {{cb_fe, tt::DataFormat::Float32, state_tiles * state_tile, state_tile},
                 {cb_gin0, x_format, 3 * x_block, x_block},
                 {cb_gate_factor, output_format, factor_bytes, factor_bytes}}),
            shared_cb(
                gate_set,
                {{cb_fb, tt::DataFormat::Float32, state_tiles * state_tile, state_tile},
                 {cb_gin1, w_format, 4 * w_block, w_block}})};
    };
    const auto buffers_bytes = [&](uint32_t kb) {
        const auto buffers = gate_buffers(kb);
        return buffers[0].total_size + buffers[1].total_size;
    };
    const auto& workspace_shard = in.gate_workspace.memory_config().shard_spec()->shape;
    const uint32_t workspace_bytes = workspace_shard[0] * workspace_shard[1] * in.gate_workspace.element_size();
    uint32_t gate_kb = 1;
    for (const uint32_t kb : {8u, 4u, 2u}) {
        if (gate_k % kb == 0 && buffers_bytes(kb) <= workspace_bytes) {
            gate_kb = kb;
            break;
        }
    }
    TT_FATAL(
        buffers_bytes(gate_kb) <= workspace_bytes,
        "relay_chunk_scan: the output gate needs {} B of gate_workspace per core, it has {}",
        buffers_bytes(gate_kb),
        workspace_bytes);
    auto gate_cbs = gate_buffers(gate_kb);
    gate_cbs[0].buffer = gate_cbs[1].buffer = in.gate_workspace.buffer();
    gate_cbs[1].address_offset = gate_cbs[0].total_size;
    const uint32_t gate_bytes = core_chunks * kt * state_tile;
    desc.cbs = {
        // The seeds first: every core of a head takes them at the address its head's first core writes them to.
        kda_factory_detail::cb_descriptor(cb_seed, state_tiles, state_tile, tt::DataFormat::Float32, correction_set),
        kda_factory_detail::cb_descriptor(cb_tail, state_tiles, state_tile, tt::DataFormat::Float32, correction_set),
        std::move(terms_cb),
        // Every chunk's gate rows, FP32, which the matmul accumulates in place (a head core's helper writes them at the
        // same address), then the differences, which compute forms once the gate rows have become factors.
        shared_cb(
            gate_set,
            {{cb_gate, tt::DataFormat::Float32, gate_bytes, gate_bytes},
             {cb_diff, tt::DataFormat::Float32, (streamed ? 1 : core_chunks) * kt * state_tile, gate_bytes}}),
        std::move(gate_cbs[0]),
        std::move(gate_cbs[1]),
        kda_factory_detail::cb_descriptor(
            cb_progress, 1, 2 * gate_progress_words * sizeof(uint32_t), tt::DataFormat::UInt32, gate_set),
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
        landing_slot_cb(cb_p, 1),
        landing_slot_cb(cb_p_exact, 1),
    };
    desc.semaphores.push_back(
        tt::tt_metal::SemaphoreDescriptor{.id = ack_semaphore_id, .core_ranges = head_set, .initial_value = 0});
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = landed_semaphore_id, .core_ranges = correction_set, .initial_value = 0});
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
             common + 6 + static_cast<uint32_t>(w_north.size()),
             common});
        tt::tt_metal::TensorAccessorArgs(in.gate_input.buffer()).append_to(ct);
        tt::tt_metal::TensorAccessorArgs(gate_local.buffer()).append_to(ct);
        tt::tt_metal::TensorAccessorArgs(in.gate_weight.buffer()).append_to(ct);
    };
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
    reader_ct.insert(
        reader_ct.end(),
        {cb_seed,
         cb_tail,
         cb_terms,
         core_chunks,
         uint32_t(streamed),
         landed_semaphore_id,
         ack_semaphore_id,
         cores_per_head,
         grid_x,
         first_core});
    tt::tt_metal::TensorAccessorArgs(in.terms.buffer()).append_to(reader_ct);
    reader_ct.insert(reader_ct.end(), {cb_scaler, cb_gate, cb_weight, attrs.num_heads});
    tt::tt_metal::TensorAccessorArgs(in.norm_weight.buffer()).append_to(reader_ct);
    // The output gate's operands, whose common args follow the cores' NoC coordinates.
    append_gate_ct(reader_ct, 14 + static_cast<uint32_t>(correction_cores.size()));
    std::vector<uint32_t> writer_ct = {
        cb_eout,
        cb_chron,
        cb_norm,
        kt,
        num_chunks,
        ack_semaphore_id,
        cores_per_head,
        grid_x,
        core_chunks,
        attrs.num_links,
        first_core,
        attrs.num_heads};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct);
    tt::tt_metal::TensorAccessorArgs(carry.buffer()).append_to(writer_ct);
    writer_ct.insert(writer_ct.end(), slab_ct.begin(), slab_ct.end());
    // The output gate's operands, whose common args follow the senders'.
    append_gate_ct(writer_ct, 3 + 2 * attrs.num_links);
    std::vector<uint32_t> sender_ct = {cb_stage, cb_eout};
    sender_ct.insert(sender_ct.end(), chronology.begin(), chronology.end());
    sender_ct.insert(
        sender_ct.end(), {kt * state_tile, relay_chunk_scan_packet_bytes(), kt, sender_slots, ack_semaphore_id});
    tt::tt_metal::TensorAccessorArgs(in.actual_start.buffer()).append_to(sender_ct);
    tt::tt_metal::TensorAccessorArgs(carry.buffer()).append_to(sender_ct);
    sender_ct.insert(sender_ct.end(), slab_ct.begin(), slab_ct.end());
    tt::tt_metal::TensorAccessorArgs(slab_index.buffer()).append_to(sender_ct);
    sender_ct.push_back(uint32_t(unicast_peers));

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
    for (const uint32_t exact : {cb_ein_exact, cb_p_exact, cb_fb}) {
        compute_config.unpack_to_dest_mode[exact] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    // The FP32 gate rows, and an FP32 output's g, reach DST exactly.
    compute_config.unpack_to_dest_mode[cb_gate] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    if (attrs.output_dtype == tt::tt_metal::DataType::FLOAT32) {
        compute_config.unpack_to_dest_mode[cb_gate_factor] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    auto helper_compute_config = compute_config;
    tt::tt_metal::KernelDescriptor compute{
        .kernel_source = std::string(kernel_dir) + "compute/relay_chunk_scan.cpp",
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
             first_core,
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
    // The helpers' data movement: x on RISC 1 and W on RISC 0, each on its own NoC, but both on the RISC a sender
    // leaves free.
    std::array<std::vector<CoreCoord>, 2> helper_kinds;  // the free helpers, then the senders
    for (const auto& core : helper_cores) {
        helper_kinds[std::ranges::find(sender_cores, core) != sender_cores.end()].push_back(core);
    }
    const auto helper_kernel = [&](const CoreRangeSet& cores, bool moves_x, bool moves_w) {
        // The operands' common args follow the landed count.
        std::vector<uint32_t> ct = {cb_gate, num_chunks, cores_per_head, moves_x, moves_w};
        append_gate_ct(ct, 1);
        const bool riscv_1 = moves_x;
        return tt::tt_metal::KernelDescriptor{
            .kernel_source = std::string(kernel_dir) + "dataflow/helper_relay_chunk_scan.cpp",
            .core_ranges = cores,
            .compile_time_args = std::move(ct),
            .config = tt::tt_metal::DataMovementConfigDescriptor{
                .processor = riscv_1 ? tt::tt_metal::DataMovementProcessor::RISCV_1
                                     : tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = riscv_1 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0}};
    };
    std::vector<tt::tt_metal::KernelDescriptor> helpers;
    if (!helper_kinds[0].empty()) {
        const auto cores = kda_factory_detail::merged_core_set(helper_kinds[0]);
        helpers.push_back(helper_kernel(cores, true, false));
        helpers.push_back(helper_kernel(cores, false, true));
    }
    if (!helper_kinds[1].empty()) {
        helpers.push_back(helper_kernel(kda_factory_detail::merged_core_set(helper_kinds[1]), true, true));
    }
    tt::tt_metal::KernelDescriptor helper_compute{
        .kernel_source = std::string(kernel_dir) + "compute/helper_relay_chunk_scan.cpp",
        .core_ranges = helper_set,
        .compile_time_args =
            {cb_gin0,
             cb_gin1,
             cb_gate,
             core_chunks,
             kt,
             gate_kb,
             gate_k / gate_kb,
             static_cast<uint32_t>(attrs.gate_math_fidelity)},
        .config = std::move(helper_compute_config),
    };
    tt::tt_metal::KernelDescriptor sender{
        .kernel_source = std::string(kernel_dir) + "dataflow/sender_relay_chunk_scan.cpp",
        .core_ranges = sender_set,
        .compile_time_args = std::move(sender_ct),
        // NoC 0, away from the head writers' DRAM copies (the writer default, NoC 1): the relay's hops are latency
        // bound.
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_0},
    };

    const auto& semaphores = attrs.semaphore_addresses;
    const uint32_t arrival_address = semaphores[0];
    const uint32_t final_address = semaphores[1];
    const uint32_t carry_address = semaphores[2];
    // x's and W's block counts, then the landed count.
    const uint32_t x_valid = semaphores[3];
    const uint32_t w_valid = semaphores[4];
    const uint32_t gate_landed = semaphores[5];
    const auto append_gate_args = [&](tt::tt_metal::KernelDescriptor::RTArgList& args) {
        args.push_back(in.gate_input.buffer());
        args.push_back(gate_local.buffer());
        args.push_back(in.gate_weight.buffer());
        args.push_back(tp_rank);
        args.push_back(x_valid);
        args.push_back(w_valid);
        args.append(w_north);
        args.append(gate_table);
    };
    // The heads' cores take no per-core args (their dispatch would cost more than the work): each kernel finds its
    // head and share from its place in the grid, the rest is common per device.
    tt::tt_metal::KernelDescriptor::RTArgList reader_args;
    for (auto* buffer :
         {in.packed.buffer(),
          in.initial_state.buffer(),
          in.actual_start.buffer(),
          slab_index.buffer(),
          in.final_e.buffer(),
          in.final_b.buffer(),
          actual_end.buffer(),
          in.terms.buffer(),
          in.landing.buffer(),
          in.norm_weight.buffer()}) {
        reader_args.push_back(buffer);
    }
    reader_args.push_back(arrival_address);
    reader_args.push_back(final_address);
    reader_args.push_back(sp_rank);
    reader_args.push_back(gate_landed);
    for (const auto& core : correction_cores) {
        reader_args.push_back(xy(mesh->worker_core_from_logical_core(core)));
    }
    append_gate_args(reader_args);
    reader.emplace_common_runtime_args(reader_args);
    tt::tt_metal::KernelDescriptor::RTArgList writer_args;
    writer_args.push_back(output.buffer());
    writer_args.push_back(carry.buffer());
    writer_args.push_back(uint32_t(back_sender));
    for (const auto& core : sender_cores) {
        writer_args.push_back(xy(mesh->worker_core_from_logical_core(core)));
    }
    append_gate_args(writer_args);
    writer.emplace_common_runtime_args(writer_args);
    tt::tt_metal::KernelDescriptor::RTArgList helper_args;
    helper_args.push_back(gate_landed);
    append_gate_args(helper_args);
    for (auto& helper : helpers) {
        helper.emplace_common_runtime_args(helper_args);
    }
    for (uint32_t h = 0; h < batch_heads; ++h) {
        const uint32_t head_xy = xy(mesh->worker_core_from_logical_core(head_cores[h]));
        for (auto& helper : helpers) {
            if (helper.core_ranges.contains(helper_cores[h])) {
                helper.emplace_runtime_args(helper_cores[h], {h, head_xy});
            }
        }
    }
    desc.kernels = {
        std::move(reader), std::move(writer), std::move(compute), std::move(sender), std::move(helper_compute)};
    for (auto& helper : helpers) {
        desc.kernels.push_back(std::move(helper));
    }
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
        "relay_chunk_scan uses {} links; the hops towards the successor and predecessor have {} and {}",
        attrs.num_links,
        links.size(),
        back_links.size());
    for (uint32_t link = 0; link < attrs.num_links; ++link) {
        std::vector<uint32_t> heads, notify;
        for (uint32_t h = link; h < batch_heads; h += attrs.num_links) {
            const auto head = mesh->worker_core_from_logical_core(head_cores[h]);
            heads.push_back(xy(head));
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
                        "relay_chunk_scan uses {} links; a carry route from SP rank {} leaves on a direction with "
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
