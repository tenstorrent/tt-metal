// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The per-layer fabric sends of deepseek_v41_flash_d_p, ported from the model's Python prepared programs (NC.3):
// every chip's program is the Python _build* loop body for that chip, the kernels are the Python kernel strings
// moved verbatim (device/kernels), and every runtime arg the Python route patched per call (an io address or a
// per-call value) is a slot the override rewrites.

#include "layer_send_program_factory.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::layer_send {

using namespace tt::tt_metal;

namespace {

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/layer_send/device/kernels/";
constexpr uint32_t kBlocks = 4;  // snapshot blocks of 32 rows
constexpr uint32_t kBlockRows = 32;
constexpr uint32_t kZero = 0, kLocal = 1, kRecv = 2;         // home core's kind of a block (static pattern)
constexpr uint32_t kNone = 0, kFromRing = 1, kFromSlab = 2;  // port core's role for a block (static pattern)

using Arg = std::variant<uint32_t, Buffer*>;

std::string kernel_path(const char* name) { return std::string(kKernelDir) + name; }

CoreRangeSet cells(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& c : cores) {
        ranges.emplace_back(c, c);
    }
    return CoreRangeSet(ranges);
}

DataMovementConfigDescriptor dm(DataMovementProcessor processor, NOC noc) {
    return DataMovementConfigDescriptor{.processor = processor, .noc = noc};
}

CBDescriptor byte_cb(uint32_t index, uint32_t bytes, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = bytes,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index), .data_format = tt::DataFormat::UInt8, .page_size = bytes}},
    };
}

void append_accessor(std::vector<uint32_t>& ct, const Tensor& t) { TensorAccessorArgs(*t.buffer()).append_to(ct); }

// Per-core runtime args under construction: io addresses as buffer bindings, per-call values as their current value;
// both recorded as slots.
struct Args {
    uint32_t kernel;
    CoreCoord core;
    const std::vector<Tensor>& io;
    const std::vector<uint32_t>& values;
    std::vector<SlotRef>& slots;
    std::vector<Arg> a;

    void u(uint32_t v) { a.emplace_back(v); }
    void us(const std::vector<uint32_t>& vs) {
        for (uint32_t v : vs) {
            a.emplace_back(v);
        }
    }
    void io_addr(uint32_t i) {
        slots.push_back(SlotRef{.kernel = kernel, .core = core, .index = static_cast<uint32_t>(a.size()), .src = i});
        a.emplace_back(io.at(i).buffer());
    }
    void value(uint32_t j) {
        slots.push_back(
            SlotRef{.kernel = kernel, .core = core, .index = static_cast<uint32_t>(a.size()), .src = kValue + j});
        a.emplace_back(values.at(j));
    }
    // ttnn.setup_fabric_connection: the connection's args (and its semaphores appended to the program)
    void fabric(
        tt::tt_metal::distributed::MeshDevice& mesh,
        const tt::tt_metal::distributed::MeshCoordinate& src,
        const tt::tt_metal::distributed::MeshCoordinate& peer,
        uint32_t link,
        ProgramDescriptor& prog) {
        std::vector<uint32_t> f;
        tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
            mesh,
            mesh.get_fabric_node_id(src),
            mesh.get_fabric_node_id(peer),
            link,
            prog,
            core,
            f,
            tt::CoreType::WORKER);
        us(f);
    }
    void emit(KernelDescriptor& k) { k.emplace_runtime_args(core, a); }
};

uint32_t node_mesh_id(tt::tt_metal::distributed::MeshDevice& mesh, uint32_t r, uint32_t c) {
    return static_cast<uint32_t>(*mesh.get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(r, c)).mesh_id);
}
uint32_t node_chip_id(tt::tt_metal::distributed::MeshDevice& mesh, uint32_t r, uint32_t c) {
    return static_cast<uint32_t>(mesh.get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(r, c)).chip_id);
}

// ---------------------------------------------------------------------------------------------------- window sends
// consts: [page_bytes, seg, nseg, carry, p0, batch, slot | copy_cores, copy_batch, ring_pieces, window_row0 (fused)]
ProgramDescriptor window(
    const SendPlan& p,
    const std::vector<Tensor>& io,
    const std::vector<uint32_t>& values,
    uint32_t q,
    std::vector<SlotRef>& slots,
    bool fused) {
    auto& mesh = *io.at(0).device();
    const auto& k = p.consts;
    const uint32_t page = k[0], carry = k[3], batch = k[5], slot = k[6];
    const uint32_t n = p.rows * p.cols;
    const uint32_t r = q / p.cols, c = q % p.cols;
    const CoreCoord cc{0, 0};
    const CoreRangeSet crs = cells({cc});
    const Hop& h = p.hops.at(q).at(0);
    const CoreCoord v = mesh.worker_core_from_logical_core(cc);
    ProgramDescriptor prog;
    prog.cbs.push_back(byte_cb(0, (batch * slot) + 64, crs));
    // [region, tail, (rows,) last, sem, px, py, mesh_id, chip_id, connection..]
    std::vector<uint32_t> ct{k[0], k[1], k[2], k[3], k[4], k[5]};
    append_accessor(ct, io.at(0));
    append_accessor(ct, io.at(1));
    Args s{.kernel = 0, .core = cc, .io = io, .values = values, .slots = slots, .a = {}};
    s.io_addr(0);
    s.io_addr(1);
    if (fused) {
        const auto q_rows = static_cast<uint32_t>(io.at(2).logical_shape()[-2]);
        append_accessor(ct, io.at(2));
        ct.push_back(q_rows - carry);
        s.io_addr(2);
    }
    s.us({static_cast<uint32_t>(q == n - 1), p.sems.at(0), static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)});
    s.us({node_mesh_id(mesh, h.dst_r, h.dst_c), node_chip_id(mesh, h.dst_r, h.dst_c)});
    if (fused) {
        prog.cbs.push_back(byte_cb(1, (k[8] * page) + 64, CoreRangeSet(CoreRange({0, 0}, {k[7] - 1, 0}))));
    }
    s.fabric(
        mesh,
        tt::tt_metal::distributed::MeshCoordinate(r, c),
        tt::tt_metal::distributed::MeshCoordinate(h.peer_r, h.peer_c),
        h.link,
        prog);
    KernelDescriptor send;
    send.kernel_source = kernel_path(fused ? "window_send_fused.cpp" : "window_send.cpp");
    send.core_ranges = crs;
    send.compile_time_args = ct;
    send.config = dm(DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);
    s.emit(send);
    prog.kernels.push_back(std::move(send));
    if (fused) {
        const uint32_t copy_cores = k[7], copy_batch = k[8], pieces = k[9], row0 = k[10];
        const auto q_rows = static_cast<uint32_t>(io.at(2).logical_shape()[-2]);
        std::vector<uint32_t> cct{page, copy_batch, copy_cores, 1 + pieces};
        append_accessor(cct, io.at(2));
        append_accessor(cct, io.at(0));
        append_accessor(cct, io.at(3));
        KernelDescriptor copy;
        copy.kernel_source = kernel_path("window_copy.cpp");
        copy.core_ranges = CoreRangeSet(CoreRange({0, 0}, {copy_cores - 1, 0}));
        copy.compile_time_args = cct;
        copy.config = dm(DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);
        for (uint32_t j = 0; j < copy_cores; ++j) {
            // [rows, region, ring, k, (0, 128, q_rows): the region's segment, (src, dst, rows) x ring pieces]
            Args a{.kernel = 1, .core = CoreCoord{j, 0}, .io = io, .values = values, .slots = slots, .a = {}};
            a.io_addr(2);
            a.io_addr(0);
            a.io_addr(3);
            a.us({j, 0, row0, q_rows});
            for (uint32_t i = 0; i < 3 * pieces; ++i) {
                a.value(i);
            }
            a.emit(copy);
        }
        prog.kernels.push_back(std::move(copy));
    }
    return prog;
}

// ------------------------------------------------------------------------------------------------- snapshot sends
int32_t step(uint32_t n, uint32_t a, uint32_t b, bool wrap) {
    if (wrap) {
        const uint32_t f = (b + n - a) % n;
        return f * 2 <= n ? 1 : -1;
    }
    return b > a ? 1 : -1;
}

// snapshot_send.tree: the parent of chip (r, c) in the broadcast tree from owner (r0, c0)
std::pair<uint32_t, uint32_t> parent(const SendPlan& p, uint32_t r, uint32_t c, uint32_t r0, uint32_t c0) {
    const uint32_t R = p.rows, C = p.cols;
    if (c != c0) {
        return {r, static_cast<uint32_t>((static_cast<int32_t>(c + C) + step(C, c, c0, p.wrap_x != 0)) % C)};
    }
    return {static_cast<uint32_t>((static_cast<int32_t>(r + R) + step(R, r, r0, p.wrap_y != 0)) % R), c};
}

struct ChipPattern {
    std::vector<uint32_t> kinds = std::vector<uint32_t>(kBlocks, kZero);
    std::map<uint32_t, std::vector<uint32_t>> ports;  // dir -> roles [4]
    std::vector<uint32_t> masks = std::vector<uint32_t>(kBlocks, 0);
};

// SnapshotSend.plan for chip q
ChipPattern pattern_plan(const SendPlan& p, uint32_t q) {
    const uint32_t n = p.rows * p.cols;
    ChipPattern out;
    for (uint32_t b = 0; b < kBlocks; ++b) {
        const uint32_t code = p.pattern.at(b);
        if (code == 0xFF || code == 0xFE) {
            out.kinds[b] = code == 0xFF ? kZero : kLocal;
            continue;
        }
        const uint32_t r0 = code / p.cols, c0 = code % p.cols;
        out.kinds[b] = q == code ? kLocal : kRecv;
        // every child of q in the tree of code: the port of q (first in mesh_plan's order) whose peer it is
        for (uint32_t ch = 0; ch < n; ++ch) {
            if (ch == code) {
                continue;
            }
            const auto [pr, pc] = parent(p, ch / p.cols, ch % p.cols, r0, c0);
            if (pr * p.cols + pc != q) {
                continue;
            }
            const Hop* port = nullptr;
            for (const Hop& h : p.hops.at(q)) {
                if (h.peer_r * p.cols + h.peer_c == ch) {
                    port = &h;
                    break;
                }
            }
            TT_FATAL(port != nullptr, "snapshot send: chip {} -> {} is not a fabric neighbour", q, ch);
            auto& roles = out.ports.try_emplace(port->dir, std::vector<uint32_t>(kBlocks, kNone)).first->second;
            roles[b] = q == code ? kFromRing : kFromSlab;
        }
    }
    for (uint32_t b = 0; b < kBlocks; ++b) {
        uint32_t k = 0;
        for (const auto& [d, roles] : out.ports) {
            if (roles[b] == kFromSlab) {
                out.masks[b] |= 1u << k;
            }
            ++k;
        }
    }
    return out;
}

// consts: [page, k (rows a packet) | wait_limit (multi)]
ProgramDescriptor snapshot(
    const SendPlan& p,
    const std::vector<Tensor>& io,
    const std::vector<uint32_t>& values,
    uint32_t q,
    std::vector<SlotRef>& slots) {
    auto& mesh = *io.at(0).device();
    const uint32_t page = p.consts.at(0), krows = p.consts.at(1);
    const uint32_t r = q / p.cols, c = q % p.cols;
    const bool multi = p.kind == SendKind::SNAP_MULTI, dyn = p.kind == SendKind::SNAP_DYN;
    const bool stat = p.kind == SendKind::SNAP_STATIC;
    const uint32_t ns = multi ? static_cast<uint32_t>(io.size() / 2) : 1;
    const CoreCoord hc{0, 0};
    const CoreCoord hv = mesh.worker_core_from_logical_core(hc);
    // the ports this program runs (sorted by direction): every neighbour, or (static) those with a block
    std::vector<const Hop*> used;
    ChipPattern pat;
    if (stat) {
        pat = pattern_plan(p, q);
        for (const auto& [d, roles] : pat.ports) {
            for (const Hop& h : p.hops.at(q)) {
                if (h.dir == d) {
                    used.push_back(&h);
                    break;
                }
            }
        }
    } else {
        for (const Hop& h : p.hops.at(q)) {
            used.push_back(&h);
        }
        std::sort(used.begin(), used.end(), [](const Hop* a, const Hop* b) { return a->dir < b->dir; });
    }
    std::vector<CoreCoord> pcs;
    for (const Hop* h : used) {
        pcs.push_back(CoreCoord{1 + h->dir, 0});
    }
    std::vector<CoreCoord> all{hc};
    all.insert(all.end(), pcs.begin(), pcs.end());
    const CoreRangeSet hcrs = cells({hc});
    const CoreRangeSet pcrs = cells(pcs);
    const CoreRangeSet all_crs = cells(all);

    ProgramDescriptor prog;
    prog.cbs.push_back(byte_cb(0, ((kBlockRows + 1) * page) + 64, hcrs));
    if (!pcs.empty()) {
        prog.cbs.push_back(byte_cb(0, (kBlockRows * page) + 64, pcrs));
    }
    const uint32_t nsem = multi ? 1 : kBlocks;
    for (uint32_t b = 0; b < nsem; ++b) {
        prog.semaphores.push_back(SemaphoreDescriptor{.id = b, .core_ranges = all_crs, .initial_value = 0});
    }
    std::vector<uint32_t> hct{page};
    std::vector<uint32_t> pct{page, krows};
    if (multi) {
        hct.insert(hct.end(), {ns, p.consts.at(2)});
        pct.insert(pct.end(), {ns, p.consts.at(2)});
    }
    append_accessor(hct, io.at(0));
    append_accessor(hct, io.at(multi ? ns : 1));
    append_accessor(pct, io.at(0));
    append_accessor(pct, io.at(multi ? ns : 1));
    const std::vector<uint32_t> geo{p.rows, p.cols, p.wrap_x, p.wrap_y};
    std::vector<uint32_t> sems(p.sems.begin(), p.sems.begin() + (multi ? 4 * ns : kBlocks));

    // the io head: [ring, slab, page0, ...values]: static (page0, 4 ring rows), dyn (page0, rows, codes),
    // multi [ring0, slab0, (ring, slab, page0, rows, codes) x ns]
    const auto head = [&](Args& a) {
        if (multi) {
            a.io_addr(0);
            a.io_addr(ns);
            for (uint32_t s = 0; s < ns; ++s) {
                a.io_addr(s);
                a.io_addr(ns + s);
                for (uint32_t j = 0; j < 3; ++j) {
                    a.value((3 * s) + j);
                }
            }
            return;
        }
        a.io_addr(0);
        a.io_addr(1);
        for (uint32_t j = 0; j < (stat ? 5u : 3u); ++j) {
            a.value(j);
        }
    };

    Args h{.kernel = 0, .core = hc, .io = io, .values = values, .slots = slots, .a = {}};
    head(h);
    if (stat) {
        h.us(pat.kinds);
        h.us(pat.masks);
        h.us(sems);
        h.u(static_cast<uint32_t>(used.size()));
        for (const auto& pc : pcs) {
            const CoreCoord w = mesh.worker_core_from_logical_core(pc);
            h.us({static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)});
        }
    } else {
        h.us({r, c});
        h.us(geo);
        if (dyn) {
            h.us(sems);
        }
        h.u(static_cast<uint32_t>(used.size()));
        for (size_t i = 0; i < used.size(); ++i) {
            const CoreCoord w = mesh.worker_core_from_logical_core(pcs[i]);
            h.us({static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y), used[i]->peer_r, used[i]->peer_c});
        }
        if (multi) {
            h.us(sems);
        }
    }
    KernelDescriptor home;
    home.kernel_source = kernel_path(stat ? "snap_home.cpp" : dyn ? "snap_home_dyn.cpp" : "snap_home_multi.cpp");
    home.core_ranges = hcrs;
    home.compile_time_args = hct;
    home.config = dm(DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);
    h.emit(home);
    prog.kernels.push_back(std::move(home));
    if (pcs.empty()) {
        return prog;
    }
    KernelDescriptor port;
    port.kernel_source = kernel_path(stat ? "snap_port.cpp" : dyn ? "snap_port_dyn.cpp" : "snap_port_multi.cpp");
    port.core_ranges = pcrs;
    port.compile_time_args = pct;
    port.config = dm(DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);
    for (size_t i = 0; i < used.size(); ++i) {
        const Hop& hp = *used[i];
        Args a{.kernel = 1, .core = pcs[i], .io = io, .values = values, .slots = slots, .a = {}};
        head(a);
        if (stat) {
            a.us(pat.ports.at(hp.dir));
            a.us(sems);
        } else {
            a.us({r, c});
            a.us(geo);
            a.us({hp.peer_r, hp.peer_c});
            if (dyn) {
                a.us(sems);
            }
        }
        a.us({static_cast<uint32_t>(hv.x), static_cast<uint32_t>(hv.y)});
        a.us({node_mesh_id(mesh, hp.peer_r, hp.peer_c), node_chip_id(mesh, hp.peer_r, hp.peer_c)});
        if (multi) {
            a.us(sems);
        }
        a.fabric(
            mesh,
            tt::tt_metal::distributed::MeshCoordinate(r, c),
            tt::tt_metal::distributed::MeshCoordinate(hp.peer_r, hp.peer_c),
            hp.link,
            prog);
        a.emit(port);
    }
    prog.kernels.push_back(std::move(port));
    return prog;
}

// ---------------------------------------------------------------------------------------------------- glued send
// consts: [page, seg, nseg, window, batch, slot, q_rows, segs, copy_batch, ring_pieces]
// io: [region, rows, tail_0 .. tail_{S-1}, ring_0 .. ring_{S-1}]; values: 4 S carry values, 3 RP S ring pieces
ProgramDescriptor glued(
    const SendPlan& p,
    const std::vector<Tensor>& io,
    const std::vector<uint32_t>& values,
    uint32_t q,
    std::vector<SlotRef>& slots) {
    auto& mesh = *io.at(0).device();
    const auto& k = p.consts;
    const uint32_t page = k[0], window = k[3], batch = k[4], slot = k[5], q_rows = k[6], S = k[7];
    const uint32_t copy_batch = k[8], pieces = k[9];
    const uint32_t n = p.rows * p.cols;
    const uint32_t r = q / p.cols, c = q % p.cols;
    const bool send = q < n - 1, recv = q > 0;
    const CoreCoord cc{0, 0};
    const CoreRangeSet crs = cells({cc});
    const CoreCoord v = mesh.worker_core_from_logical_core(cc);
    ProgramDescriptor prog;
    prog.cbs.push_back(byte_cb(0, (batch * slot) + 64, crs));
    prog.cbs.push_back(byte_cb(1, (copy_batch * page) + 64, crs));
    std::vector<uint32_t> ct0{page, k[1], k[2], window, batch, q_rows - window, window + q_rows, S};
    append_accessor(ct0, io.at(0));
    append_accessor(ct0, io.at(1));
    for (uint32_t s = 0; s < S; ++s) {
        append_accessor(ct0, io.at(2 + s));
    }
    std::vector<uint32_t> ct1{page, copy_batch, pieces, S, q_rows, window};
    append_accessor(ct1, io.at(1));
    append_accessor(ct1, io.at(0));
    append_accessor(ct1, io.at(2 + S));
    // RISCV_0: [region, rows, sem, px, py, mesh_id, chip_id, tail x S, (active, tail, page, rot) x S, connection..]
    Args a0{.kernel = 0, .core = cc, .io = io, .values = values, .slots = slots, .a = {}};
    a0.io_addr(0);
    a0.io_addr(1);
    a0.us({p.sems.at(0), static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)});
    if (send) {
        const Hop& h = p.hops.at(q).at(0);
        a0.us({node_mesh_id(mesh, h.dst_r, h.dst_c), node_chip_id(mesh, h.dst_r, h.dst_c)});
    } else {
        a0.us({0, 0});
    }
    for (uint32_t s = 0; s < S; ++s) {
        a0.io_addr(2 + s);
    }
    for (uint32_t j = 0; j < 4 * S; ++j) {
        a0.value(j);
    }
    if (send) {
        const Hop& h = p.hops.at(q).at(0);
        a0.fabric(
            mesh,
            tt::tt_metal::distributed::MeshCoordinate(r, c),
            tt::tt_metal::distributed::MeshCoordinate(h.peer_r, h.peer_c),
            h.link,
            prog);
    }
    KernelDescriptor k0;
    k0.kernel_source = kernel_path("glued_send.cpp");
    k0.core_ranges = crs;
    k0.compile_time_args = ct0;
    k0.defines = {{"G_SEND", send ? "1" : "0"}, {"G_RECV", recv ? "1" : "0"}};
    k0.config = dm(DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);
    a0.emit(k0);
    // RISCV_1: [rows, region, ring x S, (own row, ring row, rows) x RP x S]
    Args a1{.kernel = 1, .core = cc, .io = io, .values = values, .slots = slots, .a = {}};
    a1.io_addr(1);
    a1.io_addr(0);
    for (uint32_t s = 0; s < S; ++s) {
        a1.io_addr(2 + S + s);
    }
    for (uint32_t j = 0; j < 3 * pieces * S; ++j) {
        a1.value((4 * S) + j);
    }
    KernelDescriptor k1;
    k1.kernel_source = kernel_path("glued_ring.cpp");
    k1.core_ranges = crs;
    k1.compile_time_args = ct1;
    k1.config = dm(DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);
    a1.emit(k1);
    prog.kernels.push_back(std::move(k0));
    prog.kernels.push_back(std::move(k1));
    return prog;
}

}  // namespace

ProgramDescriptor LayerSendProgramFactory::create_descriptor(
    const LayerSendParams& attrs,
    const LayerSendInputs& tensor_args,
    const ttnn::MeshCoordinate& coord,
    std::vector<SlotRef>& slots) {
    const SendPlan& p = *attrs.plan;
    const uint32_t q = (coord[0] * p.cols) + (coord.dims() > 1 ? coord[1] : 0);
    switch (p.kind) {
        case SendKind::WINDOW: return window(p, tensor_args.io, attrs.values, q, slots, false);
        case SendKind::WINDOW_FUSED: return window(p, tensor_args.io, attrs.values, q, slots, true);
        case SendKind::SNAP_STATIC:
        case SendKind::SNAP_DYN:
        case SendKind::SNAP_MULTI: return snapshot(p, tensor_args.io, attrs.values, q, slots);
        case SendKind::GLUED: return glued(p, tensor_args.io, attrs.values, q, slots);
    }
    TT_THROW("layer_send: unknown kind {}", static_cast<uint32_t>(p.kind));
}

LayerSendMeshWorkloadFactory::cached_mesh_workload_t LayerSendMeshWorkloadFactory::create_mesh_workload(
    const LayerSendParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const LayerSendInputs& tensor_args,
    Tensor& /*output*/) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (const auto& coord : tensor_coords.coords()) {
        std::vector<SlotRef> slots;
        Program program{LayerSendProgramFactory::create_descriptor(attrs, tensor_args, coord, slots)};
        shared_variables_t sv;
        sv.patches.reserve(slots.size());
        for (const auto& s : slots) {
            sv.patches.push_back(
                Patch{.args = &GetRuntimeArgs(program, s.kernel, s.core), .index = s.index, .src = s.src});
        }
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(program));
        shared.emplace(ttnn::MeshCoordinateRange(coord), std::move(sv));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void LayerSendMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const LayerSendParams& attrs,
    const LayerSendInputs& tensor_args,
    Tensor& /*output*/) {
    // The key pins every other arg (geometry, routes, semaphores, connections). A call writes its io addresses and
    // its values (the Python route's io slots and value slots) into every program.
    const auto& io = tensor_args.io;
    std::vector<uint32_t> addr(io.size());
    for (size_t i = 0; i < io.size(); ++i) {
        addr[i] = io[i].buffer()->address();
    }
    const auto& values = attrs.values;
    for (auto& [range, sv] : cached_workload.shared_variables) {
        for (const Patch& p : sv.patches) {
            (*p.args)[p.index] = p.src >= kValue ? values[p.src - kValue] : addr[p.src];
        }
    }
}

}  // namespace ttnn::operations::layer_send
