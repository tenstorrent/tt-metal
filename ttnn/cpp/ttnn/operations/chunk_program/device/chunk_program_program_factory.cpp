// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The per-chunk prepared programs of deepseek_v41_flash_d_p, ported from the model's Python prepared programs (NC.7):
// each range's program is the Python _build loop body for that range, the kernels are the Python kernel strings moved
// to device/kernels (verbatim; GluedRows takes NPAIR tables, the derive kernel one accessor config for its outputs),
// and every runtime arg the Python route patched per call (an io address or a per-call value) is a slot the override
// rewrites.

#include "chunk_program_program_factory.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::chunk_program {

using namespace tt::tt_metal;

namespace {

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/chunk_program/device/kernels/";

using Arg = std::variant<uint32_t, Buffer*>;

std::string kernel_path(const char* name) { return std::string(kKernelDir) + name; }

uint32_t align64(uint32_t b) { return (b + 63) & ~63u; }

CoreRangeSet rect(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
    return CoreRangeSet(CoreRange(CoreCoord{x0, y0}, CoreCoord{x1, y1}));
}

CoreRangeSet cells(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& c : cores) {
        ranges.emplace_back(c, c);
    }
    return CoreRangeSet(ranges);
}

DataMovementConfigDescriptor dm0() {
    return DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
}

CBDescriptor byte_cb(uint32_t bytes, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = bytes,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = 0, .data_format = tt::DataFormat::UInt8, .page_size = bytes}},
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

// common runtime args that are all per-call values [first, first + n)
void common_values(
    KernelDescriptor& k,
    uint32_t kernel,
    const std::vector<uint32_t>& values,
    uint32_t first,
    uint32_t n,
    std::vector<SlotRef>& slots) {
    k.common_runtime_args.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        slots.push_back(SlotRef{.kernel = kernel, .core = {}, .index = i, .src = kValue + first + i, .common = true});
        k.common_runtime_args.push_back(values.at(first + i));
    }
}

KernelDescriptor dm_kernel(const char* name, const CoreRangeSet& crs, std::vector<uint32_t> ct) {
    KernelDescriptor k;
    k.kernel_source = kernel_path(name);
    k.core_ranges = crs;
    k.compile_time_args = std::move(ct);
    k.config = dm0();
    return k;
}

// ------------------------------------------------------------------------------------------------------ page copy
// consts: [page, batch, cores]; io [src, dst]; values [s0, d0, n, nb, ss, ds] (tt/page_copy.PageCopy._build)
ProgramDescriptor page_copy(
    const ChunkPlan& p,
    const std::vector<Tensor>& io,
    const std::vector<uint32_t>& values,
    std::vector<SlotRef>& slots) {
    const uint32_t page = p.consts.at(0), batch = p.consts.at(1), nc = p.consts.at(2);
    const CoreRangeSet crs = rect(0, 0, nc - 1, 0);
    ProgramDescriptor prog;
    prog.cbs.push_back(byte_cb(64 + (batch * align64(page)), crs));
    std::vector<uint32_t> ct{page, batch, nc};
    append_accessor(ct, io.at(0));
    append_accessor(ct, io.at(1));
    KernelDescriptor k = dm_kernel("page_copy.cpp", crs, ct);
    for (uint32_t c = 0; c < nc; ++c) {
        Args a{.kernel = 0, .core = CoreCoord{c, 0}, .io = io, .values = values, .slots = slots, .a = {}};
        a.io_addr(0);
        a.io_addr(1);
        a.u(c);
        for (uint32_t j = 0; j < 6; ++j) {
            a.value(j);
        }
        a.emit(k);
    }
    prog.kernels.push_back(std::move(k));
    return prog;
}

// ----------------------------------------------------------------------------------------------------- glued rows
// consts: [page, pages a unit, batch, cores, values a mesh row, pairs]; io (table, out) x pairs; values: every mesh
// row's pieces (common args; tt/lane_rows.GluedRows._build, common=True)
ProgramDescriptor glued_rows(
    const ChunkPlan& p,
    const std::vector<Tensor>& io,
    const std::vector<uint32_t>& values,
    uint32_t r,
    std::vector<SlotRef>& slots) {
    const auto& k = p.consts;
    const uint32_t page = k.at(0), batch = k.at(2), nc = k.at(3), gw = k.at(4), npair = k.at(5);
    TT_FATAL(npair >= 1 && npair <= 2, "chunk_program glued_rows: 1 or 2 pairs ({})", npair);
    const CoreRangeSet crs = rect(0, 0, nc - 1, 0);
    ProgramDescriptor prog;
    prog.cbs.push_back(byte_cb(64 + (batch * align64(page)), crs));
    std::vector<uint32_t> ct{page, k.at(1), batch, nc, gw};
    for (uint32_t i = 0; i < 2 * npair; ++i) {
        append_accessor(ct, io.at(i));
    }
    KernelDescriptor kd = dm_kernel("glued_rows.cpp", crs, ct);
    kd.defines = {{"NPAIR", std::to_string(npair)}};
    for (uint32_t c = 0; c < nc; ++c) {
        Args a{.kernel = 0, .core = CoreCoord{c, 0}, .io = io, .values = values, .slots = slots, .a = {}};
        for (uint32_t i = 0; i < 2 * npair; ++i) {
            a.io_addr(i);
        }
        a.us({c, r});
        a.emit(kd);
    }
    common_values(kd, 0, values, 0, p.rows * gw, slots);
    prog.kernels.push_back(std::move(kd));
    return prog;
}

// --------------------------------------------------------------------------------------------------- ids reorders
// IDS_REORDER consts: [units a mesh row, SP, vocab - 1]; io [gathered input, dst]; values [iu, cs, valid]
// GLUED_IDS consts: [W, SP, vocab - 1, header words, block, cap, align]; io [input (gathered unless align), dst]
ProgramDescriptor ids(
    const ChunkPlan& p,
    const std::vector<Tensor>& io,
    const std::vector<uint32_t>& values,
    uint32_t r,
    std::vector<SlotRef>& slots) {
    const auto& k = p.consts;
    const bool glued = p.kind == ChunkKind::GLUED_IDS;
    const CoreRangeSet crs = rect(0, 0, 0, 0);
    ProgramDescriptor prog;
    uint32_t size = 0;
    if (glued) {
        const uint32_t w = k.at(0), sp = k.at(1), hdr = k.at(3), align = k.at(6);
        size = 64 + (4 * (((align != 0u ? 1 : sp) * (w + hdr)) + w));
    } else {
        size = 64 + (k.at(0) * 128);
    }
    prog.cbs.push_back(byte_cb(size, crs));
    std::vector<uint32_t> ct(k.begin(), k.end());
    append_accessor(ct, io.at(0));
    append_accessor(ct, io.at(1));
    KernelDescriptor kd = dm_kernel(glued ? "glued_ids.cpp" : "ids_reorder.cpp", crs, ct);
    Args a{.kernel = 0, .core = CoreCoord{0, 0}, .io = io, .values = values, .slots = slots, .a = {}};
    a.io_addr(0);
    a.io_addr(1);
    if (!glued) {
        a.value(0);
        a.value(1);
        a.value(2);
    }
    a.u(r);
    a.emit(kd);
    prog.kernels.push_back(std::move(kd));
    return prog;
}

// --------------------------------------------------------------------------------------------------------- derive
// consts: [chunk, SP, TP, R, block, align]; groups: the sections, each
//   [code (0 keep, 1 region, 2 ckeep, 3 idx, 4 srv), cand, bsel, ckeep_fast, cores, a0..a5, outputs, first output io,
//    values, first value]
// io [header, the sections' outputs]; values [header byte offset, the sections' values]. Every section runs the
// derive kernel (tt/glued_derive.Derive._build) on its own cores of the one program: a lone section on the Python
// route's cores ((0, 0) .. (min(cores, 8) - 1, cores / 8)); more sections side by side.
ProgramDescriptor derive(
    const ChunkPlan& p,
    const std::vector<Tensor>& io,
    const std::vector<uint32_t>& values,
    uint32_t q,
    std::vector<SlotRef>& slots) {
    const auto& k = p.consts;
    const uint32_t chunk = k.at(0), sp = k.at(1), tp = k.at(2), R = k.at(3), block = k.at(4), align = k.at(5);
    const auto grid = io.at(0).device()->compute_with_storage_grid_size();
    ProgramDescriptor prog;
    std::vector<CoreRange> all;
    uint32_t x = 0, y = 0;
    const uint32_t hpage = io.at(0).buffer()->page_size();
    for (size_t si = 0; si < p.groups.size(); ++si) {
        const auto& s = p.groups[si];
        const uint32_t code = s.at(0), nc = s.at(4), nout = s.at(11), o0 = s.at(12), nvals = s.at(13), v0 = s.at(14);
        const uint32_t cx = std::min<uint32_t>(nc, 8), cy = nc / cx;
        TT_FATAL(cx * cy == nc, "chunk_program derive: {} cores", nc);
        uint32_t x0 = 0, y0 = 0;
        if (nc > 1) {
            if (x > 0) {
                x = 0;
                ++y;
            }
            x0 = 0;
            y0 = y;
            y += cy;
        } else {
            x0 = x;
            y0 = y;
            if (++x == 8) {
                x = 0;
                ++y;
            }
        }
        TT_FATAL(
            x0 + cx <= grid.x && y0 + cy <= grid.y,
            "chunk_program derive: sections past the {}x{} grid",
            grid.x,
            grid.y);
        const CoreRange cr(CoreCoord{x0, y0}, CoreCoord{x0 + cx - 1, y0 + cy - 1});
        all.push_back(cr);
        std::vector<uint32_t> ct{chunk, sp, tp, R, block, nc};
        for (uint32_t i = 0; i < 6; ++i) {
            ct.push_back(s.at(5 + i));
        }
        append_accessor(ct, io.at(0));
        append_accessor(ct, io.at(o0));
        KernelDescriptor kd = dm_kernel("derive.cpp", CoreRangeSet(cr), ct);
        auto flag = [](bool b) { return std::string(b ? "1" : "0"); };
        kd.defines = {
            {"D_KEEP", flag(code == 0)},
            {"D_REGION", flag(code == 1)},
            {"D_CKEEP", flag(code == 2)},
            {"D_IDX", flag(code == 3)},
            {"D_SRV", flag(code == 4)},
            {"D_CAND", flag(code == 3 && s.at(1) != 0)},
            {"D_BSEL", flag(code == 3 && s.at(2) != 0)},
            {"CKEEP_FAST", flag(code == 2 && s.at(3) != 0)},
            {"NOUT", std::to_string(nout)},
            {"ALIGN_ROWS", std::to_string(align)},
            {"P0", std::to_string(io.at(o0).buffer()->page_size())},
            {"P1", std::to_string(nout > 1 ? io.at(o0 + 1).buffer()->page_size() : 4)},
            {"HPAGE_BYTES", std::to_string(hpage)},
        };
        if (code == 3) {
            kd.defines.emplace_back("TAB_ARG", std::to_string(nout));
        }
        const uint32_t kernel = static_cast<uint32_t>(prog.kernels.size());
        for (uint32_t kk = 0; kk < nc; ++kk) {
            Args a{
                .kernel = kernel,
                .core = CoreCoord{x0 + (kk % cx), y0 + (kk / cx)},
                .io = io,
                .values = values,
                .slots = slots,
                .a = {}};
            a.io_addr(0);
            for (uint32_t i = 0; i < nout; ++i) {
                a.io_addr(o0 + i);
            }
            a.us({q, kk});
            a.value(0);
            for (uint32_t v = 0; v < nvals; ++v) {
                a.value(v0 + v);
            }
            a.emit(kd);
        }
        prog.kernels.push_back(std::move(kd));
    }
    prog.cbs.push_back(byte_cb(64 + 128 + 16384, CoreRangeSet(all)));
    return prog;
}

// ----------------------------------------------------------------------------------------------------- owned send
// consts: [page, K, N, C, axis, L, NS, NW, R, B, SPIN, home x, home y, port cores, (x, y) x port cores] (logical);
// groups[q]: chip q's ports (peer row, peer col, link id, link index) x ports, in tt/owned_gather.OwnedSend._build's
// order; sems: [started, arrived]. io [out, a record buffer (its accessor args only)]; values: the common values
// [record buffers, lo, hi, map words], the same in both kernels.
ProgramDescriptor owned_send(
    const ChunkPlan& p,
    const std::vector<Tensor>& io,
    const std::vector<uint32_t>& values,
    uint32_t q,
    std::vector<SlotRef>& slots) {
    auto& mesh = *io.at(0).device();
    const auto& k = p.consts;
    const uint32_t page = k.at(0), K = k.at(1), n = k.at(2), B = k.at(9);
    const uint32_t r = q / p.cols, c = q % p.cols;
    const CoreCoord hc{k.at(11), k.at(12)};
    const uint32_t ncores = k.at(13);
    std::vector<CoreCoord> port_cores;
    for (uint32_t i = 0; i < ncores; ++i) {
        port_cores.push_back(CoreCoord{k.at(14 + (2 * i)), k.at(15 + (2 * i))});
    }
    const auto& g = p.groups.at(q);
    const uint32_t nports = static_cast<uint32_t>(g.size() / 4);
    TT_FATAL(nports <= ncores, "chunk_program owned_send: {} ports, {} port cores", nports, ncores);
    std::vector<CoreCoord> pcs(port_cores.begin(), port_cores.begin() + nports);
    std::vector<CoreCoord> allc{hc};
    allc.insert(allc.end(), pcs.begin(), pcs.end());
    const CoreRangeSet all_crs = cells(allc);
    const CoreRangeSet hcrs = cells({hc});
    const CoreRangeSet pcrs = cells(pcs);
    ProgramDescriptor prog;
    prog.cbs.push_back(byte_cb((2 * B * K * page) + 64, all_crs));
    prog.semaphores.push_back(SemaphoreDescriptor{.id = 0, .core_ranges = all_crs, .initial_value = 0});
    std::vector<uint32_t> ct(k.begin(), k.begin() + 11);
    append_accessor(ct, io.at(0));
    append_accessor(ct, io.at(1));
    const uint32_t nv = (3 * k.at(6)) + k.at(7);
    const CoreCoord hv = mesh.worker_core_from_logical_core(hc);
    std::vector<uint32_t> ids(n);
    for (uint32_t d = 0; d < n; ++d) {
        ids[d] = static_cast<uint32_t>(
            mesh.get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(d / p.cols, d % p.cols)).chip_id);
    }
    // home: [out, me, started, arrived, nports, (px, py) x nports]
    KernelDescriptor kh = dm_kernel("owned_home.cpp", hcrs, ct);
    {
        Args a{.kernel = 0, .core = hc, .io = io, .values = values, .slots = slots, .a = {}};
        a.io_addr(0);
        a.us({q, p.sems.at(0), p.sems.at(1), nports});
        for (uint32_t i = 0; i < nports; ++i) {
            const CoreCoord v = mesh.worker_core_from_logical_core(pcs[i]);
            a.us({static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)});
        }
        a.emit(kh);
    }
    common_values(kh, 0, values, 0, nv, slots);
    // ports: [out, me, peer chip, mesh id, link, hx, hy, started, arrived, chip id x N, connection..]
    KernelDescriptor kp = dm_kernel("owned_port.cpp", pcrs, ct);
    for (uint32_t i = 0; i < nports; ++i) {
        const uint32_t pr = g[4 * i], pc = g[(4 * i) + 1], link_id = g[(4 * i) + 2], li = g[(4 * i) + 3];
        const auto pn = mesh.get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(pr, pc));
        Args a{.kernel = 1, .core = pcs[i], .io = io, .values = values, .slots = slots, .a = {}};
        a.io_addr(0);
        a.us({q, static_cast<uint32_t>(pn.chip_id), static_cast<uint32_t>(*pn.mesh_id), li});
        a.us({static_cast<uint32_t>(hv.x), static_cast<uint32_t>(hv.y), p.sems.at(0), p.sems.at(1)});
        a.us(ids);
        a.fabric(
            mesh,
            tt::tt_metal::distributed::MeshCoordinate(r, c),
            tt::tt_metal::distributed::MeshCoordinate(pr, pc),
            link_id,
            prog);
        a.emit(kp);
    }
    common_values(kp, 1, values, 0, nv, slots);
    prog.kernels.push_back(std::move(kh));
    prog.kernels.push_back(std::move(kp));
    return prog;
}

}  // namespace

ProgramDescriptor ChunkProgramProgramFactory::create_descriptor(
    const ChunkProgramParams& attrs,
    const ChunkProgramInputs& tensor_args,
    const ttnn::MeshCoordinate& coord,
    std::vector<SlotRef>& slots) {
    const ChunkPlan& p = *attrs.plan;
    const uint32_t r = coord[0];
    const uint32_t q = (coord[0] * p.cols) + (coord.dims() > 1 ? coord[1] : 0);
    const auto& io = tensor_args.io;
    switch (p.kind) {
        case ChunkKind::PAGE_COPY: return page_copy(p, io, attrs.values, slots);
        case ChunkKind::GLUED_ROWS: return glued_rows(p, io, attrs.values, r, slots);
        case ChunkKind::IDS_REORDER:
        case ChunkKind::GLUED_IDS: return ids(p, io, attrs.values, r, slots);
        case ChunkKind::DERIVE: return derive(p, io, attrs.values, q, slots);
        case ChunkKind::OWNED_SEND: return owned_send(p, io, attrs.values, q, slots);
    }
    TT_THROW("chunk_program: unknown kind {}", static_cast<uint32_t>(p.kind));
}

ChunkProgramMeshWorkloadFactory::cached_mesh_workload_t ChunkProgramMeshWorkloadFactory::create_mesh_workload(
    const ChunkProgramParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ChunkProgramInputs& tensor_args,
    Tensor& /*output*/) {
    const ChunkPlan& p = *attrs.plan;
    // the ranges of the Python MeshProgramDescriptor: the whole mesh, an SP row each, or a chip each
    std::vector<ttnn::MeshCoordinateRange> ranges;
    const uint32_t R = p.rows, C = p.cols;
    TT_FATAL(tensor_coords.coords().size() == R * C, "chunk_program: io on every chip of the {}x{} mesh", R, C);
    switch (p.kind) {
        case ChunkKind::PAGE_COPY:
            ranges.emplace_back(ttnn::MeshCoordinate(0, 0), ttnn::MeshCoordinate(R - 1, C - 1));
            break;
        case ChunkKind::GLUED_ROWS:
        case ChunkKind::IDS_REORDER:
        case ChunkKind::GLUED_IDS:
            for (uint32_t r = 0; r < R; ++r) {
                ranges.emplace_back(ttnn::MeshCoordinate(r, 0), ttnn::MeshCoordinate(r, C - 1));
            }
            break;
        case ChunkKind::DERIVE:
        case ChunkKind::OWNED_SEND:
            for (const auto& coord : tensor_coords.coords()) {
                ranges.emplace_back(coord);
            }
            break;
    }
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (const auto& range : ranges) {
        std::vector<SlotRef> slots;
        Program program{ChunkProgramProgramFactory::create_descriptor(attrs, tensor_args, range.start_coord(), slots)};
        shared_variables_t sv;
        sv.patches.reserve(slots.size());
        for (const auto& s : slots) {
            auto* args =
                s.common ? &GetCommonRuntimeArgs(program, s.kernel) : &GetRuntimeArgs(program, s.kernel, s.core);
            sv.patches.push_back(Patch{.args = args, .index = s.index, .src = s.src});
        }
        workload.add_program(range, std::move(program));
        shared.emplace(range, std::move(sv));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void ChunkProgramMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ChunkProgramParams& attrs,
    const ChunkProgramInputs& tensor_args,
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

}  // namespace ttnn::operations::chunk_program
