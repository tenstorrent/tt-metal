// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused tail's programs, one a chip (TailEngine, TailA2A):
//  * no per-call arena tensor: the engine's regions are program CBs; the landing CB comes first, on the engine and
//    the port cores, so it sits at the same address there and on every chip, and the kernels read the slots' bases
//    with get_write_ptr instead of taking them as runtime args;
//  * the fence's ready / arrival words are persistent word tensors of the resources (not global semaphores);
//  * TailA2A creates no port on a direction without a neighbour.

#include "fused_tail_program_factory.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <set>
#include <unordered_map>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::moe_block::fused_tail {

namespace fused_tail_factory_detail {

using tt::tt_metal::Buffer;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::DataMovementProcessor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::NOC;
using tt::tt_metal::ProgramDescriptor;

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_tail/device/kernels/";
constexpr uint32_t kPage = 2048;

// TailEngine's CB ids (the engine kernels' constants) and the ports' (compile-time args).
enum EngineCb : uint8_t {
    CB_W = 0,
    CB_A = 1,
    CB_PART = 2,
    CB_PMY = 3,
    CB_ONES = 4,
    CB_ACCM = 5,
    CB_XSEND = 6,
    CB_SQM = 7,
    CB_XRECV = 8,
    CB_SE = 9,
    CB_OUT = 10,
    CB_L0 = 11,
    CB_RM = 12,
    CB_PSTAGE = 13,
    CB_PCOUNT = 14,
    CB_POLL = 15,
};
// TailA2A's.
enum A2ACb : uint8_t { CB_A2A_RM = 0, CB_A2A_T = 1, CB_A2A_SE = 2, CB_A2A_OUT = 3, CB_A2A_CH = 4 };

KernelDescriptor kernel(const std::string& file, const CoreRangeSet& cores, KernelDescriptor::ConfigDescriptor config) {
    KernelDescriptor k;
    k.kernel_source = std::string(kKernelDir) + file;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.config = std::move(config);
    return k;
}

DataMovementConfigDescriptor dm(DataMovementProcessor risc, NOC noc) {
    return DataMovementConfigDescriptor{.processor = risc, .noc = noc};
}

CBDescriptor cb(uint8_t index, const CoreRangeSet& cores, uint32_t total, tt::DataFormat fmt, uint32_t page) {
    return CBDescriptor{
        .total_size = total,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{.buffer_index = index, .data_format = fmt, .page_size = page}},
    };
}

// (TailEngine's CB check: a block addressed from the CB's pointer must not straddle its end - every core pushes one
// size there and the dataflow ends read page by page, so each size must divide the CB)
void check_blocks(const char* name, uint32_t size, uint32_t page, const std::vector<uint32_t>& blocks) {
    for (uint32_t blk : std::set<uint32_t>(blocks.begin(), blocks.end())) {
        TT_FATAL(
            ring_pages(size / page, {blk}) == size / page,
            "fused_tail (engine): CB {} ({} pages) takes blocks of {}: one would straddle its end",
            name,
            size / page,
            blk);
    }
}

void append_accessor(KernelDescriptor& k, const Buffer* b) {
    tt::tt_metal::TensorAccessorArgs(*b).append_to(k.compile_time_args);
}

uint32_t u32(size_t v) { return static_cast<uint32_t>(v); }
uint32_t xy(const CoreCoord& v) { return u32(v.x | v.y << 16); }

CoreRangeSet cells(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.emplace_back(c, c);
    }
    return CoreRangeSet(ranges);
}

std::vector<ttnn::MeshCoordinate> coords_of(const ttnn::MeshCoordinateRangeSet& set) {
    std::vector<ttnn::MeshCoordinate> out;
    for (const auto& range : set.ranges()) {
        for (const auto& c : range) {
            out.push_back(c);
        }
    }
    return out;
}

uint32_t index_of(const std::vector<uint32_t>& v, uint32_t x) {
    return static_cast<uint32_t>(std::find(v.begin(), v.end(), x) - v.begin());
}

// The buffers the runtime args address, by source (address_patches.hpp; nullptr: absent).
std::array<Buffer*, 9> sources_of(const FusedTailInputs& a, const Tensor& output) {
    const auto opt = [](const std::optional<Tensor>& t) { return t.has_value() ? t->buffer() : nullptr; };
    return {
        a.y_own.buffer(),
        opt(a.shared_expert),
        opt(a.w_up),
        output.buffer(),
        a.ready.buffer(),
        a.words.buffer(),
        opt(a.port_epoch),
        opt(a.arrival),
        opt(a.staging)};
}

// The cached workload of one chip's program descriptors, with each program's address patches.
TailEngineFactory::cached_mesh_workload_t to_cached(
    tt::tt_metal::WorkloadDescriptor descriptors, const FusedTailInputs& args, const Tensor& output) {
    const auto sources = sources_of(args, output);
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, TailEngineFactory::shared_variables_t> shared;
    for (auto& [range, desc] : descriptors.programs) {
        TailEngineFactory::shared_variables_t sv{.patches = take_address_patches(desc, sources)};
        workload.add_program(range, tt::tt_metal::Program(desc));
        shared.emplace(range, std::move(sv));
    }
    return {std::move(workload), std::move(shared)};
}

void patch_cached(
    TailEngineFactory::cached_mesh_workload_t& cached, const FusedTailInputs& args, const Tensor& output) {
    const auto sources = sources_of(args, output);
    for (auto& [range, sv] : cached.shared_variables) {
        apply_address_patches(cached.workload.get_programs().at(range), sv.patches, sources);
    }
}

tt::tt_metal::WorkloadDescriptor engine_workload(
    const FusedTailParams& attrs,
    const FusedTailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords);
tt::tt_metal::WorkloadDescriptor a2a_workload(
    const FusedTailParams& attrs,
    const FusedTailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords);

}  // namespace fused_tail_factory_detail
using namespace fused_tail_factory_detail;

TailEngineFactory::cached_mesh_workload_t TailEngineFactory::create_mesh_workload(
    const FusedTailParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const FusedTailInputs& tensor_args,
    Tensor& output) {
    return to_cached(engine_workload(attrs, tensor_args, output, tensor_coords), tensor_args, output);
}

void TailEngineFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const FusedTailParams& /*attrs*/,
    const FusedTailInputs& tensor_args,
    Tensor& output) {
    patch_cached(cached, tensor_args, output);
}

TailA2AFactory::cached_mesh_workload_t TailA2AFactory::create_mesh_workload(
    const FusedTailParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const FusedTailInputs& tensor_args,
    Tensor& output) {
    return to_cached(a2a_workload(attrs, tensor_args, output, tensor_coords), tensor_args, output);
}

void TailA2AFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const FusedTailParams& /*attrs*/,
    const FusedTailInputs& tensor_args,
    Tensor& output) {
    patch_cached(cached, tensor_args, output);
}

// ---------------------------------------------------------------------------------------------------------------------
// TailEngine
// ---------------------------------------------------------------------------------------------------------------------

tt::tt_metal::WorkloadDescriptor fused_tail_factory_detail::engine_workload(
    const FusedTailParams& attrs,
    const FusedTailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& plan = *attrs.engine_plan;
    const auto& g = plan.geom;
    auto* mesh = args.y_own.device();
    const uint32_t GX = g.GX, GY = g.GY, PAIRS = g.PAIRS, G = g.chunks;
    const auto w_fmt = tt::tt_metal::datatype_to_dataformat_converter(args.w_up->dtype());
    const auto bf16 = tt::DataFormat::Float16_b, fp32 = tt::DataFormat::Float32;

    Buffer* y = args.y_own.buffer();
    Buffer* w = args.w_up->buffer();
    Buffer* se = args.shared_expert->buffer();
    Buffer* out = output.buffer();
    Buffer* words = args.words.buffer();
    Buffer* ready = args.ready.buffer();
    Buffer* epoch = args.port_epoch->buffer();

    auto virt = [&](uint32_t cx, uint32_t cy) { return mesh->worker_core_from_logical_core(CoreCoord(cx, cy)); };
    auto node = [&](uint32_t r, uint32_t c) { return mesh->get_fabric_node_id(ttnn::MeshCoordinate(r, c)); };
    auto schedule = [&](uint32_t p, TailDir d) {
        if (!row_neighbor(p, G, plan.ring, d)) {
            return std::vector<Unit>{};
        }
        const auto s = gather_schedule(p, G, plan.ring);
        return d == TailDir::Fwd ? s.first : s.second;
    };

    const CoreRangeSet& engine = plan.engine_cores;
    const CoreRangeSet& ports = plan.port_cores;
    const CoreRangeSet both = engine.merge(ports);
    const uint32_t PSTAGE = 3 * kPortGroup * plan.RP;

    // CBs: the landing slots first (engine and ports: one address everywhere), then the engine's, then the ports'.
    // (W' is resident: its K ranges pushed in order, never popped - no wrap)
    std::vector<CBDescriptor> cbs;
    auto engine_cb = [&](uint8_t i,
                         const char* name,
                         uint32_t size,
                         tt::DataFormat fmt,
                         uint32_t page,
                         std::vector<uint32_t> blocks) {
        check_blocks(name, size, page, blocks);
        cbs.push_back(cb(i, i == CB_L0 ? both : engine, size, fmt, page));
    };
    // (the landing slots of every step, SHMAX pages each: own steps tilized here, the others landed by the gather)
    engine_cb(CB_L0, "land", g.NSTMAX * g.SHMAX * kPage, bf16, kPage, {g.SHMAX});
    engine_cb(CB_W, "w", g.KH * g.PNMAX * g.wtb, w_fmt, g.wtb, {1});
    engine_cb(CB_A, "ring", g.NSLOT * g.KH * kPage, bf16, kPage, {g.KH});
    engine_cb(CB_PART, "part", g.NSLOT * GX * 4096, fp32, 4096, {GX});
    engine_cb(CB_PMY, "pmy", 4096, fp32, 4096, {1});
    engine_cb(CB_ONES, "ones", kPage, bf16, kPage, {1});
    engine_cb(CB_ACCM, "accm", g.FNR * 4096, fp32, 4096, g.FNS);
    engine_cb(CB_XSEND, "xsend", 2 * kXT * 4096, fp32, 4096, {kXT});
    engine_cb(CB_SQM, "sqm", 2 * 4096, fp32, 4096, {1});
    engine_cb(CB_XRECV, "xrecv", g.NSTMAX * kXT * 4096, fp32, 4096, {kXT});
    engine_cb(CB_SE, "se", g.FNR * kPage, bf16, kPage, g.FNS);
    engine_cb(CB_OUT, "out", g.FNR * kPage, bf16, kPage, g.FNS);
    // (my own shares' row-major source: pushes of my share's n tiles - no wrap within a call)
    engine_cb(CB_RM, "rm", g.NOWNMAX * g.SHMAX * kPage, bf16, kPage, {1});
    // the ports' staging, their group counts and their poll words
    cbs.push_back(cb(CB_PSTAGE, ports, PSTAGE * kPage, bf16, kPage));
    cbs.push_back(cb(CB_PCOUNT, ports, 4 * 16, tt::DataFormat::UInt32, 16));
    cbs.push_back(cb(CB_POLL, ports, 16 + g.RT * GX * 16, tt::DataFormat::UInt32, 16));

    const std::vector<uint32_t> ct = {g.KH, g.PNMAX, g.RT, g.chunks, GX, g.wtb, g.NSLOT, g.KT, kXT, PAIRS, g.NSTMAX};
    std::vector<uint32_t> wkb;
    for (const auto& [k0, k1] : g.wk) {
        wkb.push_back(k0);
    }
    wkb.push_back(g.KH);
    std::vector<uint32_t> akf;
    for (const auto& [k0, k1] : g.ak) {
        akf.push_back(k0);
        akf.push_back(k1);
    }
    std::vector<uint32_t> xs0;  // (the engine's NoC x by column)
    for (uint32_t c = 0; c < GX; ++c) {
        xs0.push_back(u32(virt(c, 0).x));
    }
    const CoreCoord g0 = virt(0, 0), g1 = virt(GX - 1, GY - 1);  // (the W' gate: lead core (0, 0), the NoC rectangle)
    const std::vector<uint32_t> gate = {
        xy(g0),
        u32(std::min(g0.x, g1.x)),
        u32(std::min(g0.y, g1.y)),
        u32(std::max(g0.x, g1.x)),
        u32(std::max(g0.y, g1.y)),
        GX * GY};
    const uint32_t wreg = g.region_tiles();
    const uint32_t pct_wg = kLandGroup;

    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& coord : coords_of(tensor_coords)) {
        const uint32_t r = coord[0], p = coord[1];
        const auto order = chunk_order(p, g.chunks, plan.ring);
        const auto relays = engine_relays(p, G, plan.ring, g);

        ProgramDescriptor desc;
        desc.cbs = ProgramDescriptor::CBDescriptors(cbs.begin(), cbs.end());

        auto reader =
            kernel("tail_engine/tail_engine_reader.cpp", engine, dm(DataMovementProcessor::RISCV_0, NOC::NOC_0));
        reader.compile_time_args = ct;
        append_accessor(reader, y);
        append_accessor(reader, se);
        auto writer =
            kernel("tail_engine/tail_engine_writer.cpp", engine, dm(DataMovementProcessor::RISCV_1, NOC::NOC_1));
        writer.compile_time_args = ct;
        append_accessor(writer, out);
        ComputeConfigDescriptor comp{
            .math_fidelity = attrs.math_fidelity,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .math_approx_mode = false,
        };
        comp.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        for (uint8_t i : {CB_ACCM, CB_SQM, CB_XRECV}) {
            comp.unpack_to_dest_mode[i] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
        auto compute = kernel("tail_engine/tail_engine_compute.cpp", engine, comp);
        compute.compile_time_args = {
            g.KH,
            GX,
            g.SHMAX,
            std::bit_cast<uint32_t>(static_cast<float>(1.0 / g.Hr)),
            std::bit_cast<uint32_t>(plan.norm_eps),
            kXT};

        for (uint32_t cx = 0; cx < GX; ++cx) {
            const uint32_t pn = g.pn[cx];
            const auto [ak0, ak1] = g.ak[cx];
            for (uint32_t cy = 0; cy < GY; ++cy) {
                const uint32_t pp = cy % PAIRS, h = cy / PAIRS;
                const auto [fo, fn] = g.fin[cx][h];
                const CoreCoord core(cx, cy);
                const uint32_t vx = u32(virt(cx, cy).x), vy = u32(virt(cx, cy).y);
                const uint32_t yt = u32(virt(cx, h * PAIRS).y), yb = u32(virt(cx, h * PAIRS + PAIRS - 1).y);
                const auto [wk0, wk1] = g.wk[pp];
                const uint32_t reg = (h * GX + cx) * PAIRS + pp;

                KernelDescriptor::RTArgList b;
                for (uint32_t x : {pp, pn, g.kh[h].first}) {
                    b.push_back(x);
                }
                b.push_back(w);
                for (uint32_t x : {reg % plan.nbank, (reg / plan.nbank) * wreg * g.wtb, wk0, wk1, vx, yt, yb}) {
                    b.push_back(x);
                }
                b.push_back(y);
                b.push_back(ak0);
                b.push_back(ak1);
                b.push_back(se);
                for (uint32_t x : {g.NTO, g.n0[cx] + fo, fn, g.nst[pp], g.nown[pp]}) {
                    b.push_back(x);
                }
                b.append(gate);
                b.push_back(static_cast<uint32_t>(cx == 0 && cy == 0));
                b.append(order);
                reader.emplace_runtime_args(core, b);

                KernelDescriptor::RTArgList n;
                for (uint32_t x :
                     {pp, pn, g.n0[cx] + fo, fn, ak0, ak1, u32(virt(0, cy).x), u32(virt(GX - 1, cy).x), cx}) {
                    n.push_back(x);
                }
                n.push_back(out);
                for (uint32_t x : {g.NTO, g.SHMAX, vx, vy, u32(virt(cx, (cy + PAIRS) % GY).y)}) {
                    n.push_back(x);
                }
                n.push_back(words);
                n.push_back(g.nst[pp]);
                n.push_back(g.nown[pp]);
                n.append(order);
                for (uint32_t c2 = 0; c2 < GX; ++c2) {
                    n.push_back(u32(virt(c2, cy).x));
                }
                n.append(wkb);
                const auto it = relays.find({cx, cy});
                std::vector<uint32_t> rl = it == relays.end() ? std::vector<uint32_t>{} : it->second;
                n.push_back(static_cast<uint32_t>(rl.size()));
                rl.resize(g.NSTMAX, 0);  // (a fixed length)
                n.append(rl);
                writer.emplace_runtime_args(core, n);

                std::vector<uint32_t> c = {pn, fo, fn, ak0, ak1, g.nst[pp], g.nown[pp], 0};
                c.insert(c.end(), wkb.begin() + 1, wkb.end());
                compute.runtime_args.emplace_back(core, std::move(c));
            }
        }

        // ---- the gather's ports ----
        const std::vector<uint32_t> pct = {
            g.RT, GX, plan.RP, g.SHMAX, g.KT, kPortGroup, PAIRS, pct_wg, CB_PSTAGE, CB_PCOUNT, CB_POLL, CB_L0};
        auto port_reader =
            kernel("tail_engine/tail_engine_port_reader.cpp", ports, dm(DataMovementProcessor::RISCV_1, NOC::NOC_0));
        port_reader.compile_time_args = pct;
        auto port_writer =
            kernel("tail_engine/tail_engine_port_writer.cpp", ports, dm(DataMovementProcessor::RISCV_0, NOC::NOC_1));
        port_writer.compile_time_args = pct;
        std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> fabric_args;
        for (const auto& [key, core] : plan.ports) {
            const auto [dd, l] = key;
            const auto peer = row_neighbor(p, G, plan.ring, dd);
            const auto sends = schedule(p, dd);
            std::vector<uint32_t> ys;
            for (uint32_t q = 0; q < PAIRS; ++q) {
                ys.push_back(u32(virt(0, q + PAIRS * l).y));
            }
            KernelDescriptor::RTArgList pr;
            pr.push_back(words);
            pr.push_back(static_cast<uint32_t>(sends.size()));
            pr.push_back(epoch);
            for (const auto& u : sends) {
                pr.push_back(u.part | index_of(order, u.pos) << 8);
            }
            pr.append(akf);
            pr.append(xs0);
            pr.append(ys);
            port_reader.emplace_runtime_args(core, pr);

            // (my ready to the peer's port that writes into this chip; it waits for it before writing my landing slots)
            const bool send_ready = peer.has_value() && !schedule(*peer, opposite(dd)).empty();
            const CoreCoord peer_port = plan.ports.at({opposite(dd), l});
            const CoreCoord rc = send_ready ? virt(peer_port.x, peer_port.y) : virt(0, 0);
            KernelDescriptor::RTArgList ps;
            ps.push_back(words);
            ps.push_back(ready);
            for (uint32_t x : {static_cast<uint32_t>(send_ready), u32(rc.x), u32(rc.y)}) {
                ps.push_back(x);
            }
            const bool linked = peer.has_value() && (!sends.empty() || send_ready);
            std::vector<uint32_t> tail;
            if (linked) {
                const auto pnode = node(r, *peer);
                const auto peer_order = chunk_order(*peer, g.chunks, plan.ring);
                ps.push_back(*pnode.mesh_id);
                ps.push_back(static_cast<uint32_t>(pnode.chip_id));
                ps.push_back(static_cast<uint32_t>(sends.size()));
                for (const auto& u : sends) {
                    ps.push_back(u.part | index_of(peer_order, u.pos) << 8);
                }
                const auto& links = plan.links.at({ChipCoord{r, p}, dd});
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    node(r, p), pnode, links.at(l), desc, core, tail);
            } else {
                for (uint32_t x : {0u, 0u, 0u}) {
                    ps.push_back(x);
                }
            }
            ps.append(akf);
            ps.append(xs0);
            ps.append(ys);
            ps.append(tail);
            port_writer.emplace_runtime_args(core, ps);
        }

        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(writer));
        desc.kernels.push_back(std::move(compute));
        desc.kernels.push_back(std::move(port_reader));
        desc.kernels.push_back(std::move(port_writer));
        for (uint32_t id = 0; id < 5; ++id) {  // SX, SFREE, SW, SG, SGO
            desc.semaphores.push_back(
                tt::tt_metal::SemaphoreDescriptor{.id = id, .core_ranges = engine, .initial_value = 0});
        }
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(desc)});
    }
    return workload;
}

// ---------------------------------------------------------------------------------------------------------------------
// TailA2A
// ---------------------------------------------------------------------------------------------------------------------

tt::tt_metal::WorkloadDescriptor fused_tail_factory_detail::a2a_workload(
    const FusedTailParams& attrs,
    const FusedTailInputs& args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& plan = *attrs.a2a_plan;
    const auto& g = plan.geom;
    const auto& sc = plan.sched;
    auto* mesh = args.y_own.device();
    const auto bf16 = tt::DataFormat::Float16_b;

    Buffer* y = args.y_own.buffer();
    Buffer* se = args.shared_expert.has_value() ? args.shared_expert->buffer() : nullptr;
    Buffer* out = output.buffer();
    Buffer* z = args.staging->buffer();
    Buffer* zr = args.words.buffer();
    Buffer* arr = args.arrival->buffer();
    Buffer* rdy = args.ready.buffer();

    auto virt = [&](const CoreCoord& c) { return mesh->worker_core_from_logical_core(c); };
    auto node = [&](uint32_t r, uint32_t c) { return mesh->get_fabric_node_id(ttnn::MeshCoordinate(r, c)); };
    const CoreRangeSet work = cells(plan.workers);
    const uint32_t W = g.W, has_se = se != nullptr ? 1u : 0u;

    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& coord : coords_of(tensor_coords)) {
        const uint32_t r = coord[0], p = coord[1];
        const ChipCoord ch{r, p};
        // (this chip's port cores: the port kernels and their CB go only there)
        std::vector<std::pair<TailSchedule::Port, CoreCoord>> my_ports;
        std::vector<CoreCoord> port_list;
        for (const auto& [key, core] : plan.ports) {
            if (std::get<0>(key) == ch) {
                my_ports.push_back({{p, std::get<1>(key), std::get<2>(key)}, core});
                port_list.push_back(core);
            }
        }
        const CoreRangeSet pcrs = cells(port_list);

        ProgramDescriptor desc;
        // (every block W pages - the port's RP x GROUP: no block straddles a ring's end)
        desc.cbs.push_back(cb(CB_A2A_RM, work, ring_pages(2 * W, {W}) * kPage, bf16, kPage));
        desc.cbs.push_back(cb(CB_A2A_T, work, ring_pages(W, {W}) * kPage, bf16, kPage));
        desc.cbs.push_back(cb(CB_A2A_SE, work, ring_pages(2 * W, {W}) * kPage, bf16, kPage));
        desc.cbs.push_back(cb(CB_A2A_OUT, work, ring_pages(2 * W, {W}) * kPage, bf16, kPage));
        desc.cbs.push_back(cb(CB_A2A_CH, pcrs, ring_pages(2 * g.group * g.RP, {g.group * g.RP}) * kPage, bf16, kPage));

        auto wreader =
            kernel("tail_a2a/tail_a2a_worker_reader.cpp", work, dm(DataMovementProcessor::RISCV_1, NOC::NOC_0));
        wreader.compile_time_args = {W, has_se};
        append_accessor(wreader, y);
        append_accessor(wreader, se != nullptr ? se : y);
        auto wcompute = kernel(
            "tail_a2a/tail_a2a_worker_compute.cpp",
            work,
            ComputeConfigDescriptor{
                .math_fidelity = tt::tt_metal::MathFidelity::HiFi2,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false});
        wcompute.compile_time_args = {W, has_se};
        auto wwriter =
            kernel("tail_a2a/tail_a2a_worker_writer.cpp", work, dm(DataMovementProcessor::RISCV_0, NOC::NOC_1));
        wwriter.compile_time_args = {W};
        append_accessor(wwriter, out);
        append_accessor(wwriter, z);
        const std::vector<uint32_t> pct = {CB_A2A_CH, kPage, g.RP, g.NBK, g.group, kIncEvery};
        auto preader =
            kernel("tail_a2a/tail_a2a_port_reader.cpp", pcrs, dm(DataMovementProcessor::RISCV_1, NOC::NOC_0));
        preader.compile_time_args = pct;
        append_accessor(preader, z);
        auto psender =
            kernel("tail_a2a/tail_a2a_port_writer.cpp", pcrs, dm(DataMovementProcessor::RISCV_0, NOC::NOC_1));
        psender.compile_time_args = pct;
        append_accessor(psender, out);

        for (const auto& [port, core] : my_ports) {
            const auto [pp, d, l] = port;
            const auto ds = sc.dests(p, d);
            KernelDescriptor::RTArgList pr;
            pr.push_back(z);
            pr.push_back(g.NW);
            pr.push_back(zr);
            for (uint32_t x : {g.RT, g.NTH, g.NTB, l, plan.num_links, static_cast<uint32_t>(ds.size())}) {
                pr.push_back(x);
            }
            for (const auto& u : ds) {
                pr.push_back(u.pos | u.part << 8);
            }
            preader.emplace_runtime_args(core, pr);

            const auto rs = sc.readies(port);
            std::set<uint32_t> dest_chips;
            for (const auto& u : ds) {
                dest_chips.insert(u.pos);
            }
            KernelDescriptor::RTArgList ps;
            ps.push_back(out);
            for (uint32_t x : {p, g.RT, g.NTB, l, plan.num_links}) {
                ps.push_back(x);
            }
            ps.push_back(arr);
            ps.push_back(sc.expect(port));
            ps.push_back(rdy);
            ps.push_back(static_cast<uint32_t>(dest_chips.size()));
            ps.push_back(static_cast<uint32_t>(rs.size()));
            for (const auto& [xp, xd, xl] : rs) {
                const auto n = node(r, xp);
                ps.push_back(xy(virt(plan.ports.at({ChipCoord{r, xp}, xd, xl}))));
                ps.push_back(static_cast<uint32_t>(n.chip_id));
                ps.push_back(*n.mesh_id);
                ps.push_back(sc.hops(p, xp, d));
            }
            ps.push_back(static_cast<uint32_t>(ds.size()));
            for (const auto& u : ds) {
                const auto n = node(r, u.pos);
                ps.push_back(u.pos | u.part << 8);
                ps.push_back(static_cast<uint32_t>(n.chip_id));
                ps.push_back(*n.mesh_id);
                ps.push_back(sc.hops(p, u.pos, d));
                ps.push_back(g.RT * sc.runs(u.part, l));
                ps.push_back(xy(virt(plan.ports.at({ChipCoord{r, u.pos}, opposite(d), l}))));
            }
            if (!ds.empty() || !rs.empty()) {
                const auto peer = row_neighbor(p, g.G, plan.ring, d);
                TT_FATAL(
                    peer.has_value(),
                    "fused_tail (a2a): ({}, {}) has no {} link",
                    r,
                    p,
                    d == TailDir::Fwd ? "fwd" : "bwd");
                const auto& links = plan.links.at({ch, d});
                std::vector<uint32_t> fabric;
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    node(r, p), node(r, *peer), links.at(l), desc, core, fabric);
                ps.append(fabric);
            }
            psender.emplace_runtime_args(core, ps);
        }

        // (the ports that carry something: the workers' z-ready words; a port with no destination neither waits nor
        // re-arms)
        std::vector<uint32_t> pxy;
        for (const auto& [port, core] : my_ports) {
            if (!sc.dests(p, std::get<1>(port)).empty()) {
                pxy.push_back(xy(virt(core)));
            }
        }
        for (uint32_t i = 0; i < plan.workers.size(); ++i) {
            const CoreCoord& core = plan.workers[i];
            const uint32_t c0 = i * W;
            KernelDescriptor::RTArgList rd;
            rd.push_back(y);
            if (se != nullptr) {
                rd.push_back(se);
            } else {
                rd.push_back(0u);
            }
            for (uint32_t x : {c0, g.RT, g.NTH}) {
                rd.push_back(x);
            }
            wreader.emplace_runtime_args(core, rd);
            wcompute.runtime_args.emplace_back(core, std::vector<uint32_t>{g.RT});
            KernelDescriptor::RTArgList wr;
            wr.push_back(out);
            wr.push_back(z);
            for (uint32_t x : {c0, g.RT, g.NTH, g.NTB, p}) {
                wr.push_back(x);
            }
            wr.push_back(zr);
            wr.push_back(static_cast<uint32_t>(pxy.size()));
            wr.append(pxy);
            wwriter.emplace_runtime_args(core, wr);
        }

        desc.kernels.push_back(std::move(wreader));
        desc.kernels.push_back(std::move(wcompute));
        desc.kernels.push_back(std::move(wwriter));
        desc.kernels.push_back(std::move(preader));
        desc.kernels.push_back(std::move(psender));
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(desc)});
    }
    return workload;
}

}  // namespace ttnn::operations::experimental::moe_block::fused_tail
