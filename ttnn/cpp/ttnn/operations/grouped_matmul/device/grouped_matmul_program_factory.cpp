// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_program_factory.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

#include "grouped_matmul_planner.hpp"
#include "kernels/grouped_matmul_args.hpp"

namespace ttnn::operations::grouped_matmul {

using namespace tt::tt_metal;
using namespace ::grouped_matmul;

namespace {

constexpr const char* kGmmKernelDir = "ttnn/cpp/ttnn/operations/grouped_matmul/device/kernels/";

// grouped_matmul.py NUM_CB_SLOTS: the length of the unpack-to-dest mode table.
constexpr uint32_t kUnpackModeSlots = 64;

std::string gmm_kernel_path(const char* name) { return std::string(kGmmKernelDir) + name; }

struct TileBytes {
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t o = 0;
    uint32_t f32 = 0;
};

TileBytes gmm_tile_bytes(const Tensor& x, const Tensor& w, const Tensor& out) {
    return TileBytes{
        .a = tt::tt_metal::tile_size(x.dtype()),
        .b = tt::tt_metal::tile_size(w.dtype()),
        .o = tt::tt_metal::tile_size(out.dtype()),
        .f32 = tt::tt_metal::tile_size(DataType::FLOAT32),
    };
}

CBDescriptor gmm_cb(uint32_t idx, DataType dtype, uint32_t page, uint32_t pages, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = page * pages,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(idx),
            .data_format = datatype_to_dataformat_converter(dtype),
            .page_size = page,
        }},
    };
}

void add_gmm_cbs(
    ProgramDescriptor& desc,
    const Tensor& x,
    const Tensor& w,
    const Tensor& out,
    const TileBytes& tb,
    uint32_t depth,
    uint32_t mt,
    uint32_t bw,
    uint32_t pn_max,
    const CoreRangeSet& cores) {
    TT_FATAL(
        cb::OUT < hal::get_num_dataflow_buffers(),
        "grouped_matmul: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        cb::OUT,
        hal::get_num_dataflow_buffers());
    desc.cbs.push_back(gmm_cb(cb::IN0, x.dtype(), tb.a, depth * mt * bw, cores));
    desc.cbs.push_back(gmm_cb(cb::IN1, w.dtype(), tb.b, depth * bw * pn_max, cores));
    desc.cbs.push_back(gmm_cb(cb::PART, DataType::FLOAT32, tb.f32, mt * pn_max, cores));
    desc.cbs.push_back(gmm_cb(cb::OUT, out.dtype(), tb.o, mt * pn_max, cores));
}

void add_gmm_semaphores(ProgramDescriptor& desc, const CoreRangeSet& cores) {
    const CoreCoord first = cores.ranges().front().start_coord;
    for (const uint32_t expected : {sem::READY, sem::VALID}) {
        const auto id = desc.find_available_semaphore_id(first, tt::CoreType::WORKER);
        TT_FATAL(id.has_value() && *id == expected, "grouped_matmul: semaphore id {} unavailable", expected);
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = *id, .core_type = tt::CoreType::WORKER, .core_ranges = cores, .initial_value = 0});
    }
}

// The compute config both paths build (rt already clamped).
ComputeConfigDescriptor gmm_compute_config(const GroupedMatmulParams& attrs, uint32_t rt, uint32_t pn_max) {
    const bool fp32 = attrs.fp32_dest_acc_en;
    const uint32_t half = fp32 ? 4 : 8;
    const uint32_t dst = half * (rt * pn_max > half ? 2 : 1);
    TT_FATAL(rt * pn_max <= dst, "grouped_matmul: rt {} x {} column tiles exceed DEST ({} tiles)", rt, pn_max, dst);
    ComputeConfigDescriptor cfg;
    cfg.math_fidelity = attrs.math_fidelity;
    cfg.fp32_dest_acc_en = fp32;
    cfg.math_approx_mode = attrs.math_approx_mode;
    cfg.dst_full_sync_en = rt * pn_max > half;
    if (fp32) {
        cfg.unpack_to_dest_mode.assign(kUnpackModeSlots, UnpackToDestMode::Default);
        cfg.unpack_to_dest_mode[cb::PART] = UnpackToDestMode::UnpackToDestFp32;
    }
    return cfg;
}

CoreRangeSet merged_cores(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& c : cores) {
        ranges.emplace_back(c, c);
    }
    return CoreRangeSet(ranges).merge_ranges();
}

void add_reader_diag(KernelDescriptor& k, const GroupedMatmulParams& attrs) {
    if (attrs.diag_no_w) {
        k.defines.emplace_back("GMM_DIAG_NO_W", "1");
    }
    if (attrs.diag_no_x) {
        k.defines.emplace_back("GMM_DIAG_NO_X", "1");
    }
}

uint32_t interleaved_group_count(const GroupedMatmulParams& attrs, const GroupedMatmulInputs& t) {
    const uint32_t nt = t.w.logical_shape()[3] / tt::constants::TILE_WIDTH;
    const uint32_t gx = std::min(attrs.grid_x, nt);
    return (gx == 0 || nt % gx == 0) ? 1 : 2;  // column_split: widths base and base + 1
}

uint32_t bank_group_count(const GroupedMatmulParams& attrs) {
    if (attrs.one_group) {
        return 1;
    }
    std::vector<uint32_t> pns;
    for (const auto& s : attrs.bank_layout->slots) {
        if (std::find(pns.begin(), pns.end(), s.pn) == pns.end()) {
            pns.push_back(s.pn);
        }
    }
    return static_cast<uint32_t>(pns.size());
}

ProgramDescriptor interleaved_descriptor(
    const GroupedMatmulParams& attrs, const GroupedMatmulInputs& t, const Tensor& out) {
    const Tensor& x = t.x;
    const Tensor& w = t.w;
    const auto& xs = x.logical_shape();
    const auto& ws = w.logical_shape();
    const TileBytes tb = gmm_tile_bytes(x, w, out);
    InterleavedPlanInputs in{
        .x_rank = static_cast<uint32_t>(xs.rank()),
        .w_rank = static_cast<uint32_t>(ws.rank()),
        .a_tile_bytes = tb.a,
        .b_tile_bytes = tb.b,
        .o_tile_bytes = tb.o,
        .f32_tile_bytes = tb.f32,
        .grid_x = attrs.grid_x,
        .grid_y = attrs.grid_y,
        .block_w = attrs.block_w,
    };
    if (xs.rank() == 4 && ws.rank() == 4) {
        in.x0 = xs[0], in.x1 = xs[1], in.x2 = xs[2], in.x3 = xs[3];
        in.w0 = ws[0], in.w1 = ws[1], in.w2 = ws[2], in.w3 = ws[3];
    }
    const InterleavedPlan* p = solve_interleaved_plan(in);
    TT_FATAL(p != nullptr, "grouped_matmul: no layout for {} x {}", xs, ws);
    const uint32_t G = p->G, MT = p->MT, KT = p->KT, NT = p->NT, GX = p->GX, BW = p->BW;
    const uint32_t pn_max = p->pn_max();
    std::vector<uint32_t> n0s(GX, 0);
    for (uint32_t i = 1; i < GX; ++i) {
        n0s[i] = n0s[i - 1] + p->pns[i - 1];
    }
    const CoreRangeSet allc(CoreRange(CoreCoord(0, 0), CoreCoord(GX - 1, G - 1)));

    ProgramDescriptor desc;
    add_gmm_cbs(desc, x, w, out, tb, attrs.depth, MT, BW, pn_max, allc);
    add_gmm_semaphores(desc, allc);

    auto* dev = x.device();
    TT_FATAL(attrs.depth >= 2, "grouped_matmul: depth {} (>= 2)", attrs.depth);
    const uint32_t rt = std::max(1u, std::min(attrs.rt, MT));
    const ComputeConfigDescriptor cfg = gmm_compute_config(attrs, rt, pn_max);

    // One group per distinct width, in order of first appearance.
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> groups;
    for (uint32_t cx = 0; cx < GX; ++cx) {
        const uint32_t pn = p->pns[cx];
        auto it = std::find_if(groups.begin(), groups.end(), [pn](const auto& g) { return g.first == pn; });
        if (it == groups.end()) {
            groups.emplace_back(pn, std::vector<uint32_t>{cx});
        } else {
            it->second.push_back(cx);
        }
    }

    for (const auto& [pn, cols] : groups) {
        std::vector<CoreRange> ranges;
        ranges.reserve(cols.size());
        for (const uint32_t cx : cols) {
            ranges.emplace_back(CoreCoord(cx, 0), CoreCoord(cx, G - 1));
        }
        const CoreRangeSet crs = CoreRangeSet(ranges).merge_ranges();

        KernelDescriptor reader;
        reader.kernel_source = gmm_kernel_path("gmm_reader.cpp");
        reader.core_ranges = crs;
        reader.compile_time_args.assign(reader_ct::COUNT, 0u);
        reader.compile_time_args[reader_ct::MT] = MT;
        reader.compile_time_args[reader_ct::KT] = KT;
        reader.compile_time_args[reader_ct::NT] = NT;
        reader.compile_time_args[reader_ct::BW] = BW;
        reader.compile_time_args[reader_ct::PN] = pn;
        reader.compile_time_args[reader_ct::GX] = GX;
        reader.compile_time_args[reader_ct::CB_IN0] = cb::IN0;
        reader.compile_time_args[reader_ct::CB_IN1] = cb::IN1;
        reader.compile_time_args[reader_ct::IN0_TB] = tb.a;
        reader.compile_time_args[reader_ct::IN1_TB] = tb.b;
        reader.compile_time_args[reader_ct::PNX] = pn_max;
        TensorAccessorArgs(x.buffer()).append_to(reader.compile_time_args);
        TensorAccessorArgs(w.buffer()).append_to(reader.compile_time_args);
        reader.emplace_common_runtime_args({x.buffer(), w.buffer()});
        add_reader_diag(reader, attrs);
        reader.config = ReaderConfigDescriptor{};

        KernelDescriptor writer;
        writer.kernel_source = gmm_kernel_path("gmm_writer.cpp");
        writer.core_ranges = crs;
        writer.compile_time_args.assign(writer_ct::COUNT, 0u);
        writer.compile_time_args[writer_ct::MT] = MT;
        writer.compile_time_args[writer_ct::NT] = NT;
        writer.compile_time_args[writer_ct::PN] = pn;
        writer.compile_time_args[writer_ct::CB_OUT] = cb::OUT;
        writer.compile_time_args[writer_ct::OUT_TB] = tb.o;
        TensorAccessorArgs(out.buffer()).append_to(writer.compile_time_args);
        writer.emplace_common_runtime_args({out.buffer()});
        writer.config = WriterConfigDescriptor{};

        KernelDescriptor compute;
        compute.kernel_source = gmm_kernel_path("gmm_compute.cpp");
        compute.core_ranges = crs;
        compute.compile_time_args.assign(compute_ct::COUNT, 0u);
        compute.compile_time_args[compute_ct::MT] = MT;
        compute.compile_time_args[compute_ct::BW] = BW;
        compute.compile_time_args[compute_ct::PN] = pn;
        compute.compile_time_args[compute_ct::NBLK] = KT / BW;
        compute.compile_time_args[compute_ct::CB_IN0] = cb::IN0;
        compute.compile_time_args[compute_ct::CB_IN1] = cb::IN1;
        compute.compile_time_args[compute_ct::CB_PART] = cb::PART;
        compute.compile_time_args[compute_ct::CB_OUT] = cb::OUT;
        compute.compile_time_args[compute_ct::RT] = rt;
        compute.compile_time_args[compute_ct::PNX] = pn_max;
        if (attrs.diag_no_compute) {
            compute.defines.emplace_back("GMM_DIAG_NO_COMPUTE", "1");
        }
        compute.config = cfg;

        reader.runtime_args.reserve(cols.size() * G);
        writer.runtime_args.reserve(cols.size() * G);
        for (const uint32_t cx : cols) {
            for (uint32_t g = 0; g < G; ++g) {
                const CoreCoord s = dev->worker_core_from_logical_core(CoreCoord(0, g));
                KernelDescriptor::CoreRuntimeArgs rd(reader_rt::COUNT, 0u);
                rd[reader_rt::G] = g;
                rd[reader_rt::N0] = n0s[cx];
                rd[reader_rt::SENDER] = cx == 0 ? 1u : 0u;
                if (GX > 1) {
                    const CoreCoord lo = dev->worker_core_from_logical_core(CoreCoord(1, g));
                    const CoreCoord hi = dev->worker_core_from_logical_core(CoreCoord(GX - 1, g));
                    rd[reader_rt::MC_X0] = lo.x;
                    rd[reader_rt::MC_X1] = hi.x;
                    rd[reader_rt::MC_Y] = lo.y;
                }
                rd[reader_rt::SENDER_X] = s.x;
                rd[reader_rt::SENDER_Y] = s.y;
                reader.runtime_args.emplace_back(CoreCoord(cx, g), std::move(rd));
                KernelDescriptor::CoreRuntimeArgs wr(writer_rt::COUNT, 0u);
                wr[writer_rt::G] = g;
                wr[writer_rt::N0] = n0s[cx];
                writer.runtime_args.emplace_back(CoreCoord(cx, g), std::move(wr));
            }
        }
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(writer));
        desc.kernels.push_back(std::move(compute));
    }
    return desc;
}

ProgramDescriptor bank_descriptor(const GroupedMatmulParams& attrs, const GroupedMatmulInputs& t, const Tensor& out) {
    const Tensor& x = t.x;
    const Tensor& w = t.w;
    const BankLayout& layout = *attrs.bank_layout;
    const auto& xs = x.logical_shape();
    const uint32_t G = xs[1], M = xs[2], K = xs[3];
    TT_FATAL(
        G == layout.G && K / tt::constants::TILE_WIDTH == layout.KT && K % tt::constants::TILE_WIDTH == 0,
        "grouped_matmul: {} vs layout G {} KT {}",
        xs,
        layout.G,
        layout.KT);
    TT_FATAL(!layout.slots.empty(), "grouped_matmul: bank layout without slots");
    const TileBytes tb = gmm_tile_bytes(x, w, out);
    const uint32_t MT = M / tt::constants::TILE_HEIGHT, KT = layout.KT, NT = layout.NT;
    uint32_t pn_max = 0;
    for (const auto& s : layout.slots) {
        pn_max = std::max(pn_max, s.pn);
    }
    const uint32_t BW = solve_bank_block_w(MT, KT, pn_max, tb.a, tb.b, tb.o, tb.f32, attrs.block_w);
    TT_FATAL(BW != 0 && KT % BW == 0, "grouped_matmul: bank path, block_w {}", attrs.block_w);
    TT_FATAL(attrs.depth >= 2, "grouped_matmul: depth {} (>= 2)", attrs.depth);
    TT_FATAL(
        attrs.w_inflight >= 1 && attrs.w_inflight <= std::min(attrs.depth, 8u),
        "grouped_matmul: w_inflight {} (1 .. min(depth, 8))",
        attrs.w_inflight);
    TT_FATAL(
        attrs.x_ahead <= std::min(3u, attrs.depth - 1),
        "grouped_matmul: x_ahead {} (0 .. min(3, depth - 1))",
        attrs.x_ahead);

    std::vector<CoreCoord> all_cores;
    all_cores.reserve(layout.slots.size());
    for (const auto& s : layout.slots) {
        all_cores.emplace_back(s.core_x, s.core_y);
    }
    const CoreRangeSet allc = merged_cores(all_cores);

    ProgramDescriptor desc;
    add_gmm_cbs(desc, x, w, out, tb, attrs.depth, MT, BW, pn_max, allc);
    add_gmm_semaphores(desc, allc);

    auto* dev = x.device();
    const uint32_t rt = std::max(1u, std::min(attrs.rt, MT));
    const ComputeConfigDescriptor cfg = gmm_compute_config(attrs, rt, pn_max);

    std::vector<std::pair<uint32_t, std::vector<const BankSlot*>>> groups;
    for (const auto& s : layout.slots) {
        const uint32_t key = attrs.one_group ? 0 : s.pn;
        auto it = std::find_if(groups.begin(), groups.end(), [key](const auto& g) { return g.first == key; });
        if (it == groups.end()) {
            groups.emplace_back(key, std::vector<const BankSlot*>{&s});
        } else {
            it->second.push_back(&s);
        }
    }

    for (const auto& [pn, slots] : groups) {
        std::vector<CoreCoord> cores;
        cores.reserve(slots.size());
        for (const BankSlot* s : slots) {
            cores.emplace_back(s->core_x, s->core_y);
        }
        const CoreRangeSet crs = merged_cores(cores);

        KernelDescriptor reader;
        reader.kernel_source = gmm_kernel_path("gmm_bank_reader.cpp");
        reader.core_ranges = crs;
        reader.compile_time_args.assign(bank_reader_ct::COUNT, 0u);
        reader.compile_time_args[bank_reader_ct::MT] = MT;
        reader.compile_time_args[bank_reader_ct::KT] = KT;
        reader.compile_time_args[bank_reader_ct::BW] = BW;
        reader.compile_time_args[bank_reader_ct::PN] = pn;
        reader.compile_time_args[bank_reader_ct::PNX] = pn_max;
        reader.compile_time_args[bank_reader_ct::CB_IN0] = cb::IN0;
        reader.compile_time_args[bank_reader_ct::CB_IN1] = cb::IN1;
        reader.compile_time_args[bank_reader_ct::IN0_TB] = tb.a;
        reader.compile_time_args[bank_reader_ct::IN1_TB] = tb.b;
        reader.compile_time_args[bank_reader_ct::DEPTH] = attrs.depth;
        reader.compile_time_args[bank_reader_ct::W_INFLIGHT] = attrs.w_inflight;
        reader.compile_time_args[bank_reader_ct::X_AHEAD] = attrs.x_ahead;
        TensorAccessorArgs(x.buffer()).append_to(reader.compile_time_args);
        reader.emplace_common_runtime_args({x.buffer(), w.buffer()});
        add_reader_diag(reader, attrs);
        reader.config = ReaderConfigDescriptor{};

        KernelDescriptor writer;
        writer.kernel_source = gmm_kernel_path("gmm_writer.cpp");
        writer.core_ranges = crs;
        writer.compile_time_args.assign(writer_ct::COUNT, 0u);
        writer.compile_time_args[writer_ct::MT] = MT;
        writer.compile_time_args[writer_ct::NT] = NT;
        writer.compile_time_args[writer_ct::PN] = pn;
        writer.compile_time_args[writer_ct::CB_OUT] = cb::OUT;
        writer.compile_time_args[writer_ct::OUT_TB] = tb.o;
        TensorAccessorArgs(out.buffer()).append_to(writer.compile_time_args);
        writer.emplace_common_runtime_args({out.buffer()});
        writer.config = WriterConfigDescriptor{};

        KernelDescriptor compute;
        compute.kernel_source = gmm_kernel_path("gmm_compute.cpp");
        compute.core_ranges = crs;
        compute.compile_time_args.assign(compute_ct::COUNT, 0u);
        compute.compile_time_args[compute_ct::MT] = MT;
        compute.compile_time_args[compute_ct::BW] = BW;
        compute.compile_time_args[compute_ct::PN] = pn;
        compute.compile_time_args[compute_ct::NBLK] = KT / BW;
        compute.compile_time_args[compute_ct::CB_IN0] = cb::IN0;
        compute.compile_time_args[compute_ct::CB_IN1] = cb::IN1;
        compute.compile_time_args[compute_ct::CB_PART] = cb::PART;
        compute.compile_time_args[compute_ct::CB_OUT] = cb::OUT;
        compute.compile_time_args[compute_ct::RT] = rt;
        compute.compile_time_args[compute_ct::PNX] = pn_max;
        if (attrs.diag_no_compute) {
            compute.defines.emplace_back("GMM_DIAG_NO_COMPUTE", "1");
        }
        compute.config = cfg;
        if (attrs.one_group) {
            reader.defines.emplace_back("GMM_PN_RT", "1");
            writer.defines.emplace_back("GMM_PN_RT", "1");
            compute.defines.emplace_back("GMM_PN_RT", "1");
        }

        reader.runtime_args.reserve(slots.size());
        writer.runtime_args.reserve(slots.size());
        for (const BankSlot* s : slots) {
            const BankRect& r = layout.rect[s->cluster];
            const BankCore& sc = layout.sender[s->cluster];
            const CoreCoord lo = dev->worker_core_from_logical_core(CoreCoord(r.x0, r.y0));
            const CoreCoord hi = dev->worker_core_from_logical_core(CoreCoord(r.x1, r.y1));
            const CoreCoord snd = dev->worker_core_from_logical_core(CoreCoord(sc.x, sc.y));
            const CoreCoord core(s->core_x, s->core_y);
            KernelDescriptor::CoreRuntimeArgs rd(
                attrs.one_group ? bank_reader_rt::COUNT_PN_RT : bank_reader_rt::COUNT, 0u);
            rd[bank_reader_rt::G] = s->g;
            rd[bank_reader_rt::TILE0] = s->tile0;
            rd[bank_reader_rt::BANK] = s->bank;
            rd[bank_reader_rt::SENDER] = s->sender ? 1u : 0u;
            rd[bank_reader_rt::MC_X0] = lo.x;
            rd[bank_reader_rt::MC_Y0] = lo.y;
            rd[bank_reader_rt::MC_X1] = hi.x;
            rd[bank_reader_rt::MC_Y1] = hi.y;
            rd[bank_reader_rt::NDEST] = (r.x1 - r.x0 + 1) * (r.y1 - r.y0 + 1);
            rd[bank_reader_rt::SENDER_X] = snd.x;
            rd[bank_reader_rt::SENDER_Y] = snd.y;
            KernelDescriptor::CoreRuntimeArgs wr(attrs.one_group ? writer_rt::COUNT_PN_RT : writer_rt::COUNT, 0u);
            wr[writer_rt::G] = s->g;
            wr[writer_rt::N0] = s->n0;
            if (attrs.one_group) {
                rd[bank_reader_rt::PN] = s->pn;
                wr[writer_rt::PN] = s->pn;
                KernelDescriptor::CoreRuntimeArgs cp(compute_rt::COUNT_PN_RT, 0u);
                cp[compute_rt::PN] = s->pn;
                compute.runtime_args.emplace_back(core, std::move(cp));
            }
            reader.runtime_args.emplace_back(core, std::move(rd));
            writer.runtime_args.emplace_back(core, std::move(wr));
        }
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(writer));
        desc.kernels.push_back(std::move(compute));
    }
    return desc;
}

}  // namespace

ProgramDescriptor GroupedMatmulProgramFactory::create_descriptor(
    const GroupedMatmulParams& attrs, const GroupedMatmulInputs& tensors, Tensor& output) {
    if (attrs.bank_layout.has_value()) {
        return bank_descriptor(attrs, tensors, output);
    }
    return interleaved_descriptor(attrs, tensors, output);
}

void GroupedMatmulProgramFactory::override_runtime_arguments(
    Program& program,
    const GroupedMatmulParams& attrs,
    const GroupedMatmulInputs& tensors,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The key (every attribute and the tensor specs) pins all compile-time and per-core args; only the three
    // common address slots of each kernel group change. A slot already holding the call's address is left alone.
    const uint32_t x_addr = tensors.x.buffer()->address();
    const uint32_t w_addr = tensors.w.buffer()->address();
    const uint32_t y_addr = output.buffer()->address();
    const uint32_t groups =
        attrs.bank_layout.has_value() ? bank_group_count(attrs) : interleaved_group_count(attrs, tensors);
    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t base = g * kernel_role::KERNELS_PER_GROUP;
        auto& reader = GetCommonRuntimeArgs(program, base + kernel_role::READER);
        if (reader[reader_common::X_ADDR] != x_addr) {
            reader[reader_common::X_ADDR] = x_addr;
        }
        if (reader[reader_common::W_ADDR] != w_addr) {
            reader[reader_common::W_ADDR] = w_addr;
        }
        auto& writer = GetCommonRuntimeArgs(program, base + kernel_role::WRITER);
        if (writer[writer_common::Y_ADDR] != y_addr) {
            writer[writer_common::Y_ADDR] = y_addr;
        }
    }
}

}  // namespace ttnn::operations::grouped_matmul
