// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "distributed_rms_norm_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "distributed_rms_norm_planner.hpp"
#include "kernels/distributed_rms_norm_args.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::distributed_rms_norm {

using namespace tt::tt_metal;
namespace args = ::distributed_rms_norm;
using tt::tt_metal::distributed::MeshCoordinate;

namespace {

constexpr const char* kDrmsKernelDir = "ttnn/cpp/ttnn/operations/distributed_rms_norm/device/kernels/";

std::string drms_kernel_path(const char* name) { return std::string(kDrmsKernelDir) + name; }

// Python packs a double into fp32 bits (struct.pack("<f", value)).
uint32_t fp32_bits(double value) { return std::bit_cast<uint32_t>(static_cast<float>(value)); }

uint32_t pack_xy(const CoreCoord& c) { return (static_cast<uint32_t>(c.x) << 16) | static_cast<uint32_t>(c.y); }

CBDescriptor plain_cb(
    uint32_t index, tt::DataFormat format, uint32_t page_size, uint32_t pages, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = page_size * pages,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index), .data_format = format, .page_size = page_size}},
    };
}

// The compute-core CB inventory (detail::cb_inventory, the same list the L1 fit summed), in its order.
void add_compute_cbs(
    ProgramDescriptor& desc,
    const detail::Plan& plan,
    DataType input_dtype,
    DataType output_dtype,
    std::optional<DataType> gamma_dtype) {
    const detail::Geometry& g = plan.g;
    TT_FATAL(
        !g.has_gamma || gamma_dtype.has_value(),
        "distributed_rms_norm: the plan has gamma but no gamma tensor was given");
    const auto inventory = detail::cb_inventory(
        g,
        plan.block_rows,
        plan.x_depth,
        datatype_to_dataformat_converter(input_dtype),
        datatype_to_dataformat_converter(output_dtype),
        gamma_dtype.has_value() ? datatype_to_dataformat_converter(*gamma_dtype) : tt::DataFormat::Invalid);
    for (const detail::CbEntry& cb : inventory) {
        desc.cbs.push_back(plain_cb(cb.index, cb.format, cb.page_size, cb.pages, plan.compute_cores));
    }
}

// Pushes a semaphore over `cores` taking the lowest id free on `probe` (a core of `cores`, whose ids match every
// other core's: the program semaphores are pushed in order on an empty descriptor). The kernels know the ids from
// the args header, so the id the descriptor hands out must be that one.
void push_semaphore(ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t expected_id) {
    const CoreCoord probe = cores.ranges().front().start_coord;
    const std::optional<uint32_t> id = desc.find_available_semaphore_id(probe, tt::CoreType::WORKER);
    TT_FATAL(
        id.has_value() && *id == expected_id,
        "distributed_rms_norm: semaphore id {} expected, the descriptor has {}",
        expected_id,
        id.value_or(0u));
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = *id, .core_type = tt::CoreType::WORKER, .core_ranges = cores, .initial_value = 0});
}

// Row-group members per root as row-runs (lo, hi, count): consecutive x on one row.
using MemberRun = std::tuple<CoreCoord, CoreCoord, uint32_t>;
std::map<CoreCoord, std::vector<MemberRun>> member_runs_of(const detail::Plan& plan) {
    std::vector<const detail::CoreWork*> order;
    order.reserve(plan.work.size());
    for (const auto& w : plan.work) {
        order.push_back(&w);
    }
    std::stable_sort(order.begin(), order.end(), [](const auto* a, const auto* b) { return a->index < b->index; });
    std::map<CoreCoord, std::vector<MemberRun>> runs_of;
    for (const auto* w : order) {
        if (w->index == 0) {
            runs_of[w->core];
            continue;
        }
        auto& runs = runs_of[w->root];
        const CoreCoord c = w->core;
        if (!runs.empty() && std::get<1>(runs.back()).y == c.y && std::get<1>(runs.back()).x + 1 == c.x) {
            std::get<1>(runs.back()) = c;
            std::get<2>(runs.back()) += 1;
        } else {
            runs.emplace_back(c, c, 1u);
        }
    }
    return runs_of;
}

}  // namespace

ProgramDescriptor DistributedRmsNormProgramFactory::create_descriptor(
    const DistributedRmsNormParams& attrs,
    const DistributedRmsNormInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        args::cb::MAX_ID < hal::get_num_dataflow_buffers(),
        "distributed_rms_norm: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        args::cb::MAX_ID,
        hal::get_num_dataflow_buffers());
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "distributed_rms_norm: one program per mesh coordinate");
    TT_FATAL(t.scratch.has_value(), "distributed_rms_norm: the exchange landing (scratch_tensor) must be resolved");
    const MeshCoordinate& coord = *mesh_dispatch_coordinate;
    auto* mesh = t.input.device();

    const detail::Plan& plan = detail::solve_plan(detail::plan_inputs(attrs, t, output));
    const detail::Geometry& g = plan.g;
    const uint32_t G = attrs.group_size;
    const uint32_t L = attrs.num_links;
    const uint32_t R = plan.block_rows;
    const uint32_t Rw = plan.wave_rows;
    const uint32_t nwpb = plan.waves_per_block;
    const uint32_t num_waves = plan.num_waves;
    const uint32_t K = plan.landing_waves;
    TT_FATAL(nwpb <= args::sem::MAX_WAVES_PER_BLOCK, "distributed_rms_norm: {} waves per block", nwpb);
    TT_FATAL(plan.ports.size() == L, "distributed_rms_norm: {} port pairs planned for {} links", plan.ports.size(), L);

    const detail::MeshExchange& exchange =
        detail::mesh_exchange(*mesh, attrs.cluster_axis, attrs.ring, attrs.exchange_mode);
    const detail::ChipSchedule& ch = exchange.chips.at(coord);
    const detail::ExchangeSemaphores& sems = detail::exchange_semaphores(*mesh, G);
    TT_FATAL(
        sems.arrival_addrs.size() >= G,
        "distributed_rms_norm: {} arrival semaphores for G={}",
        sems.arrival_addrs.size(),
        G);
    const uint32_t payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    auto virt = [&](const CoreCoord& c) { return mesh->worker_core_from_logical_core(c); };
    auto port = [&](uint32_t d, uint32_t l) { return plan.ports.at(l)[d]; };

    Buffer* input_buffer = t.input.buffer();
    Buffer* output_buffer = output.buffer();
    Buffer* gamma_buffer = t.gamma.has_value() ? t.gamma->buffer() : nullptr;
    Buffer* landing_buffer = t.scratch->buffer();

    ProgramDescriptor desc;

    // ---- semaphores (before the fabric connections take theirs) ----
    for (uint32_t s = 0; s < args::sem::GATHER_BASE + nwpb; ++s) {
        push_semaphore(desc, plan.all_cores, s);
    }
    std::array<uint32_t, args::sem::NUM_OUT_SPLIT> sem_out{};
    for (uint32_t i = 0; i < args::sem::NUM_OUT_SPLIT; ++i) {
        sem_out[i] = args::sem::GATHER_BASE + nwpb + i;
        push_semaphore(desc, plan.compute_cores, sem_out[i]);
    }

    // ---- circular buffers ----
    add_compute_cbs(
        desc,
        plan,
        t.input.dtype(),
        output.dtype(),
        t.gamma.has_value() ? std::optional<DataType>(t.gamma->dtype()) : std::nullopt);
    if (g.sharded) {
        desc.cbs.push_back(ttnn::cb_descriptor_from_sharded_tensor(static_cast<uint8_t>(args::cb::X_TILES), t.input));
    }
    if (g.out_native) {
        desc.cbs.push_back(ttnn::cb_descriptor_from_sharded_tensor(static_cast<uint8_t>(args::cb::OUTPUT), output));
    }

    // ---- compile-time args ----
    const uint32_t w_valid_last =
        (g.W % tt::constants::TILE_WIDTH) != 0 ? g.W % tt::constants::TILE_WIDTH : tt::constants::TILE_WIDTH;
    const uint32_t es_zero = t.input.dtype() == DataType::BFLOAT8_B ? 0u : g.es;
    const auto& out_split = plan.out_split;

    std::vector<uint32_t> reader_ct(args::reader_ct::COUNT);
    reader_ct[args::reader_ct::BLOCK_ROWS] = R;
    reader_ct[args::reader_ct::W_CHUNK_TILES] = g.C;
    reader_ct[args::reader_ct::NUM_W_CHUNKS] = g.num_w_chunks;
    reader_ct[args::reader_ct::WT] = g.Wt;
    reader_ct[args::reader_ct::WS] = g.WS;
    reader_ct[args::reader_ct::X_DEPTH] = plan.x_depth;
    reader_ct[args::reader_ct::SHARDED] = static_cast<uint32_t>(g.sharded);
    reader_ct[args::reader_ct::COMBINE] = static_cast<uint32_t>(g.combine);
    reader_ct[args::reader_ct::G_W] = g.G_w;
    reader_ct[args::reader_ct::WAVE_ROWS] = Rw;
    reader_ct[args::reader_ct::WAVES_PER_BLOCK] = nwpb;
    reader_ct[args::reader_ct::GROUP_DEVICES] = G;
    reader_ct[args::reader_ct::T_IN] = g.T_in;
    reader_ct[args::reader_ct::T_I] = g.T_i;
    reader_ct[args::reader_ct::W_VALID_LAST] = w_valid_last;
    reader_ct[args::reader_ct::ES_ZERO] = es_zero;
    reader_ct[args::reader_ct::MASKED] = static_cast<uint32_t>(g.masked);
    reader_ct[args::reader_ct::NUM_WAVES] = num_waves;
    reader_ct[args::reader_ct::INTERM_FP32] = static_cast<uint32_t>(g.interm_fp32);
    reader_ct[args::reader_ct::READ_WAVE_GATE] = static_cast<uint32_t>(detail::READ_WAVE_GATE);
    reader_ct[args::reader_ct::LANDING_WAVES] = K;
    reader_ct[args::reader_ct::OUT_SPLIT_X] = out_split[0];
    reader_ct[args::reader_ct::NOC0_BANKS_WEST] = out_split[1];
    reader_ct[args::reader_ct::NOC0_BANKS_EAST] = out_split[2];
    reader_ct[args::reader_ct::SEM_OUT_READY] = sem_out[0];
    reader_ct[args::reader_ct::SEM_OUT_DONE] = sem_out[1];
    reader_ct[args::reader_ct::SEM_OUT_ADDR] = sem_out[2];
    reader_ct[args::reader_ct::T_OUT] = g.T_out;
    TensorAccessorArgs(input_buffer).append_to(reader_ct);
    TensorAccessorArgs(output_buffer).append_to(reader_ct);
    reader_ct.push_back(static_cast<uint32_t>(g.fin_compact()));

    std::vector<uint32_t> writer_ct(args::writer_ct::COUNT);
    writer_ct[args::writer_ct::HAS_GAMMA] = static_cast<uint32_t>(g.has_gamma);
    writer_ct[args::writer_ct::GAMMA_IS_TILE] = static_cast<uint32_t>(g.gamma_is_tile);
    writer_ct[args::writer_ct::BLOCK_ROWS] = R;
    writer_ct[args::writer_ct::W_CHUNK_TILES] = g.C;
    writer_ct[args::writer_ct::NUM_W_CHUNKS] = g.num_w_chunks;
    writer_ct[args::writer_ct::WT] = g.Wt;
    writer_ct[args::writer_ct::WS] = g.WS;
    writer_ct[args::writer_ct::OUT_NATIVE] = static_cast<uint32_t>(g.out_native);
    writer_ct[args::writer_ct::T_OUT] = g.T_out;
    writer_ct[args::writer_ct::T_G] = g.T_g;
    writer_ct[args::writer_ct::ES_G] = g.es_g;
    writer_ct[args::writer_ct::WP] = g.Wp;
    writer_ct[args::writer_ct::GAMMA_RESIDENT] = static_cast<uint32_t>(g.gamma_resident);
    writer_ct[args::writer_ct::OUT_SPLIT_X] = out_split[0];
    writer_ct[args::writer_ct::NOC0_BANKS_WEST] = out_split[1];
    writer_ct[args::writer_ct::NOC0_BANKS_EAST] = out_split[2];
    writer_ct[args::writer_ct::SEM_OUT_READY] = sem_out[0];
    writer_ct[args::writer_ct::SEM_OUT_DONE] = sem_out[1];
    writer_ct[args::writer_ct::SEM_OUT_ADDR] = sem_out[2];
    TensorAccessorArgs(output_buffer).append_to(writer_ct);
    TensorAccessorArgs(gamma_buffer).append_to(writer_ct);

    std::vector<uint32_t> compute_ct(args::compute_ct::COUNT);
    compute_ct[args::compute_ct::HAS_GAMMA] = static_cast<uint32_t>(g.has_gamma);
    compute_ct[args::compute_ct::BLOCK_ROWS] = R;
    compute_ct[args::compute_ct::W_CHUNK_TILES] = g.C;
    compute_ct[args::compute_ct::NUM_W_CHUNKS] = g.num_w_chunks;
    compute_ct[args::compute_ct::WS] = g.WS;
    compute_ct[args::compute_ct::OUT_NATIVE] = static_cast<uint32_t>(g.out_native);
    compute_ct[args::compute_ct::COMBINE] = static_cast<uint32_t>(g.combine);
    compute_ct[args::compute_ct::MASKED] = static_cast<uint32_t>(g.masked);
    compute_ct[args::compute_ct::G_W] = g.G_w;
    compute_ct[args::compute_ct::SCALE_BLOCK_TILES] = detail::SCALE_BLOCK_TILES;
    compute_ct[args::compute_ct::WAVE_ROWS] = Rw;
    compute_ct[args::compute_ct::WAVES_PER_BLOCK] = nwpb;
    compute_ct[args::compute_ct::GROUP_DEVICES] = G;
    compute_ct[args::compute_ct::GAMMA_RESIDENT] = static_cast<uint32_t>(g.gamma_resident);
    compute_ct[args::compute_ct::FIN_COMPACT] = static_cast<uint32_t>(g.fin_compact());

    std::vector<uint32_t> port_ct(args::port_ct::COUNT);
    port_ct[args::port_ct::GROUP_DEVICES] = G;
    port_ct[args::port_ct::NUM_WAVES] = num_waves;
    port_ct[args::port_ct::PAYLOAD] = payload;
    port_ct[args::port_ct::NUM_HEADERS] = args::MAX_HEADERS;
    port_ct[args::port_ct::SEM_OWN_STATS] = args::sem::OWN_STATS;
    port_ct[args::port_ct::SEM_OWN_COUNTED] = args::sem::OWN_COUNTED;
    port_ct[args::port_ct::OWN_AHEAD] = static_cast<uint32_t>(detail::PORT_OWN_AHEAD);
    port_ct[args::port_ct::LANDING_WAVES] = K;

    // ---- kernels (kernel_index order) ----
    const auto roots = plan.roots();
    const bool esw_on = std::any_of(plan.esw_offsets.begin(), plan.esw_offsets.end(), [](uint32_t o) { return o > 0; });
    std::map<CoreCoord, uint32_t> root_off;
    std::map<CoreCoord, std::pair<uint32_t, uint32_t>> root_link;
    for (uint32_t gi = 0; gi < roots.size(); ++gi) {
        root_off[roots[gi]->core] = plan.esw_offsets.at(gi);
        root_link[roots[gi]->core] = {gi % L, gi / L};
    }
    const auto member_runs = member_runs_of(plan);

    std::vector<uint32_t> send_dirs;
    for (uint32_t d : {args::dir::FWD, args::dir::BWD}) {
        if (!ch.sends[d].empty()) {
            send_dirs.push_back(d);
        }
    }

    KernelDescriptor reader;
    reader.kernel_source = drms_kernel_path("distributed_rms_norm_reader.cpp");
    reader.core_ranges = plan.compute_cores;
    reader.compile_time_args = std::move(reader_ct);
    if (esw_on) {
        reader.defines = {{"DRMS_ESW_OFFSETS", "1"}};
    }
    reader.config = ReaderConfigDescriptor{};
    reader.emplace_common_runtime_args(
        {input_buffer,
         landing_buffer,
         plan.seg_stride,
         output_buffer,
         static_cast<uint32_t>(ch.entries[args::dir::FWD].size()),
         static_cast<uint32_t>(ch.entries[args::dir::BWD].size()),
         ch.position,
         static_cast<uint32_t>(send_dirs.size())});

    KernelDescriptor writer;
    writer.kernel_source = drms_kernel_path("distributed_rms_norm_writer.cpp");
    writer.core_ranges = plan.compute_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};
    writer.emplace_common_runtime_args({output_buffer, gamma_buffer});

    KernelDescriptor compute;
    compute.kernel_source = drms_kernel_path("distributed_rms_norm_compute.cpp");
    compute.core_ranges = plan.compute_cores;
    compute.compile_time_args = std::move(compute_ct);
    // Resolved by the entry (the Python's _compute_config: the lossless fp32 unpack of cb::STATS_GATHER included).
    compute.config = attrs.compute_config;
    // EPS_BITS is a runtime scalar left out of the program key; override_runtime_arguments re-applies it.
    std::vector<uint32_t> compute_common(args::compute_common::COUNT);
    compute_common[args::compute_common::INV_W_BITS] = fp32_bits(1.0 / static_cast<double>(G * g.W));
    compute_common[args::compute_common::EPS_BITS] = fp32_bits(attrs.epsilon);
    compute.common_runtime_args = std::move(compute_common);

    TT_FATAL(
        reader.common_runtime_args.size() == args::reader_common::COUNT &&
            writer.common_runtime_args.size() == args::writer_common::COUNT,
        "distributed_rms_norm: common runtime argument layout is out of sync with distributed_rms_norm_args.hpp");
    // After reader_common::COUNT: the chip's entry slot ids, forward then backward (the same on every core).
    for (uint32_t d : {args::dir::FWD, args::dir::BWD}) {
        reader.common_runtime_args.insert(reader.common_runtime_args.end(), ch.entries[d].begin(), ch.entries[d].end());
    }

    auto slot_port = [&](uint32_t q, uint32_t l) -> CoreCoord {
        if (q == ch.position) {
            TT_FATAL(!send_dirs.empty(), "distributed_rms_norm: own slot {} has no sending port", q);
            return port(send_dirs.front(), l);
        }
        for (uint32_t d : {args::dir::FWD, args::dir::BWD}) {
            const auto& entries = ch.entries[d];
            if (std::find(entries.begin(), entries.end(), q) != entries.end()) {
                return port(d, l);
            }
        }
        TT_THROW("distributed_rms_norm: slot {} lands nowhere on chip {}", q, coord);
    };
    // Then one row per link (the same on every core of the link): its sending ports, the ports a root reports
    // collected waves to, then the G packed slot-port coords.
    for (uint32_t l = 0; l < L; ++l) {
        std::vector<uint32_t> row(args::reader_link_row::COUNT, 0u);
        row[args::reader_link_row::SEND_PORT_0] = !send_dirs.empty() ? pack_xy(virt(port(send_dirs[0], l))) : 0u;
        row[args::reader_link_row::SEND_PORT_1] = send_dirs.size() > 1 ? pack_xy(virt(port(send_dirs[1], l))) : 0u;
        // bounded landing: the link's ports that send readies (return credits) for this chip
        uint32_t n_free = 0;
        for (uint32_t d : {args::dir::FWD, args::dir::BWD}) {
            if (!ch.ready_out[d].empty()) {
                row[args::reader_link_row::FREE_PORT_0 + n_free++] = pack_xy(virt(port(d, l)));
            }
        }
        row[args::reader_link_row::N_FREE] = n_free;
        reader.common_runtime_args.insert(reader.common_runtime_args.end(), row.begin(), row.end());
        for (uint32_t q = 0; q < G; ++q) {
            reader.common_runtime_args.push_back(pack_xy(virt(slot_port(q, l))));
        }
    }

    reader.runtime_args.reserve(plan.work.size());
    writer.runtime_args.reserve(plan.work.size());
    compute.runtime_args.reserve(plan.work.size());
    for (const auto& w : plan.work) {
        const CoreCoord rv = virt(w.root);
        const bool is_root = w.index == 0;
        const auto [l, r_l] = is_root ? root_link.at(w.core) : std::pair<uint32_t, uint32_t>{0, 0};

        std::vector<uint32_t> rt(args::reader_rt::COUNT);
        rt[args::reader_rt::START_TILE_ROW] = w.start_tile_row;
        rt[args::reader_rt::ROWS] = w.rows;
        rt[args::reader_rt::W0] = w.w0;
        rt[args::reader_rt::V] = w.V;
        rt[args::reader_rt::IS_ROOT] = static_cast<uint32_t>(is_root);
        rt[args::reader_rt::MY_INDEX] = w.index;
        rt[args::reader_rt::ROOT_X] = rv.x;
        rt[args::reader_rt::ROOT_Y] = rv.y;
        rt[args::reader_rt::REC_OFF] = r_l * Rw * args::RECORD_BYTES;
        rt[args::reader_rt::LINK] = l;
        if (esw_on) {
            const uint32_t o = root_off.at(w.root);
            std::vector<uint32_t> esw(args::esw_rt::COUNT);
            esw[args::esw_rt::WAVE_OFF] = o;
            esw[args::esw_rt::READ_START] = static_cast<uint32_t>(
                static_cast<double>(plan.rows_before.at(o) * g.Wt * g.T_in) / detail::ESW_READ_GBPS *
                detail::AICLK_GHZ);
            rt.insert(rt.end(), esw.begin(), esw.end());
        }
        static const std::vector<MemberRun> kNoRuns;
        const auto& runs = is_root ? member_runs.at(w.core) : kNoRuns;
        TT_FATAL(runs.size() + 1 <= std::max<uint32_t>(g.G_w, 1), "distributed_rms_norm: too many member runs");
        rt.push_back(static_cast<uint32_t>(runs.size()));
        for (const auto& [lo, hi, cnt] : runs) {
            rt.push_back(pack_xy(virt(lo)));
            rt.push_back(pack_xy(virt(hi)));
            rt.push_back(cnt);
        }
        rt.resize(rt.size() + (args::MEMBER_RUN_WORDS * (g.G_w - 1 - runs.size())), 0u);
        reader.runtime_args.emplace_back(w.core, std::move(rt));

        const bool masked_core = g.masked && w.w0 + w.V == g.Wt;
        std::vector<uint32_t> crt(args::compute_rt::COUNT);
        crt[args::compute_rt::ROWS] = w.rows;
        crt[args::compute_rt::SLICE_VALID] = w.V;
        crt[args::compute_rt::IS_ROOT] = static_cast<uint32_t>(is_root);
        crt[args::compute_rt::MASKED_CORE] = static_cast<uint32_t>(masked_core);
        compute.runtime_args.emplace_back(w.core, std::move(crt));

        std::vector<uint32_t> wrt(args::writer_rt::COUNT);
        wrt[args::writer_rt::START_TILE_ROW] = w.start_tile_row;
        wrt[args::writer_rt::ROWS] = w.rows;
        wrt[args::writer_rt::W0] = w.w0;
        wrt[args::writer_rt::V] = w.V;
        writer.runtime_args.emplace_back(w.core, std::move(wrt));
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));

    // ---- ports ----
    KernelDescriptor port_kernel;
    port_kernel.kernel_source = drms_kernel_path("distributed_rms_norm_port.cpp");
    port_kernel.core_ranges = plan.port_cores;
    port_kernel.compile_time_args = std::move(port_ct);
    port_kernel.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_1,
    };
    // READY_ADDR is the persistent ready semaphore: stable for the mesh's life, never rewritten on a hit.
    port_kernel.emplace_common_runtime_args({landing_buffer, plan.seg_stride, sems.ready_addr});
    TT_FATAL(
        port_kernel.common_runtime_args.size() == args::port_common::COUNT,
        "distributed_rms_norm: port common runtime argument layout is out of sync with distributed_rms_norm_args.hpp");

    const tt::tt_fabric::FabricNodeId& my_node = exchange.nodes.at(coord);
    const std::array<std::optional<MeshCoordinate>, 2> peer_of{ch.next, ch.prev};
    const uint32_t credit_waves = num_waves - K;
    for (uint32_t l = 0; l < L; ++l) {
        std::vector<uint32_t> link_roots;
        for (uint32_t gi = 0; gi < roots.size(); ++gi) {
            if (gi % L == l) {
                link_roots.push_back(pack_xy(virt(roots[gi]->core)));
            }
        }
        const auto n_link_roots = static_cast<uint32_t>(link_roots.size());
        for (uint32_t d : {args::dir::FWD, args::dir::BWD}) {
            const CoreCoord core = port(d, l);
            const auto& sends = ch.sends[d];
            const auto& entries = ch.entries[d];
            const auto& ready_out = ch.ready_out[d];
            const auto& dests = ch.dests[d];
            const uint32_t n_wait = ch.n_wait[d];
            const bool has_items = !sends.empty();
            const bool has_conn = has_items || !ready_out.empty();
            const CoreCoord pc = virt(core);
            const uint32_t n_roots = has_items ? n_link_roots : 0;
            const uint32_t n_notify = entries.empty() ? 0 : n_link_roots;
            const uint32_t n_counted = (has_items && num_waves > 1) ? n_link_roots : 0;
            const auto n_relay = static_cast<uint32_t>(
                std::count_if(sends.begin(), sends.end(), [](const auto& s) { return s.wait != 0; }));
            TT_FATAL(
                dests.size() <= args::MAX_HEADERS, "distributed_rms_norm: {} packet headers on a port", dests.size());
            const uint32_t sib = d == args::dir::FWD ? args::dir::BWD : args::dir::FWD;
            const auto sib_relays = static_cast<uint32_t>(
                std::count_if(ch.sends[sib].begin(), ch.sends[sib].end(), [](const auto& s) { return s.wait != 0; }));

            std::vector<uint32_t> pa(args::port_rt::COUNT);
            pa[args::port_rt::N_DEST] = static_cast<uint32_t>(dests.size());
            pa[args::port_rt::N_SEND] = static_cast<uint32_t>(sends.size());
            pa[args::port_rt::N_IN] = static_cast<uint32_t>(entries.size());
            pa[args::port_rt::N_ROOTS] = n_roots;
            pa[args::port_rt::N_READY_OUT] = static_cast<uint32_t>(ready_out.size());
            pa[args::port_rt::N_READY_WAIT] = n_wait;
            pa[args::port_rt::MY_X] = pc.x;
            pa[args::port_rt::MY_Y] = pc.y;
            pa[args::port_rt::N_NOTIFY] = n_notify;
            pa[args::port_rt::HAS_CONN] = static_cast<uint32_t>(has_conn);
            pa[args::port_rt::N_COUNTED] = n_counted;
            pa[args::port_rt::NOTIFY_QUANTUM] = 1;  // notify the roots per landed entry (lamp L4)
            pa[args::port_rt::LANDED_SEM_ID] = d == args::dir::FWD ? args::sem::LANDED_FWD : args::sem::LANDED_BWD;
            pa[args::port_rt::N_CREDIT_OUT] = (!ready_out.empty() && credit_waves > 0) ? credit_waves : 0u;
            pa[args::port_rt::CREDIT_THR] =
                (ready_out.empty() ? 0u : n_link_roots) + ((!ready_out.empty() && sib_relays > 0) ? 1u : 0u);
            pa[args::port_rt::RELAY_DONE_TO] = (credit_waves > 0 && n_relay > 0) ? pack_xy(virt(port(sib, l))) : 0u;
            pa[args::port_rt::N_CREDIT_IN] = (has_items && credit_waves > 0) ? credit_waves * n_wait : 0u;
            for (uint32_t q : entries) {
                pa.push_back(sems.arrival_addrs.at(q));
            }
            for (const auto& dst : dests) {
                const tt::tt_fabric::FabricNodeId& dn = exchange.nodes.at(dst.coord);
                pa.push_back(static_cast<uint32_t>(dn.chip_id));
                pa.push_back(static_cast<uint32_t>(*dn.mesh_id));
                pa.push_back(dst.hops);
                pa.push_back(dst.mcast_code);
            }
            for (const auto& s : sends) {
                pa.push_back(s.slot);
                pa.push_back(s.header);
                pa.push_back(sems.arrival_addrs.at(s.slot));
                pa.push_back(s.wait);
            }
            for (const auto& r : ready_out) {
                const CoreCoord rc = virt(port(r.source_dir, l));
                pa.push_back(r.header);
                pa.push_back(rc.x);
                pa.push_back(rc.y);
            }
            pa.insert(pa.end(), plan.seg_wave.at(l).begin(), plan.seg_wave.at(l).end());
            if (std::max(n_roots, n_notify) > 0) {
                pa.insert(pa.end(), link_roots.begin(), link_roots.end());
            }
            if (has_conn) {
                TT_FATAL(peer_of[d].has_value(), "distributed_rms_norm: port {} of chip {} has no peer", d, coord);
                const tt::tt_fabric::FabricNodeId& peer_node = exchange.nodes.at(*peer_of[d]);
                const auto link_ids = tt::tt_fabric::get_forwarding_link_indices(my_node, peer_node);
                TT_FATAL(l < link_ids.size(), "distributed_rms_norm: link {} of {} usable", l, link_ids.size());
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    my_node, peer_node, link_ids[l], desc, core, pa, tt::CoreType::WORKER);
            }
            port_kernel.runtime_args.emplace_back(core, std::move(pa));
        }
    }
    desc.kernels.push_back(std::move(port_kernel));
    return desc;
}

void DistributedRmsNormProgramFactory::override_runtime_arguments(
    Program& program,
    const DistributedRmsNormParams& attrs,
    const DistributedRmsNormInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The key pins every per-core slot, SEG_STRIDE and INV_W_BITS; READY_ADDR is persistent per-mesh state. A hit
    // writes the tensor addresses, the landing address, EPS_BITS and re-points the shard-backed CBs.
    const uint32_t input_addr = t.input.buffer()->address();
    const uint32_t output_addr = output.buffer()->address();
    const uint32_t gamma_addr = t.gamma.has_value() ? t.gamma->buffer()->address() : 0u;
    const uint32_t landing_addr = t.scratch.has_value() ? t.scratch->buffer()->address() : 0u;

    auto& reader = GetCommonRuntimeArgs(program, args::kernel_index::READER);
    reader[args::reader_common::INPUT_ADDR] = input_addr;
    reader[args::reader_common::LANDING_ADDR] = landing_addr;
    reader[args::reader_common::OUTPUT_ADDR] = output_addr;
    auto& writer = GetCommonRuntimeArgs(program, args::kernel_index::WRITER);
    writer[args::writer_common::OUTPUT_ADDR] = output_addr;
    writer[args::writer_common::GAMMA_ADDR] = gamma_addr;
    GetCommonRuntimeArgs(program, args::kernel_index::COMPUTE)[args::compute_common::EPS_BITS] =
        fp32_bits(attrs.epsilon);
    GetCommonRuntimeArgs(program, args::kernel_index::PORT)[args::port_common::LANDING_ADDR] = landing_addr;

    for (const auto& circular_buffer : program.circular_buffers()) {
        if (!circular_buffer->globally_allocated()) {
            continue;
        }
        const auto& indices = circular_buffer->buffer_indices();
        if (indices.contains(args::cb::X_TILES)) {
            UpdateDynamicCircularBufferAddress(program, circular_buffer->id(), *t.input.buffer());
        } else if (indices.contains(args::cb::OUTPUT)) {
            UpdateDynamicCircularBufferAddress(program, circular_buffer->id(), *output.buffer());
        }
    }
}

}  // namespace ttnn::operations::distributed_rms_norm
