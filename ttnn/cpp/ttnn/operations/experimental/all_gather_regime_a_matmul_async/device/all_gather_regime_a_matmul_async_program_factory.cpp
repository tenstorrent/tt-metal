// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Phase A (DRAM-staged) fused all-gather + regime_a_matmul, ONE device's program.
//
// Milestone (this file): D=2 FULL-GATHER-BARRIER correctness (REGIME_A_AGMM_TASK3_BLUEPRINT.md §"Bring-up").
//   1. Allocate a per-device DRAM gather buffer [M, K_global] (output slot 1, a mesh tensor => same address on
//      every device, so the injector can address remote gather slots with this device's TensorAccessor).
//   2. Fabric injector (mux v2, one link toward the forward neighbour): read this device's in0 shard, write it
//      into the LOCAL gather slice, fabric-unicast it into the neighbour's gather buffer at the same global-K
//      offset, then atomic-inc the neighbour's gather_progress GlobalSemaphore once the shard has flushed.
//   3. Full-gather barrier: the injector waits gather_progress >= D-1 (all remote shards landed locally), then
//      fans out a local gather_ready semaphore to every regime_a compute core.
//   4. Replicated regime_a compute engine (DEFAULT placement / bank-order rings — no diagnostics, no fabric
//      hop-distance) reads the fully-populated gather buffer as its in0 and computes [M, N]. Its in0 reader is
//      the COPIED in0_ring_reduce_writer.cpp, gated with AGMM_FULL_GATHER_BARRIER to wait on gather_ready.
//
// Streaming (per-transport readiness, overlap), D>2, and bidirectional ring are Task 3 milestone #18.

#include "all_gather_regime_a_matmul_async_program_factory.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_plan.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

namespace ra = ttnn::operations::experimental::regime_a_matmul::plan;

constexpr const char* kIn1ReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/kernels/in1_reader.cpp";
constexpr const char* kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/device/kernels/compute.cpp";
// COPY of regime_a's in0 ring/reduce writer, gated with AGMM_FULL_GATHER_BARRIER (gather-buffer in0 + barrier).
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async/device/kernels/in0_ring_reduce_writer.cpp";
constexpr const char* kInjectorKernel =
    "ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async/device/kernels/dm_in0_injector.cpp";

constexpr uint32_t kTileBytesBf16 = 2048u;
constexpr uint32_t kTileBytesFp32 = 4096u;

uint32_t largest_div(uint32_t v, uint32_t cap) {
    if (v == 0) {
        return 1u;
    }
    for (uint32_t d = std::min(cap, v); d >= 1; --d) {
        if (v % d == 0) {
            return d;
        }
    }
    return 1u;
}

void mkcb(Program& program, const CoreRangeSet& crs, uint32_t idx, uint32_t ntiles, tt::DataFormat df, uint32_t tsz) {
    CircularBufferConfig c(ntiles * tsz, {{idx, df}});
    c.set_page_size(idx, tsz);
    CreateCircularBuffer(program, crs, c);
}

uint32_t align_up(uint32_t v, uint32_t a) { return ((v + a - 1u) / a) * a; }

}  // namespace

ttnn::device_operation::CachedProgram<AllGatherRegimeAMatmulAsyncProgramFactory::shared_variables_t>
AllGatherRegimeAMatmulAsyncProgramFactory::create_at(
    const AllGatherRegimeAMatmulAsyncParams& op,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    Program program = CreateProgram();

    const auto& in0_shard = tensor_args.input_tensor;   // [M, K_local] on THIS device
    const auto& in1 = tensor_args.weight_tensor;         // [K_global, N] (replicated, 8-bank width-shard)
    Tensor& out = output_tensors.at(0);                  // [M, N]
    Tensor& gather = output_tensors.at(1);               // [M, K_global] DRAM interleaved (mesh: same addr all dev)

    auto* mesh_device = in0_shard.device();  // Tensor::device() returns MeshDevice*
    TT_FATAL(mesh_device != nullptr, "all_gather_regime_a_matmul_async requires a MeshDevice");

    const uint32_t D = op.d;
    // Milestone #17: D=2 forward-neighbour full-gather barrier. D>2 / ring is milestone #18.
    TT_FATAL(D == 2, "all_gather_regime_a_matmul_async Phase A currently implements D=2 (got D={})", D);

    const uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        in0_shard, mesh_coordinate, op.cluster_axis);
    const std::optional<ttnn::MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        in0_shard, mesh_coordinate, 1, op.topology, op.cluster_axis);
    const std::optional<ttnn::MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        in0_shard, mesh_coordinate, -1, op.topology, op.cluster_axis);
    // D=2: the single "other" device is whichever neighbour exists (forward for device 0, backward for the
    // wrap/end device under linear topology). Both directions reach the other device in one hop.
    const std::optional<ttnn::MeshCoordinate>& other_coord = forward_coord.has_value() ? forward_coord : backward_coord;
    TT_FATAL(other_coord.has_value(), "D=2 requires a neighbour along the cluster axis");

    // ============================ regime_a compute engine (replicated default path) =========================
    // Build the regime_a plan for the FULL matmul [M, K_global] x [K_global, N] using the GATHER buffer's shape
    // as in0 (the planner reads only logical shapes). make_and_build_plan does NOT call get_worker_noc_hop_distance
    // (only the factory's ring/placement diagnostics do — which we deliberately skip here), so it is mesh-safe.
    auto planres = make_and_build_plan(mesh_device, gather, in1, op.regime_a_config);
    TT_FATAL(planres.ok(), "regime_a planner rejected config: {}", planres.error);
    const ra::ExecutionPlan& P = *planres.plan;
    const ra::Geometry& geo = P.geo;
    const ra::CbSizes& cb = P.cb;

    // Resolve the SAME config make_and_build_plan used (auto-select when config=None), so Pk/Ns/Sm/kb match.
    const uint32_t Mt_r = (static_cast<uint32_t>(gather.logical_shape()[-2]) + 31u) / 32u;
    const uint32_t Kt_r = (static_cast<uint32_t>(gather.logical_shape()[-1]) + 31u) / 32u;
    const uint32_t Nt_r = (static_cast<uint32_t>(in1.logical_shape()[-1]) + 31u) / 32u;
    const RegimeAMatmulConfig cfg = op.regime_a_config.value_or(auto_select_config(Mt_r, Kt_r, Nt_r));
    const uint32_t Pk = cfg.k_slices ? cfg.k_slices : 1u;
    const uint32_t Sm = cfg.m_slices ? cfg.m_slices : 1u;
    const uint32_t kb = cfg.k_block_tiles ? cfg.k_block_tiles : 1u;
    const uint32_t use_reduce = (Pk > 1u) ? 1u : 0u;

    // Core range sets: all compute cores + split-NoC groups (g0 = noc 0, g1 = noc 1).
    std::set<CoreRange> all_set, g0_set, g1_set;
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> core_noc;
    cores.reserve(geo.num_cores);
    core_noc.reserve(geo.num_cores);
    for (const auto& cp : P.cores) {
        CoreCoord c{cp.coord.x, cp.coord.y};
        cores.push_back(c);
        core_noc.push_back(cp.noc);
        all_set.insert(CoreRange(c, c));
        (cp.noc ? g1_set : g0_set).insert(CoreRange(c, c));
    }
    CoreRangeSet all_cores(all_set);
    CoreRangeSet g0(g0_set);
    CoreRangeSet g1(g1_set);

    // Circular buffers (regime_a spec §5), on all compute cores.
    mkcb(program, all_cores, 0, cb.cb0_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
    mkcb(program, all_cores, 1, cb.cb1_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
    mkcb(program, all_cores, 2, cb.cb2_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
    mkcb(program, all_cores, 3, cb.cb3_tiles, tt::DataFormat::Float32, kTileBytesFp32);
    if (cb.cb7_tiles > 0u) {
        mkcb(program, all_cores, 7, cb.cb7_tiles, tt::DataFormat::Float16_b, kTileBytesBf16);
    }

    // Semaphores. regime_a's ring/reduce sems + the full-gather-barrier fan-out sem (injector -> readers).
    const uint32_t fwd_sem = CreateSemaphore(program, all_cores, 0u);
    const uint32_t red_sem = CreateSemaphore(program, all_cores, 0u);
    const uint32_t redfree_sem = CreateSemaphore(program, all_cores, 0u);
    uint32_t in1valid_sem = 0u, in1ready_sem = 0u;
    if (Sm > 1u) {
        in1valid_sem = CreateSemaphore(program, all_cores, 0u);
        in1ready_sem = CreateSemaphore(program, all_cores, 0u);
    }
    // gather_ready lives on the compute cores AND the injector core (union), so the injector can address it.

    // ---- in1 reader compile args (in1_reader.cpp order) ----
    std::vector<uint32_t> rct = {
        kb, geo.N_sub, geo.W, geo.G, kTileBytesBf16, geo.N_bpc, geo.in1_shard_stride_n, in1valid_sem, in1ready_sem};

    auto mk = [&](const char* src,
                  const CoreRangeSet& g,
                  DataMovementProcessor proc,
                  NOC noc,
                  const std::vector<uint32_t>& ct,
                  const std::map<std::string, std::string>& defs) -> KernelHandle {
        if (g.num_cores() == 0) {
            return 0;
        }
        return CreateKernel(
            program, src, g, DataMovementConfig{.processor = proc, .noc = noc, .compile_args = ct, .defines = defs});
    };

    // ---- writer (COPIED gated in0 ring/reduce writer) compile args ----
    // Base layout identical to regime_a; AGMM_FULL_GATHER_BARRIER appends the gather_ready sem id at the end.
    std::map<std::string, std::string> wdefs;
    wdefs["AGMM_FULL_GATHER_BARRIER"] = "1";
    std::vector<uint32_t> wct = {
        geo.M_block_capacity,   // 0
        kb,                     // 1
        geo.N_sub,              // 2
        geo.K_num_blocks_eff,   // 3
        kTileBytesBf16,         // 4
        geo.in0_stride_k,       // 5  (== Kt_global; regime_a reads in0 with the full-K row stride)
        geo.out_stride_n,       // 6
        geo.W,                  // 7
        geo.G,                  // 8
        fwd_sem,                // 9
        red_sem,                // 10
        geo.N_bpc,              // 11
        redfree_sem,            // 12
        use_reduce};            // 13
    TensorAccessorArgs(*gather.buffer()).append_to(wct);  // in0 accessor -> GATHER buffer
    TensorAccessorArgs(*out.buffer()).append_to(wct);     // out accessor

    KernelHandle readerA = mk(kIn1ReaderKernel, g0, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, rct, {});
    KernelHandle readerB = mk(kIn1ReaderKernel, g1, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, rct, {});
    KernelHandle writerA = mk(kWriterKernel, g0, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default, wct, wdefs);
    KernelHandle writerB = mk(kWriterKernel, g1, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default, wct, wdefs);

    // ---- compute compile args ----
    const uint32_t sbh = largest_div(geo.M_block_capacity, 2u);
    const uint32_t sbw = largest_div(geo.N_sub, 4u / sbh);
    std::vector<uint32_t> cct = {
        geo.K_num_blocks_eff, geo.M_block_capacity, kb, geo.N_sub, 1u, geo.N_bpc, sbh, sbw};
    std::map<std::string, std::string> cdefs = {{"REDUCE_K", "1"}, {"IN0_KSLICE_RESIDENT", "1"}};
    KernelHandle compute = CreateKernel(
        program,
        kComputeKernel,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = false,
            .math_approx_mode = false,
            .compile_args = cct,
            .defines = cdefs});

    // ============================ fabric injector (mux v2, forward neighbour) ==============================
    // Reserve one injector worker core + one mux core from the grid, avoiding the regime_a compute cores.
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    std::set<std::pair<uint32_t, uint32_t>> used;
    for (const auto& c : cores) {
        used.insert({c.x, c.y});
    }
    auto take_free_core = [&]() -> CoreCoord {
        for (int y = (int)grid.y - 1; y >= 0; --y) {
            for (int x = (int)grid.x - 1; x >= 0; --x) {
                if (!used.count({(uint32_t)x, (uint32_t)y})) {
                    used.insert({(uint32_t)x, (uint32_t)y});
                    return CoreCoord{(uint32_t)x, (uint32_t)y};
                }
            }
        }
        TT_THROW("all_gather_regime_a_matmul_async: no free core for fabric mux/injector");
    };
    const CoreCoord mux_logical = take_free_core();
    const CoreCoord inj_logical = take_free_core();
    const CoreRangeSet inj_crs(CoreRange(inj_logical, inj_logical));

    // gather_ready fan-out semaphore, created on compute cores UNION injector core so both share its L1 address.
    std::set<CoreRange> ready_set = all_set;
    ready_set.insert(CoreRange(inj_logical, inj_logical));
    const uint32_t gather_ready_sem = CreateSemaphore(program, CoreRangeSet(ready_set), 0u);

    // Injector L1 scratch: payload tile CB (c_0) + packet-header CB (c_1).
    const uint32_t aligned_hdr = align_up((uint32_t)tt::tt_fabric::get_tt_fabric_packet_header_size_bytes(),
                                          (uint32_t)hal::get_l1_alignment());
    mkcb(program, inj_crs, 0, 1, tt::DataFormat::Float16_b, kTileBytesBf16);  // payload (one tile)
    {
        CircularBufferConfig hc(aligned_hdr, {{1, tt::DataFormat::Float16_b}});
        hc.set_page_size(1, aligned_hdr);
        CreateCircularBuffer(program, inj_crs, hc);
    }

    // mux v2 toward the forward neighbour.
    const auto src_node = mesh_device->get_fabric_node_id(mesh_coordinate);
    const auto dst_node = mesh_device->get_fabric_node_id(other_coord.value());
    const auto link_indices = tt::tt_fabric::get_forwarding_link_indices(src_node, dst_node);
    TT_FATAL(!link_indices.empty(), "no fabric forwarding link from src to forward neighbour");
    const uint32_t channel_buf_bytes = align_up(aligned_hdr + kTileBytesBf16, (uint32_t)hal::get_l1_alignment());
    tt::tt_fabric::FabricMuxV2Config mux_config(
        /*num_channels=*/1,
        /*num_buffers_per_channel=*/(uint8_t)op.num_buffers_per_channel,
        /*channel_buffer_size_bytes=*/channel_buf_bytes,
        /*base_l1_address=*/mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1));
    tt::tt_fabric::add_fabric_mux_v2_to_program(
        program, mux_config, mux_logical, src_node, dst_node, link_indices.front(), NOC::RISCV_0_default);
    const CoreCoord mux_virtual = mesh_device->worker_core_from_logical_core(mux_logical);

    // one-hop unicast to the (single) forward neighbour for D=2.
    const uint32_t num_hops = 1u;

    // Injector compile args: TensorAccessorArgs(in0 shard) then TensorAccessorArgs(gather buffer).
    std::vector<uint32_t> ict;
    TensorAccessorArgs(*in0_shard.buffer()).append_to(ict);
    TensorAccessorArgs(*gather.buffer()).append_to(ict);
    KernelHandle injector = CreateKernel(
        program,
        kInjectorKernel,
        inj_crs,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ict, .defines = {}});

    // ============================ runtime args ============================
    const uint32_t in1_addr = in1.buffer()->address();
    const uint32_t out_addr = out.buffer()->address();
    const uint32_t gather_addr = gather.buffer()->address();

    auto phys = [&](uint32_t core_idx) {
        const auto& c = P.cores[core_idx].coord;
        return mesh_device->worker_core_from_logical_core(CoreCoord{c.x, c.y});
    };

    for (uint32_t i = 0; i < geo.num_cores; ++i) {
        const ra::CorePlan& cp = P.cores[i];
        const KernelHandle rh = cp.noc ? readerB : readerA;
        const KernelHandle wh = cp.noc ? writerB : writerA;

        std::vector<uint32_t> ra_rt = {in1_addr, cp.bank, cp.ring_pos, cp.k_start, cp.n_local, cp.valid_k, cp.valid_n};
        if (Sm == 1u) {
            ra_rt.push_back(2u);
            ra_rt.push_back(0u);
        } else if (cp.mm == 0u) {
            ra_rt.push_back(1u);
            ra_rt.push_back(Sm - 1u);
            for (uint32_t s = 1; s < Sm; ++s) {
                auto p = phys(i + s);
                ra_rt.push_back(p.x);
                ra_rt.push_back(p.y);
            }
        } else {
            ra_rt.push_back(0u);
            ra_rt.push_back(1u);
            auto p = phys(i - cp.mm);
            ra_rt.push_back(p.x);
            ra_rt.push_back(p.y);
        }
        SetRuntimeArgs(program, rh, cores[i], ra_rt);

        auto fwd_next = phys(cp.ring_next_idx);
        auto red_next = phys(cp.red_next_idx);
        auto red_prev = phys(cp.red_prev_idx);
        std::vector<uint32_t> wa = {
            gather_addr,             // 0  in0_addr == gather buffer base
            out_addr,                // 1
            cp.m_start,              // 2
            cp.n_start,              // 3
            cp.k_start,              // 4
            cp.ring_pos,             // 5
            fwd_next.x,              // 6
            fwd_next.y,              // 7
            red_next.x,              // 8
            red_next.y,              // 9
            cp.is_bottom ? 1u : 0u,  // 10
            cp.is_top ? 1u : 0u,     // 11
            red_prev.x,              // 12
            red_prev.y,              // 13
            cp.valid_k,              // 14
            cp.valid_m,              // 15
            cp.valid_n,              // 16
            gather_ready_sem,        // 17 (AGMM_FULL_GATHER_BARRIER) fan-out sem id
            D - 1u};                 // 18 (AGMM_FULL_GATHER_BARRIER) expected remote-shard count
        SetRuntimeArgs(program, wh, cores[i], wa);

        std::vector<uint32_t> ca = {0u, geo.M_block_capacity, 0u, geo.N_bpc * geo.N_sub, cp.is_bottom ? 1u : 0u};
        SetRuntimeArgs(program, compute, cores[i], ca);
    }

    // ---- injector runtime args ----
    // in0 shard has K_local tiles; this device owns global K-tile offset device_index * Kt_local.
    const uint32_t Kt_global = geo.Kt;
    const uint32_t Kt_local = Kt_global / D;
    const uint32_t k_tile_global_base = device_index * Kt_local;
    // gather_progress GlobalSemaphore (remote shards landed here). address() is uniform across the mesh.
    const uint32_t progress_addr = op.d > 1 ? tensor_args.multi_device_global_semaphore.at(0).address() : 0u;
    // neighbour's injector core == our injector core (symmetric programs, uniform harvesting on the galaxy).
    const CoreCoord inj_virtual = mesh_device->worker_core_from_logical_core(inj_logical);

    std::vector<uint32_t> inj_rt = {
        in0_shard.buffer()->address(),  // a0 in0 shard base
        gather_addr,                    // a1 local gather base
        geo.Mt,                         // a2 Mt
        Kt_local,                       // a3 Kt_local
        Kt_global,                      // a4 Kt_global (gather row stride)
        k_tile_global_base,             // a5 global K-tile base of this shard
        kTileBytesBf16,                 // a6 tile bytes
        num_hops,                       // a7 hops to forward neighbour
        progress_addr,                  // a8 gather_progress GlobalSemaphore L1 address (on neighbour)
        (uint32_t)inj_virtual.x,        // a9 neighbour injector core x (== ours; where its gather_progress lives)
        (uint32_t)inj_virtual.y,        // a10 neighbour injector core y
        gather_ready_sem,               // a11 local fan-out sem id
        D - 1u,                         // a12 expected remote-shard count before fan-out
        geo.num_cores};                 // a13 number of compute cores to fan out to
    // followed by each compute core's (x,y) for the local gather_ready fan-out.
    for (uint32_t i = 0; i < geo.num_cores; ++i) {
        auto p = phys(i);
        inj_rt.push_back(p.x);
        inj_rt.push_back(p.y);
    }
    // then the mux v2 client connection args (packet-header/payload come from CBs c_1/c_0 in-kernel).
    const uint32_t flow_control_sem = CreateSemaphore(program, inj_logical, 0u);
    const uint32_t teardown_sem = CreateSemaphore(program, inj_logical, 0u);
    mux_config.append_client_connection_rt_args(
        mux_virtual,
        /*logical_channel_id=*/0,
        tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{flow_control_sem, teardown_sem},
        inj_rt);
    SetRuntimeArgs(program, injector, inj_logical, inj_rt);

    return ttnn::device_operation::CachedProgram<shared_variables_t>{
        std::move(program),
        shared_variables_t{
            .num_cores = geo.num_cores,
            .cores = std::move(cores),
            .core_noc = std::move(core_noc),
            .readerA = readerA,
            .readerB = readerB,
            .writerA = writerA,
            .writerB = writerB,
            .compute = compute,
            .injector_cores = {inj_logical},
            .injector = injector}};
}

void AllGatherRegimeAMatmulAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherRegimeAMatmulAsyncParams& /*operation_attributes*/,
    const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    const uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t out_addr = output_tensors.at(0).buffer()->address();
    const uint32_t gather_addr = output_tensors.at(1).buffer()->address();
    const uint32_t in0_addr = tensor_args.input_tensor.buffer()->address();

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& sv = cached_workload.shared_variables.at(range);
        bool has_g0 = false, has_g1 = false;
        for (const auto n : sv.core_noc) {
            (n ? has_g1 : has_g0) = true;
        }
        auto* readerA_args = has_g0 ? &GetRuntimeArgs(program, sv.readerA) : nullptr;
        auto* readerB_args = has_g1 ? &GetRuntimeArgs(program, sv.readerB) : nullptr;
        auto* writerA_args = has_g0 ? &GetRuntimeArgs(program, sv.writerA) : nullptr;
        auto* writerB_args = has_g1 ? &GetRuntimeArgs(program, sv.writerB) : nullptr;
        for (uint32_t i = 0; i < sv.num_cores; ++i) {
            const CoreCoord& core = sv.cores[i];
            const bool b = sv.core_noc[i] != 0u;
            (*(b ? readerB_args : readerA_args))[core.x][core.y][0] = in1_addr;
            auto& wa = (*(b ? writerB_args : writerA_args))[core.x][core.y];
            wa[0] = gather_addr;
            wa[1] = out_addr;
        }
        // injector: a0 in0 shard, a1 local gather base.
        auto& inj_args = GetRuntimeArgs(program, sv.injector);
        const CoreCoord& ic = sv.injector_cores.at(0);
        inj_args[ic.x][ic.y][0] = in0_addr;
        inj_args[ic.x][ic.y][1] = gather_addr;
    }
}

}  // namespace ttnn::experimental::prim
