// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fabric_reduce_scatter_program_factory.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/indestructible.hpp>

#include "kernels/fabric_reduce_scatter_args.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::fabric_reduce_scatter {

using namespace tt::tt_metal;
using namespace ::fabric_reduce_scatter;

namespace {

constexpr const char* kFrsKernelDir = "ttnn/cpp/ttnn/operations/fabric_reduce_scatter/device/kernels/";

std::string frs_kernel_path(const char* name) { return std::string(kFrsKernelDir) + name; }

CoreRangeSet merged_set(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& c : cores) {
        ranges.emplace_back(c, c);
    }
    return CoreRangeSet(ranges).merge_ranges();
}

std::vector<CoreCoord> plan_cores(const plan::Plan& p) {
    std::vector<CoreCoord> cores;
    for (const auto& ch : p.chips) {
        cores.insert(cores.end(), ch.ports.begin(), ch.ports.end());
        cores.insert(cores.end(), ch.finals.begin(), ch.finals.end());
    }
    std::sort(cores.begin(), cores.end());
    cores.erase(std::unique(cores.begin(), cores.end()), cores.end());
    return cores;
}

using InputsKey = std::tuple<std::optional<uint32_t>, uint32_t, uint32_t, uint32_t, uint32_t>;
InputsKey inputs_key(const plan::PlanInputs& in) {
    return {
        in.cluster_axis,
        static_cast<uint32_t>(in.topology),
        in.num_links,
        static_cast<uint32_t>(in.scheme),
        static_cast<uint32_t>(in.placement)};
}

DataMovementConfigDescriptor dm_config(DataMovementProcessor processor, NOC noc) {
    return DataMovementConfigDescriptor{.processor = processor, .noc = noc};
}

}  // namespace

plan::PlanInputs plan_inputs(const FabricReduceScatterParams& attrs) {
    return plan::PlanInputs{
        .cluster_axis = attrs.cluster_axis,
        .topology = attrs.topology,
        .num_links = attrs.num_links,
        .scheme = attrs.scheme,
        .placement = attrs.placement,
    };
}

const MeshState& get_mesh_state(distributed::MeshDevice& mesh_device, const plan::PlanInputs& inputs) {
    // Keyed by the mesh id and shape (reshape keeps the id) plus the inputs; the semaphores by the core set.
    using FastKey = std::tuple<uint32_t, uint32_t, uint32_t, InputsKey>;
    using SemKey = std::tuple<uint32_t, uint32_t, uint32_t, std::vector<CoreCoord>>;
    static ttsl::Indestructible<std::map<FastKey, const MeshState*>> fast;
    static ttsl::Indestructible<std::map<SemKey, std::unique_ptr<MeshState>>> states;
    const auto shape = mesh_device.shape();
    const uint32_t rows = shape[0];
    const uint32_t cols = shape.dims() > 1 ? shape[1] : 1;
    const FastKey fkey{static_cast<uint32_t>(mesh_device.id()), rows, cols, inputs_key(inputs)};
    if (auto it = fast.get().find(fkey); it != fast.get().end()) {
        return *it->second;
    }
    const plan::Plan& p = plan::get_plan(mesh_device, inputs);
    auto cores = plan_cores(p);
    SemKey skey{static_cast<uint32_t>(mesh_device.id()), rows, cols, cores};
    auto sit = states.get().find(skey);
    if (sit == states.get().end()) {
        const CoreRangeSet crs = merged_set(cores);
        auto sem_a = ttnn::global_semaphore::create_global_semaphore(&mesh_device, crs, 0);
        auto sem_b = ttnn::global_semaphore::create_global_semaphore(&mesh_device, crs, 0);
        auto ready = ttnn::global_semaphore::create_global_semaphore(&mesh_device, crs, 0);
        // every counter zero before any chip's first call increments a neighbour's
        tt::tt_metal::distributed::Synchronize(&mesh_device, std::nullopt);
        const auto a_addr = static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(sem_a));
        const auto b_addr = static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(sem_b));
        const auto r_addr = static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(ready));
        auto state = std::make_unique<MeshState>(MeshState{
            .plan = &p,
            .sem_a = std::move(sem_a),
            .sem_b = std::move(sem_b),
            .ready = std::move(ready),
            .sem_a_addr = a_addr,
            .sem_b_addr = b_addr,
            .ready_addr = r_addr,
        });
        sit = states.get().emplace(std::move(skey), std::move(state)).first;
    }
    // The semaphores may be shared by inputs with the same core set; the plan is this inputs' plan.
    if (sit->second->plan != &p) {
        auto copy = std::make_unique<MeshState>(MeshState{
            .plan = &p,
            .sem_a = sit->second->sem_a,
            .sem_b = sit->second->sem_b,
            .ready = sit->second->ready,
            .sem_a_addr = sit->second->sem_a_addr,
            .sem_b_addr = sit->second->sem_b_addr,
            .ready_addr = sit->second->ready_addr,
        });
        static ttsl::Indestructible<std::vector<std::unique_ptr<MeshState>>> aliases;
        aliases.get().push_back(std::move(copy));
        fast.get().emplace(fkey, aliases.get().back().get());
        return *aliases.get().back();
    }
    fast.get().emplace(fkey, sit->second.get());
    return *sit->second;
}

ProgramDescriptor FabricReduceScatterProgramFactory::create_descriptor(
    const FabricReduceScatterParams& attrs,
    const FabricReduceScatterInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "fabric_reduce_scatter: one program per mesh coordinate");
    TT_FATAL(tensor_args.scratch.has_value(), "fabric_reduce_scatter: scratch_tensor missing");
    TT_FATAL(
        cb::OUT < hal::get_num_dataflow_buffers(),
        "fabric_reduce_scatter: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        cb::OUT,
        hal::get_num_dataflow_buffers());
    const Tensor& input = tensor_args.input;
    const Tensor& scratch = *tensor_args.scratch;
    auto* mesh_device = input.device();
    const MeshState& state = get_mesh_state(*mesh_device, plan_inputs(attrs));
    const plan::Plan& p = *state.plan;

    const auto& coord = *mesh_dispatch_coordinate;
    const uint32_t cols = attrs.mesh_cols;
    const uint32_t ci = (coord[0] * cols) + (coord.dims() > 1 ? coord[1] : 0);
    const auto coord_of = [cols](uint32_t idx) { return distributed::MeshCoordinate(idx / cols, idx % cols); };
    const plan::ChipPlan& ch = p.chips.at(ci);

    const uint32_t G = p.group_size;
    const uint32_t num_links = p.num_links;
    const auto page_bytes = static_cast<uint32_t>(input.buffer()->aligned_page_size());
    const auto dram = mesh_device->dram_grid_size();
    const auto num_banks = static_cast<uint32_t>(dram.x * dram.y);
    const auto& shape = input.padded_shape();
    const auto [tile_h, tile_w] = input.tensor_spec().tile().get_tile_shape();
    uint32_t blk_pages = 0;
    uint32_t col_wt = 0;
    uint32_t col_wb = 0;
    if (attrs.dim == 2) {
        blk_pages = (shape[-2] / G / tile_h) * (shape[-1] / tile_w);
    } else {
        col_wt = shape[-1] / tile_w;
        col_wb = col_wt / G;
        blk_pages = (shape[-2] / tile_h) * col_wb;
    }
    const uint32_t run_pages = std::max<uint32_t>(
        1, static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / page_bytes));
    const uint32_t chunk_bytes = run_pages * page_bytes;
    const uint32_t group = std::max<uint32_t>(1, std::min<uint32_t>(kMaxGroup, kCbBytes / (2 * chunk_bytes)));
    const uint32_t lanes = p.n_rings * num_links;
    TT_FATAL(
        2 * lanes <= num_banks, "{} rings x {} links need at least {} DRAM banks", p.n_rings, num_links, 2 * lanes);

    ProgramDescriptor desc;
    std::vector<CoreCoord> port_and_final(ch.ports.begin(), ch.ports.end());
    port_and_final.insert(port_and_final.end(), ch.finals.begin(), ch.finals.end());
    const CoreRangeSet pf_set = merged_set(port_and_final);
    const CoreRangeSet port_set = merged_set(ch.ports);
    const CoreRangeSet final_set = merged_set(ch.finals);
    const auto data_format = datatype_to_dataformat_converter(input.dtype());
    for (uint32_t idx : {cb::OWN, cb::IN, cb::IN2, cb::OUT}) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = 2 * group * chunk_bytes,
            .core_ranges = idx == cb::IN2 ? final_set : pf_set,
            .format_descriptors = {CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = data_format, .page_size = page_bytes}},
        });
    }

    // Compile-time args.
    std::vector<uint32_t> ct_args(ct::COUNT);
    ct_args[ct::PAGE_BYTES] = page_bytes;
    ct_args[ct::RUN_PAGES] = run_pages;
    ct_args[ct::NUM_BANKS] = num_banks;
    ct_args[ct::GROUP] = group;
    ct_args[ct::INC_EVERY] = kIncEvery;
    std::vector<uint32_t> reader_ct = ct_args;
    reader_ct.resize(read_ct::COUNT);
    reader_ct[read_ct::COL_WT] = col_wt;
    reader_ct[read_ct::COL_WB] = col_wb;
    TensorAccessorArgs(input.buffer()).append_to(reader_ct);
    TensorAccessorArgs(scratch.buffer()).append_to(reader_ct);
    std::vector<uint32_t> sender_ct = ct_args;
    TensorAccessorArgs(scratch.buffer()).append_to(sender_ct);
    std::vector<uint32_t> writer_ct = ct_args;
    TensorAccessorArgs(output.buffer()).append_to(writer_ct);
    std::vector<uint32_t> compute_ct_args(compute_ct::COUNT);
    compute_ct_args[compute_ct::RUN_PAGES] = run_pages;

    ComputeConfigDescriptor compute_config{};
    const char* compute_source = "compute_add.cpp";
    if (attrs.exact_fp32_add) {
        compute_source = "compute_exact_add.cpp";
        compute_config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
        compute_config.unpack_to_dest_mode.assign(hal::get_num_dataflow_buffers(), UnpackToDestMode::Default);
        for (uint32_t idx : {cb::OWN, cb::IN, cb::IN2}) {
            compute_config.unpack_to_dest_mode[idx] = UnpackToDestMode::UnpackToDestFp32;
        }
    }

    KernelDescriptor port_reader;
    port_reader.kernel_source = frs_kernel_path("port_reader.cpp");
    port_reader.core_ranges = port_set;
    port_reader.compile_time_args = reader_ct;
    port_reader.emplace_common_runtime_args({input.buffer(), scratch.buffer(), blk_pages, state.sem_a_addr});
    port_reader.config = dm_config(DataMovementProcessor::RISCV_1, NOC::RISCV_0_default);

    KernelDescriptor port_compute;
    port_compute.kernel_source = frs_kernel_path(compute_source);
    port_compute.core_ranges = port_set;
    port_compute.compile_time_args = compute_ct_args;
    port_compute.config = compute_config;

    KernelDescriptor port_sender;
    port_sender.kernel_source = frs_kernel_path("port_sender.cpp");
    port_sender.core_ranges = port_set;
    port_sender.compile_time_args = sender_ct;
    port_sender.emplace_common_runtime_args({scratch.buffer(), blk_pages, state.sem_a_addr, state.ready_addr});
    port_sender.config = dm_config(DataMovementProcessor::RISCV_0, NOC::RISCV_1_default);

    KernelDescriptor final_reader;
    final_reader.kernel_source = frs_kernel_path("final_reader.cpp");
    final_reader.core_ranges = final_set;
    final_reader.compile_time_args = reader_ct;
    final_reader.emplace_common_runtime_args(
        {input.buffer(), scratch.buffer(), blk_pages, G, state.sem_a_addr, state.sem_b_addr});
    final_reader.config = dm_config(DataMovementProcessor::RISCV_1, NOC::RISCV_0_default);

    KernelDescriptor final_compute;
    final_compute.kernel_source = frs_kernel_path(compute_source);
    final_compute.core_ranges = final_set;
    final_compute.compile_time_args = compute_ct_args;
    final_compute.config = compute_config;

    KernelDescriptor final_writer;
    final_writer.kernel_source = frs_kernel_path("final_writer.cpp");
    final_writer.core_ranges = final_set;
    final_writer.compile_time_args = writer_ct;
    final_writer.emplace_common_runtime_args({output.buffer(), blk_pages, state.sem_a_addr, state.sem_b_addr});
    final_writer.config = dm_config(DataMovementProcessor::RISCV_0, NOC::RISCV_1_default);

    TT_FATAL(
        port_reader.common_runtime_args.size() == port_reader_common::COUNT &&
            port_sender.common_runtime_args.size() == port_sender_common::COUNT &&
            final_reader.common_runtime_args.size() == final_reader_common::COUNT &&
            final_writer.common_runtime_args.size() == final_writer_common::COUNT,
        "fabric_reduce_scatter: common runtime argument layout is out of sync with fabric_reduce_scatter_args.hpp");

    const auto virt = [mesh_device](const CoreCoord& c) { return mesh_device->worker_core_from_logical_core(c); };
    const auto me = mesh_device->get_fabric_node_id(coord);

    // Ports, in the plan's (ring, direction, link) order.
    for (uint32_t j = 0; j < p.n_rings; ++j) {
        const plan::RingPosition& rg = ch.rings.at(j);
        for (uint32_t d = 0; d < 2; ++d) {
            const auto dir = static_cast<plan::Direction>(d);
            const std::optional<uint32_t> peer = dir == plan::FWD ? rg.next : rg.prev;
            const uint32_t opp = 1 - d;
            for (uint32_t l = 0; l < num_links; ++l) {
                const CoreCoord& core = ch.ports.at(plan::port_index(j, d, l, num_links));
                const uint32_t lane = (j * num_links) + l;
                const plan::PortSchedule sched = plan::rs_schedule(p, ci, j, dir);
                const auto nblocks = static_cast<uint32_t>(sched.blocks.size());
                const uint32_t lead = sched.lead;
                const uint32_t relays = nblocks - lead;
                const uint32_t full = plan::chunks_per_shard(blk_pages, lane, lanes, num_banks, run_pages);

                std::vector<uint32_t> pr(port_reader_rt::COUNT, 0u);
                pr[port_reader_rt::FIRST_BANK] = lane;
                pr[port_reader_rt::BANK_STRIDE] = lanes;
                pr[port_reader_rt::FULL] = full;
                pr[port_reader_rt::LEAD] = lead;
                pr[port_reader_rt::NUM_BLOCKS] = nblocks;
                for (const auto& [block, slot] : sched.blocks) {
                    pr.push_back(block);
                }
                port_reader.runtime_args.emplace_back(core, std::move(pr));

                const bool direct = lead == nblocks;  // every block starts here: the reader feeds the sender
                std::vector<uint32_t> pc(compute_rt::COUNT, 0u);
                pc[compute_rt::N] = direct ? 0 : relays * full;
                pc[compute_rt::HAS_B] = 1;
                pc[compute_rt::HAS_C] = 0;
                pc[compute_rt::N_COPY] = direct ? 0 : lead * full;
                port_compute.runtime_args.emplace_back(core, std::move(pc));

                std::vector<uint32_t> ps(port_sender_rt::COUNT, 0u);
                ps[port_sender_rt::FIRST_BANK] = lane;
                ps[port_sender_rt::BANK_STRIDE] = lanes;
                ps[port_sender_rt::FINAL_SEM_ADDR] = dir == plan::FWD ? state.sem_a_addr : state.sem_b_addr;
                ps[port_sender_rt::EXPECT_IN] = plan::increments(relays * full, kIncEvery);
                ps[port_sender_rt::FULL] = full;
                const bool send_ready = plan::needs_ready(p, ci, j, dir);
                if (send_ready) {
                    const CoreCoord rc = virt(p.chips.at(*peer).ports.at(plan::port_index(j, opp, l, num_links)));
                    ps[port_sender_rt::SEND_READY] = 1;
                    ps[port_sender_rt::READY_X] = rc.x;
                    ps[port_sender_rt::READY_Y] = rc.y;
                }
                const bool has_links = peer.has_value() && !rg.links[d].empty();
                if (has_links && nblocks > 0) {
                    const plan::ChipPlan& pch = p.chips.at(*peer);
                    const CoreCoord pcore = virt(pch.ports.at(plan::port_index(j, d, l, num_links)));
                    const CoreCoord fc0 = virt(pch.finals.at(2 * lane));
                    const CoreCoord fc1 = virt(pch.finals.at((2 * lane) + 1));
                    ps[port_sender_rt::PEER_X] = pcore.x;
                    ps[port_sender_rt::PEER_Y] = pcore.y;
                    ps[port_sender_rt::FINAL_X] = fc0.x;
                    ps[port_sender_rt::FINAL_Y] = fc0.y;
                    ps[port_sender_rt::FINAL_X1] = fc1.x;
                    ps[port_sender_rt::FINAL_Y1] = fc1.y;
                    ps[port_sender_rt::FULL_F0] =
                        plan::chunks_per_shard(blk_pages, lane, 2 * lanes, num_banks, run_pages);
                    ps[port_sender_rt::FULL_F1] =
                        plan::chunks_per_shard(blk_pages, lane + lanes, 2 * lanes, num_banks, run_pages);
                    ps[port_sender_rt::DST_MESH_ID] = p.fabric_mesh_ids.at(*peer);
                    ps[port_sender_rt::DST_CHIP_ID] = p.fabric_chip_ids.at(*peer);
                    ps[port_sender_rt::NUM_BLOCKS] = nblocks;
                    for (const auto& [block, slot] : sched.blocks) {
                        ps.push_back(block | (slot << BLOCK_SLOT_SHIFT));
                    }
                } else if (send_ready) {
                    ps[port_sender_rt::DST_MESH_ID] = p.fabric_mesh_ids.at(*peer);
                    ps[port_sender_rt::DST_CHIP_ID] = p.fabric_chip_ids.at(*peer);
                }
                if ((has_links && nblocks > 0) || send_ready) {
                    TT_FATAL(has_links, "fabric_reduce_scatter: a ready to a peer without a link");
                    const auto dst = mesh_device->get_fabric_node_id(coord_of(*peer));
                    tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                        *mesh_device, me, dst, rg.links[d].at(l), desc, core, ps, tt::CoreType::WORKER);
                }
                port_sender.runtime_args.emplace_back(core, std::move(ps));
            }
        }
    }

    // Final cores: 2 lane + h owns banks lane + h lanes, stride 2 lanes.
    for (uint32_t i = 0; i < ch.finals.size(); ++i) {
        const CoreCoord& core = ch.finals[i];
        const uint32_t lane = i / 2;
        const uint32_t h = i % 2;
        const uint32_t j = lane / num_links;
        const plan::RingPosition& rg = ch.rings.at(j);
        const bool has_f = rg.prev.has_value() && !plan::rs_schedule(p, *rg.prev, j, plan::FWD).blocks.empty();
        const bool has_b = rg.next.has_value() && !plan::rs_schedule(p, *rg.next, j, plan::BWD).blocks.empty();
        const uint32_t first = lane + (h * lanes);
        const uint32_t fstride = 2 * lanes;
        const uint32_t n = plan::chunks_per_shard(blk_pages, first, fstride, num_banks, run_pages);

        std::vector<uint32_t> fr(final_reader_rt::COUNT, 0u);
        fr[final_reader_rt::FIRST_BANK] = first;
        fr[final_reader_rt::BANK_STRIDE] = fstride;
        fr[final_reader_rt::OWN_SLOT] = ch.out;
        fr[final_reader_rt::FWD_SLOT] = ch.out;
        fr[final_reader_rt::HAS_FWD] = has_f ? 1 : 0;
        fr[final_reader_rt::HAS_BWD] = has_b ? 1 : 0;
        fr[final_reader_rt::NUM_CHUNKS] = n;
        final_reader.runtime_args.emplace_back(core, std::move(fr));

        std::vector<uint32_t> fc(compute_rt::COUNT, 0u);
        fc[compute_rt::N] = n;
        fc[compute_rt::HAS_B] = has_f ? 1 : 0;
        fc[compute_rt::HAS_C] = has_b ? 1 : 0;
        fc[compute_rt::N_COPY] = 0;
        final_compute.runtime_args.emplace_back(core, std::move(fc));

        std::vector<uint32_t> fw(final_writer_rt::COUNT, 0u);
        fw[final_writer_rt::FIRST_BANK] = first;
        fw[final_writer_rt::BANK_STRIDE] = fstride;
        fw[final_writer_rt::NUM_CHUNKS] = n;
        fw[final_writer_rt::EXPECT_A] = has_f ? plan::increments(n, kIncEvery) : 0;
        fw[final_writer_rt::EXPECT_B] = has_b ? plan::increments(n, kIncEvery) : 0;
        final_writer.runtime_args.emplace_back(core, std::move(fw));
    }

    desc.kernels.resize(kernel_index::COUNT);
    desc.kernels[kernel_index::PORT_READER] = std::move(port_reader);
    desc.kernels[kernel_index::PORT_COMPUTE] = std::move(port_compute);
    desc.kernels[kernel_index::PORT_SENDER] = std::move(port_sender);
    desc.kernels[kernel_index::FINAL_READER] = std::move(final_reader);
    desc.kernels[kernel_index::FINAL_COMPUTE] = std::move(final_compute);
    desc.kernels[kernel_index::FINAL_WRITER] = std::move(final_writer);
    return desc;
}

FabricReduceScatterMeshWorkloadFactory::cached_mesh_workload_t
FabricReduceScatterMeshWorkloadFactory::create_mesh_workload(
    const FabricReduceScatterParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const FabricReduceScatterInputs& tensor_args,
    Tensor& output) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (const auto& coord : tensor_coords.coords()) {
        Program program{FabricReduceScatterProgramFactory::create_descriptor(attrs, tensor_args, output, coord)};
        shared_variables_t sv{
            .port_reader = GetCommonRuntimeArgs(program, kernel_index::PORT_READER),
            .port_sender = GetCommonRuntimeArgs(program, kernel_index::PORT_SENDER),
            .final_reader = GetCommonRuntimeArgs(program, kernel_index::FINAL_READER),
            .final_writer = GetCommonRuntimeArgs(program, kernel_index::FINAL_WRITER),
            .applied_addresses = {},
        };
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(program));
        shared.emplace(ttnn::MeshCoordinateRange(coord), std::move(sv));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void FabricReduceScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const FabricReduceScatterParams& /*attrs*/,
    const FabricReduceScatterInputs& tensor_args,
    Tensor& output) {
    // The key pins every other slot (geometry, schedules, semaphore addresses: per-mesh state that keeps its
    // create-time values). A hit writes only the input, scratch and output addresses, the same on every program.
    TT_FATAL(tensor_args.scratch.has_value(), "fabric_reduce_scatter: scratch_tensor missing");
    const uint32_t in_addr = tensor_args.input.buffer()->address();
    const uint32_t scr_addr = tensor_args.scratch->buffer()->address();
    const uint32_t out_addr = output.buffer()->address();
    const std::vector<uint32_t> addresses{in_addr, scr_addr, out_addr};
    auto& first = cached_workload.shared_variables.begin()->second;
    if (first.applied_addresses == addresses) {
        return;
    }
    first.applied_addresses.clear();
    for (auto& [range, sv] : cached_workload.shared_variables) {
        auto& pr = sv.port_reader.get();
        pr[port_reader_common::IN_ADDR] = in_addr;
        pr[port_reader_common::SCR_ADDR] = scr_addr;
        sv.port_sender.get()[port_sender_common::SCR_ADDR] = scr_addr;
        auto& fr = sv.final_reader.get();
        fr[final_reader_common::IN_ADDR] = in_addr;
        fr[final_reader_common::SCR_ADDR] = scr_addr;
        sv.final_writer.get()[final_writer_common::OUT_ADDR] = out_addr;
    }
    first.applied_addresses = addresses;
}

}  // namespace ttnn::operations::fabric_reduce_scatter
