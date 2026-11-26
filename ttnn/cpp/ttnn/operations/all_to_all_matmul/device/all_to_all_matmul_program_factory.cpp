// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_matmul_program_factory.hpp"

#include <array>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "all_to_all_matmul_device_operation.hpp"
#include "all_to_all_matmul_planner.hpp"
#include "kernels/all_to_all_matmul_args.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::all_to_all_matmul {

using namespace tt::tt_metal;
using namespace ::all_to_all_matmul;

namespace {

constexpr const char* kFactoryKernelDir = "ttnn/cpp/ttnn/operations/all_to_all_matmul/device/kernels/";

std::string factory_kernel_path(const char* name) { return std::string(kFactoryKernelDir) + name; }

CBDescriptor plain_cb(
    uint32_t id, uint32_t total_size, const CoreRangeSet& cores, DataType dtype, uint32_t page_size, const Tile& tile) {
    return CBDescriptor{
        .total_size = total_size,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(id),
            .data_format = datatype_to_dataformat_converter(dtype),
            .page_size = page_size,
            .tile = TileDescriptor(tile)}},
    };
}

uint32_t pack_nbw(const SlotWork& s) {
    TT_FATAL(
        s.nb0 <= nbw_bits::FIELD_MASK && s.nb1 <= nbw_bits::FIELD_MASK && s.m0 <= nbw_bits::FIELD_MASK &&
            s.m1 <= nbw_bits::FIELD_MASK,
        "all_to_all_matmul: slot fields exceed 8 bits");
    return (s.nb0 << nbw_bits::NB0_SHIFT) | (s.nb1 << nbw_bits::NB1_SHIFT) | (s.m0 << nbw_bits::M0_SHIFT) |
           (s.m1 << nbw_bits::M1_SHIFT);
}

}  // namespace

ProgramDescriptor AllToAllMatmulProgramFactory::create_descriptor(
    const AllToAllMatmulParams& attrs,
    const AllToAllMatmulInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "all_to_all_matmul: one program per mesh coordinate");
    TT_FATAL(tensor_args.scratch.has_value(), "all_to_all_matmul: the landing scratch_tensor is required");
    TT_FATAL(
        cb::RETILE_STAGE < hal::get_num_dataflow_buffers(),
        "all_to_all_matmul: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        cb::RETILE_STAGE,
        hal::get_num_dataflow_buffers());

    const Tensor& input = tensor_args.input;
    const Tensor& weight = tensor_args.weight;
    const Tensor& scratch = *tensor_args.scratch;
    auto* mesh_device = input.device();
    const auto& coord = *mesh_dispatch_coordinate;

    const MeshTopology& topology = mesh_topology(mesh_device, attrs.cluster_axis, attrs.num_links);
    const PlanInputs inputs = plan_inputs_of(attrs, input, weight);
    const Plan& plan = solve_plan(topology, inputs);
    const MeshResources& res = mesh_resources(mesh_device, topology, plan);
    const ChipPlan& chip = plan.chips.at(coord);

    const uint32_t in_tile = inputs.in_tile_bytes;
    const uint32_t w_tile = inputs.w_tile_bytes;
    const Tile& in_tile_shape = input.tensor_spec().tile();
    const Tile& w_tile_shape = weight.tensor_spec().tile();
    const auto virt = [&](const CoreCoord& c) { return mesh_device->worker_core_from_logical_core(c); };
    TT_FATAL(
        res.arrival_addrs.size() >= plan.slots_per_core_max && res.ready_addrs.size() >= plan.n_ready,
        "all_to_all_matmul: per-mesh semaphores fewer than the plan needs");

    ProgramDescriptor desc;

    // ---- circular buffers ----
    desc.cbs.push_back(ttnn::cb_descriptor_from_sharded_tensor(static_cast<uint8_t>(cb::A_LANDED), scratch));
    desc.cbs.push_back(plain_cb(
        cb::WEIGHT,
        plan.weight_depth * plan.Kt * plan.block_n * w_tile,
        plan.active_cores,
        weight.dtype(),
        w_tile,
        w_tile_shape));
    desc.cbs.push_back(plain_cb(
        cb::OUT,
        kOutDepth * plan.out_subblock_h * plan.block_n * in_tile,
        plan.active_cores,
        input.dtype(),
        in_tile,
        in_tile_shape));
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * plan.batch_chunks * plan.packet_bytes,
        .core_ranges = chip.port_cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb::PORT_CHUNKS),
            .data_format = datatype_to_dataformat_converter(input.dtype()),
            .page_size = plan.packet_bytes}},
    });
    if (plan.retile) {
        // One descriptor per core set (the compute and port cores do not form one rectangle).
        for (const CoreRangeSet* crs : {&plan.active_cores, &chip.port_cores}) {
            desc.cbs.push_back(
                plain_cb(cb::RETILE_STAGE, plan.stage_tiles * in_tile, *crs, input.dtype(), in_tile, in_tile_shape));
        }
    }

    // ---- compute-core reader ----
    KernelDescriptor reader;
    reader.kernel_source = factory_kernel_path("all_to_all_matmul_reader.cpp");
    reader.core_ranges = plan.active_cores;
    reader.compile_time_args.assign(reader_ct::COUNT, 0u);
    {
        auto& ct = reader.compile_time_args;
        ct[reader_ct::CB_A_LANDED] = cb::A_LANDED;
        ct[reader_ct::CB_WEIGHT] = cb::WEIGHT;
        ct[reader_ct::KT] = plan.Kt;
        ct[reader_ct::NT] = plan.Nt;
        ct[reader_ct::BLOCK_N] = plan.block_n;
        ct[reader_ct::NUM_N_BLOCKS] = plan.num_n_blocks;
        ct[reader_ct::B] = inputs.B;
        ct[reader_ct::MQ_T] = plan.Mq_t;
        ct[reader_ct::ST] = plan.St;
        ct[reader_ct::HL] = inputs.Hl;
        ct[reader_ct::IN_TILE_BYTES] = in_tile;
        ct[reader_ct::W_TILE_BYTES] = w_tile;
        ct[reader_ct::MQ] = plan.Mq;
        ct[reader_ct::CB_RETILE_STAGE] = cb::RETILE_STAGE;
        ct[reader_ct::WEIGHT_CHUNK_TILES] = plan.weight_chunk_tiles;
        ct[reader_ct::W_PREFETCH_BLOCKS] = plan.w_prefetch;
    }
    TensorAccessorArgs(*input.buffer()).append_to(reader.compile_time_args);
    TensorAccessorArgs(*weight.buffer()).append_to(reader.compile_time_args);
    {
        KernelDescriptor::RTArgList common;
        common.reserve(reader_common::ARRIVAL_BASE + plan.slots_per_core_max);
        common.push_back(input.buffer());    // IN_ADDR
        common.push_back(weight.buffer());   // W_ADDR
        common.push_back(chip.q);            // MY_Q
        common.push_back(scratch.buffer());  // LANDING_ADDR
        for (uint32_t s = 0; s < plan.slots_per_core_max; ++s) {
            common.push_back(res.arrival_addrs[s]);  // ARRIVAL_BASE + s
        }
        reader.emplace_common_runtime_args(common);
    }
    reader.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = kComputeReaderNoc};

    // ---- compute ----
    KernelDescriptor compute;
    compute.kernel_source = factory_kernel_path("all_to_all_matmul_compute.cpp");
    compute.core_ranges = plan.active_cores;
    compute.compile_time_args.assign(compute_ct::COUNT, 0u);
    {
        auto& ct = compute.compile_time_args;
        ct[compute_ct::CB_A_LANDED] = cb::A_LANDED;
        ct[compute_ct::CB_WEIGHT] = cb::WEIGHT;
        ct[compute_ct::CB_OUT] = cb::OUT;
        ct[compute_ct::BLOCK_M] = plan.block_m;
        ct[compute_ct::KT] = plan.Kt;
        ct[compute_ct::BLOCK_N] = plan.block_n;
        ct[compute_ct::NUM_N_BLOCKS] = plan.num_n_blocks;
        ct[compute_ct::OUT_SUBBLOCK_H] = plan.out_subblock_h;
        ct[compute_ct::OUT_SUBBLOCK_W] = plan.out_subblock_w;
    }
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = attrs.math_fidelity,
        .fp32_dest_acc_en = attrs.fp32_dest_acc_en,
        .math_approx_mode = attrs.math_approx_mode,
    };

    // ---- compute-core writer ----
    KernelDescriptor writer;
    writer.kernel_source = factory_kernel_path("all_to_all_matmul_writer.cpp");
    writer.core_ranges = plan.active_cores;
    writer.compile_time_args.assign(writer_ct::COUNT, 0u);
    {
        auto& ct = writer.compile_time_args;
        ct[writer_ct::CB_OUT] = cb::OUT;
        ct[writer_ct::TILE_BYTES] = in_tile;
        ct[writer_ct::BLOCK_M] = plan.block_m;
        ct[writer_ct::MQ_T] = plan.Mq_t;
        ct[writer_ct::NT] = plan.Nt;
        ct[writer_ct::BLOCK_N] = plan.block_n;
        ct[writer_ct::NUM_N_BLOCKS] = plan.num_n_blocks;
        ct[writer_ct::OUT_SUBBLOCK_H] = plan.out_subblock_h;
        ct[writer_ct::H] = inputs.H;
        ct[writer_ct::OUT_CHUNK_TILES] = kOutChunkTiles;
    }
    TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.emplace_common_runtime_args({output.buffer()});
    writer.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = kComputeWriterNoc};
    TT_FATAL(
        writer.common_runtime_args.size() == writer_common::COUNT,
        "all_to_all_matmul: writer common args out of sync with all_to_all_matmul_args.hpp");

    // per-core args of the compute cores
    const size_t n_active = chip.cores.size();
    reader.runtime_args.reserve(n_active);
    compute.runtime_args.reserve(n_active);
    writer.runtime_args.reserve(n_active);
    for (const CoreWork& cw : chip.cores) {
        const auto n_slots = static_cast<uint32_t>(cw.slots.size());
        std::vector<uint32_t> rd(reader_rt::SLOTS_BASE + (n_slots * reader_slot::COUNT), 0u);
        std::vector<uint32_t> cp(compute_rt::NBW_BASE + n_slots, 0u);
        std::vector<uint32_t> wr(writer_rt::SLOTS_BASE + (n_slots * writer_slot::COUNT), 0u);
        rd[reader_rt::N_SLOTS] = n_slots;
        cp[compute_rt::N_SLOTS] = n_slots;
        wr[writer_rt::N_SLOTS] = n_slots;
        for (uint32_t s = 0; s < n_slots; ++s) {
            const SlotWork& sw = cw.slots[s];
            const uint32_t nbw = pack_nbw(sw);
            const uint32_t r0 = reader_rt::SLOTS_BASE + (s * reader_slot::COUNT);
            rd[r0 + reader_slot::WORD] = (sw.h & reader_slot::HEAD_MASK) |
                                         (static_cast<uint32_t>(sw.wait) << reader_slot::WAIT_SHIFT) |
                                         (sw.pre << reader_slot::PRE_SHIFT);
            rd[r0 + reader_slot::NBW] = nbw;
            if (sw.helper.has_value()) {
                const CoreCoord hv = virt(*sw.helper);
                rd[r0 + reader_slot::FWD] = 1;
                rd[r0 + reader_slot::FWD_X] = hv.x;
                rd[r0 + reader_slot::FWD_Y] = hv.y;
                rd[r0 + reader_slot::FWD_SLOT_OFFSET] = sw.helper_slot * plan.slot_bytes;
                rd[r0 + reader_slot::FWD_ARRIVAL_ADDR] = res.arrival_addrs.at(sw.helper_slot);
            }
            rd[r0 + reader_slot::PACE] = sw.pace;
            cp[compute_rt::NBW_BASE + s] = nbw;
            const uint32_t w0 = writer_rt::SLOTS_BASE + (s * writer_slot::COUNT);
            wr[w0 + writer_slot::HEAD] = sw.h;
            wr[w0 + writer_slot::NBW] = nbw;
        }
        reader.runtime_args.emplace_back(cw.core, std::move(rd));
        compute.runtime_args.emplace_back(cw.core, std::move(cp));
        writer.runtime_args.emplace_back(cw.core, std::move(wr));
    }

    // ---- port reader ----
    KernelDescriptor port_reader;
    port_reader.kernel_source = factory_kernel_path("all_to_all_matmul_port_reader.cpp");
    port_reader.core_ranges = chip.port_cores;
    port_reader.compile_time_args.assign(port_reader_ct::COUNT, 0u);
    {
        auto& ct = port_reader.compile_time_args;
        ct[port_reader_ct::CB_PORT_CHUNKS] = cb::PORT_CHUNKS;
        ct[port_reader_ct::TILE_BYTES] = in_tile;
        ct[port_reader_ct::PACKET_TILES] = plan.packet_tiles;
        ct[port_reader_ct::BATCH_CHUNKS] = plan.batch_chunks;
        ct[port_reader_ct::B] = inputs.B;
        ct[port_reader_ct::MQ_T] = plan.Mq_t;
        ct[port_reader_ct::KT] = plan.Kt;
        ct[port_reader_ct::ST] = plan.St;
        ct[port_reader_ct::HL] = inputs.Hl;
        ct[port_reader_ct::MQ] = plan.Mq;
        ct[port_reader_ct::CB_RETILE_STAGE] = cb::RETILE_STAGE;
        ct[port_reader_ct::PACK] = static_cast<uint32_t>(plan.pack);
        ct[port_reader_ct::WINDOW] = plan.port_window;
    }
    TensorAccessorArgs(*input.buffer()).append_to(port_reader.compile_time_args);
    port_reader.emplace_common_runtime_args({input.buffer()});
    port_reader.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = kPortReaderNoc};
    TT_FATAL(
        port_reader.common_runtime_args.size() == port_reader_common::COUNT,
        "all_to_all_matmul: port reader common args out of sync with all_to_all_matmul_args.hpp");

    // ---- port sender ----
    KernelDescriptor port_sender;
    port_sender.kernel_source = factory_kernel_path("all_to_all_matmul_port_sender.cpp");
    port_sender.core_ranges = chip.port_cores;
    port_sender.compile_time_args.assign(port_sender_ct::COUNT, 0u);
    {
        auto& ct = port_sender.compile_time_args;
        ct[port_sender_ct::CB_PORT_CHUNKS] = cb::PORT_CHUNKS;
        ct[port_sender_ct::TILE_BYTES] = in_tile;
        ct[port_sender_ct::PACKET_TILES] = plan.packet_tiles;
        ct[port_sender_ct::BATCH_CHUNKS] = plan.batch_chunks;
        ct[port_sender_ct::SLOT_TILES] = plan.slot_tiles;
        ct[port_sender_ct::MAX_HEADERS] = kMaxPacketHeaders;
        ct[port_sender_ct::PACK] = static_cast<uint32_t>(plan.pack);
        ct[port_sender_ct::FENCE_PER_DEST] = static_cast<uint32_t>(kFencePerDest);
    }
    port_sender.emplace_common_runtime_args({scratch.buffer()});
    port_sender.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = kPortSenderNoc};
    TT_FATAL(
        port_sender.common_runtime_args.size() == port_sender_common::COUNT,
        "all_to_all_matmul: port sender common args out of sync with all_to_all_matmul_args.hpp");

    const auto src_node = mesh_device->get_fabric_node_id(coord);
    // Fabric connections are appended to the descriptor (they may add semaphores to it): collect the args first.
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> sender_args;
    sender_args.reserve(chip.ports.size());
    for (const PortWork& pw : chip.ports) {
        std::vector<uint32_t> rd(port_reader_rt::ENTRIES_BASE, 0u);
        rd[port_reader_rt::N_ENTRIES] = static_cast<uint32_t>(pw.entries.size());
        rd.insert(rd.end(), pw.entries.begin(), pw.entries.end());
        port_reader.runtime_args.emplace_back(pw.core, std::move(rd));

        TT_FATAL(
            pw.dests.size() <= kMaxPacketHeaders,
            "all_to_all_matmul: a port with {} destinations exceeds the header pool",
            pw.dests.size());
        std::vector<uint32_t> args(port_sender_rt::DESTS_BASE, 0u);
        args[port_sender_rt::HAS_CONN] = static_cast<uint32_t>(pw.has_conn);
        args[port_sender_rt::N_DEST] = static_cast<uint32_t>(pw.dests.size());
        args[port_sender_rt::HDRS_PER_DEST] = pw.hdrs_per_dest;
        for (const PortDest& dest : pw.dests) {
            const auto node = mesh_device->get_fabric_node_id(dest.coord);
            std::array<uint32_t, port_sender_dest::COUNT> rec{};
            rec[port_sender_dest::CHIP_ID] = static_cast<uint32_t>(node.chip_id);
            rec[port_sender_dest::MESH_ID] = static_cast<uint32_t>(*node.mesh_id);
            rec[port_sender_dest::READY_ADDR] =
                dest.ready_index.has_value() ? res.ready_addrs.at(*dest.ready_index) : 0u;
            args.insert(args.end(), rec.begin(), rec.end());
        }
        args.push_back(static_cast<uint32_t>(pw.readies.size()));
        for (const PortReady& ready : pw.readies) {
            const CoreCoord rc = virt(ready.target_port_core);
            std::array<uint32_t, port_sender_ready::COUNT> rec{};
            rec[port_sender_ready::DEST_INDEX] = ready.dest_index;
            rec[port_sender_ready::X] = rc.x;
            rec[port_sender_ready::Y] = rc.y;
            rec[port_sender_ready::ADDR] = res.ready_addrs.at(ready.ready_index);
            args.insert(args.end(), rec.begin(), rec.end());
        }
        args.push_back(static_cast<uint32_t>(pw.subs.size()));
        for (const PortSub& sub : pw.subs) {
            const CoreCoord cv = virt(sub.core);
            std::array<uint32_t, port_sender_sub::COUNT> rec{};
            rec[port_sender_sub::WORD0] =
                (sub.dest_index & port_sender_sub::DEST_MASK) | (sub.tiles << port_sender_sub::TILES_SHIFT);
            rec[port_sender_sub::X] = cv.x;
            rec[port_sender_sub::Y] = cv.y;
            rec[port_sender_sub::SLOT_OFFSET] = sub.slot * plan.slot_bytes;
            rec[port_sender_sub::ARRIVAL_ADDR] = res.arrival_addrs.at(sub.slot);
            args.insert(args.end(), rec.begin(), rec.end());
        }
        if (pw.has_conn) {
            const auto dst_node = mesh_device->get_fabric_node_id(pw.peer);
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                src_node, dst_node, pw.link, desc, pw.core, args, tt::CoreType::WORKER);
        }
        sender_args.emplace_back(pw.core, std::move(args));
    }
    for (auto& [core, args] : sender_args) {
        port_sender.runtime_args.emplace_back(core, std::move(args));
    }

    // Kernel order is kernel_index::{READER, COMPUTE, WRITER, PORT_READER, PORT_SENDER}.
    desc.kernels.reserve(kernel_index::COUNT);
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(port_reader));
    desc.kernels.push_back(std::move(port_sender));
    return desc;
}

void AllToAllMatmulProgramFactory::override_runtime_arguments(
    Program& program,
    const AllToAllMatmulParams& /*attrs*/,
    const AllToAllMatmulInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The key pins every per-core arg (slot work, entries, routes, semaphore addresses, slot offsets) and every
    // compile-time arg. A hit rewrites the common tensor-address slots and the landing CB, only where they moved.
    const auto in_addr = tensor_args.input.buffer()->address();
    const auto w_addr = tensor_args.weight.buffer()->address();
    const auto out_addr = output.buffer()->address();
    Buffer* scratch_buffer = tensor_args.scratch->buffer();
    const auto landing_addr = scratch_buffer->address();

    auto& reader = GetCommonRuntimeArgs(program, kernel_index::READER);
    const bool landing_moved = reader[reader_common::LANDING_ADDR] != landing_addr;
    if (reader[reader_common::IN_ADDR] != in_addr) {
        reader[reader_common::IN_ADDR] = in_addr;
    }
    if (reader[reader_common::W_ADDR] != w_addr) {
        reader[reader_common::W_ADDR] = w_addr;
    }
    auto& writer = GetCommonRuntimeArgs(program, kernel_index::WRITER);
    if (writer[writer_common::OUT_ADDR] != out_addr) {
        writer[writer_common::OUT_ADDR] = out_addr;
    }
    auto& port_reader = GetCommonRuntimeArgs(program, kernel_index::PORT_READER);
    if (port_reader[port_reader_common::IN_ADDR] != in_addr) {
        port_reader[port_reader_common::IN_ADDR] = in_addr;
    }
    if (landing_moved) {
        for (const auto& circular_buffer : program.circular_buffers()) {
            if (circular_buffer->globally_allocated() && circular_buffer->buffer_indices().contains(cb::A_LANDED)) {
                UpdateDynamicCircularBufferAddress(program, circular_buffer->id(), *scratch_buffer);
            }
        }
        GetCommonRuntimeArgs(program, kernel_index::PORT_SENDER)[port_sender_common::LANDING_ADDR] = landing_addr;
        reader[reader_common::LANDING_ADDR] = landing_addr;
    }
}

}  // namespace ttnn::operations::all_to_all_matmul
