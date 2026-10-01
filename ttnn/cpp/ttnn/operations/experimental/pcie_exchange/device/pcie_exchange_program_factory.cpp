// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pcie_exchange_program_factory.hpp"

#include <set>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/pcie_p2p.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "pcie_exchange_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::pcie_exchange {

namespace {

constexpr uint32_t CB_IN = tt::CBIndex::c_0;
constexpr uint32_t CB_LANDING = tt::CBIndex::c_1;
constexpr uint32_t CB_OUT = tt::CBIndex::c_2;
// bfp8 payload mode: the compute kernel packs its bf16 shard to bfp8 twice, one copy for the dataflow kernel to
// send (c_3) and one it adds itself (c_4), so a CB never has two consumers.
constexpr uint32_t CB_STAGE_DM = tt::CBIndex::c_3;
constexpr uint32_t CB_STAGE_MATH = tt::CBIndex::c_4;
constexpr uint32_t BFP8_TILE_BYTES = 1088;

struct Geometry {
    uint32_t ntiles;
    uint32_t tile_bytes;
    uint32_t shard_bytes;
};

Geometry geometry(const Tensor& in) {
    const auto& ss = in.memory_config().shard_spec().value();
    Geometry g;
    if (in.layout() == Layout::TILE) {
        g.ntiles = (ss.shape[0] / tt::constants::TILE_HEIGHT) * (ss.shape[1] / tt::constants::TILE_WIDTH);
        g.tile_bytes = tt::tile_size(datatype_to_dataformat_converter(in.dtype()));
        g.shard_bytes = g.ntiles * g.tile_bytes;
    } else {
        // ROW_MAJOR (ALL_GATHER only): the shard is moved as one block; CBs are unused but must stay consistent.
        g.shard_bytes = ss.shape[0] * ss.shape[1] * in.element_size();
        g.ntiles = 1;
        g.tile_bytes = g.shard_bytes;
    }
    return g;
}

// Runtime args shared by both kernels' views of the per-core layout. Index constants are used by override.
constexpr uint32_t RT_IN = 0, RT_OUT = 1, RT_XB = 2, RT_CTRL = 12;  // RT 13 = payload bytes (constant per program)

uint32_t ctrl_address(const PcieExchangeInputs& tensor_args, uint32_t xb_addr, uint32_t shard_bytes) {
    return tensor_args.ctrl.has_value() ? tensor_args.ctrl->buffer()->address() : xb_addr + 2 * shard_bytes;
}

}  // namespace

PcieExchangeProgramFactory::cached_program_t PcieExchangeProgramFactory::create_at(
    const PcieExchangeParams& args,
    const ttnn::MeshCoordinate& coord,
    const PcieExchangeInputs& tensor_args,
    Tensor& output) {
    namespace p2p = tt::tt_metal::experimental::pcie_p2p;
    const auto& in = tensor_args.input;
    const auto& xb = tensor_args.exchange;
    auto* mesh = in.device();
    IDevice* dev = mesh->get_device(coord);
    const auto my_chip = dev->id();
    // Peer: the other device of the 1x2 mesh.
    auto peer_chip = my_chip;
    for (auto* d : mesh->get_devices()) {
        if (d->id() != my_chip) {
            peer_chip = d->id();
        }
    }
    TT_FATAL(peer_chip != my_chip, "pcie_exchange: no peer device");
    const uint32_t my_idx = p2p::mesh_index(my_chip);
    const uint32_t peer_idx = p2p::mesh_index(peer_chip);

    Program program{};
    const auto cores = PcieExchangeDeviceOperation::shard_cores(in);
    const uint32_t n = cores.size();
    const Geometry g = geometry(in);
    const bool reduce = args.mode == PcieExchangeMode::ALL_REDUCE;
    const auto df = datatype_to_dataformat_converter(in.dtype());

    std::set<CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.insert(CoreRange(c, c));
    }
    CoreRangeSet all_cores(ranges);

    shared_variables_t sv;
    sv.cores = cores;
    sv.has_compute = reduce;

    const bool bfp8 = args.bfp8_payload;
    const uint32_t payload_tile_bytes = bfp8 ? BFP8_TILE_BYTES : g.tile_bytes;
    const uint32_t payload_bytes = g.ntiles * payload_tile_bytes;
    const auto landing_df = bfp8 ? tt::DataFormat::Bfp8_b : df;
    if (reduce) {
        auto cb_in_cfg = CircularBufferConfig(g.shard_bytes, {{CB_IN, df}})
                             .set_page_size(CB_IN, g.tile_bytes)
                             .set_globally_allocated_address(*in.buffer());
        sv.cb_in = CreateCircularBuffer(program, all_cores, cb_in_cfg);
        auto cb_land_cfg = CircularBufferConfig(2 * payload_bytes, {{CB_LANDING, landing_df}})
                               .set_page_size(CB_LANDING, payload_tile_bytes)
                               .set_globally_allocated_address(*xb.buffer());
        sv.cb_landing = CreateCircularBuffer(program, all_cores, cb_land_cfg);
        if (bfp8) {
            auto cb_stage_dm = CircularBufferConfig(payload_bytes, {{CB_STAGE_DM, landing_df}})
                                   .set_page_size(CB_STAGE_DM, payload_tile_bytes);
            CreateCircularBuffer(program, all_cores, cb_stage_dm);
            auto cb_stage_math = CircularBufferConfig(payload_bytes, {{CB_STAGE_MATH, landing_df}})
                                     .set_page_size(CB_STAGE_MATH, payload_tile_bytes);
            CreateCircularBuffer(program, all_cores, cb_stage_math);
        }
        auto cb_out_cfg = CircularBufferConfig(g.shard_bytes, {{CB_OUT, df}})
                              .set_page_size(CB_OUT, g.tile_bytes)
                              .set_globally_allocated_address(*output.buffer());
        sv.cb_out = CreateCircularBuffer(program, all_cores, cb_out_cfg);
    }

    sv.dm_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/pcie_exchange/device/kernels/dataflow/pcie_exchange_dm.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {bfp8 ? 1u : 0u, CB_IN, CB_LANDING, CB_STAGE_DM}});
    if (reduce) {
        sv.compute_kernel = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/pcie_exchange/device/kernels/compute/pcie_exchange_add.cpp",
            all_cores,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .compile_args = {CB_IN, CB_LANDING, CB_OUT, CB_STAGE_DM, CB_STAGE_MATH, bfp8 ? 1u : 0u}});
    }

    std::vector<CoreCoord> out_cores;
    if (!reduce) {
        out_cores = corerange_to_cores(output.memory_config().shard_spec().value().grid, std::nullopt, true);
    }
    const uint32_t in_addr = in.buffer()->address();
    const uint32_t out_addr = output.buffer()->address();
    const uint32_t xb_addr = xb.buffer()->address();
    const uint32_t ctrl_addr = ctrl_address(tensor_args, xb_addr, g.shard_bytes);
    for (uint32_t i = 0; i < n; ++i) {
        const auto& core = cores[i];
        const uint64_t win = p2p::remote_l1_noc_addr(my_chip, peer_chip, core, 0);
        CoreCoord dst_a{0, 0}, dst_b{0, 0};
        if (!reduce) {
            dst_a = dev->worker_core_from_logical_core(out_cores[my_idx * n + i]);
            dst_b = dev->worker_core_from_logical_core(out_cores[peer_idx * n + i]);
        }
        std::vector<uint32_t> rt = {
            in_addr,
            out_addr,
            xb_addr,
            g.shard_bytes,
            g.ntiles,
            reduce ? 0u : 1u,
            uint32_t(win & 0xffffffffu),
            uint32_t(win >> 32),
            uint32_t(dst_a.x),
            uint32_t(dst_a.y),
            uint32_t(dst_b.x),
            uint32_t(dst_b.y),
            ctrl_addr,
            payload_bytes,
        };
        SetRuntimeArgs(program, sv.dm_kernel, core, rt);
        if (reduce) {
            SetRuntimeArgs(program, sv.compute_kernel, core, {g.ntiles, ctrl_addr});
        }
    }
    return {std::move(program), std::move(sv)};
}

PcieExchangeProgramFactory::cached_mesh_workload_t PcieExchangeProgramFactory::create_mesh_workload(
    const PcieExchangeParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const PcieExchangeInputs& tensor_args,
    Tensor& output) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached = create_at(args, coord, tensor_args, output);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached.shared_variables));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void PcieExchangeProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const PcieExchangeParams& args,
    const PcieExchangeInputs& tensor_args,
    Tensor& output) {
    (void)args;
    const auto& in = tensor_args.input;
    const auto& xb = tensor_args.exchange;
    const Geometry g = geometry(in);
    const uint32_t in_addr = in.buffer()->address();
    const uint32_t out_addr = output.buffer()->address();
    const uint32_t xb_addr = xb.buffer()->address();
    const uint32_t ctrl_addr = ctrl_address(tensor_args, xb_addr, g.shard_bytes);
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& sv = cached_workload.shared_variables.at(range);
        auto& dm_args = GetRuntimeArgs(program, sv.dm_kernel);
        for (const auto& core : sv.cores) {
            auto& rt = dm_args[core.x][core.y];
            rt[RT_IN] = in_addr;
            rt[RT_OUT] = out_addr;
            rt[RT_XB] = xb_addr;
            rt[RT_CTRL] = ctrl_addr;
        }
        if (sv.has_compute) {
            auto& c_args = GetRuntimeArgs(program, sv.compute_kernel);
            for (const auto& core : sv.cores) {
                c_args[core.x][core.y][1] = ctrl_addr;
            }
            UpdateDynamicCircularBufferAddress(program, sv.cb_in, *in.buffer());
            UpdateDynamicCircularBufferAddress(program, sv.cb_landing, *xb.buffer());
            UpdateDynamicCircularBufferAddress(program, sv.cb_out, *output.buffer());
        }
    }
}

}  // namespace ttnn::operations::experimental::pcie_exchange
