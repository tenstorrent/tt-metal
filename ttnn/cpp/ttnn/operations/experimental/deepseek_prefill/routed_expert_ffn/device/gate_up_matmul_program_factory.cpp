// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_up_matmul_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/core_coord.hpp>
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

namespace {

// CB indices — kept in sync with the kernel files.
constexpr tt::CBIndex CB_IN0 = tt::CBIndex::c_0;        // x (activations)
constexpr tt::CBIndex CB_IN1_GATE = tt::CBIndex::c_1;   // gate_proj weight tiles
constexpr tt::CBIndex CB_IN1_UP = tt::CBIndex::c_2;     // up_proj weight tiles
constexpr tt::CBIndex CB_GATE_INT = tt::CBIndex::c_3;   // gate accumulator (intermediate)
constexpr tt::CBIndex CB_UP_INT = tt::CBIndex::c_4;     // up   accumulator (intermediate)
constexpr tt::CBIndex CB_SILU_GATE = tt::CBIndex::c_5;  // silu(gate) intermediate
constexpr tt::CBIndex CB_ACT_OUT = tt::CBIndex::c_6;    // fused silu output

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::vector<CoreCoord> flatten(const CoreRangeSet& crs) {
    std::vector<CoreCoord> out;
    for (const auto& range : crs.ranges()) {
        for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
            for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                out.push_back({x, y});
            }
        }
    }
    return out;
}

static uint32_t tile_start(uint32_t core_idx, uint32_t n_g1, uint32_t b1, uint32_t b2, uint32_t tile_step) {
    if (core_idx < n_g1) {
        return core_idx * b1 * tile_step;
    }
    return (n_g1 * b1 + (core_idx - n_g1) * b2) * tile_step;
}

// ── CB creation ───────────────────────────────────────────────────────────────
// Creates all seven CBs for one group.  CB_SILU_GATE holds the silu(gate)
// intermediate and CB_ACT_OUT holds the fused output written to DRAM.
static void create_cbs(
    tt::tt_metal::Program& program,
    const CoreRangeSet& cores,
    uint32_t n_blocks_local,
    uint32_t in0_block_size,
    uint32_t in1_block_size,
    uint32_t in0_tile_size,
    uint32_t in1_tile_size,
    uint32_t interm_tile_size,
    uint32_t out_tile_size,
    uint32_t Mt_block_size,
    uint32_t Nt_block_size,
    tt::DataFormat in0_df,
    tt::DataFormat in1_df,
    tt::DataFormat interm_df,
    tt::DataFormat out_df) {
    uint32_t full_N_local = n_blocks_local * Nt_block_size;
    uint32_t full_out_tiles = Mt_block_size * full_N_local;

    tt::tt_metal::CreateCircularBuffer(
        program,
        cores,
        tt::tt_metal::CircularBufferConfig(2 * in0_block_size * in0_tile_size, {{CB_IN0, in0_df}})
            .set_page_size(CB_IN0, in0_tile_size));

    tt::tt_metal::CreateCircularBuffer(
        program,
        cores,
        tt::tt_metal::CircularBufferConfig(2 * in1_block_size * in1_tile_size, {{CB_IN1_GATE, in1_df}})
            .set_page_size(CB_IN1_GATE, in1_tile_size));

    tt::tt_metal::CreateCircularBuffer(
        program,
        cores,
        tt::tt_metal::CircularBufferConfig(2 * in1_block_size * in1_tile_size, {{CB_IN1_UP, in1_df}})
            .set_page_size(CB_IN1_UP, in1_tile_size));

    tt::tt_metal::CreateCircularBuffer(
        program,
        cores,
        tt::tt_metal::CircularBufferConfig(full_out_tiles * interm_tile_size, {{CB_GATE_INT, interm_df}})
            .set_page_size(CB_GATE_INT, interm_tile_size));

    tt::tt_metal::CreateCircularBuffer(
        program,
        cores,
        tt::tt_metal::CircularBufferConfig(full_out_tiles * interm_tile_size, {{CB_UP_INT, interm_df}})
            .set_page_size(CB_UP_INT, interm_tile_size));

    // CB_SILU_GATE: silu(gate) intermediate temp used by compute kernel.
    tt::tt_metal::CreateCircularBuffer(
        program,
        cores,
        tt::tt_metal::CircularBufferConfig(full_out_tiles * out_tile_size, {{CB_SILU_GATE, out_df}})
            .set_page_size(CB_SILU_GATE, out_tile_size));

    // CB_ACT_OUT: fused silu output; drained by NCRISC to DRAM.
    tt::tt_metal::CreateCircularBuffer(
        program,
        cores,
        tt::tt_metal::CircularBufferConfig(full_out_tiles * out_tile_size, {{CB_ACT_OUT, out_df}})
            .set_page_size(CB_ACT_OUT, out_tile_size));
}

// ── Kernel creation per quadrant group ───────────────────────────────────────
// Creates sender BRISC (x=0 cores), optional receiver BRISC (x>0 cores),
// NCRISC (all cores), and TRISC (all cores).
//
// When n_n_cores == 1: no mcast, sender kernel reads DRAM on all cores.
// When n_n_cores  > 1 and x0 == 0:
//   - reader_x_mcast_sender.cpp on x=0 column (num_mcast_dests = n_n_cores-1)
//   - reader_x_mcast_receiver.cpp on x=1..x1 columns
// When n_n_cores  > 1 and x0 > 0 (TR/BR quadrant, no sender in this group):
//   - reader_x_mcast_receiver.cpp on all cores in the quadrant
static KernelGroupInfo create_kernels_for_group(
    tt::tt_metal::Program& program,
    const CoreRangeSet& full_core_set,
    uint32_t x0,
    uint32_t x1,  // logical x bounds of this quadrant
    uint32_t y0,
    uint32_t y1,         // logical y bounds of this quadrant
    uint32_t n_n_cores,  // total N columns in the grid
    uint32_t m_blocks_local,
    uint32_t n_blocks_local,
    uint32_t K_num_blocks,
    const GateUpMatmulParams& p,
    uint32_t in0_tile_size,
    uint32_t in1_tile_size,
    uint32_t out_tile_size,
    const GateUpMatmulInputs& inputs,
    Tensor& output) {
    KernelGroupInfo kg;

    // ── Separate sender (x=0) from receiver (x>0) cores ──────────────────────
    bool has_sender = (x0 == 0);
    uint32_t num_mcast_dests = n_n_cores - 1;  // 0 when n_n_cores == 1

    // All cores in this quadrant.
    kg.all_cores = flatten(full_core_set);

    // Compile-time args shared by both sender variants (and also the receiver).
    // Layout: [K_num_blocks, m_blocks_local, Mt_block_size, Kt_block_size, in0_tile_size, num_mcast_dests]
    std::vector<uint32_t> rx_ct_base = {
        K_num_blocks, m_blocks_local, p.M_block_size, p.K_block_size, in0_tile_size, num_mcast_dests};

    if (num_mcast_dests == 0) {
        // Single-column grid (n_n_cores == 1): all cores are senders reading x from DRAM.
        // Or: mcast disabled — each core reads x independently (same as original reader_x.cpp).
        std::vector<uint32_t> sender_ct = rx_ct_base;
        tt::tt_metal::TensorAccessorArgs(inputs.x.buffer()).append_to(sender_ct);
        kg.sender_x_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/device/kernels/"
            "reader_x_mcast_sender.cpp",
            full_core_set,
            tt::tt_metal::ReaderDataMovementConfig(sender_ct, {}));
        kg.sender_cores = flatten(full_core_set);
    } else {
        // ── reader_x_mcast_sender (BRISC, x=0 column if present) ─────────────────
        if (has_sender) {
            std::vector<uint32_t> sender_ct = rx_ct_base;
            tt::tt_metal::TensorAccessorArgs(inputs.x.buffer()).append_to(sender_ct);

            CoreRangeSet sender_core_set(CoreRange({0, y0}, {0, y1}));
            kg.sender_x_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/device/kernels/"
                "reader_x_mcast_sender.cpp",
                sender_core_set,
                tt::tt_metal::ReaderDataMovementConfig(sender_ct, {}));
            kg.sender_cores = flatten(sender_core_set);
        }

        // ── reader_x_mcast_receiver (BRISC, x>0 cores in this quadrant) ──────────
        bool has_receiver_here = (x1 > 0 || (!has_sender));
        if (has_receiver_here) {
            uint32_t rx0 = has_sender ? 1 : x0;  // first receiver x in this group
            if (rx0 <= x1) {
                CoreRangeSet receiver_core_set(CoreRange({rx0, y0}, {x1, y1}));
                kg.receiver_x_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/device/kernels/"
                    "reader_x_mcast_receiver.cpp",
                    receiver_core_set,
                    tt::tt_metal::ReaderDataMovementConfig(rx_ct_base, {}));
                kg.receiver_cores = flatten(receiver_core_set);
            }
        }
    }

    // ── reader_weights_writer_outputs (NCRISC, all cores) ────────────────────
    // ct[0]=K_num_blocks, ct[1]=m_blocks_local, ct[2]=n_blocks_local,
    // ct[3]=Mt_block_size, ct[4]=Kt_block_size, ct[5]=Nt_block_size,
    // ct[6]=in1_tile_size, ct[7]=out_tile_size, ct[8]=Nt,
    // ct[9..]=TensorAccessors(gate_proj, up_proj, act_out)
    // rt[0]=gate_proj_addr, rt[1]=up_proj_addr, rt[2]=act_out_addr,
    //    rt[3]=m_tile_start, rt[4]=n_tile_start
    std::vector<uint32_t> rw_ct = {
        K_num_blocks,
        m_blocks_local,
        n_blocks_local,
        p.M_block_size,
        p.K_block_size,
        p.N_block_size,
        in1_tile_size,
        out_tile_size,
        p.Nt};
    tt::tt_metal::TensorAccessorArgs(inputs.gate_proj.buffer()).append_to(rw_ct);
    tt::tt_metal::TensorAccessorArgs(inputs.up_proj.buffer()).append_to(rw_ct);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(rw_ct);

    kg.reader_weights_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/device/kernels/"
        "reader_weights_writer_outputs.cpp",
        full_core_set,
        tt::tt_metal::WriterDataMovementConfig(rw_ct, {}));

    // ── compute_dual_gate_up (TRISC, all cores) ───────────────────────────────
    std::vector<uint32_t> cp_ct = {
        K_num_blocks,
        m_blocks_local,
        n_blocks_local,
        p.M_block_size,
        p.K_block_size,
        p.N_block_size,
        p.subblock_h,
        p.subblock_w};

    kg.compute_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/device/kernels/"
        "compute_dual_gate_up.cpp",
        full_core_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = cp_ct});

    return kg;
}

// ── Main program builder ──────────────────────────────────────────────────────

struct CreatedProgram {
    tt::tt_metal::Program program;
    GateUpMatmulSharedVariables shared_variables;
};

CreatedProgram create_program(const GateUpMatmulParams& p, const GateUpMatmulInputs& inputs, Tensor& output) {
    tt::tt_metal::Program program{};

    // ── Data formats ─────────────────────────────────────────────────────────
    auto in0_df = tt::tt_metal::datatype_to_dataformat_converter(inputs.x.dtype());
    auto in1_df = tt::tt_metal::datatype_to_dataformat_converter(inputs.gate_proj.dtype());
    auto interm_df = tt::DataFormat::Float16_b;
    auto out_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    uint32_t in0_tile_size = tt::tile_size(in0_df);
    uint32_t in1_tile_size = tt::tile_size(in1_df);
    uint32_t interm_tile_size = tt::tile_size(interm_df);
    uint32_t out_tile_size = tt::tile_size(out_df);

    // ── Block counts ─────────────────────────────────────────────────────────
    uint32_t K_num_blocks = p.Kt / p.K_block_size;
    uint32_t M_num_blocks = p.Mt / p.M_block_size;
    uint32_t N_num_blocks = p.Nt / p.N_block_size;

    uint32_t in0_block_size = p.M_block_size * p.K_block_size;
    uint32_t in1_block_size = p.K_block_size * p.N_block_size;

    // ── 2-D core grid: x-axis = N split, y-axis = M split ────────────────────
    auto* device = inputs.x.device();
    // CoreCoord grid = device->compute_with_storage_grid_size();

    uint32_t n_n_cores = 8;
    // uint32_t n_n_cores = std::min(N_num_blocks, (uint32_t)grid.x);
    uint32_t n_m_cores = 8;
    // uint32_t n_m_cores = std::min(M_num_blocks, (uint32_t)grid.y);

    uint32_t n_b1 = (N_num_blocks + n_n_cores - 1) / n_n_cores;
    uint32_t n_b2 = N_num_blocks / n_n_cores;
    uint32_t n_n_g1 = N_num_blocks % n_n_cores;

    uint32_t m_b1 = (M_num_blocks + n_m_cores - 1) / n_m_cores;
    uint32_t m_b2 = M_num_blocks / n_m_cores;
    uint32_t n_m_g1 = M_num_blocks % n_m_cores;

    // ── Semaphores for mcast synchronisation ─────────────────────────────────
    // sender_semaphore: receivers atomically increment this; sender waits for
    //   the count to reach num_mcast_dests, then resets it to 0.
    // receiver_semaphore: sender multicasts VALID into this; receiver waits
    //   for VALID then resets to INVALID before the next block.
    // Both are created on all cores so every core has the same L1 offset.
    // When n_n_cores == 1 (no mcast), these semaphores are unused.
    CoreRangeSet all_core_set(CoreRange({0, 0}, {n_n_cores - 1, n_m_cores - 1}));
    uint32_t sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_core_set, 0);
    uint32_t receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_core_set, INVALID);

    // ── Build kernels per quadrant ────────────────────────────────────────────
    struct GroupDef {
        uint32_t x0, x1, y0, y1;
        uint32_t m_blocks_local;
        uint32_t n_blocks_local;
    };

    std::vector<GroupDef> group_defs;
    auto push = [&](uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1, uint32_t mb, uint32_t nb) {
        if (mb > 0 && nb > 0) {
            group_defs.push_back({x0, x1, y0, y1, mb, nb});
        }
    };

    if (n_m_g1 > 0 && n_n_g1 > 0) {
        push(0, n_n_g1 - 1, 0, n_m_g1 - 1, m_b1, n_b1);  // TL
    }
    if (n_m_g1 > 0 && n_n_g1 < n_n_cores) {
        push(n_n_g1, n_n_cores - 1, 0, n_m_g1 - 1, m_b1, n_b2);  // TR
    }
    if (n_m_g1 < n_m_cores && n_n_g1 > 0) {
        push(0, n_n_g1 - 1, n_m_g1, n_m_cores - 1, m_b2, n_b1);  // BL
    }
    if (n_m_g1 < n_m_cores && n_n_g1 < n_n_cores) {
        push(n_n_g1, n_n_cores - 1, n_m_g1, n_m_cores - 1, m_b2, n_b2);  // BR
    }

    GateUpMatmulSharedVariables sv;

    for (auto& gd : group_defs) {
        CoreRangeSet core_set(CoreRange({gd.x0, gd.y0}, {gd.x1, gd.y1}));

        create_cbs(
            program,
            core_set,
            gd.n_blocks_local,
            in0_block_size,
            in1_block_size,
            in0_tile_size,
            in1_tile_size,
            interm_tile_size,
            out_tile_size,
            p.M_block_size,
            p.N_block_size,
            in0_df,
            in1_df,
            interm_df,
            out_df);

        auto kg = create_kernels_for_group(
            program,
            core_set,
            gd.x0,
            gd.x1,
            gd.y0,
            gd.y1,
            n_n_cores,
            gd.m_blocks_local,
            gd.n_blocks_local,
            K_num_blocks,
            p,
            in0_tile_size,
            in1_tile_size,
            out_tile_size,
            inputs,
            output);

        // ── NCRISC runtime args (all cores) ───────────────────────────────────
        for (const auto& core : kg.all_cores) {
            uint32_t m_tile_start = tile_start(core.y, n_m_g1, m_b1, m_b2, p.M_block_size);
            uint32_t n_tile_start = tile_start(core.x, n_n_g1, n_b1, n_b2, p.N_block_size);

            tt::tt_metal::SetRuntimeArgs(
                program,
                kg.reader_weights_id,
                core,
                {inputs.gate_proj.buffer()->address(),
                 inputs.up_proj.buffer()->address(),
                 output.buffer()->address(),
                 m_tile_start,
                 n_tile_start});
        }

        // ── Sender runtime args ───────────────────────────────────────────────
        // rt[0]=x_addr, rt[1]=m_tile_start
        // If num_mcast_dests > 0:
        //   rt[2..5]=mcast dest NOC coords, rt[6]=sender_sem_id, rt[7]=recv_sem_id
        uint32_t num_mcast_dests = n_n_cores - 1;
        for (const auto& core : kg.sender_cores) {
            uint32_t m_tile_start = tile_start(core.y, n_m_g1, m_b1, m_b2, p.M_block_size);
            std::vector<uint32_t> sender_rt = {inputs.x.buffer()->address(), m_tile_start};

            if (num_mcast_dests > 0) {
                // Physical coords of first receiver (logical (1, core.y)) and
                // last receiver (logical (n_n_cores-1, core.y)) in this M-row.
                CoreCoord recv_start_phys = device->worker_core_from_logical_core({1, core.y});
                CoreCoord recv_end_phys = device->worker_core_from_logical_core({n_n_cores - 1, core.y});
                sender_rt.push_back(recv_start_phys.x);
                sender_rt.push_back(recv_start_phys.y);
                sender_rt.push_back(recv_end_phys.x);
                sender_rt.push_back(recv_end_phys.y);
                sender_rt.push_back(sender_semaphore_id);
                sender_rt.push_back(receiver_semaphore_id);
            }

            tt::tt_metal::SetRuntimeArgs(program, kg.sender_x_id, core, sender_rt);
        }

        // ── Receiver runtime args ─────────────────────────────────────────────
        // rt[0]=sender_noc_x, rt[1]=sender_noc_y, rt[2]=sender_sem_id, rt[3]=recv_sem_id
        for (const auto& core : kg.receiver_cores) {
            CoreCoord sender_phys = device->worker_core_from_logical_core({0, core.y});
            tt::tt_metal::SetRuntimeArgs(
                program,
                kg.receiver_x_id,
                core,
                {(uint32_t)sender_phys.x, (uint32_t)sender_phys.y, sender_semaphore_id, receiver_semaphore_id});
        }

        sv.groups.push_back(std::move(kg));
    }

    return CreatedProgram{std::move(program), std::move(sv)};
}

}  // namespace

GateUpMatmulProgramFactory::cached_mesh_workload_t GateUpMatmulProgramFactory::create_mesh_workload(
    const GateUpMatmulParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const GateUpMatmulInputs& inputs,
    tensor_return_value_t& outputs) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_vars;

    for (const auto& coord : tensor_coords.coords()) {
        auto result = create_program(attrs, inputs, outputs);
        auto coord_range = ttnn::MeshCoordinateRange(coord);
        workload.add_program(coord_range, std::move(result.program));
        shared_vars.emplace(coord_range, std::move(result.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_vars)};
}

void GateUpMatmulProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const GateUpMatmulParams& /*attrs*/,
    const GateUpMatmulInputs& inputs,
    tensor_return_value_t& outputs) {
    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& sv = cached_workload.shared_variables.at(coord_range);

        for (auto& group : sv.groups) {
            for (const auto& core : group.all_cores) {
                auto& rw_args = GetRuntimeArgs(program, group.reader_weights_id, core);
                rw_args[0] = inputs.gate_proj.buffer()->address();
                rw_args[1] = inputs.up_proj.buffer()->address();
                rw_args[2] = outputs.buffer()->address();
                // rw_args[3] = m_tile_start — unchanged
                // rw_args[4] = n_tile_start — unchanged
            }

            for (const auto& core : group.sender_cores) {
                auto& sx_args = GetRuntimeArgs(program, group.sender_x_id, core);
                sx_args[0] = inputs.x.buffer()->address();
                // sx_args[1] = m_tile_start — unchanged
                // sx_args[2..7] = mcast coords and semaphore IDs — unchanged
            }

            // Receiver cores (group.receiver_cores): no buffer addresses to update.
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
