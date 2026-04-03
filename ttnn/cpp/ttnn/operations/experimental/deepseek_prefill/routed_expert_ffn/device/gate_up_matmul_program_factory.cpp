// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_up_matmul_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

namespace {

// CB indices — kept in sync with the kernel files.
constexpr tt::CBIndex CB_IN0 = tt::CBIndex::c_0;       // x
constexpr tt::CBIndex CB_IN1_GATE = tt::CBIndex::c_1;  // gate_proj tiles
constexpr tt::CBIndex CB_IN1_UP = tt::CBIndex::c_2;    // up_proj tiles
constexpr tt::CBIndex CB_GATE_INT = tt::CBIndex::c_3;  // gate intermediate (accumulator)
constexpr tt::CBIndex CB_UP_INT = tt::CBIndex::c_4;    // up   intermediate (accumulator)
constexpr tt::CBIndex CB_GATE_OUT = tt::CBIndex::c_5;  // gate output tiles
constexpr tt::CBIndex CB_UP_OUT = tt::CBIndex::c_6;    // up   output tiles

struct CreatedProgram {
    tt::tt_metal::Program program;
    GateUpMatmulSharedVariables shared_variables;
};

CreatedProgram create_program(
    const GateUpMatmulParams& p, const GateUpMatmulInputs& inputs, std::array<Tensor, 2>& outputs) {
    tt::tt_metal::Program program{};

    // ---------------------------------------------------------------
    // Determine core, device and data formats
    // ---------------------------------------------------------------
    CoreCoord core{0, 0};
    CoreRangeSet core_set(std::vector{CoreRange(core, core)});

    auto in0_df = tt::tt_metal::datatype_to_dataformat_converter(inputs.x.dtype());
    auto in1_df = tt::tt_metal::datatype_to_dataformat_converter(inputs.gate_proj.dtype());

    // Intermediate uses fp32 if dest-acc is enabled, bfloat16 otherwise.
    // For now default to bfloat16 (fine for bfloat4/bfloat8 inputs).
    auto interm_df = tt::DataFormat::Float16_b;
    auto out_df = tt::tt_metal::datatype_to_dataformat_converter(outputs[0].dtype());

    uint32_t in0_tile_size = tt::tile_size(in0_df);
    uint32_t in1_tile_size = tt::tile_size(in1_df);
    uint32_t interm_tile_size = tt::tile_size(interm_df);
    uint32_t out_tile_size = tt::tile_size(out_df);

    // ---------------------------------------------------------------
    // Derived counts
    // ---------------------------------------------------------------
    uint32_t K_num_blocks = p.K_tiles / p.K_block_tiles;
    uint32_t N_num_blocks = p.N_tiles / p.N_block_tiles;
    uint32_t M_num_blocks = p.M_tiles / p.M_block_tiles;

    uint32_t in0_block_tiles = p.M_block_tiles * p.K_block_tiles;
    uint32_t in1_block_tiles = p.K_block_tiles * p.N_block_tiles;
    uint32_t full_out_tiles = p.M_block_tiles * p.N_tiles;  // full N width per M block

    // ---------------------------------------------------------------
    // Circular buffers
    // ---------------------------------------------------------------
    // in0: one k-block at a time (M_block × K_block tiles), double-buffered
    {
        uint32_t cb_size = 2 * in0_block_tiles * in0_tile_size;
        auto cfg = tt::tt_metal::CircularBufferConfig(cb_size, {{CB_IN0, in0_df}}).set_page_size(CB_IN0, in0_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, core_set, cfg);
    }
    // in1_gate: one (k_block × n_block) at a time, double-buffered
    {
        uint32_t cb_size = 2 * in1_block_tiles * in1_tile_size;
        auto cfg = tt::tt_metal::CircularBufferConfig(cb_size, {{CB_IN1_GATE, in1_df}})
                       .set_page_size(CB_IN1_GATE, in1_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, core_set, cfg);
    }
    // in1_up: same
    {
        uint32_t cb_size = 2 * in1_block_tiles * in1_tile_size;
        auto cfg =
            tt::tt_metal::CircularBufferConfig(cb_size, {{CB_IN1_UP, in1_df}}).set_page_size(CB_IN1_UP, in1_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, core_set, cfg);
    }
    // gate_interm: holds partial sums for the full N width of one M block
    {
        uint32_t cb_size = full_out_tiles * interm_tile_size;
        auto cfg = tt::tt_metal::CircularBufferConfig(cb_size, {{CB_GATE_INT, interm_df}})
                       .set_page_size(CB_GATE_INT, interm_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, core_set, cfg);
    }
    // up_interm: same
    {
        uint32_t cb_size = full_out_tiles * interm_tile_size;
        auto cfg = tt::tt_metal::CircularBufferConfig(cb_size, {{CB_UP_INT, interm_df}})
                       .set_page_size(CB_UP_INT, interm_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, core_set, cfg);
    }
    // gate_out / up_out: one M block worth of output
    {
        uint32_t cb_size = full_out_tiles * out_tile_size;
        auto cfg = tt::tt_metal::CircularBufferConfig(cb_size, {{CB_GATE_OUT, out_df}})
                       .set_page_size(CB_GATE_OUT, out_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, core_set, cfg);
    }
    {
        uint32_t cb_size = full_out_tiles * out_tile_size;
        auto cfg =
            tt::tt_metal::CircularBufferConfig(cb_size, {{CB_UP_OUT, out_df}}).set_page_size(CB_UP_OUT, out_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, core_set, cfg);
    }

    // ---------------------------------------------------------------
    // Build compile-time args for each kernel
    // ---------------------------------------------------------------

    // --- BRISC: reader_x ---
    // ct args: [M_num_blocks, K_num_blocks, M_block_tiles, K_block_tiles, in0_tile_size,
    //           <TensorAccessor for x>]
    std::vector<uint32_t> reader_x_ct_args = {
        M_num_blocks, K_num_blocks, p.M_block_tiles, p.K_block_tiles, in0_tile_size};
    tt::tt_metal::TensorAccessorArgs(inputs.x.buffer()).append_to(reader_x_ct_args);

    auto reader_x_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/device/kernels/reader_x.cpp",
        core_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_x_ct_args, {}));

    // --- NCRISC: reader_weights_writer ---
    // ct args: [M_num_blocks, K_num_blocks, N_num_blocks, M_block_tiles, K_block_tiles, N_block_tiles,
    //           in1_tile_size, out_tile_size, N_tiles,
    //           <TensorAccessor gate_proj>, <TensorAccessor up_proj>,
    //           <TensorAccessor gate_out>, <TensorAccessor up_out>]
    std::vector<uint32_t> rw_ct_args = {
        M_num_blocks,
        K_num_blocks,
        N_num_blocks,
        p.M_block_tiles,
        p.K_block_tiles,
        p.N_block_tiles,
        in1_tile_size,
        out_tile_size,
        p.N_tiles};
    tt::tt_metal::TensorAccessorArgs(inputs.gate_proj.buffer()).append_to(rw_ct_args);
    tt::tt_metal::TensorAccessorArgs(inputs.up_proj.buffer()).append_to(rw_ct_args);
    tt::tt_metal::TensorAccessorArgs(outputs[0].buffer()).append_to(rw_ct_args);
    tt::tt_metal::TensorAccessorArgs(outputs[1].buffer()).append_to(rw_ct_args);

    auto reader_weights_writer_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/device/kernels/"
        "reader_weights_writer.cpp",
        core_set,
        tt::tt_metal::WriterDataMovementConfig(rw_ct_args, {}));

    // --- TRISC: compute_dual_gate_up ---
    // ct args: [M_num_blocks, K_num_blocks, N_num_blocks,
    //           M_block_tiles, K_block_tiles, N_block_tiles, N_tiles,
    //           subblock_h, subblock_w]
    std::vector<uint32_t> compute_ct_args = {
        M_num_blocks,
        K_num_blocks,
        N_num_blocks,
        p.M_block_tiles,
        p.K_block_tiles,
        p.N_block_tiles,
        p.N_tiles,
        p.subblock_h,
        p.subblock_w};

    auto compute_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/device/kernels/"
        "compute_dual_gate_up.cpp",
        core_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_ct_args});

    // ---------------------------------------------------------------
    // Runtime args
    // ---------------------------------------------------------------
    tt::tt_metal::SetRuntimeArgs(program, reader_x_id, core, {inputs.x.buffer()->address()});

    tt::tt_metal::SetRuntimeArgs(
        program,
        reader_weights_writer_id,
        core,
        {inputs.gate_proj.buffer()->address(),
         inputs.up_proj.buffer()->address(),
         outputs[0].buffer()->address(),
         outputs[1].buffer()->address()});

    // Compute kernel has no runtime args (all info is in compile-time args).

    return CreatedProgram{
        std::move(program), GateUpMatmulSharedVariables{reader_x_id, reader_weights_writer_id, compute_id, core}};
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
        shared_vars.emplace(coord_range, result.shared_variables);
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

        auto& x_args = GetRuntimeArgs(program, sv.reader_x_id, sv.core);
        x_args[0] = inputs.x.buffer()->address();

        auto& rw_args = GetRuntimeArgs(program, sv.reader_weights_writer_id, sv.core);
        rw_args[0] = inputs.gate_proj.buffer()->address();
        rw_args[1] = inputs.up_proj.buffer()->address();
        rw_args[2] = outputs[0].buffer()->address();
        rw_args[3] = outputs[1].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
