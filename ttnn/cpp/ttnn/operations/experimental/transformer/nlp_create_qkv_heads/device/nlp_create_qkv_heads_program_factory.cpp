// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "nlp_create_qkv_heads_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {

constexpr const char* INTERLEAVED_READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
    "reader_tm_tile_layout_nlp_create_qkv_heads.cpp";
constexpr const char* INTERLEAVED_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
    "writer_tm_tile_layout_nlp_create_qkv_heads.cpp";
constexpr const char* COMPUTE_TRANSPOSE_KERNEL_PATH = "ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp";
constexpr const char* SHARDED_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
    "reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts NlpCreateHeadsDeviceOperation::Interleaved::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& input_tensor = tensor_args.input_tensor_q;
    std::optional<const Tensor> input_tensor_kv = tensor_args.input_tensor_kv;
    const uint32_t num_q_heads = operation_attributes.num_q_heads;
    const uint32_t num_kv_heads = operation_attributes.num_kv_heads;
    const uint32_t head_dim = operation_attributes.head_dim;
    const bool transpose_k_heads = operation_attributes.transpose_k_heads;
    auto& output = tensor_return_value;
    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    const auto& input_shape = input_tensor.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = input_tensor.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    tt_metal::Buffer* in1_buffer = nullptr;
    if (read_from_input_tensor_kv) {
        in1_buffer = input_tensor_kv.value().buffer();
        TT_ASSERT(in1_buffer->size() % single_tile_size == 0);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_w_tiles = input_shape[3] / TILE_WIDTH;
    uint32_t in1_w_tiles = 0;
    if (read_from_input_tensor_kv) {
        in1_w_tiles = input_tensor_kv.value().padded_shape()[3] / TILE_WIDTH;
    }

    // Per output tensor args
    // Output shape for Q is: [B, num_q_heads, s, head_dim], shuffled from [B, 1, s, num_q_heads * head_dim]
    // Output shape for K/V is: [B, num_kv_heads, s, head_dim], shuffled from [B, 1, s, num_kv_heads * head_dim]
    // NOTE: Output h and w dims are identical for Q, K, V, so any arg that is related to these dims for q_* can be
    // shared for K, V
    uint32_t q_out_h_tiles = input_shape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = head_dim / TILE_WIDTH;  // tiles along head_dim
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = num_q_heads * q_out_HtWt;
    uint32_t kv_out_CHtWt = num_kv_heads * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;
    uint32_t kv_num_tiles = num_kv_heads * q_out_w_tiles;

    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of in0_w_tiles per core
    uint32_t num_blocks = input_shape[0] * input_shape[1] * input_shape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    ////////////////////////////////////////////////////////////////////////////
    //                      Output tensors
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Tensor& q = std::get<0>(output);
    tt_metal::Tensor& k = std::get<1>(output);
    tt_metal::Tensor& v = std::get<2>(output);

    TT_ASSERT(q.buffer() != nullptr, "Output q buffer should be allocated on device!");
    TT_ASSERT(k.buffer() != nullptr, "Output k buffer should be allocated on device!");
    TT_ASSERT(v.buffer() != nullptr, "Output v buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    // CB indices kept identical to the legacy program: cb1 carries Q/V (and K when not transposing); cb0 is
    // the K input for compute and cb16 the transposed K output, used only when transpose_k_heads is set.
    [[maybe_unused]] constexpr uint32_t src1_cb_index =
        1;  // cb0 is needed for compute if we want to use generic transpose_wh compute kernel
    [[maybe_unused]] constexpr uint32_t src0_cb_index = 0;
    [[maybe_unused]] constexpr uint32_t out_cb_index = 16;

    // Create circular buffers (DFBs)
    uint32_t micro_block_size = 1;                 // Num tiles to read/wait for in reader and writer
    uint32_t cb_num_tiles = micro_block_size * 4;  // Quadruple buffer everything

    // TODO: Investigate perf allocating full in0_w_tiles with double buffer
    // uint32_t cb1_num_tiles = in0_w_tiles * 2; // double buffer; this runs out of space for generic shapes
    uint32_t cb1_num_tiles = cb_num_tiles;

    std::vector<m2::DataflowBufferSpec> dfbs;
    dfbs.push_back(m2::DataflowBufferSpec{
        .unique_id = m2::DFBSpecName{"qv"},
        .entry_size = single_tile_size,
        .num_entries = cb1_num_tiles,
        .data_format_metadata = cb_data_format,
    });
    if (transpose_k_heads) {
        // If we transpose_k_heads:
        // - reader will write to cb0, instead of cb1
        // - compute will wait on cb0 and write to cb16
        // - writer will wait on cb 16, instead of cb1
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"k_in"},
            .entry_size = single_tile_size,
            .num_entries = cb_num_tiles,
            .data_format_metadata = cb_data_format,
        });
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"k_out"},
            .entry_size = single_tile_size,
            .num_entries = cb_num_tiles,
            .data_format_metadata = cb_data_format,
        });
    }

    // ---- kernels ----
    m2::KernelSpec::CompilerOptions reader_opts;
    m2::KernelSpec::CompilerOptions writer_opts;
    if (transpose_k_heads) {
        reader_opts.defines.emplace("TRANSPOSE_K_HEADS", "1");
        writer_opts.defines.emplace("TRANSPOSE_K_HEADS", "1");
    }
    if (read_from_input_tensor_kv) {
        reader_opts.defines.emplace("READ_FROM_INPUT_TENSOR_KV", "1");
    }

    std::vector<m2::KernelSpec> kernels;

    if (transpose_k_heads) {
        // For FLOAT32 input, enable fp32 dest accumulation so the JIT data-format selection
        // resolves the unpack-dst CB to Tf32 (10-bit mantissa) instead of Float16_b (7-bit
        // mantissa). Mirrors the per-dtype promotion in eltwise unary/binary primitives.
        const bool fp32_dest_acc_en = input_tensor.dtype() == tt_metal::DataType::FLOAT32;

        // Compute kernel reads K from cb0 (k_in) and writes transposed K to cb16 (k_out). The legacy
        // factory split the compute kernel across two core groups so each carries its own per-block tile
        // count as a compile-time arg; we preserve that by emitting one KernelSpec per non-empty group.
        m2::KernelSpec compute_1{
            .unique_id = m2::KernelSpecName{"compute_1"},
            .source = std::filesystem::path{COMPUTE_TRANSPOSE_KERNEL_PATH},
            .dfb_bindings =
                {m2::ConsumerOf(m2::DFBSpecName{"k_in"}, "k_in"), m2::ProducerOf(m2::DFBSpecName{"k_out"}, "k_out")},
            .compile_time_args = {{"per_core_block_cnt", num_blocks_per_core_group_1 * kv_num_tiles}},
            .hw_config = m2::ComputeHardwareConfig{.fp32_dest_acc_en = fp32_dest_acc_en},
        };
        kernels.push_back(std::move(compute_1));

        if (core_group_2.num_cores() > 0) {
            m2::KernelSpec compute_2{
                .unique_id = m2::KernelSpecName{"compute_2"},
                .source = std::filesystem::path{COMPUTE_TRANSPOSE_KERNEL_PATH},
                .dfb_bindings =
                    {m2::ConsumerOf(m2::DFBSpecName{"k_in"}, "k_in"),
                     m2::ProducerOf(m2::DFBSpecName{"k_out"}, "k_out")},
                .compile_time_args = {{"per_core_block_cnt", num_blocks_per_core_group_2 * kv_num_tiles}},
                .hw_config = m2::ComputeHardwareConfig{.fp32_dest_acc_en = fp32_dest_acc_en},
            };
            kernels.push_back(std::move(compute_2));
        }
    }

    // Reader: produces Q/V into the qv DFB, and K into either qv (no transpose) or k_in (transpose).
    // When not transposing, K shares the qv DFB. Metal 2.0 forbids a kernel binding the same DFB twice in
    // the same role, so we emit a single qv ProducerOf and the kernel aliases dfb::cb_k -> dfb::cb_qv under
    // the (absent) TRANSPOSE_K_HEADS guard. When transposing, K gets its own k_in DFB and a distinct binding.
    std::vector<m2::DFBBinding> reader_dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"qv"}, "cb_qv")};
    if (transpose_k_heads) {
        reader_dfb_bindings.push_back(m2::ProducerOf(m2::DFBSpecName{"k_in"}, "cb_k"));
    }
    std::vector<m2::TensorBinding> reader_tensor_bindings = {
        m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"in0"}, .accessor_name = "in0"}};
    if (read_from_input_tensor_kv) {
        reader_tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"in1"}, .accessor_name = "in1"});
    }

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{INTERLEAVED_READER_KERNEL_PATH},
        .compiler_options = std::move(reader_opts),
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args = {{"q_num_tiles", q_num_tiles}, {"kv_num_tiles", kv_num_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_blocks", "in0_tensor_tile_id", "in1_tensor_tile_id"}},
        // Reader on NCRISC (RISCV_1 / NOC1), writer on BRISC — mirrors the legacy Reader/Writer configs so
        // the two data-movement kernels don't collide on the same DM processor.
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    // Writer: consumes Q/V from qv, K from qv (no transpose) or k_out (transpose), writes to Q/K/V outputs.
    // Same single-binding rule as the reader: in the non-transpose case the writer binds qv once and the
    // kernel aliases dfb::cb_k -> dfb::cb_qv; when transposing it consumes K from the dedicated k_out DFB.
    std::vector<m2::DFBBinding> writer_dfb_bindings = {m2::ConsumerOf(m2::DFBSpecName{"qv"}, "cb_qv")};
    if (transpose_k_heads) {
        writer_dfb_bindings.push_back(m2::ConsumerOf(m2::DFBSpecName{"k_out"}, "cb_k"));
    }

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{INTERLEAVED_WRITER_KERNEL_PATH},
        .compiler_options = std::move(writer_opts),
        .dfb_bindings = std::move(writer_dfb_bindings),
        // TODO: Q, K, V doesn't necessarily need to be the same output mem config
        .tensor_bindings =
            {m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"q"}, .accessor_name = "q"},
             m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"k"}, .accessor_name = "k"},
             m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"v"}, .accessor_name = "v"}},
        .compile_time_args =
            {{"q_out_h_tiles", q_out_h_tiles},
             {"q_out_w_tiles", q_out_w_tiles},
             {"q_out_HtWt", q_out_HtWt},
             {"q_out_c", num_q_heads},
             {"kv_out_c", num_kv_heads}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_blocks", "q_out_h_dim", "q_out_tensor_tile_id", "k_out_tensor_tile_id", "v_out_tensor_tile_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));

    // ---- ProgramSpec ----
    m2::ProgramSpec spec;
    spec.name = "nlp_create_qkv_heads_interleaved";
    spec.kernels = std::move(kernels);
    spec.dataflow_buffers = std::move(dfbs);
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"in0"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"q"}, .spec = q.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"k"}, .spec = k.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"v"}, .spec = v.tensor_spec()}};
    // The optional kv input gets a TensorParameter only when present; structural, so it's fine for it to
    // differ across cache entries (extract is gated on the same has_value() everywhere).
    if (read_from_input_tensor_kv) {
        spec.tensor_parameters.push_back(m2::TensorParameter{
            .unique_id = m2::TensorParamName{"in1"}, .spec = input_tensor_kv.value().tensor_spec()});
    }

    // WorkUnit: reader + writer (+ compute kernels when transposing) on all_cores. The compute kernels are
    // placed on their own core groups (split_work_to_cores), but the WorkUnitSpec carries the full
    // node set; per-kernel placement is derived from the kernel's runtime-arg coverage.
    m2::WorkUnitSpec work_unit{.name = "nlp_create_qkv_heads", .target_nodes = all_cores};
    if (transpose_k_heads) {
        work_unit.kernels.push_back(m2::KernelSpecName{"compute_1"});
        if (core_group_2.num_cores() > 0) {
            work_unit.kernels.push_back(m2::KernelSpecName{"compute_2"});
        }
    }
    work_unit.kernels.push_back(m2::KernelSpecName{"reader"});
    work_unit.kernels.push_back(m2::KernelSpecName{"writer"});
    spec.work_units = {std::move(work_unit)};

    // ---- run-args (degenerate: complete set) ----
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * q_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t v_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t k_out_tensor_tile_id = transpose_k_heads
                                            ? (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + q_out_h_dim
                                            : v_out_tensor_tile_id;

        reader_args.runtime_arg_values.push_back(
            {core,
             {{"num_blocks", num_blocks_per_core},
              {"in0_tensor_tile_id", num_blocks_written * in0_w_tiles},
              {"in1_tensor_tile_id", num_blocks_written * in1_w_tiles}}});

        writer_args.runtime_arg_values.push_back(
            {core,
             {{"num_blocks", num_blocks_per_core},
              {"q_out_h_dim", q_out_h_dim},
              {"q_out_tensor_tile_id", q_out_tensor_tile_id},
              {"k_out_tensor_tile_id", k_out_tensor_tile_id},
              {"v_out_tensor_tile_id", v_out_tensor_tile_id}}});

        num_blocks_written += num_blocks_per_core;
    }

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args.push_back(std::move(reader_args));
    run_params.kernel_run_args.push_back(std::move(writer_args));
    run_params.tensor_args.emplace(
        m2::TensorParamName{"in0"}, m2::ProgramRunArgs::TensorArgument{std::cref(input_tensor.mesh_tensor())});
    if (read_from_input_tensor_kv) {
        run_params.tensor_args.emplace(
            m2::TensorParamName{"in1"},
            m2::ProgramRunArgs::TensorArgument{std::cref(input_tensor_kv.value().mesh_tensor())});
    }
    run_params.tensor_args.emplace(
        m2::TensorParamName{"q"}, m2::ProgramRunArgs::TensorArgument{std::cref(q.mesh_tensor())});
    run_params.tensor_args.emplace(
        m2::TensorParamName{"k"}, m2::ProgramRunArgs::TensorArgument{std::cref(k.mesh_tensor())});
    run_params.tensor_args.emplace(
        m2::TensorParamName{"v"}, m2::ProgramRunArgs::TensorArgument{std::cref(v.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_params)};
}

namespace {

// Shared structural derivation for the Sharded variant. A pure function of the operation attributes +
// the input/output tensor layouts — the per-core source ADDRESSES (which can change per dispatch) are
// computed separately, in create_per_enqueue_args.
struct ShardedDerived {
    CoreRangeSet q_cores;
    CoreRangeSet k_cores;
    uint32_t q_num_tiles = 0;
    uint32_t k_num_tiles = 0;
    uint32_t v_num_tiles = 0;
    uint32_t single_tile_size = 0;
    tt::DataFormat cb_data_format = tt::DataFormat::Invalid;
    uint32_t num_cores = 0;
    uint32_t num_cores_x = 0;
    uint32_t num_cores_y = 0;
};

ShardedDerived derive_sharded(
    const NlpCreateHeadsDeviceOperation::operation_attributes_t& /*operation_attributes*/,
    const NlpCreateHeadsDeviceOperation::tensor_args_t& tensor_args,
    NlpCreateHeadsDeviceOperation::tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input_tensor_q;
    ShardedDerived d;
    d.cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    d.single_tile_size = tt::tile_size(d.cb_data_format);

    // CB indices c_16/c_17/c_18 are no longer baked here — the kernel reads them from its DFB binding
    // tokens (dfb::cb_q_out / dfb::cb_kv_out). The per-tensor tile counts and core grids remain structural.
    auto q_shard_spec = std::get<0>(output).shard_spec().value();
    d.q_cores = q_shard_spec.grid;
    d.q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;

    auto k_shard_spec = std::get<1>(output).shard_spec().value();
    d.k_cores = k_shard_spec.grid;
    d.k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    auto v_shard_spec = std::get<0>(output).shard_spec().value();
    d.v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    d.num_cores = std::max(d.q_cores.num_cores(), d.k_cores.num_cores());
    auto core_grid = d.q_cores.bounding_box();
    d.num_cores_x = core_grid.end_coord.x + 1;
    d.num_cores_y = core_grid.end_coord.y + 1;
    return d;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts NlpCreateHeadsDeviceOperation::Sharded::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& output = tensor_return_value;
    const auto d = derive_sharded(operation_attributes, tensor_args, output);

    // Q/K/V output DFBs back onto the output tensors (borrowed memory). c_16 / c_17 / c_18.
    std::vector<m2::DataflowBufferSpec> dfbs = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"q_out"},
            .entry_size = d.single_tile_size,
            .num_entries = d.q_num_tiles,
            .data_format_metadata = d.cb_data_format,
            .borrowed_from = m2::TensorParamName{"q"},
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"k_out"},
            .entry_size = d.single_tile_size,
            .num_entries = d.k_num_tiles,
            .data_format_metadata = d.cb_data_format,
            .borrowed_from = m2::TensorParamName{"k"},
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"v_out"},
            .entry_size = d.single_tile_size,
            .num_entries = d.v_num_tiles,
            .data_format_metadata = d.cb_data_format,
            .borrowed_from = m2::TensorParamName{"v"},
        },
    };

    // Reader produces Q (cb_q_out) and K (cb_kv_out); writer produces Q (cb_q_out) and V (cb_kv_out). Same
    // source compiled twice, bound to different output DFBs as the second endpoint, exactly as the legacy
    // program did with different CB-index compile-time args. The kernel reads every per-core runtime value
    // positionally (incl. a variable-length NOC coord array), so they are all passed as varargs; the CB
    // ids come from the DFB binding tokens (dfb::).
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{SHARDED_KERNEL_PATH},
        .dfb_bindings =
            {m2::ProducerOf(m2::DFBSpecName{"q_out"}, "cb_q_out"),
             m2::ProducerOf(m2::DFBSpecName{"k_out"}, "cb_kv_out")},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
        .advanced_options = m2::KernelAdvancedOptions{.num_runtime_varargs = 19 + d.num_cores_x + d.num_cores_y},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{SHARDED_KERNEL_PATH},
        .dfb_bindings =
            {m2::ProducerOf(m2::DFBSpecName{"q_out"}, "cb_q_out"),
             m2::ProducerOf(m2::DFBSpecName{"v_out"}, "cb_kv_out")},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
        .advanced_options = m2::KernelAdvancedOptions{.num_runtime_varargs = 19 + d.num_cores_x + d.num_cores_y},
    };

    m2::ProgramSpec spec;
    spec.name = "nlp_create_qkv_heads_sharded";
    spec.kernels = {std::move(reader), std::move(writer)};
    spec.dataflow_buffers = std::move(dfbs);
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"q"}, .spec = std::get<0>(output).tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"k"}, .spec = std::get<1>(output).tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"v"}, .spec = std::get<2>(output).tensor_spec()}};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "nlp_create_qkv_heads_sharded",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
        .target_nodes = d.q_cores}};

    // All runtime args (incl. the input addresses) are per-enqueue; create_program_artifacts contributes
    // no run-args of its own (only the tensor bindings, supplied by create_per_enqueue_args).
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = m2::ProgramRunArgs{}};
}

m2::ProgramRunArgs NlpCreateHeadsDeviceOperation::Sharded::create_per_enqueue_args(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const auto& input_tensor = tensor_args.input_tensor_q;
    const auto& input_tensor_kv = tensor_args.input_tensor_kv;
    auto& output = tensor_return_value;
    auto head_dim = operation_attributes.head_dim;
    auto num_q_heads = operation_attributes.num_q_heads;
    auto num_kv_heads = operation_attributes.num_kv_heads;

    tt_metal::IDevice* device = input_tensor.device();
    const auto d = derive_sharded(operation_attributes, tensor_args, output);

    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * d.single_tile_size;

    uint32_t per_core_out_q_heads = num_q_heads / d.q_cores.num_cores();
    uint32_t per_risc0_out_q_heads = div_up(per_core_out_q_heads, 2);
    uint32_t per_risc1_out_q_heads = per_core_out_q_heads / 2;
    uint32_t per_core_in_q_heads = num_q_heads / input_tensor.shard_spec().value().num_cores();

    uint32_t per_core_out_kv_heads = num_kv_heads / d.k_cores.num_cores();
    uint32_t per_core_in_kv_heads =
        num_kv_heads / (read_from_input_tensor_kv ? input_tensor_kv.value().shard_spec().value().num_cores()
                                                  : input_tensor.shard_spec().value().num_cores());

    uint32_t q_base_addr = input_tensor.buffer()->address();
    uint32_t k_base_addr = 0;
    if (read_from_input_tensor_kv) {
        k_base_addr = input_tensor_kv.value().buffer()->address();
    } else {
        k_base_addr = q_base_addr + per_core_in_q_heads * head_tiles * d.single_tile_size;
    }
    uint32_t v_base_addr = k_base_addr + (per_core_in_kv_heads * head_tiles * d.single_tile_size);

    const uint32_t num_cores = d.num_cores;
    const uint32_t num_cores_x = d.num_cores_x;
    const uint32_t num_cores_y = d.num_cores_y;

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(num_cores_x);
    for (uint32_t x = 0; x < num_cores_x; ++x) {
        noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(num_cores_y);
    for (uint32_t y = 0; y < num_cores_y; ++y) {
        noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    uint32_t remote_q_head_start_idx = 0;
    uint32_t remote_kv_head_start_idx = 0;
    uint32_t q_x = 0, q_y = 0, kv_x = 0, kv_y = 0;
    uint32_t q_start_addr = q_base_addr;
    uint32_t k_start_addr = k_base_addr;
    uint32_t v_start_addr = v_base_addr;

    uint32_t remote_q_read = 0;
    uint32_t remote_kv_read = 0;

    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        bool read_kv_heads = i < d.k_cores.num_cores();
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(19 + num_cores_x + num_cores_y);
        reader_runtime_args = {
            head_size,
            per_risc0_out_q_heads,
            per_core_in_q_heads,
            remote_q_head_start_idx,
            q_x,
            q_y,
            q_base_addr,
            q_start_addr,
            0,
            read_kv_heads,
            per_core_out_kv_heads,
            per_core_in_kv_heads,
            remote_kv_head_start_idx,
            kv_x,
            kv_y,
            k_base_addr,
            k_start_addr,
            d.k_num_tiles,
            num_cores_x,
        };
        reader_runtime_args.insert(reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
        reader_runtime_args.insert(reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());

        remote_q_read += per_risc0_out_q_heads;
        q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
        q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
        remote_q_head_start_idx = (remote_q_head_start_idx + per_risc0_out_q_heads) % per_core_in_q_heads;
        q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;

        reader_args.advanced_options.runtime_varargs.emplace(core, reader_runtime_args);

        reader_runtime_args[1] = per_risc1_out_q_heads;
        reader_runtime_args[3] = remote_q_head_start_idx;
        reader_runtime_args[4] = q_x;
        reader_runtime_args[5] = q_y;
        reader_runtime_args[7] = q_start_addr;
        reader_runtime_args[8] = per_risc0_out_q_heads * head_size;

        if (per_risc1_out_q_heads > 0) {
            remote_q_read += per_risc1_out_q_heads;
            q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
            q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
            remote_q_head_start_idx = (per_risc1_out_q_heads + remote_q_head_start_idx) % per_core_in_q_heads;
            q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;
        }

        if (read_kv_heads) {
            reader_runtime_args[15] = v_base_addr;
            reader_runtime_args[16] = v_start_addr;
            remote_kv_read += per_core_out_kv_heads;
            kv_y = (remote_kv_read / per_core_in_kv_heads) / num_cores_x;
            kv_x = (remote_kv_read / per_core_in_kv_heads) % num_cores_x;
            remote_kv_head_start_idx = (remote_kv_head_start_idx + per_core_out_kv_heads) % per_core_in_kv_heads;
            k_start_addr = k_base_addr + remote_kv_head_start_idx * head_size;
            v_start_addr = v_base_addr + remote_kv_head_start_idx * head_size;
        }

        writer_args.advanced_options.runtime_varargs.emplace(core, std::move(reader_runtime_args));
    }

    m2::ProgramRunArgs args;
    args.kernel_run_args.push_back(std::move(reader_args));
    args.kernel_run_args.push_back(std::move(writer_args));
    args.tensor_args.emplace(
        m2::TensorParamName{"q"}, m2::ProgramRunArgs::TensorArgument{std::cref(std::get<0>(output).mesh_tensor())});
    args.tensor_args.emplace(
        m2::TensorParamName{"k"}, m2::ProgramRunArgs::TensorArgument{std::cref(std::get<1>(output).mesh_tensor())});
    args.tensor_args.emplace(
        m2::TensorParamName{"v"}, m2::ProgramRunArgs::TensorArgument{std::cref(std::get<2>(output).mesh_tensor())});
    return args;
}

}  // namespace ttnn::operations::experimental::transformer
