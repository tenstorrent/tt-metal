// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attention_qkv_decode_device_operation.hpp"

#include <array>
#include <bit>
#include <map>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek::attention_qkv_decode {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

constexpr const char* kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/attention_qkv_decode/device/kernels/";

const TensorParamName TOKENS{"tokens"};
const TensorParamName W_QA{"w_q_a"};
const TensorParamName W_KV{"w_kv"};
const TensorParamName W_QB{"w_q_b"};
const TensorParamName Q_A_GAMMA{"q_a_gamma"};
const TensorParamName KV_GAMMA{"kv_gamma"};
const TensorParamName COS{"cos"};
const TensorParamName SIN{"sin"};
const TensorParamName KV_CACHE{"kv_cache"};
const TensorParamName CACHE_POS{"cache_pos"};
const TensorParamName Q{"q"};
const TensorParamName W_COMP{"w_comp"};
const TensorParamName COMP_OUT{"comp_output"};

const DFBSpecName IN0_DFB{"in0"};
const DFBSpecName W_PROJ_DFB{"w_proj"};
// With pipes, kv's relay: a relay's producer kernel must span every receiver of the pipes it covers.
const DFBSpecName W_KV_DFB{"w_kv"};
const DFBSpecName W_QB_DFB{"w_qb"};
const DFBSpecName X_DFB{"x"};
const DFBSpecName ROW_DFB{"qkv_row"};
const DFBSpecName MM_OUT_DFB{"mm_out"};
const DFBSpecName STAT_DFB{"stat"};
const DFBSpecName SCALE_DFB{"scale"};
const DFBSpecName TMP_DFB{"tmp"};
const DFBSpecName GAMMA_DFB{"gamma"};
const DFBSpecName Q_A_GAMMA_DFB{"q_a_gamma"};
const DFBSpecName CS_DFB{"cs"};
const DFBSpecName CS_RAW_DFB{"cs_raw"};
const DFBSpecName OUT_DFB{"out"};
const DFBSpecName CACHE_TILE_DFB{"cache_tile"};
const DFBSpecName POS_DFB{"pos"};
const DFBSpecName W_COMP_DFB{"w_comp"};
const DFBSpecName COMP_X_DFB{"comp_x"};

const SemaphoreSpecName ARRIVE_SEM{"arrive"};
const SemaphoreSpecName READY_SEM{"ready"};

const KernelSpecName QA_READER{"qa_reader"};
const KernelSpecName KV_READER{"kv_reader"};
const KernelSpecName QA_COMPUTE{"qa_compute"};
const KernelSpecName KV_COMPUTE{"kv_compute"};
const KernelSpecName KV_WRITER{"kv_writer"};
const KernelSpecName QB_READER{"qb_reader"};
const KernelSpecName QB_COMPUTE{"qb_compute"};
const KernelSpecName QB_WRITER{"qb_writer"};

DataMovementHardwareConfig dm_config(DataMovementProcessor processor, NOC noc) {
    return DataMovementHardwareConfig{
        .config_1xx = DataMovementHardwareConfig::DataMovement1XXConfig{.processor = processor, .noc = noc},
    };
}

ComputeHardwareConfig compute_config() {
    // custom_mm is LoFi (exact for a bf4 weight against bf16 activations) and needs a 16-bit DST.
    return ComputeHardwareConfig{.fpu_math_fidelity = MathFidelity::LoFi, .enable_32_bit_dest = false};
}

std::string kernel_path(const char* file) { return std::string(kKernelDir) + file; }

}  // namespace

ttnn::device_operation::ProgramArtifacts AttentionQkvDecodeDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes, const tensor_args_t& t, tensor_return_value_t& q_out) {
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;
    auto* device = t.tokens.device();

    const uint32_t k_tiles = t.tokens.logical_shape()[-1] / TILE_W;
    const uint32_t q_lora_tiles = t.w_q_a.logical_shape()[-1] / TILE_W;
    const uint32_t head_dim = t.w_kv.logical_shape()[-1];
    const uint32_t head_tiles = head_dim / TILE_W;
    const uint32_t nc_tiles = t.w_q_b.logical_shape()[-1] / TILE_W / kNumQbCores;
    const uint32_t rope_tiles = t.cos.logical_shape()[-1] / TILE_W;
    // The gathered row: q_a, kv, then q_a times its RMSNorm gamma (what q_b multiplies).
    const uint32_t scaled_q_a_first = q_lora_tiles + head_tiles;
    const uint32_t row_tiles = scaled_q_a_first + q_lora_tiles;

    const tt::tt_metal::Tile row_tile({1, TILE_W});
    const tt::DataFormat bf16 = tt::DataFormat::Float16_b;
    const uint32_t row_tile_bytes = row_tile.get_tile_size(bf16);
    const tt::DataFormat w_format = datatype_to_dataformat_converter(t.w_q_a.dtype());
    const tt::DataFormat w_qb_format = datatype_to_dataformat_converter(t.w_q_b.dtype());
    const uint32_t w_tile_bytes = tt::tile_size(w_format);
    const uint32_t w_qb_tile_bytes = tt::tile_size(w_qb_format);
    TT_FATAL(
        datatype_to_dataformat_converter(t.w_kv.dtype()) == w_format,
        "attention_qkv_decode: w_q_a and w_kv must share a dtype");
    const uint32_t cache_tile_bytes = tt::tile_size(bf16);
    const uint32_t pos_page_bytes = t.cache_pos.buffer()->aligned_page_size();

    // Roles. Each list is in slab order: core i reads shard i of its weight.
    const std::vector<CoreCoord> qb_cores =
        corerange_to_cores(CoreRangeSet(CoreRange({0, 0}, {kRowCoresX - 1, kQbCoresY - 1})), std::nullopt, true);
    const std::vector<CoreCoord> kv_cores =
        corerange_to_cores(CoreRangeSet(CoreRange({0, kQbCoresY}, {kRowCoresX - 1, kGridY - 1})), std::nullopt, true);
    const std::vector<CoreCoord> qa_cores =
        corerange_to_cores(CoreRangeSet(CoreRange({kRowCoresX, 0}, {kGridX - 1, kGridY - 1})), std::nullopt, true);
    TT_FATAL(qa_cores.size() == q_lora_tiles && kv_cores.size() == head_tiles, "attention_qkv_decode: core roles");
    // q_b core i also computes column tile comp_slot[i] of tokens @ w_comp when is_comp_core[i].
    const bool has_comp = t.w_comp.has_value();
    std::vector<bool> is_comp_core(qb_cores.size(), false);
    std::vector<uint32_t> comp_slot(qb_cores.size(), 0);
    if (has_comp) {
        const auto indices = comp_core_indices(t.w_comp->logical_shape()[-1] / TILE_W);
        for (uint32_t j = 0; j < indices.size(); ++j) {
            is_comp_core[indices[j]] = true;
            comp_slot[indices[j]] = j;
        }
    }

    const CoreCoord hub = qa_cores.front();
    const CoreCoord hub_phys = device->worker_core_from_logical_core(hub);
    const CoreCoord row_start_phys = device->worker_core_from_logical_core({0, 0});
    const CoreCoord row_end_phys = device->worker_core_from_logical_core({kRowCoresX - 1, kGridY - 1});
    const uint32_t num_row_dests = kRowCoresX * kGridY;

    Group<KernelSpec> kernels;
    ProgramRunArgs run_args;

    // With pipes, each core's whole weight slab is one entry of its role's ring, and the weight DFBs are
    // relays over those rings rather than buffers the readers fill from DRAM.
    const auto& prefetcher_pipes = operation_attributes.prefetcher_pipes;
    const bool use_pipes = !prefetcher_pipes.empty();
    const std::array<uint32_t, kNumWeightRoles> slab_bytes = {
        k_tiles * w_tile_bytes, k_tiles * w_tile_bytes, q_lora_tiles * nc_tiles * w_qb_tile_bytes};
    std::array<std::vector<PrefetcherPipeParamName>, kNumWeightRoles> role_pipe_names;
    std::array<uint32_t, kNumWeightRoles> role_ring_size{};
    Group<PrefetcherPipeParameter> pipe_parameters;
    for (size_t p = 0; p < prefetcher_pipes.size(); ++p) {
        PrefetcherPipe& pipe = *prefetcher_pipes[p];
        uint32_t role = 0;
        while (!weight_role_cores(static_cast<WeightRole>(role)).contains(pipe.receiver_cores())) {
            ++role;
        }
        const PrefetcherPipeParamName name{fmt::format("w_pipe_{}", p)};
        role_pipe_names[role].push_back(name);
        role_ring_size[role] = pipe.ring_size();
        pipe_parameters.push_back(PrefetcherPipeParameter{
            .unique_id = name,
            .receivers = pipe.receiver_cores(),
            .ring_size = pipe.ring_size(),
            .entry_size = slab_bytes[role],
        });
        run_args.advanced_options.prefetcher_pipe_args.emplace(name, PrefetcherPipeArgument{pipe});
    }
    auto relay_tiles = [&](WeightRole role, uint32_t tile_bytes) {
        const uint32_t ring = role_ring_size[static_cast<uint32_t>(role)];
        return (ring - ring % slab_bytes[static_cast<uint32_t>(role)]) / tile_bytes;
    };
    const auto& qa_pipe_names = role_pipe_names[static_cast<uint32_t>(WeightRole::QA)];
    const auto& kv_pipe_names = role_pipe_names[static_cast<uint32_t>(WeightRole::KV)];
    const auto& qb_pipe_names = role_pipe_names[static_cast<uint32_t>(WeightRole::QB)];

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    auto row_dfb = [&](const DFBSpecName& name, uint32_t num_entries) {
        return DataflowBufferSpec{
            .unique_id = name,
            .entry_size = row_tile_bytes,
            .num_entries = num_entries,
            .data_format_metadata = bf16,
            .tile_format_metadata = row_tile,
        };
    };
    // Declared widest node set first: every core, then the q_b and kv cores, then q_a and kv, q_b and
    // q_a, then a single role. A DFB takes the highest watermark among its own core ranges, and an
    // allocation on a smaller range does not always raise a wider range it intersects, so a wide DFB
    // declared after a narrow one can land on top of it.
    Group<DataflowBufferSpec> dataflow_buffers = {
        // Gathered on the hub, multicast to the q_b and kv cores. One address on every core, so a
        // producer addresses the hub's copy with its own write pointer.
        row_dfb(ROW_DFB, row_tiles),
        row_dfb(STAT_DFB, 1),
        row_dfb(SCALE_DFB, 1),
        row_dfb(TMP_DFB, rope_tiles),
        // cos tiles then sin tiles, row 0 of the tables' 32x32 tiles as 1x32 tiles.
        row_dfb(CS_DFB, 2 * rope_tiles),
        // Row 0 of each table face, staged as whole DRAM-aligned 64-byte reads.
        row_dfb(CS_RAW_DFB, 2 * 2 * rope_tiles),
        row_dfb(OUT_DFB, nc_tiles),
        // The tokens replica, read in place as K 1x32 tiles.
        DataflowBufferSpec{
            .unique_id = IN0_DFB,
            .entry_size = row_tile_bytes,
            .num_entries = k_tiles,
            .data_format_metadata = bf16,
            .tile_format_metadata = row_tile,
            .borrowed_from = TOKENS,
        },
        DataflowBufferSpec{
            .unique_id = W_PROJ_DFB,
            .entry_size = w_tile_bytes,
            .num_entries = use_pipes ? relay_tiles(WeightRole::QA, w_tile_bytes) : k_tiles,
            .data_format_metadata = w_format,
            .advanced_options = {.prefetcher_pipe_relays = qa_pipe_names},
        },
        // A q_a core's tile of q_a, then the same times gamma; a kv core's tile of kv.
        row_dfb(X_DFB, 2),
        row_dfb(MM_OUT_DFB, nc_tiles),
        DataflowBufferSpec{
            .unique_id = W_QB_DFB,
            .entry_size = w_qb_tile_bytes,
            .num_entries = use_pipes ? relay_tiles(WeightRole::QB, w_qb_tile_bytes) : q_lora_tiles * nc_tiles,
            .data_format_metadata = w_qb_format,
            .advanced_options = {.prefetcher_pipe_relays = qb_pipe_names},
        },
        DataflowBufferSpec{
            .unique_id = Q_A_GAMMA_DFB,
            .entry_size = row_tile_bytes,
            .num_entries = 1,
            .data_format_metadata = bf16,
            .tile_format_metadata = row_tile,
            .borrowed_from = Q_A_GAMMA,
        },
        DataflowBufferSpec{
            .unique_id = GAMMA_DFB,
            .entry_size = row_tile_bytes,
            .num_entries = 1,
            .data_format_metadata = bf16,
            .tile_format_metadata = row_tile,
            .borrowed_from = KV_GAMMA,
        },
        DataflowBufferSpec{
            .unique_id = CACHE_TILE_DFB,
            .entry_size = cache_tile_bytes,
            .num_entries = 1,
            .data_format_metadata = bf16,
        },
        DataflowBufferSpec{
            .unique_id = POS_DFB,
            .entry_size = std::max(pos_page_bytes, row_tile_bytes),
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::Int32,
        },
    };
    if (use_pipes) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = W_KV_DFB,
            .entry_size = w_tile_bytes,
            .num_entries = relay_tiles(WeightRole::KV, w_tile_bytes),
            .data_format_metadata = w_format,
            .advanced_options = {.prefetcher_pipe_relays = kv_pipe_names},
        });
    }
    if (has_comp) {
        // On the q_b cores, after every wider DFB. Only the compressor-column cores fill them.
        const tt::DataFormat w_comp_format = datatype_to_dataformat_converter(t.w_comp->dtype());
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = W_COMP_DFB,
            .entry_size = tt::tile_size(w_comp_format),
            .num_entries = k_tiles,
            .data_format_metadata = w_comp_format,
        });
        dataflow_buffers.push_back(row_dfb(COMP_X_DFB, 1));
    }

    const CoreRangeSet all_cores(CoreRange({0, 0}, {kGridX - 1, kGridY - 1}));
    Group<SemaphoreSpec> semaphores = {
        SemaphoreSpec{.unique_id = ARRIVE_SEM, .target_nodes = all_cores},
        SemaphoreSpec{.unique_id = READY_SEM, .target_nodes = all_cores},
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t inv_q_lora_bits = std::bit_cast<uint32_t>(1.0f / static_cast<float>(q_lora_tiles * TILE_W));
    const uint32_t inv_head_dim_bits = std::bit_cast<uint32_t>(1.0f / static_cast<float>(head_dim));
    const uint32_t eps_bits = std::bit_cast<uint32_t>(operation_attributes.epsilon);

    // ---- q_a / kv projection readers (RISCV_1, NOC 0): weight slab, row gather, hub multicast ----
    const KernelSpec::CompileTimeArgs proj_reader_cta = {
        {"k_tiles", k_tiles},
        {"row_tiles", row_tiles},
        {"row_tile_bytes", row_tile_bytes},
        {"hub_noc_x", static_cast<uint32_t>(hub_phys.x)},
        {"hub_noc_y", static_cast<uint32_t>(hub_phys.y)},
        {"row_start_noc_x", static_cast<uint32_t>(row_start_phys.x)},
        {"row_start_noc_y", static_cast<uint32_t>(row_start_phys.y)},
        {"row_end_noc_x", static_cast<uint32_t>(row_end_phys.x)},
        {"row_end_noc_y", static_cast<uint32_t>(row_end_phys.y)},
        {"num_row_dests", num_row_dests},
        {"scaled_q_a_first", scaled_q_a_first},
    };
    for (const bool is_kv : {false, true}) {
        const auto& name = is_kv ? KV_READER : QA_READER;
        KernelSpec reader{
            .unique_id = name,
            .source = kernel_path("dataflow/reader_qkv_proj.cpp"),
            .dfb_bindings =
                {ProducerOf(IN0_DFB, "in0"),
                 ProducerOf(use_pipes && is_kv ? W_KV_DFB : W_PROJ_DFB, "w"),
                 ConsumerOf(X_DFB, "x"),
                 ProducerOf(ROW_DFB, "qkv_row")},
            .semaphore_bindings =
                {SemaphoreBinding{.semaphore_spec_name = ARRIVE_SEM, .accessor_name = "arrive"},
                 SemaphoreBinding{.semaphore_spec_name = READY_SEM, .accessor_name = "ready"}},
            .compile_time_args = proj_reader_cta,
            .runtime_arg_schema = {.runtime_arg_names = {"shard_id", "slot", "is_hub"}},
            .hw_config = dm_config(DataMovementProcessor::RISCV_1, NOC::NOC_0),
        };
        if (use_pipes) {
            reader.compiler_options.defines.emplace("W_PIPE", "1");
            // One accessor names every pipe the relay covers; the one whose receiver is this core is bound.
            reader.advanced_options.prefetcher_pipe_bindings = {
                {.pipe_parameter_names = is_kv ? kv_pipe_names : qa_pipe_names, .accessor_name = "w"}};
        } else {
            reader.tensor_bindings.push_back(
                TensorBinding{.tensor_parameter_name = is_kv ? W_KV : W_QA, .accessor_name = "w"});
        }
        if (is_kv) {
            reader.compiler_options.defines.emplace("IS_KV", "1");
        } else {
            reader.dfb_bindings.push_back(ProducerOf(Q_A_GAMMA_DFB, "q_a_gamma"));
        }
        KernelRunArgs reader_run_args{.kernel = name};
        const auto& cores = is_kv ? kv_cores : qa_cores;
        for (uint32_t i = 0; i < cores.size(); ++i) {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                cores[i],
                {{"shard_id", i},
                 {"slot", is_kv ? q_lora_tiles + i : i},
                 {"is_hub", static_cast<uint32_t>(cores[i] == hub)}});
        }
        kernels.push_back(std::move(reader));
        run_args.kernel_run_args.push_back(std::move(reader_run_args));
    }

    // ---- q_a / kv projection compute ----
    for (const bool is_kv : {false, true}) {
        const auto& name = is_kv ? KV_COMPUTE : QA_COMPUTE;
        KernelSpec compute{
            .unique_id = name,
            .source = kernel_path("compute/compute_qkv_proj.cpp"),
            .dfb_bindings =
                {ConsumerOf(IN0_DFB, "in0"),
                 ConsumerOf(use_pipes && is_kv ? W_KV_DFB : W_PROJ_DFB, "w"),
                 ProducerOf(X_DFB, "x"),
                 ConsumerOf(ROW_DFB, "qkv_row")},
            .compile_time_args = {{"k_tiles", k_tiles}},
            .hw_config = compute_config(),
        };
        compute.compiler_options.opt_level = KernelBuildOptLevel::O3;
        if (use_pipes) {
            compute.compiler_options.defines.emplace("W_PIPE", "1");
        }
        KernelRunArgs compute_run_args{.kernel = name};
        if (is_kv) {
            compute.compiler_options.defines.emplace("IS_KV", "1");
            for (const auto& dfb : {STAT_DFB, SCALE_DFB, TMP_DFB}) {
                compute.dfb_bindings.push_back(ProducerOf(dfb, dfb.get()));
                compute.dfb_bindings.push_back(ConsumerOf(dfb, dfb.get()));
            }
            compute.dfb_bindings.push_back(ConsumerOf(GAMMA_DFB, "gamma"));
            compute.dfb_bindings.push_back(ConsumerOf(CS_DFB, "cs"));
            compute.dfb_bindings.push_back(ProducerOf(OUT_DFB, "out"));
            compute.compile_time_args.emplace("row_tiles", row_tiles);
            compute.compile_time_args.emplace("kv_first_tile", q_lora_tiles);
            compute.compile_time_args.emplace("kv_tiles", head_tiles);
            compute.compile_time_args.emplace("inv_n_bits", inv_head_dim_bits);
            compute.compile_time_args.emplace("eps_bits", eps_bits);
            compute.runtime_arg_schema.runtime_arg_names = {"kv_index", "do_rope"};
            for (uint32_t i = 0; i < kv_cores.size(); ++i) {
                AddRuntimeArgsForNode(
                    compute_run_args.runtime_arg_values,
                    kv_cores[i],
                    {{"kv_index", i}, {"do_rope", static_cast<uint32_t>(i + rope_tiles >= head_tiles)}});
            }
        } else {
            compute.dfb_bindings.push_back(ProducerOf(MM_OUT_DFB, "mm_out"));
            compute.dfb_bindings.push_back(ConsumerOf(MM_OUT_DFB, "mm_out"));
            compute.dfb_bindings.push_back(ConsumerOf(Q_A_GAMMA_DFB, "q_a_gamma"));
        }
        kernels.push_back(std::move(compute));
        run_args.kernel_run_args.push_back(std::move(compute_run_args));
    }

    // ---- kv writer (RISCV_0, NOC 1): RoPE table rows, ring tile read-modify-write ----
    {
        KernelSpec writer{
            .unique_id = KV_WRITER,
            .source = kernel_path("dataflow/writer_kv_cache.cpp"),
            .dfb_bindings =
                {ProducerOf(GAMMA_DFB, "gamma"),
                 ProducerOf(CS_DFB, "cs"),
                 ProducerOf(CS_RAW_DFB, "cs_raw"),
                 ConsumerOf(CS_RAW_DFB, "cs_raw"),
                 ConsumerOf(OUT_DFB, "out"),
                 ProducerOf(CACHE_TILE_DFB, "cache_tile"),
                 ConsumerOf(CACHE_TILE_DFB, "cache_tile"),
                 ProducerOf(POS_DFB, "pos"),
                 ConsumerOf(POS_DFB, "pos")},
            .tensor_bindings =
                {TensorBinding{.tensor_parameter_name = COS, .accessor_name = "cos"},
                 TensorBinding{.tensor_parameter_name = SIN, .accessor_name = "sin"},
                 TensorBinding{.tensor_parameter_name = KV_CACHE, .accessor_name = "kv_cache"},
                 TensorBinding{.tensor_parameter_name = CACHE_POS, .accessor_name = "cache_pos"}},
            .compile_time_args =
                {{"head_tiles", head_tiles},
                 {"rope_tiles", rope_tiles},
                 {"row_tile_bytes", row_tile_bytes},
                 {"cache_tile_bytes", cache_tile_bytes},
                 {"pos_page_bytes", pos_page_bytes}},
            .runtime_arg_schema = {.runtime_arg_names = {"kv_index", "do_rope"}},
            .hw_config = dm_config(DataMovementProcessor::RISCV_0, NOC::NOC_1),
        };
        KernelRunArgs writer_run_args{.kernel = KV_WRITER};
        for (uint32_t i = 0; i < kv_cores.size(); ++i) {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                kv_cores[i],
                {{"kv_index", i}, {"do_rope", static_cast<uint32_t>(i + rope_tiles >= head_tiles)}});
        }
        kernels.push_back(std::move(writer));
        run_args.kernel_run_args.push_back(std::move(writer_run_args));
    }

    // A q_b core holds nc_tiles of one head; the head's last core holds its rope tiles.
    auto qb_does_rope = [&](uint32_t i) { return static_cast<uint32_t>(((i + 1) * nc_tiles) % head_tiles == 0); };

    // With w_comp the q_b kernels also take the compressor-column path (COMP), on the cores is_comp marks: they first
    // compute their tile of tokens @ w_comp from their own DRAM slab and write it into comp_output. One kernel per
    // role still spans every q_b core, as the relay over q_b's pipes needs of its producer.
    // ---- q_b reader (RISCV_1, NOC 0): (comp: its w_comp slab), weight slab, then the multicast row ----
    KernelSpec qb_reader{
        .unique_id = QB_READER,
        .source = kernel_path("dataflow/reader_q_b.cpp"),
        .dfb_bindings = {ProducerOf(W_QB_DFB, "w"), ProducerOf(ROW_DFB, "qkv_row")},
        .semaphore_bindings = {SemaphoreBinding{.semaphore_spec_name = READY_SEM, .accessor_name = "ready"}},
        .compile_time_args = {{"w_tiles", q_lora_tiles * nc_tiles}, {"row_tiles", row_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = {"shard_id"}},
        .hw_config = dm_config(DataMovementProcessor::RISCV_1, NOC::NOC_0),
    };
    if (use_pipes) {
        qb_reader.compiler_options.defines.emplace("W_PIPE", "1");
        qb_reader.advanced_options.prefetcher_pipe_bindings = {
            {.pipe_parameter_names = qb_pipe_names, .accessor_name = "w"}};
    } else {
        qb_reader.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = W_QB, .accessor_name = "w"});
    }

    // ---- q_b compute: (comp: tokens @ w_comp), matmul, q_a's RMSNorm scale, RoPE ----
    KernelSpec qb_compute{
        .unique_id = QB_COMPUTE,
        .source = kernel_path("compute/compute_q_b.cpp"),
        .dfb_bindings =
            {ConsumerOf(ROW_DFB, "qkv_row"),
             ConsumerOf(W_QB_DFB, "w"),
             ConsumerOf(CS_DFB, "cs"),
             ProducerOf(OUT_DFB, "out")},
        .compile_time_args =
            {{"q_lora_tiles", q_lora_tiles},
             {"row_tiles", row_tiles},
             {"scaled_q_a_first", scaled_q_a_first},
             {"nc_tiles", nc_tiles},
             {"inv_n_bits", inv_q_lora_bits},
             {"eps_bits", eps_bits}},
        .runtime_arg_schema = {.runtime_arg_names = {"do_rope"}},
        .hw_config = compute_config(),
    };
    qb_compute.compiler_options.opt_level = KernelBuildOptLevel::O3;
    if (use_pipes) {
        qb_compute.compiler_options.defines.emplace("W_PIPE", "1");
    }
    for (const auto& dfb : {MM_OUT_DFB, STAT_DFB, SCALE_DFB, TMP_DFB}) {
        qb_compute.dfb_bindings.push_back(ProducerOf(dfb, dfb.get()));
        qb_compute.dfb_bindings.push_back(ConsumerOf(dfb, dfb.get()));
    }

    // ---- q_b writer (RISCV_0, NOC 1): RoPE table rows, (comp: its comp_output tile), q into SDPA's layout ----
    KernelSpec qb_writer{
        .unique_id = QB_WRITER,
        .source = kernel_path("dataflow/writer_q_b.cpp"),
        .dfb_bindings =
            {ProducerOf(CS_DFB, "cs"),
             ProducerOf(CS_RAW_DFB, "cs_raw"),
             ConsumerOf(CS_RAW_DFB, "cs_raw"),
             ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = COS, .accessor_name = "cos"},
             TensorBinding{.tensor_parameter_name = SIN, .accessor_name = "sin"},
             TensorBinding{.tensor_parameter_name = Q, .accessor_name = "q"}},
        .compile_time_args = {{"nc_tiles", nc_tiles}, {"rope_tiles", rope_tiles}, {"row_tile_bytes", row_tile_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"do_rope", "q_row", "q_col_bytes"}},
        .hw_config = dm_config(DataMovementProcessor::RISCV_0, NOC::NOC_1),
    };

    if (has_comp) {
        for (KernelSpec* kernel : {&qb_reader, &qb_compute, &qb_writer}) {
            kernel->compiler_options.defines.emplace("COMP", "1");
            kernel->runtime_arg_schema.runtime_arg_names.push_back("is_comp");
        }
        qb_reader.dfb_bindings.push_back(ProducerOf(IN0_DFB, "in0"));
        qb_reader.dfb_bindings.push_back(ProducerOf(W_COMP_DFB, "w_comp"));
        qb_reader.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = W_COMP, .accessor_name = "w_comp"});
        qb_reader.compile_time_args.emplace("k_tiles", k_tiles);
        qb_reader.runtime_arg_schema.runtime_arg_names.push_back("comp_shard_id");
        qb_compute.dfb_bindings.push_back(ConsumerOf(IN0_DFB, "in0"));
        qb_compute.dfb_bindings.push_back(ConsumerOf(W_COMP_DFB, "w_comp"));
        qb_compute.dfb_bindings.push_back(ProducerOf(COMP_X_DFB, "comp_x"));
        qb_compute.compile_time_args.emplace("k_tiles", k_tiles);
        qb_writer.dfb_bindings.push_back(ConsumerOf(COMP_X_DFB, "comp_x"));
        qb_writer.tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = COMP_OUT, .accessor_name = "comp_output"});
        qb_writer.runtime_arg_schema.runtime_arg_names.push_back("comp_col_bytes");
    }

    KernelRunArgs qb_reader_run_args{.kernel = QB_READER};
    KernelRunArgs qb_compute_run_args{.kernel = QB_COMPUTE};
    KernelRunArgs qb_writer_run_args{.kernel = QB_WRITER};
    for (uint32_t i = 0; i < qb_cores.size(); ++i) {
        const uint32_t first_tile = i * nc_tiles;
        AddRuntimeArgsForNode(qb_reader_run_args.runtime_arg_values, qb_cores[i], {{"shard_id", i}});
        AddRuntimeArgsForNode(qb_compute_run_args.runtime_arg_values, qb_cores[i], {{"do_rope", qb_does_rope(i)}});
        AddRuntimeArgsForNode(
            qb_writer_run_args.runtime_arg_values,
            qb_cores[i],
            {{"do_rope", qb_does_rope(i)},
             {"q_row", first_tile / head_tiles},
             {"q_col_bytes", (first_tile % head_tiles) * row_tile_bytes}});
        if (has_comp) {
            const uint32_t is_comp = static_cast<uint32_t>(is_comp_core[i]);
            AddRuntimeArgsForNode(
                qb_reader_run_args.runtime_arg_values,
                qb_cores[i],
                {{"is_comp", is_comp}, {"comp_shard_id", comp_slot[i]}});
            AddRuntimeArgsForNode(qb_compute_run_args.runtime_arg_values, qb_cores[i], {{"is_comp", is_comp}});
            AddRuntimeArgsForNode(
                qb_writer_run_args.runtime_arg_values,
                qb_cores[i],
                {{"is_comp", is_comp}, {"comp_col_bytes", comp_slot[i] * row_tile_bytes}});
        }
    }
    for (auto* kernel : {&qb_reader, &qb_compute, &qb_writer}) {
        kernels.push_back(std::move(*kernel));
    }
    for (auto* kernel_run_args : {&qb_reader_run_args, &qb_compute_run_args, &qb_writer_run_args}) {
        run_args.kernel_run_args.push_back(std::move(*kernel_run_args));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Work units
    ////////////////////////////////////////////////////////////////////////////
    auto range_set = [](const std::vector<CoreCoord>& cores) {
        std::vector<CoreRange> ranges;
        ranges.reserve(cores.size());
        for (const auto& core : cores) {
            ranges.emplace_back(core, core);
        }
        return CoreRangeSet(ranges).merge_ranges();
    };
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{.name = "q_a", .kernels = {QA_READER, QA_COMPUTE}, .target_nodes = range_set(qa_cores)},
        WorkUnitSpec{.name = "kv", .kernels = {KV_READER, KV_COMPUTE, KV_WRITER}, .target_nodes = range_set(kv_cores)},
        WorkUnitSpec{.name = "q_b", .kernels = {QB_READER, QB_COMPUTE, QB_WRITER}, .target_nodes = range_set(qb_cores)},
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensors
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::pair<TensorParamName, const Tensor*>> tensors = {
        {TOKENS, &t.tokens},
        {Q_A_GAMMA, &t.q_a_gamma},
        {KV_GAMMA, &t.kv_gamma},
        {COS, &t.cos},
        {SIN, &t.sin},
        {KV_CACHE, &t.kv_cache},
        {CACHE_POS, &t.cache_pos},
        {Q, &q_out},
    };
    if (!use_pipes) {
        tensors.insert(tensors.end(), {{W_QA, &t.w_q_a}, {W_KV, &t.w_kv}, {W_QB, &t.w_q_b}});
    }
    if (has_comp) {
        tensors.insert(tensors.end(), {{W_COMP, &*t.w_comp}, {COMP_OUT, &*t.comp_output}});
    }
    Group<TensorParameter> tensor_parameters;
    for (const auto& [name, tensor] : tensors) {
        tensor_parameters.push_back(TensorParameter{.unique_id = name, .spec = tensor->tensor_spec()});
        run_args.tensor_args.emplace(name, tensor->mesh_tensor());
    }

    ProgramSpec spec{
        .name = "attention_qkv_decode",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
        .advanced_options = {.prefetcher_pipe_parameters = std::move(pipe_parameters)},
    };
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::experimental::deepseek::attention_qkv_decode
