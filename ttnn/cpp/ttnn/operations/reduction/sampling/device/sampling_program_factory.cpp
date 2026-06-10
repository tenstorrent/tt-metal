// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/sampling/device/sampling_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {

constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/reader_values_indices_tensor.cpp";
constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp";
constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp";

using immutable_info_t = SamplingProgramFactory::immutable_info_t;

// Bytes per element of a dtype — mirrors Tensor::element_size() so derive_from_info can reconstruct the
// row/stick sizes from the TensorSpec alone (no live Tensor).
uint32_t element_size_of(DataType dt) {
    switch (dt) {
        case DataType::BFLOAT16: return sizeof(uint16_t);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::UINT8: return sizeof(uint8_t);
        default: TT_THROW("sampling: unsupported data type for element size");
    }
}

// Everything the ProgramSpec + run-args derive from. A pure function of the input/output tensor layouts
// + the (optional sub-)grid — i.e. the immutable structure. The only per-dispatch variation is the seed
// and the tensor addresses; neither appears here.
struct SamplingDerived {
    tt::DataFormat input_values_df;
    tt::DataFormat input_indices_df;
    tt::DataFormat index_df;  // UInt16
    tt::DataFormat k_df;
    tt::DataFormat p_df;
    tt::DataFormat temp_df;
    tt::DataFormat scalar_df;

    uint32_t input_values_tile_size = 0;
    uint32_t index_tile_size = 0;
    uint32_t scalar_tile_size = 0;
    uint32_t rand_tile_size = 0;

    uint32_t Ht = 0;
    uint32_t Wt = 0;
    uint32_t logWt = 0;
    uint32_t tile_height = 0;
    uint32_t tile_width = 0;
    uint32_t num_cores = 0;

    uint32_t aligned_final_indices_rm_unit_size = 0;
    uint32_t aligned_out0_unit_size = 0;
    uint32_t k_chunk_size = 0;
    uint32_t p_chunk_size = 0;
    uint32_t temp_chunk_size = 0;
    bool use_32bit_index = false;

    CoreRangeSet core_grid;
    std::vector<CoreCoord> cores;
};

// Reconstruct the full structural derivation from the ImmutableInfo alone — the six tensor specs + the
// grid + the optional sub-core grid. No live Tensor, no addresses, no seed.
SamplingDerived derive_from_info(const immutable_info_t& info) {
    SamplingDerived derived;
    // The bitonic top-k LLK carries sort indices through the dest register, and the index load/store
    // width is tied to fp32_dest_acc_en (INT32 when enabled, LO16 otherwise). WH/BH use the cheaper
    // 16-bit (UInt16) path with fp32 dest accumulation disabled; every other architecture (e.g. Quasar,
    // which also lacks UInt16/UInt32 DFB metadata support) uses 32-bit (Int32) index intermediates with
    // fp32 dest accumulation enabled. Gated on !(WH || BH) so new architectures default to 32-bit.
    derived.use_32bit_index = !(info.arch == tt::ARCH::WORMHOLE_B0 || info.arch == tt::ARCH::BLACKHOLE);
    derived.input_values_df = datatype_to_dataformat_converter(info.input_values_spec.data_type());
    derived.input_indices_df = datatype_to_dataformat_converter(info.input_indices_spec.data_type());
    derived.index_df = derived.use_32bit_index ? tt::DataFormat::Int32 : tt::DataFormat::UInt16;
    derived.k_df = datatype_to_dataformat_converter(info.k_spec.data_type());
    derived.p_df = datatype_to_dataformat_converter(info.p_spec.data_type());
    derived.temp_df = datatype_to_dataformat_converter(info.temp_spec.data_type());
    derived.scalar_df =
        (info.input_values_spec.data_type() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

    derived.input_values_tile_size = tile_size(derived.input_values_df);
    derived.index_tile_size = tile_size(derived.index_df);
    derived.scalar_tile_size = tile_size(derived.scalar_df);
    derived.rand_tile_size = tile_size(tt::DataFormat::Float16_b);

    const auto input_shape = info.input_values_spec.logical_shape();
    derived.tile_height = info.input_values_spec.tile().get_height();
    derived.tile_width = info.input_values_spec.tile().get_width();
    derived.Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / derived.tile_height;
    derived.Wt = input_shape[3] / derived.tile_width;
    derived.logWt = static_cast<uint32_t>(std::log2(derived.Wt));
    derived.num_cores = derived.Ht * derived.tile_height;

    derived.core_grid = tt::tt_metal::num_cores_to_corerangeset(derived.num_cores, info.grid, true);
    if (info.sub_core_grids.has_value()) {
        derived.core_grid = info.sub_core_grids.value();
    }
    derived.cores = corerange_to_cores(derived.core_grid, derived.num_cores, true);

    derived.aligned_final_indices_rm_unit_size =
        derived.Wt * derived.tile_width * element_size_of(info.input_indices_spec.data_type());
    derived.aligned_out0_unit_size = derived.Ht * derived.tile_height * element_size_of(info.output_spec.data_type());
    derived.k_chunk_size = derived.num_cores * 4;  // uint32
    derived.p_chunk_size = derived.num_cores * 2;  // bf16
    derived.temp_chunk_size = derived.num_cores * 2;
    return derived;
}

}  // namespace

// extract_immutable_info — the cache key + sole input to create_program_artifacts. The structural
// projection of the request (the six tensor specs + grid + sub-core grid). Deliberately EXCLUDES the
// seed, so calls differing only in seed share a cache entry and the seed can never leak into the spec.
SamplingProgramFactory::immutable_info_t SamplingProgramFactory::extract_immutable_info(
    const SamplingParams& attrs, const SamplingInputs& tensor_args) {
    // Output spec — mirrors SamplingDeviceOperation::compute_output_specs.
    TensorSpec output_spec =
        tensor_args.preallocated_output.has_value()
            ? tensor_args.preallocated_output->tensor_spec()
            : TensorSpec(
                  ttnn::Shape({1, 1, 1, tensor_args.input_values.logical_shape()[2]}),
                  TensorLayout(
                      DataType::UINT32, PageConfig(Layout::ROW_MAJOR), tensor_args.input_values.memory_config()));

    return immutable_info_t{
        .input_values_spec = tensor_args.input_values.tensor_spec(),
        .input_indices_spec = tensor_args.input_indices.tensor_spec(),
        .k_spec = tensor_args.k.tensor_spec(),
        .p_spec = tensor_args.p.tensor_spec(),
        .temp_spec = tensor_args.temp.tensor_spec(),
        .output_spec = std::move(output_spec),
        .grid = tensor_args.input_values.device()->compute_with_storage_grid_size(),
        .sub_core_grids = attrs.sub_core_grids,
        .arch = tensor_args.input_values.device()->arch(),
    };
}

ttnn::device_operation::ProgramArtifacts SamplingProgramFactory::create_program_artifacts(
    const immutable_info_t& info) {
    const auto derived = derive_from_info(info);

    constexpr uint32_t num_cb_unit = 2;
    constexpr uint32_t cb_in_units = 2 * num_cb_unit;

    // ---- 18 DFBs (formerly CBs c_0..c_17). entry_size = page_size, num_entries = total/page. ----
    auto dfb = [](const char* id, uint32_t entry, uint32_t n, tt::DataFormat f) {
        return m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{id}, .entry_size = entry, .num_entries = n, .data_format_metadata = f};
    };
    std::vector<m2::DataflowBufferSpec> dfbs = {
        dfb("input_values", derived.input_values_tile_size, cb_in_units, derived.input_values_df),
        dfb("index", derived.index_tile_size, cb_in_units, derived.index_df),
        dfb("scaler_max", derived.scalar_tile_size, 1, derived.scalar_df),
        dfb("scaler_sum", derived.scalar_tile_size, 1, derived.scalar_df),
        dfb("topk_mask", derived.input_values_tile_size, cb_in_units, derived.input_values_df),
        dfb("input_transposed", derived.input_values_tile_size, derived.Wt, derived.input_values_df),
        dfb("index_transposed", derived.index_tile_size, derived.Wt, derived.index_df),
        dfb("values", derived.input_values_tile_size, num_cb_unit, derived.input_values_df),
        dfb("local_vals", derived.input_values_tile_size, num_cb_unit, derived.input_values_df),
        dfb("output_ind", derived.index_tile_size, num_cb_unit, derived.index_df),
        dfb("cur_max", derived.input_values_tile_size, derived.Ht, derived.input_values_df),
        dfb("cur_sum", derived.input_values_tile_size, derived.Ht, derived.input_values_df),
        dfb("rand_tile", derived.rand_tile_size, 1, tt::DataFormat::Float16_b),
        dfb("final_indices",
            derived.aligned_final_indices_rm_unit_size,
            derived.Ht * derived.tile_height,
            derived.input_indices_df),
        dfb("output", derived.aligned_out0_unit_size, 1, derived.index_df),
        dfb("k", derived.k_chunk_size, 1, derived.k_df),
        dfb("p", derived.p_chunk_size, 1, derived.p_df),
        dfb("temp", derived.temp_chunk_size, 1, derived.temp_df),
    };

    // ---- kernels ----
    auto P = [](const char* dfb_id) { return m2::ProducerOf(m2::DFBSpecName{dfb_id}, dfb_id); };
    auto C = [](const char* dfb_id) { return m2::ConsumerOf(m2::DFBSpecName{dfb_id}, dfb_id); };
    auto TB = [](const char* name) {
        return m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{name}, .accessor_name = name};
    };

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{READER_KERNEL_PATH},
        .dfb_bindings = {P("input_values"), P("index"), P("final_indices")},
        .tensor_bindings = {TB("input_values"), TB("input_indices")},
        .compile_time_args =
            {{"Ht", derived.Ht},
             {"Wt", derived.Wt},
             {"final_indices_stick_size", derived.aligned_final_indices_rm_unit_size},
             {"tile_height", derived.tile_height},
             {"use_32bit_index", static_cast<uint32_t>(derived.use_32bit_index)}},
        // Reader on NCRISC (RISCV_1 / NOC1), writer on BRISC — mirrors the legacy Reader/Writer configs
        // so the two data-movement kernels don't collide on the same DM processor.
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::filesystem::path{COMPUTE_KERNEL_PATH},
        .dfb_bindings =
            {C("input_values"),
             C("index"),
             P("input_transposed"),
             C("input_transposed"),
             P("index_transposed"),
             C("index_transposed"),
             P("values"),
             C("values"),
             P("output_ind"),
             C("topk_mask"),
             C("scaler_max"),
             C("scaler_sum"),
             P("cur_max"),
             C("cur_max"),
             P("cur_sum"),
             C("cur_sum"),
             P("rand_tile"),
             P("local_vals"),
             C("temp")},
        .compile_time_args =
            {{"Ht", derived.Ht}, {"Wt", derived.Wt}, {"logWt", derived.logWt}, {"tile_width", derived.tile_width}},
        .runtime_arg_schema = {.runtime_arg_names = {"seed"}},
        // 32-bit (Int32) sort indices require fp32 dest accumulation so the top-k LLK loads/stores indices
        // in INT32 mode; the 16-bit (UInt16) path uses LO16 mode with fp32 dest acc off.
        .hw_config = m2::ComputeHardwareConfig{.fp32_dest_acc_en = derived.use_32bit_index},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{WRITER_KERNEL_PATH},
        .dfb_bindings =
            {P("output"),
             C("output"),
             P("topk_mask"),
             P("scaler_max"),
             P("scaler_sum"),
             C("final_indices"),
             C("local_vals"),
             C("output_ind"),
             C("rand_tile"),
             P("k"),
             C("k"),
             P("p"),
             C("p"),
             P("temp")},
        .tensor_bindings = {TB("output"), TB("temp"), TB("k"), TB("p")},
        .compile_time_args =
            {{"final_indices_stick_size", derived.aligned_final_indices_rm_unit_size},
             {"ids_per_batch", derived.tile_width},
             {"num_cores", derived.num_cores},
             {"use_32bit_index", static_cast<uint32_t>(derived.use_32bit_index)}},
        // core_id is a per-core RUNTIME arg, declared enqueue-invariant (it's a function of the grid,
        // identical for every dispatch that shares this cache entry).
        .runtime_arg_schema = {.runtime_arg_names = {"core_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
        .advanced_options = m2::KernelAdvancedOptions{.enqueue_invariant_runtime_args = {"core_id"}},
    };

    m2::ProgramSpec spec;
    spec.name = "sampling";
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.dataflow_buffers = std::move(dfbs);
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input_values"}, .spec = info.input_values_spec},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input_indices"}, .spec = info.input_indices_spec},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = info.output_spec},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"temp"}, .spec = info.temp_spec},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"k"}, .spec = info.k_spec},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"p"}, .spec = info.p_spec}};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "sampling",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute"}},
        .target_nodes = derived.core_grid}};

    // Enqueue-invariant run-args: the per-core core_id (user index). Set once on cache miss, retained.
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};
    for (uint32_t i = 0; i < derived.cores.size(); ++i) {
        writer_args.runtime_arg_values.push_back({derived.cores[i], {{"core_id", i}}});
    }
    // The reader has no named runtime args, but it binds tensor parameters (input_values / input_indices)
    // whose base addresses the framework fills from the run-args — so it still needs a (named-arg-empty)
    // KernelRunArgs entry to be registered.
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs invariant_args;
    invariant_args.kernel_run_args.push_back(std::move(writer_args));
    invariant_args.kernel_run_args.push_back(std::move(reader_args));

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(invariant_args)};
}

m2::ProgramRunArgs SamplingProgramFactory::create_per_enqueue_args(
    const SamplingParams& attrs,
    const SamplingInputs& tensor_args,
    Tensor& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const auto derived = derive_from_info(extract_immutable_info(attrs, tensor_args));
    const uint32_t seed = attrs.seed.has_value() ? attrs.seed.value() : 0;

    // Per-core seed (every core runs the same kernel; the seed is broadcast identically here — the
    // kernel recovers per-core distinctness internally, matching the legacy behavior).
    m2::ProgramRunArgs::KernelRunArgs compute_args{.kernel = m2::KernelSpecName{"compute"}};
    for (const auto& core : derived.cores) {
        compute_args.runtime_arg_values.push_back({core, {{"seed", seed}}});
    }

    m2::ProgramRunArgs args;
    args.kernel_run_args.push_back(std::move(compute_args));
    args.tensor_args.emplace(
        m2::TensorParamName{"input_values"},
        m2::ProgramRunArgs::TensorArgument{std::cref(tensor_args.input_values.mesh_tensor())});
    args.tensor_args.emplace(
        m2::TensorParamName{"input_indices"},
        m2::ProgramRunArgs::TensorArgument{std::cref(tensor_args.input_indices.mesh_tensor())});
    args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output_tensor.mesh_tensor())});
    args.tensor_args.emplace(
        m2::TensorParamName{"temp"}, m2::ProgramRunArgs::TensorArgument{std::cref(tensor_args.temp.mesh_tensor())});
    args.tensor_args.emplace(
        m2::TensorParamName{"k"}, m2::ProgramRunArgs::TensorArgument{std::cref(tensor_args.k.mesh_tensor())});
    args.tensor_args.emplace(
        m2::TensorParamName{"p"}, m2::ProgramRunArgs::TensorArgument{std::cref(tensor_args.p.mesh_tensor())});
    return args;
}

}  // namespace ttnn::prim
