// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/sampling/device/sampling_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

namespace ttnn::prim {

namespace {

using namespace tt::tt_metal::experimental;

constexpr auto SAMPLING_READER_SOURCE =
    "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/reader_values_indices_tensor.cpp";
constexpr auto SAMPLING_WRITER_SOURCE =
    "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp";
constexpr auto SAMPLING_COMPUTE_SOURCE =
    "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp";

// The constants below carry a SAMPLING_ prefix so this file can share a translation unit with the
// other reduction factories, which the unity build merges into one anonymous namespace. The spec-name
// strings they hold are scoped to a single ProgramSpec, so those need no prefix.
const KernelSpecName SAMPLING_READER{"reader"};
const KernelSpecName SAMPLING_COMPUTE{"compute"};

// The writer runs one instance per user, each baking its own core index as a compile-time argument, so
// it needs a KernelSpec (and a single-node WorkUnitSpec) per running core.
KernelSpecName sampling_writer_name(uint32_t core_index) {
    return KernelSpecName{"writer_" + std::to_string(core_index)};
}

// Streamed in by the reader and sorted by compute: the candidate values, and the matching index tile
// the bitonic top-k carries alongside them.
const DFBSpecName SAMPLING_INPUT_VALUES{"input_values"};
const DFBSpecName SAMPLING_INDEX{"index"};
// The row-major index tensor, read in by the reader and looked up by the writer to turn a local sorted
// position back into a global token id.
const DFBSpecName SAMPLING_FINAL_INDICES_RM{"final_indices_rm"};
// Reduce scalers: separate buffers because MAX and SUM use different tile fill layouts.
const DFBSpecName SAMPLING_SCALER_MAX{"scaler_max"};
const DFBSpecName SAMPLING_SCALER_SUM{"scaler_sum"};
// The top-k mask the writer builds from this core's k, added to the sorted values to drop everything
// outside the top k.
const DFBSpecName SAMPLING_TOPK_MASK{"topk_mask"};
// Compute-private top-k intermediates: the transposed value and index tiles the bitonic merge tree
// rewrites in place, and the sorted top-k values the softmax chain then consumes in place.
const DFBSpecName SAMPLING_INPUT_TRANSPOSED{"input_transposed"};
const DFBSpecName SAMPLING_INDEX_TRANSPOSED{"index_transposed"};
const DFBSpecName SAMPLING_VALUES{"values"};
// Compute-private softmax running statistics.
const DFBSpecName SAMPLING_CUR_MAX{"cur_max"};
const DFBSpecName SAMPLING_CUR_SUM{"cur_sum"};
// Compute results the writer reads back to do the top-p filtering and the stochastic pick.
const DFBSpecName SAMPLING_LOCAL_VALS{"local_vals"};
const DFBSpecName SAMPLING_OUTPUT_IND{"output_ind"};
const DFBSpecName SAMPLING_RAND_TILE{"rand_tile"};
// The sampled index, staged in SRAM before the writer sends this core's word to the output tensor.
const DFBSpecName SAMPLING_OUTPUT{"output"};
// Per-user k, p and temperature. Each is NOC-read whole, in one entry covering every core, so no core
// issues an unaligned read for its own value; `temp` doubles as the scalar operand compute broadcasts.
const DFBSpecName SAMPLING_K{"k"};
const DFBSpecName SAMPLING_P{"p"};
const DFBSpecName SAMPLING_TEMP{"temp"};

const TensorParamName SAMPLING_INPUT_VALUES_TENSOR{"input_values"};
const TensorParamName SAMPLING_INPUT_INDICES_TENSOR{"input_indices"};
const TensorParamName SAMPLING_OUTPUT_TENSOR{"output"};
const TensorParamName SAMPLING_TEMP_TENSOR{"temp"};
const TensorParamName SAMPLING_K_TENSOR{"k"};
const TensorParamName SAMPLING_P_TENSOR{"p"};

DataflowBufferSpec make_dfb(
    const DFBSpecName& unique_id, uint32_t entry_size, uint32_t num_entries, tt::DataFormat data_format) {
    return DataflowBufferSpec{
        .unique_id = unique_id,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = data_format,
    };
}

}  // namespace

ttnn::device_operation::ProgramArtifacts SamplingProgramFactory::create_program_artifacts(
    const SamplingParams& operation_attributes, const SamplingInputs& tensor_args, Tensor& output_tensor) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input_values_tensor = tensor_args.input_values.mesh_tensor();
    const auto& input_indices_tensor = tensor_args.input_indices.mesh_tensor();
    const auto& k = tensor_args.k.mesh_tensor();
    const auto& p = tensor_args.p.mesh_tensor();
    const auto& temp = tensor_args.temp.mesh_tensor();

    const auto& seed = operation_attributes.seed;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    uint32_t random_seed = 0;

    auto* device = &input_values_tensor.mutable_device();

    // The bitonic top-k LLK carries sort indices through the dest register, and the index
    // load/store width is tied to fp32_dest_acc_en (INT32 when enabled, LO16 otherwise). WH/BH
    // support the cheaper 16-bit (UInt16) path with fp32 dest accumulation disabled (unchanged
    // behaviour) so that WH/BH does not suffer any restrictions on the dest register. Every other
    // architecture (e.g. Quasar, which additionally lacks UInt16/UInt32 tile (DFB) metadata
    // support) uses 32-bit (Int32) index intermediates with fp32 dest accumulation enabled. This
    // is gated on !(WH || BH) so new architectures default to the safe 32-bit path.
    const bool use_32bit_index = !(device->arch() == tt::ARCH::WORMHOLE_B0 || device->arch() == tt::ARCH::BLACKHOLE);

    // The stable bitonic top-k network (ties keep the lowest candidate position, so the sampled
    // token id for an exact tie does not depend on how the network swaps equal values) is only
    // implemented in the WH/BH LLKs; the Quasar LLK static_asserts STABLE_SORT == false. Gate on
    // the architectures that implement it so new architectures fall back to the unstable network
    // instead of failing to build the kernel.
    const bool stable_sort = (device->arch() == tt::ARCH::WORMHOLE_B0 || device->arch() == tt::ARCH::BLACKHOLE);

    tt::DataFormat input_values_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_values_tensor.dtype());
    tt::DataFormat input_indices_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_indices_tensor.dtype());
    tt::DataFormat index_dfb_data_format = use_32bit_index ? tt::DataFormat::Int32 : tt::DataFormat::UInt16;
    // On the 32-bit path (e.g. Quasar), validation already requires k to be INT32 (UInt32 DFB
    // metadata is unsupported there), so the dtype-derived format is correct as-is.
    tt::DataFormat k_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(k.dtype());
    tt::DataFormat p_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(p.dtype());
    tt::DataFormat temp_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(temp.dtype());

    uint32_t input_values_tile_size = tile_size(input_values_dfb_data_format);
    uint32_t index_tile_size = tile_size(index_dfb_data_format);

    const auto& output_mesh = output_tensor.mesh_tensor();

    auto input_shape = input_values_tensor.logical_shape();
    const uint32_t tile_height = input_values_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_values_tensor.tensor_spec().tile().get_width();
    // `num_users` is the logical user count (rows in dim 2) in the range [1, 32]; validation
    // guarantees N == C == 1, so it is just input_shape[2]. The data still occupies a single padded
    // row-tile (Ht == 1) and only `num_users` cores run. Decoupling num_cores from Ht*tile_height is
    // what lets <32 users work: the old `(.../tile_height)` would integer-divide to Ht == 0 (and
    // num_cores == 0) for fewer than tile_height users.
    const uint32_t num_users = input_shape[2];
    uint32_t Ht = (num_users + tile_height - 1) / tile_height;  // == 1 for 1..32 users
    uint32_t Wt = input_shape[3] / tile_width;
    auto num_cores = num_users;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet core_grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);

    if (sub_core_grids.has_value()) {
        core_grid = sub_core_grids.value();
    }
    auto cores = corerange_to_cores(core_grid, num_cores, true);

    // `sub_core_grids` may be over-provisioned (more cores than users); only the first `num_cores`
    // cores actually run. Confine the grid to exactly those cores so we don't place a kernel (with
    // unset arguments) on the unused cores. Dataflow-buffer placement follows the bound kernels, so
    // narrowing the grid narrows the allocation too.
    if (core_grid.num_cores() != num_cores) {
        std::vector<CoreRange> active_core_ranges;
        active_core_ranges.reserve(cores.size());
        for (const auto& core : cores) {
            active_core_ranges.emplace_back(core);
        }
        core_grid = CoreRangeSet(std::move(active_core_ranges));
    }

    validate_reduce_op_program_grid(
        "Sampling",
        core_grid,
        compute_with_storage_grid_size,
        sub_core_grids.has_value() ? &sub_core_grids.value() : nullptr,
        true,
        {});

    if (seed.has_value()) {
        random_seed = seed.value();
    }

    // for streaming in input
    uint32_t num_dfb_unit = 2;
    uint32_t dfb_in_units = 2 * num_dfb_unit;

    // Reduce scaler format — MAX and SUM share it even though they need separate buffers
    tt::DataFormat scalar_df =
        (input_values_tensor.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scale_tiles = 1;
    uint32_t scalar_tile_size = tile_size(scalar_df);

    uint32_t num_out_tiles = Ht;

    // random number
    const uint32_t rand_tile_size = tile_size(tt::DataFormat::Float16_b);

    // final indices
    uint32_t final_indices_rm_unit_size = input_indices_tensor.element_size();  // 4 for int32
    uint32_t aligned_final_indices_rm_unit_size = Wt * tile_width * final_indices_rm_unit_size;

    // Output sampling indices
    uint32_t output_unit_size = output_mesh.element_size();
    uint32_t aligned_out0_unit_size = Ht * tile_height * output_unit_size;

    // k, p and temp entries span all the cores, so that no core issues an unaligned NOC read
    const uint32_t uint32_bytes = 4;
    const uint32_t bf16_bytes = 2;
    uint32_t k_chunk_size = num_cores * uint32_bytes;
    uint32_t p_chunk_size = num_cores * bf16_bytes;
    uint32_t temp_chunk_size = num_cores * bf16_bytes;

    Group<DataflowBufferSpec> dataflow_buffers{
        // Two tiles are loaded in for sampling_local_sort at a time, and we double buffer to avoid
        // stalls, so allocate four tiles of space
        make_dfb(SAMPLING_INPUT_VALUES, input_values_tile_size, dfb_in_units, input_values_dfb_data_format),
        make_dfb(SAMPLING_INDEX, index_tile_size, dfb_in_units, index_dfb_data_format),
        make_dfb(SAMPLING_SCALER_MAX, scalar_tile_size, scale_tiles, scalar_df),
        make_dfb(SAMPLING_SCALER_SUM, scalar_tile_size, scale_tiles, scalar_df),
        make_dfb(SAMPLING_TOPK_MASK, input_values_tile_size, dfb_in_units, input_values_dfb_data_format),
        // Single buffered buffer that holds the transposed input tiles
        make_dfb(SAMPLING_INPUT_TRANSPOSED, input_values_tile_size, Wt, input_values_dfb_data_format),
        // Single buffered buffer that holds the transposed index tiles
        make_dfb(SAMPLING_INDEX_TRANSPOSED, index_tile_size, Wt, index_dfb_data_format),
        // Output sampling values
        make_dfb(SAMPLING_VALUES, input_values_tile_size, num_dfb_unit, input_values_dfb_data_format),
        make_dfb(SAMPLING_LOCAL_VALS, input_values_tile_size, num_dfb_unit, input_values_dfb_data_format),
        // Output local indices
        make_dfb(SAMPLING_OUTPUT_IND, index_tile_size, num_dfb_unit, index_dfb_data_format),
        make_dfb(SAMPLING_CUR_MAX, input_values_tile_size, num_out_tiles, input_values_dfb_data_format),
        make_dfb(SAMPLING_CUR_SUM, input_values_tile_size, num_out_tiles, input_values_dfb_data_format),
        make_dfb(SAMPLING_RAND_TILE, rand_tile_size, 1, tt::DataFormat::Float16_b),
        make_dfb(
            SAMPLING_FINAL_INDICES_RM,
            aligned_final_indices_rm_unit_size,
            Ht * tile_height,
            input_indices_dfb_data_format),
        make_dfb(SAMPLING_OUTPUT, aligned_out0_unit_size, 1, index_dfb_data_format),
        make_dfb(SAMPLING_K, k_chunk_size, 1, k_dfb_data_format),
        make_dfb(SAMPLING_P, p_chunk_size, 1, p_dfb_data_format),
        make_dfb(SAMPLING_TEMP, temp_chunk_size, 1, temp_dfb_data_format),
    };

    // The reader is created once over every running core: it streams in the value and index tiles the
    // top-k consumes, plus the row-major index sticks the writer looks up.
    KernelSpec reader{
        .unique_id = SAMPLING_READER,
        .source = SAMPLING_READER_SOURCE,
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = SAMPLING_INPUT_VALUES,
                 .accessor_name = "input_values",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_FINAL_INDICES_RM,
                 .accessor_name = "input_indices",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_INDEX,
                 .accessor_name = "index",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .tensor_bindings =
            {TensorBinding{
                 .tensor_parameter_name = SAMPLING_INPUT_VALUES_TENSOR,
                 .accessor_name = "input_values",
             },
             TensorBinding{
                 .tensor_parameter_name = SAMPLING_INPUT_INDICES_TENSOR,
                 .accessor_name = "input_indices",
             }},
        .compile_time_args =
            {{"Ht", Ht},
             {"Wt", Wt},
             {"input_indices_page_size", aligned_final_indices_rm_unit_size},
             {"tile_height", tile_height},
             {"use_32bit_index", static_cast<uint32_t>(use_32bit_index)},
             {"num_users", num_users}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // 32-bit (Int32) sort indices require fp32 dest accumulation so the top-k LLK loads/stores
    // indices in INT32 mode; the 16-bit (UInt16) path uses LO16 mode with fp32 dest acc off. Every
    // other field keeps its default, matching what the compute config defaulted to before.
    const ComputeHardwareConfig compute_config{
        .enable_32_bit_dest = use_32bit_index,
    };

    // The compute kernel is identical on every running core, so one KernelSpec covers them all; its
    // node set is the union of the work units it belongs to. The five buffers it binds as both
    // PRODUCER and CONSUMER are its private top-k and softmax intermediates: it packs into them and
    // unpacks straight back out, with no other kernel involved.
    KernelSpec compute{
        .unique_id = SAMPLING_COMPUTE,
        .source = SAMPLING_COMPUTE_SOURCE,
        // O3 is the optimization level a compute kernel is built at; the CompilerOptions default (O2)
        // is the data-movement level, so compute kernels state it explicitly.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = SAMPLING_INPUT_VALUES,
                 .accessor_name = "input_values",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_INDEX,
                 .accessor_name = "index",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_INPUT_TRANSPOSED,
                 .accessor_name = "input_transposed",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_INPUT_TRANSPOSED,
                 .accessor_name = "input_transposed",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_INDEX_TRANSPOSED,
                 .accessor_name = "index_transposed",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_INDEX_TRANSPOSED,
                 .accessor_name = "index_transposed",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_VALUES,
                 .accessor_name = "values",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_VALUES,
                 .accessor_name = "values",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_OUTPUT_IND,
                 .accessor_name = "output_ind",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_TOPK_MASK,
                 .accessor_name = "topk_mask",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_SCALER_MAX,
                 .accessor_name = "scaler_max",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_SCALER_SUM,
                 .accessor_name = "scaler_sum",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_CUR_MAX,
                 .accessor_name = "cur_max",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_CUR_MAX,
                 .accessor_name = "cur_max",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_CUR_SUM,
                 .accessor_name = "cur_sum",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_CUR_SUM,
                 .accessor_name = "cur_sum",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_RAND_TILE,
                 .accessor_name = "rand_tile",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_LOCAL_VALS,
                 .accessor_name = "local_vals",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SAMPLING_TEMP,
                 .accessor_name = "temp",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             }},
        .compile_time_args =
            {{"Ht", Ht},
             {"Wt", Wt},
             {"logWt", static_cast<uint32_t>(std::log2(Wt))},
             {"seed", random_seed},
             {"tile_width", tile_width},
             {"stable_sort", static_cast<uint32_t>(stable_sort)}},
        .hw_config = ComputeHardwareConfig{compute_config},
    };

    // One writer per running core. `core_id` is the only value that differs between them, and it is a
    // compile-time argument, so each core gets its own KernelSpec rather than one shared spec.
    // `output`, `k` and `p` are bound as both PRODUCER and CONSUMER because this kernel is their only
    // toucher: `k` and `p` are NOC-read staging it then indexes by core, and `output` is a plain SRAM
    // window it fills and sends without running a FIFO operation on it at all.
    auto make_writer = [&](uint32_t core_index) {
        return KernelSpec{
            .unique_id = sampling_writer_name(core_index),
            .source = SAMPLING_WRITER_SOURCE,
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = SAMPLING_OUTPUT,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_OUTPUT,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_TOPK_MASK,
                     .accessor_name = "mask",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_SCALER_MAX,
                     .accessor_name = "scaler_max",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_SCALER_SUM,
                     .accessor_name = "scaler_sum",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_FINAL_INDICES_RM,
                     .accessor_name = "final_indices",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_LOCAL_VALS,
                     .accessor_name = "local_values",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_OUTPUT_IND,
                     .accessor_name = "local_indices",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_RAND_TILE,
                     .accessor_name = "rand",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_K,
                     .accessor_name = "k",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_K,
                     .accessor_name = "k",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_P,
                     .accessor_name = "p",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_P,
                     .accessor_name = "p",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SAMPLING_TEMP,
                     .accessor_name = "temp",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 }},
            .tensor_bindings =
                {TensorBinding{
                     .tensor_parameter_name = SAMPLING_OUTPUT_TENSOR,
                     .accessor_name = "output",
                 },
                 TensorBinding{
                     .tensor_parameter_name = SAMPLING_TEMP_TENSOR,
                     .accessor_name = "temp",
                 },
                 TensorBinding{
                     .tensor_parameter_name = SAMPLING_K_TENSOR,
                     .accessor_name = "k",
                 },
                 TensorBinding{
                     .tensor_parameter_name = SAMPLING_P_TENSOR,
                     .accessor_name = "p",
                 }},
            .compile_time_args =
                {{"final_indices_stick_size", aligned_final_indices_rm_unit_size},
                 // out_stick_size is supplied for every writer but read by none of them.
                 {"out_stick_size", aligned_out0_unit_size},
                 {"core_id", core_index},
                 {"ids_per_batch", tile_width},
                 {"num_cores", num_cores},
                 {"use_32bit_index", static_cast<uint32_t>(use_32bit_index)},
                 {"num_users", num_users}},
            .hw_config = ttnn::create_writer_datamovement_config(),
        };
    };

    Group<KernelSpec> kernels;
    kernels.reserve(2 + cores.size());
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(compute));

    // One single-node work unit per running core. The reader and the compute kernel belong to every
    // one of them, so their node sets are the union of all the running cores; each writer belongs only
    // to its own. Work units may not overlap in target nodes, which is what makes this per-node rather
    // than one work unit spanning the whole grid.
    Group<WorkUnitSpec> work_units;
    work_units.reserve(cores.size());

    for (uint32_t i = 0; i < cores.size(); ++i) {
        KernelSpecName writer_name = sampling_writer_name(i);
        kernels.push_back(make_writer(i));
        work_units.push_back(WorkUnitSpec{
            .name = "sampling_core_" + std::to_string(i),
            .kernels = {SAMPLING_READER, SAMPLING_COMPUTE, std::move(writer_name)},
            .target_nodes = NodeCoord{cores[i]},
        });
    }

    ProgramSpec spec{
        .name = "sampling",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {TensorParameter{.unique_id = SAMPLING_INPUT_VALUES_TENSOR, .spec = input_values_tensor.tensor_spec()},
             TensorParameter{.unique_id = SAMPLING_INPUT_INDICES_TENSOR, .spec = input_indices_tensor.tensor_spec()},
             TensorParameter{.unique_id = SAMPLING_OUTPUT_TENSOR, .spec = output_mesh.tensor_spec()},
             TensorParameter{.unique_id = SAMPLING_TEMP_TENSOR, .spec = temp.tensor_spec()},
             TensorParameter{.unique_id = SAMPLING_K_TENSOR, .spec = k.tensor_spec()},
             TensorParameter{.unique_id = SAMPLING_P_TENSOR, .spec = p.tensor_spec()}},
        .work_units = std::move(work_units),
    };

    // No kernel_run_args: every value these kernels used to receive at runtime was a tensor base
    // address, and those now ride on the tensor bindings.
    ProgramRunArgs run_args{
        .tensor_args =
            {{SAMPLING_INPUT_VALUES_TENSOR, input_values_tensor},
             {SAMPLING_INPUT_INDICES_TENSOR, input_indices_tensor},
             {SAMPLING_OUTPUT_TENSOR, output_mesh},
             {SAMPLING_TEMP_TENSOR, temp},
             {SAMPLING_K_TENSOR, k},
             {SAMPLING_P_TENSOR, p}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
