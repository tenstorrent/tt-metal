// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Program construction for the standalone minimal_matmul op, on the Metal 2.0 host API
// (ProgramSpecFactoryConcept).
//
// The legacy Program&-based helper minimal_matmul_factory_helper_common in
// minimal_matmul_program_factory.cpp still exists, and is now reached only by the CCL composites
// that build a fused matmul + reduce-scatter program (minimal_matmul_strided_reduce_scatter_async).
// That helper binds the *legacy* kernels; this factory binds the _metal2 forks beside them.
//
// There is deliberately no override_runtime_arguments: on the base spec concept the framework
// refreshes every tensor binding on a program-cache hit.
//

#include "minimal_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::experimental::prim {

using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::ComputeGen1Config;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DataMovementGen1Config;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelAdvancedOptions;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::SemaphoreBinding;
using tt::tt_metal::experimental::SemaphoreSpec;
using tt::tt_metal::experimental::SemaphoreSpecName;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace {

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> determine_default_block_sizes(
    uint32_t M, uint32_t K, uint32_t N, bool fp32_dest_acc_en) {
    (void)K;  // K not used for determining defaults currently
    uint32_t M_block_tiles = 8;
    uint32_t K_block_tiles = 8;
    uint32_t N_block_tiles = 8;

    uint32_t subblock_h = 2;
    uint32_t subblock_w = 2;
    if (!fp32_dest_acc_en) {
        if (N >= M) {
            subblock_h = 2;
            subblock_w = 4;
        } else {
            subblock_h = 4;
            subblock_w = 2;
        }
    }

    return {M_block_tiles, K_block_tiles, N_block_tiles, subblock_h, subblock_w};
}

// Build a linear order of cores along one axis for data movement, plus index of the current core
std::pair<std::vector<CoreCoord>, uint32_t> build_core_order_for_axis(
    const CoreCoord& core,
    bool transpose_core_grid,
    uint32_t axis_length,
    tt::tt_metal::NOC noc,
    bool axis_is_x_when_not_transposed,
    const CoreCoord& initial_endpoint) {
    std::vector<CoreCoord> order;
    order.reserve(axis_length);
    order.push_back(initial_endpoint);

    // Determine which coordinate of the current core defines its position along this axis
    const size_t current_axis_value = transpose_core_grid ? (axis_is_x_when_not_transposed ? core.y : core.x)
                                                          : (axis_is_x_when_not_transposed ? core.x : core.y);

    // Direction along the axis: increasing for NOC_0, decreasing for NOC_1
    const bool increasing = (noc == tt::tt_metal::NOC::NOC_0);

    uint32_t index_of_current = 0;  // default to 0 if axis_length == 1
    for (uint32_t worker_idx = 1; worker_idx < axis_length; ++worker_idx) {
        CoreCoord worker_core = core;
        size_t& coord_to_modify = transpose_core_grid ? (axis_is_x_when_not_transposed ? worker_core.y : worker_core.x)
                                                      : (axis_is_x_when_not_transposed ? worker_core.x : worker_core.y);

        coord_to_modify = increasing ? worker_idx : (axis_length - worker_idx);
        if (coord_to_modify == current_axis_value) {
            index_of_current = worker_idx;
        }
        order.push_back(worker_core);
    }
    return {order, index_of_current};
}

CoreCoord clamped_prev(const std::vector<CoreCoord>& order, uint32_t index) {
    return order.at(index == 0 ? 0 : index - 1);
}

CoreCoord clamped_next(const std::vector<CoreCoord>& order, uint32_t index) {
    const uint32_t last = static_cast<uint32_t>(order.size() - 1);
    return order.at(index >= last ? last : index + 1);
}

// The define builders below (and throttle_mm_perf) work in terms of std::map; KernelSpec wants a
// Table of the same pairs.
KernelSpec::CompilerOptions::Defines to_defines(const std::map<std::string, std::string>& defines) {
    // Table is a map, not a vector: no iterator-pair ctor, but it does take a range in one arg.
    return KernelSpec::CompilerOptions::Defines(defines);
}

// DFB spec names (also the kernel-side dfb:: accessor names).
const DFBSpecName DFB_IN0{"in0"};
const DFBSpecName DFB_IN1{"in1"};
const DFBSpecName DFB_OUT{"out"};
const DFBSpecName DFB_INTERMEDIATE{"intermediate"};
const DFBSpecName DFB_IN2{"in2"};
const DFBSpecName DFB_TERNARY_A{"ternary_a"};
const DFBSpecName DFB_TERNARY_B{"ternary_b"};

// Tensor parameter names (also the kernel-side tensor:: accessor names).
const TensorParamName TP_IN0{"in0"};
const TensorParamName TP_IN1{"in1"};
const TensorParamName TP_IN2{"in2"};
const TensorParamName TP_IN3{"in3"};
const TensorParamName TP_TERNARY_A{"ternary_a"};
const TensorParamName TP_TERNARY_B{"ternary_b"};

// Kernel spec names.
const KernelSpecName K_IN0_SENDER{"in0_sender"};
const KernelSpecName K_IN0_RECEIVER{"in0_receiver"};
const KernelSpecName K_IN1_SENDER{"in1_sender"};
const KernelSpecName K_IN1_RECEIVER{"in1_receiver"};
const KernelSpecName K_COMPUTE{"compute"};

std::string output_param_name(uint32_t chunk) { return "out" + std::to_string(chunk); }

DFBBinding producer_of(const DFBSpecName& dfb) {
    return DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = *dfb,
        .endpoint_type = DFBEndpointType::PRODUCER,
    };
}

DFBBinding consumer_of(const DFBSpecName& dfb) {
    return DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = *dfb,
        .endpoint_type = DFBEndpointType::CONSUMER,
    };
}

}  // namespace

ttnn::device_operation::ProgramArtifacts MinimalMatmulDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& weight_tensor = tensor_args.weight_tensor;
    const std::optional<Tensor>& bias_tensor = tensor_args.bias_tensor;
    const std::optional<Tensor>& optional_input_tensor = tensor_args.optional_input_tensor;
    const std::optional<Tensor>& fused_ternary_input_a = tensor_args.fused_ternary_input_a;
    const std::optional<Tensor>& fused_ternary_input_b = tensor_args.fused_ternary_input_b;
    const std::vector<Tensor>& output_tensors = tensor_return_value;

    const auto& fused_activation = operation_attributes.fused_activation;
    const auto& config = operation_attributes.config;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    const bool fuse_swiglu = operation_attributes.fuse_swiglu;
    const uint32_t N_chunks = static_cast<uint32_t>(operation_attributes.chunks);

    auto* device = input_tensor.device();

    // Fused concat (concat-free): in0's K is sourced from input_tensor (prefix K-tiles) then
    // optional_input_tensor (suffix), via the in0 second-source (in3) read path, instead of a
    // materialized concat. The split point is input_tensor's own K width.
    const bool two_input_split = optional_input_tensor.has_value();

    if (!config.has_value()) {
        log_debug(tt::LogOp, "No config provided, using default block sizes and core grid");
    }

    auto grid_size =
        config.has_value() ? config.value().compute_with_storage_grid_size : device->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();

    bool use_bias = bias_tensor.has_value();
    bool use_fused_ternary = fused_ternary_input_a.has_value() && fused_ternary_input_b.has_value();

    /**
     * Determine dataformats, compute kernel config
     */
    auto in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto in0_tile_size = tt::tile_size(in0_data_format);
    auto in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.dtype());
    auto in1_tile_size = tt::tile_size(in1_data_format);
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensors[0].dtype());
    auto out_tile_size = tt::tile_size(output_data_format);

    auto in2_data_format =
        use_bias ? tt::tt_metal::datatype_to_dataformat_converter(bias_tensor.value().dtype()) : in1_data_format;
    auto in2_tile_size = tt::tile_size(in2_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    // Intermediate DFB dataformat is the same datatype as DST register.
    auto intermediate_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    auto intermediate_tile_size = tt::tile_size(intermediate_data_format);

    /**
     * in0: M_tiles x K_tiles
     * in0 is divided into blocks, which are M_block_tiles x K_block_tiles
     *
     * in1: K_tiles x N_tiles
     * in1 is divided into blocks, which are K_block_tiles x N_block_tiles
     *
     * output: M_tiles x N_tiles
     * output is divided into blocks, which are M_block_tiles x N_block_tiles
     *
     * Blocks are further subdivided into subblocks. The output block is subdivided into subblock_h x subblock_w
     * subblocks. The in0 and in1 blocks are accordingly subdivided on M and N.
     */

    auto in0_tensor_shape = input_tensor.padded_shape();
    auto in1_tensor_shape = weight_tensor.padded_shape();
    // Fold activation (LHS) upper dimensions into rows: M_total = prod(upper dims) * M.
    // M is derived from input_tensor's OWN K width (K_in); for fused concat that's only the prefix
    // half, so the matmul contraction K must instead span the full weight K (both concat halves).
    uint32_t K_in = in0_tensor_shape[-1];
    uint32_t M = input_tensor.physical_volume() / K_in;
    uint32_t K = two_input_split ? static_cast<uint32_t>(in1_tensor_shape[-2]) : K_in;
    uint32_t N = in1_tensor_shape[-1];

    uint32_t M_tiles = M / tt::constants::TILE_HEIGHT;
    uint32_t K_tiles = K / tt::constants::TILE_WIDTH;
    uint32_t N_tiles = N / tt::constants::TILE_WIDTH;

    // Compute N_tiles_per_chunk for splitting
    const uint32_t N_tiles_per_chunk = N_tiles / N_chunks;

    auto [default_M_block_tiles, default_K_block_tiles, default_N_block_tiles, default_subblock_h, default_subblock_w] =
        determine_default_block_sizes(M, K, N, fp32_dest_acc_en);

    /**
     * TODO: Pick optimal subblock sizes. Currently a simple default is used.
     */
    uint32_t subblock_h = config.has_value() ? config.value().subblock_h : default_subblock_h;
    uint32_t subblock_w = config.has_value() ? config.value().subblock_w : default_subblock_w;

    uint32_t M_block_tiles = config.has_value() ? config.value().M_block_size : default_M_block_tiles;
    uint32_t K_block_tiles = config.has_value() ? config.value().K_block_size : default_K_block_tiles;
    uint32_t N_block_tiles = config.has_value() ? config.value().N_block_size : default_N_block_tiles;

    /**
     * We originally saw that for non-square outputs, N > M was significantly faster than M > N.
     * This is because originally, the in0 DM kernel was responsible for reading in0 and writing output.
     * When M > N, the in0 DM kernel has more data to read on top of its responsibility to write output.
     *
     * An optimization is to have the DM kernel with less data to read handle writes, and transpose the core_grid
     * to keep NOC usage consistent. With this optimization, N > M performance is symmetric with M > N.
     *
     * The smaller input read and mcast is always across a row of cores (x, y): (0, core_y) -> (grid_size.x-1, core_y)
     * The larger input read and mcast is always across a column of cores (x, y): (core_x, 0) -> (core_x. grid_size.y-1)
     *
     * Output is always written by DM reading the smaller input.
     *
     * Small input + output DM always runs on RISCV_1, NOC_1
     * Large input DM always runs on RISCV_0, NOC_0
     */

    auto small_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    auto small_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_1;
    auto large_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    auto large_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_0;

    // Transpose core grid if the output is wide (M > N)
    // If transpose core grid, we parallelize M on cores_x and N on cores_y and swap the NOCs and RISCVs
    bool transpose_core_grid = M > N;

    auto in0_noc = transpose_core_grid ? large_input_noc : small_input_noc;
    auto in0_risc = transpose_core_grid ? large_input_risc : small_input_risc;
    uint32_t in0_parallel_axis_cores = transpose_core_grid ? grid_size.x : grid_size.y;

    auto in1_noc = transpose_core_grid ? small_input_noc : large_input_noc;
    auto in1_risc = transpose_core_grid ? small_input_risc : large_input_risc;
    uint32_t in1_parallel_axis_cores = transpose_core_grid ? grid_size.y : grid_size.x;

    /**
     * We pad the input dimensions to the nearest multiple of the parallelization factor.
     *
     * Each core is assigned a certain number of tiles in M and N to compute.
     * Within a core, tiles are blocked by M_block_tiles and N_block_tiles.
     * Most output blocks are the full block size, but the last block in M or N can be partial.
     */
    uint32_t padded_M_tiles = tt::round_up(M_tiles, in0_parallel_axis_cores);
    uint32_t padded_K_tiles = tt::round_up(K_tiles, K_block_tiles);

    uint32_t padded_N_tiles;
    uint32_t N_tiles_per_core;
    if (fuse_swiglu) {
        // Partition on gate/up PAIRS (= output tiles), so every core's weight-tile range is
        // 2 * (pairs per core): even, and never splitting a pair across cores.
        uint32_t out_N_tiles = N_tiles / 2;
        uint32_t padded_out_N_tiles = tt::round_up(out_N_tiles, in1_parallel_axis_cores);
        padded_N_tiles = 2 * padded_out_N_tiles;
        N_tiles_per_core = 2 * (padded_out_N_tiles / in1_parallel_axis_cores);
    } else {
        padded_N_tiles = tt::round_up(N_tiles, in1_parallel_axis_cores);
        N_tiles_per_core = padded_N_tiles / in1_parallel_axis_cores;
    }

    uint32_t M_tiles_per_core = padded_M_tiles / in0_parallel_axis_cores;

    uint32_t K_blocks = padded_K_tiles / K_block_tiles;

    uint32_t M_blocks_per_core = tt::div_up(M_tiles_per_core, M_block_tiles);
    uint32_t N_blocks_per_core = tt::div_up(N_tiles_per_core, N_block_tiles);

    if (fuse_swiglu) {
        // The gate/up tile pairs are interleaved along N (gate=2p, up=2p+1). Every core's
        // N range and every N block must start on an even tile and span an even number of
        // tiles so a pair is never split across cores or blocks.
        TT_FATAL(
            N_tiles % 2 == 0 && N_tiles_per_core % 2 == 0 && N_block_tiles % 2 == 0,
            "minimal_matmul fuse_swiglu requires N_tiles ({}), N_tiles_per_core ({}) and N_block_tiles ({}) all even",
            N_tiles,
            N_tiles_per_core,
            N_block_tiles);
    }

    log_debug(tt::LogOp, "M_tiles_per_core: {}", M_tiles_per_core);
    log_debug(tt::LogOp, "N_tiles_per_core: {}", N_tiles_per_core);
    log_debug(tt::LogOp, "M_blocks_per_core: {}", M_blocks_per_core);
    log_debug(tt::LogOp, "N_blocks_per_core: {}", N_blocks_per_core);

    uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;
    uint32_t in2_block_num_tiles = N_block_tiles;

    const uint32_t double_buffer_factor = 2;
    uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    // TODO: consider not double buffering the output
    // SwiGLU emits half the N tiles per block (one per gate/up pair), so the output DFB only
    // needs to hold half a block. The intermediate DFB still holds the full (2N) block.
    uint32_t out_block_num_tiles_written = fuse_swiglu ? (out_block_num_tiles / 2) : out_block_num_tiles;
    uint32_t out_cb_num_tiles = out_block_num_tiles_written * double_buffer_factor;
    uint32_t interm_cb_num_tiles = out_block_num_tiles;  // not double buffered
    uint32_t in2_cb_num_tiles = in2_block_num_tiles;     // not double buffered

    auto core_0_0 = CoreCoord{0, 0};
    auto core_0_1 = CoreCoord{0, 1};
    auto core_1_0 = CoreCoord{1, 0};
    auto core_endx_0 = CoreCoord{grid_size.x - 1, 0};
    auto core_0_endy = CoreCoord{0, grid_size.y - 1};
    auto core_endx_endy = CoreCoord{grid_size.x - 1, grid_size.y - 1};

    auto in0_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_endx_0 : core_0_endy);
    auto in0_receiver_cores = CoreRange(transpose_core_grid ? core_0_1 : core_1_0, core_endx_endy);
    auto in1_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_0_endy : core_endx_0);
    auto in1_receiver_cores = CoreRange(transpose_core_grid ? core_1_0 : core_0_1, core_endx_endy);

    ProgramSpec spec;
    spec.name = "minimal_matmul";

    /**
     * Semaphores. Names replace the sequential ids the legacy factory handed out; the kernels bind
     * them by name. The non-zero initial values are carried over from the legacy factory.
     */
    const SemaphoreSpecName SEM_IN0_SENDER{"in0_sender"};
    const SemaphoreSpecName SEM_IN0_RECEIVER{"in0_receiver"};
    const SemaphoreSpecName SEM_IN0_VALID{"in0_valid"};
    const SemaphoreSpecName SEM_IN1_SENDER{"in1_sender"};
    const SemaphoreSpecName SEM_IN1_RECEIVER{"in1_receiver"};
    const SemaphoreSpecName SEM_IN1_VALID{"in1_valid"};

    for (const auto& [name, initial_value] : std::initializer_list<std::pair<SemaphoreSpecName, uint32_t>>{
             {SEM_IN0_SENDER, INVALID},
             {SEM_IN0_RECEIVER, INVALID},
             {SEM_IN0_VALID, VALID},
             {SEM_IN1_SENDER, INVALID},
             {SEM_IN1_RECEIVER, INVALID},
             {SEM_IN1_VALID, VALID},
         }) {
        SemaphoreSpec sem{
            .unique_id = name,
            .target_nodes = CoreRangeSet(core_grid),
        };
        sem.advanced_options.initial_value = initial_value;
        spec.semaphores.push_back(std::move(sem));
    }

    /**
     * Dataflow buffers. None are tensor-backed, so none needs a borrowed_from or a run override.
     */
    auto push_dfb = [&spec](const DFBSpecName& name, uint32_t entry_size, uint32_t num_entries, tt::DataFormat format) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = name,
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = format,
        });
    };

    push_dfb(DFB_IN0, in0_tile_size, in0_cb_num_tiles, in0_data_format);
    push_dfb(DFB_IN1, in1_tile_size, in1_cb_num_tiles, in1_data_format);
    push_dfb(DFB_OUT, out_tile_size, out_cb_num_tiles, output_data_format);
    push_dfb(DFB_INTERMEDIATE, intermediate_tile_size, interm_cb_num_tiles, intermediate_data_format);

    if (use_bias) {
        push_dfb(DFB_IN2, in2_tile_size, in2_cb_num_tiles, in2_data_format);
    }

    // Dataflow buffers for fused ternary inputs
    auto ternary_a_data_format = in1_data_format;
    auto ternary_c_data_format = in1_data_format;
    if (use_fused_ternary) {
        ternary_a_data_format = tt::tt_metal::datatype_to_dataformat_converter(fused_ternary_input_a.value().dtype());
        auto ternary_a_tile_size = tt::tile_size(ternary_a_data_format);

        TT_FATAL(ternary_a_tile_size == in1_tile_size, "ternary_a_tile_size must be equal to in1_tile_size");
        TT_FATAL(ternary_a_data_format == in1_data_format, "ternary_a_data_format must be equal to in1_data_format");
        uint32_t ternary_a_num_tiles = out_block_num_tiles;  // Same as output block, not double buffered

        push_dfb(DFB_TERNARY_A, ternary_a_tile_size, ternary_a_num_tiles, ternary_a_data_format);

        ternary_c_data_format = tt::tt_metal::datatype_to_dataformat_converter(fused_ternary_input_b.value().dtype());
        auto ternary_c_tile_size = tt::tile_size(ternary_c_data_format);
        uint32_t ternary_c_num_tiles = N_block_tiles;  // Single row (like bias), broadcast across M

        push_dfb(DFB_TERNARY_B, ternary_c_tile_size, ternary_c_num_tiles, ternary_c_data_format);
    }

    /**
     * Tensor parameters. One per distinct tensor the kernels access; the N outputs additionally get
     * a TensorBindingSequence so the kernels can walk them positionally (their count is a CTA).
     */
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TP_IN0, .spec = input_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = TP_IN1, .spec = weight_tensor.tensor_spec()});
    if (use_bias) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TP_IN2, .spec = bias_tensor.value().tensor_spec()});
    }
    if (two_input_split) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TP_IN3, .spec = optional_input_tensor.value().tensor_spec()});
    }
    if (use_fused_ternary) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TP_TERNARY_A, .spec = fused_ternary_input_a.value().tensor_spec()});
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TP_TERNARY_B, .spec = fused_ternary_input_b.value().tensor_spec()});
    }
    std::vector<std::string> output_accessor_names;
    output_accessor_names.reserve(output_tensors.size());
    for (uint32_t chunk = 0; chunk < output_tensors.size(); ++chunk) {
        const std::string name = output_param_name(chunk);
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = TensorParamName{name}, .spec = output_tensors[chunk].tensor_spec()});
        output_accessor_names.push_back(name);
    }

    std::map<std::string, std::string> defines;
    if (use_bias) {
        defines["FUSE_BIAS"] = "1";
    }

    if (fuse_swiglu) {
        defines["FUSE_SWIGLU"] = "1";
    }

    if (use_fused_ternary) {
        defines["FUSE_TERNARY"] = "1";

        // Workaround for LLK bug (https://github.com/tenstorrent/tt-llk/issues/1338)
        // - If ternary_b / gate is float32 then use unary_bcast (row broadcast) + mul_binary_tile (accurate)
        // - If ternary_b / gate is bfloat16 then use mul_tiles_bcast (row broadcast) (workaround)
        if (fused_ternary_input_b.value().dtype() == DataType::FLOAT32) {
            defines["TERNARY_B_IS_FLOAT32"] = "1";
        }
    }

    // Fused concatenation of in0: only the in0 SENDER reads from the two source buffers; the
    // receiver gets the assembled block by mcast and needs no split.
    std::map<std::string, std::string> in0_concat_defines;
    if (two_input_split) {
        in0_concat_defines = defines;
        in0_concat_defines["IN0_VIRTUAL_CONCAT"] = "1";
        // Split point = input_tensor's own K width (prefix half), in tiles.
        in0_concat_defines["IN0_K_SPLIT_TILES"] = std::to_string(K_in / tt::constants::TILE_WIDTH);
    }

    /**
     * Create kernels
     */

    bool in0_is_output_writer = !transpose_core_grid;
    bool in1_is_output_writer = transpose_core_grid;

    // is_output_writer was a CTA in the legacy kernels. It is a define now because it decides which
    // DFBs each instance binds (dfb::out on the writer, dfb::in2 / dfb::ternary_* on the other),
    // and `if constexpr` still name-looks-up the discarded branch.
    auto dm_defines = [&](bool is_output_writer, const std::map<std::string, std::string>& base) {
        std::map<std::string, std::string> d = base;
        if (is_output_writer) {
            d["IS_OUTPUT_WRITER"] = "1";
        }
        return to_defines(d);
    };

    // Shared CTA set for the in0/in1 kernels; only the input tile size (named for its own input)
    // and the injector flag differ.
    auto dm_compile_time_args =
        [&](const std::string& input_tile_size_name, uint32_t input_tile_size, bool is_injector_core) {
            return KernelSpec::CompileTimeArgs{
                {"M_tiles", M_tiles},
                {"padded_M_tiles", padded_M_tiles},
                {"K_tiles", K_tiles},
                {"padded_K_tiles", padded_K_tiles},
                {"N_tiles", N_tiles},
                {"padded_N_tiles", padded_N_tiles},
                {"M_block_tiles", M_block_tiles},
                {"K_block_tiles", K_block_tiles},
                {"N_block_tiles", N_block_tiles},
                {"M_blocks_per_core", M_blocks_per_core},
                {"N_blocks_per_core", N_blocks_per_core},
                {"out_tile_size", out_tile_size},
                {"in2_tile_size", in2_tile_size},
                {"is_injector_core", static_cast<uint32_t>(is_injector_core)},
                {"N_chunks", N_chunks},
                {"N_tiles_per_chunk", N_tiles_per_chunk},
                {input_tile_size_name, input_tile_size},
            };
        };

    // Runtime-arg names, in the order the legacy kernels read them (order is immaterial to the
    // named-arg mechanism; kept for reviewability against the legacy list).
    auto dm_runtime_arg_names = [&](const std::string& prefix) {
        KernelSpec::RuntimeArgSchema schema;
        schema.runtime_arg_names = {
            "is_sink_core",
            prefix + "_dest_noc_x",
            prefix + "_dest_noc_y",
            prefix + "_sender_noc_x",
            prefix + "_sender_noc_y",
            "M_start_tile",
            "M_end_tile",
            "N_start_tile",
            "N_end_tile",
            "defer_write_k_block",
            "max_defer_write_k_block",
        };
        if (use_fused_ternary) {
            schema.runtime_arg_names.push_back("broadcast_ternary_b");
        }
        return schema;
    };

    // Tensor bindings shared by every DM kernel: its own input, the N outputs, and the optional
    // bias / ternary tensors. The in0 kernels additionally bind the fused-concat second source.
    auto dm_tensor_bindings = [&](const TensorParamName& own_input, bool bind_in3) {
        Group<TensorBinding> bindings;
        bindings.push_back(TensorBinding{.tensor_parameter_name = own_input, .accessor_name = *own_input});
        if (use_bias) {
            bindings.push_back(TensorBinding{.tensor_parameter_name = TP_IN2, .accessor_name = *TP_IN2});
        }
        if (bind_in3) {
            bindings.push_back(TensorBinding{.tensor_parameter_name = TP_IN3, .accessor_name = *TP_IN3});
        }
        if (use_fused_ternary) {
            bindings.push_back(TensorBinding{.tensor_parameter_name = TP_TERNARY_A, .accessor_name = *TP_TERNARY_A});
            bindings.push_back(TensorBinding{.tensor_parameter_name = TP_TERNARY_B, .accessor_name = *TP_TERNARY_B});
        }
        for (const auto& name : output_accessor_names) {
            bindings.push_back(TensorBinding{.tensor_parameter_name = TensorParamName{name}, .accessor_name = name});
        }
        return bindings;
    };

    // DFB bindings for a DM kernel: it always produces its own input DFB; the output writer drains
    // the output DFB, and the other instance fills bias / ternary.
    auto dm_dfb_bindings = [&](const DFBSpecName& own_input, bool is_output_writer) {
        Group<DFBBinding> bindings;
        bindings.push_back(producer_of(own_input));
        if (is_output_writer) {
            bindings.push_back(consumer_of(DFB_OUT));
        } else {
            if (use_bias) {
                bindings.push_back(producer_of(DFB_IN2));
            }
            if (use_fused_ternary) {
                bindings.push_back(producer_of(DFB_TERNARY_A));
                bindings.push_back(producer_of(DFB_TERNARY_B));
            }
        }
        return bindings;
    };

    auto dm_semaphore_bindings =
        [](const SemaphoreSpecName& sender, const SemaphoreSpecName& receiver, const SemaphoreSpecName& valid) {
            Group<SemaphoreBinding> bindings;
            bindings.push_back(SemaphoreBinding{.semaphore_spec_name = sender, .accessor_name = *sender});
            bindings.push_back(SemaphoreBinding{.semaphore_spec_name = receiver, .accessor_name = *receiver});
            bindings.push_back(SemaphoreBinding{.semaphore_spec_name = valid, .accessor_name = *valid});
            return bindings;
        };

    // The N output bindings are also exposed as a positional sequence, so the kernel can build its
    // accessor tuple with make_tensor_accessors(tensor::outputs) without naming each one.
    KernelAdvancedOptions dm_advanced_options;
    dm_advanced_options.tensor_binding_sequences.push_back(
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "outputs", .members = output_accessor_names});

    auto make_dm_kernel = [&](const KernelSpecName& name,
                              const char* source,
                              uint32_t input_tile_size,
                              bool is_injector_core,
                              bool is_output_writer,
                              const DFBSpecName& own_input,
                              const TensorParamName& own_input_tensor,
                              bool bind_in3,
                              const std::string& rt_prefix,
                              const std::map<std::string, std::string>& base_defines,
                              tt::tt_metal::DataMovementProcessor processor,
                              tt::tt_metal::NOC noc,
                              const SemaphoreSpecName& sem_sender,
                              const SemaphoreSpecName& sem_receiver,
                              const SemaphoreSpecName& sem_valid) {
        auto cta = dm_compile_time_args(*own_input + "_tile_size", input_tile_size, is_injector_core);

        KernelSpec kernel{
            .unique_id = name,
            .source = std::filesystem::path(source),
            .compiler_options = {.defines = dm_defines(is_output_writer, base_defines)},
            .dfb_bindings = dm_dfb_bindings(own_input, is_output_writer),
            .semaphore_bindings = dm_semaphore_bindings(sem_sender, sem_receiver, sem_valid),
            .tensor_bindings = dm_tensor_bindings(own_input_tensor, bind_in3),
            .compile_time_args = std::move(cta),
            .runtime_arg_schema = dm_runtime_arg_names(rt_prefix),
            .hw_config = DataMovementGen1Config{.processor = processor, .noc = noc},
            .advanced_options = dm_advanced_options,
        };
        return kernel;
    };

    constexpr const char* kIn0Source =
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender_metal2.cpp";
    constexpr const char* kIn1Source =
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out_metal2.cpp";

    spec.kernels.push_back(make_dm_kernel(
        K_IN0_SENDER,
        kIn0Source,
        in0_tile_size,
        /*is_injector_core=*/true,
        in0_is_output_writer,
        DFB_IN0,
        TP_IN0,
        /*bind_in3=*/two_input_split,
        "in0",
        two_input_split ? in0_concat_defines : defines,
        in0_risc,
        in0_noc,
        SEM_IN0_SENDER,
        SEM_IN0_RECEIVER,
        SEM_IN0_VALID));
    spec.kernels.push_back(make_dm_kernel(
        K_IN0_RECEIVER,
        kIn0Source,
        in0_tile_size,
        /*is_injector_core=*/false,
        in0_is_output_writer,
        DFB_IN0,
        TP_IN0,
        /*bind_in3=*/false,
        "in0",
        defines,
        in0_risc,
        in0_noc,
        SEM_IN0_SENDER,
        SEM_IN0_RECEIVER,
        SEM_IN0_VALID));
    spec.kernels.push_back(make_dm_kernel(
        K_IN1_SENDER,
        kIn1Source,
        in1_tile_size,
        /*is_injector_core=*/true,
        in1_is_output_writer,
        DFB_IN1,
        TP_IN1,
        /*bind_in3=*/false,
        "in1",
        defines,
        in1_risc,
        in1_noc,
        SEM_IN1_SENDER,
        SEM_IN1_RECEIVER,
        SEM_IN1_VALID));
    spec.kernels.push_back(make_dm_kernel(
        K_IN1_RECEIVER,
        kIn1Source,
        in1_tile_size,
        /*is_injector_core=*/false,
        in1_is_output_writer,
        DFB_IN1,
        TP_IN1,
        /*bind_in3=*/false,
        "in1",
        defines,
        in1_risc,
        in1_noc,
        SEM_IN1_SENDER,
        SEM_IN1_RECEIVER,
        SEM_IN1_VALID));

    /**
     * Compute kernel.
     */
    auto compute_defines = defines;
    std::map<std::string, std::string> compute_activation_defines;
    if (fused_activation.has_value()) {
        compute_activation_defines = ttnn::operations::unary::utils::get_defines(
            fused_activation.value().op_type,
            fused_activation.value().params,
            "ACTIVATION",
            "fused_act_dst_id",
            output_tensors[0].dtype());
    }
    compute_defines.merge(compute_activation_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, compute_defines, ttnn::get_throttle_level(compute_kernel_config));

    auto compute_hw = to_compute_hardware_config(device->arch(), compute_kernel_config);
    auto& compute_gen1 = std::get<ComputeGen1Config>(compute_hw);
    compute_gen1.double_buffer_dest = true;
    if (fp32_dest_acc_en) {
        // Metal 2.0 requires an explicit unpack mode for Float32 DFBs when enable_32_bit_dest is set.
        const std::vector<std::pair<DFBSpecName, tt::DataFormat>> compute_consumed{
            {DFB_IN0, in0_data_format},
            {DFB_IN1, in1_data_format},
            {DFB_INTERMEDIATE, intermediate_data_format},
            {DFB_IN2, in2_data_format},
            {DFB_TERNARY_A, ternary_a_data_format},
            {DFB_TERNARY_B, ternary_c_data_format},
        };
        for (const auto& [dfb, format] : compute_consumed) {
            // Only DFBs this kernel actually binds may appear in unpack_modes.
            const bool bound = (dfb == DFB_IN0 || dfb == DFB_IN1 || dfb == DFB_INTERMEDIATE) ||
                               (dfb == DFB_IN2 && use_bias) ||
                               ((dfb == DFB_TERNARY_A || dfb == DFB_TERNARY_B) && use_fused_ternary);
            if (bound && format == tt::DataFormat::Float32) {
                compute_gen1.unpack_modes.emplace(dfb, tt::tt_metal::UnpackMode::UnpackToSrc);
            }
        }
    }

    Group<DFBBinding> compute_dfb_bindings;
    compute_dfb_bindings.push_back(consumer_of(DFB_IN0));
    compute_dfb_bindings.push_back(consumer_of(DFB_IN1));
    compute_dfb_bindings.push_back(producer_of(DFB_OUT));
    // The intermediate accumulator is touched only by this kernel: self-loop it.
    compute_dfb_bindings.push_back(producer_of(DFB_INTERMEDIATE));
    compute_dfb_bindings.push_back(consumer_of(DFB_INTERMEDIATE));
    if (use_bias) {
        compute_dfb_bindings.push_back(consumer_of(DFB_IN2));
    }
    if (use_fused_ternary) {
        compute_dfb_bindings.push_back(consumer_of(DFB_TERNARY_A));
        compute_dfb_bindings.push_back(consumer_of(DFB_TERNARY_B));
    }

    KernelSpec::RuntimeArgSchema compute_schema;
    compute_schema.runtime_arg_names = {"M_start_tile", "M_end_tile", "N_start_tile", "N_end_tile"};
    if (use_fused_ternary) {
        compute_schema.runtime_arg_names.push_back("fused_ternary_scalar");
        compute_schema.runtime_arg_names.push_back("broadcast_ternary_b");
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = K_COMPUTE,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute_metal2.cpp"),
        // Legacy ComputeConfigDescriptor resolves opt_level to O3; Metal 2.0 defaults to O2.
        .compiler_options =
            {.defines = to_defines(compute_defines), .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {
                {"K_num_blocks", K_blocks},
                {"M_block_tiles", M_block_tiles},
                {"K_block_tiles", K_block_tiles},
                {"N_block_tiles", N_block_tiles},
                {"M_blocks_per_core", M_blocks_per_core},
                {"N_blocks_per_core", N_blocks_per_core},
                {"subblock_h", subblock_h},
                {"subblock_w", subblock_w},
            },
        .runtime_arg_schema = compute_schema,
        .hw_config = compute_hw,
    });

    /**
     * Work units. The sender/receiver core ranges partition the grid on each axis, so each node
     * carries exactly one in0 kernel, one in1 kernel, and compute -- four distinct pairings.
     */
    auto in0_kernel_for = [&](uint32_t in1_idx) { return in1_idx == 0 ? K_IN0_SENDER : K_IN0_RECEIVER; };
    auto in1_kernel_for = [&](uint32_t in0_idx) { return in0_idx == 0 ? K_IN1_SENDER : K_IN1_RECEIVER; };

    auto add_work_unit = [&](const std::string& name, const CoreRange& nodes, uint32_t in0_idx, uint32_t in1_idx) {
        Group<KernelSpecName> kernels;
        kernels.push_back(in0_kernel_for(in1_idx));
        kernels.push_back(in1_kernel_for(in0_idx));
        kernels.push_back(K_COMPUTE);
        spec.work_units.push_back(
            WorkUnitSpec{.name = name, .kernels = std::move(kernels), .target_nodes = CoreRangeSet(nodes)});
    };

    // Representative (in0_idx, in1_idx) for each of the four regions, computed the same way the
    // per-core loop below does.
    auto idx_for = [&](const CoreCoord& c) {
        return std::pair<uint32_t, uint32_t>{
            transpose_core_grid ? static_cast<uint32_t>(c.x) : static_cast<uint32_t>(c.y),
            transpose_core_grid ? static_cast<uint32_t>(c.y) : static_cast<uint32_t>(c.x)};
    };
    for (const auto& [name, nodes] : std::initializer_list<std::pair<const char*, CoreRange>>{
             {"corner", CoreRange(core_0_0, core_0_0)},
             {"top_row", CoreRange(core_1_0, core_endx_0)},
             {"left_col", CoreRange(core_0_1, core_0_endy)},
             {"interior", CoreRange(CoreCoord{1, 1}, core_endx_endy)},
         }) {
        auto [in0_idx, in1_idx] = idx_for(nodes.start_coord);
        add_work_unit(name, nodes, in0_idx, in1_idx);
    }

    /**
     * The receiver writer cores defer their writes in order to reduce NOC congestion.
     * Further, the amount of K_blocks they defer by depends on their core coordinate.
     * If we have core_grid.x cores, we'd want to evenly stride the K_blocks they defer by.
     * For first pass, it's easy enough to use core_grid.x
     */
    uint32_t k_blocks_per_core =
        tt::div_up(K_blocks, (transpose_core_grid ? in1_parallel_axis_cores : in0_parallel_axis_cores));

    auto cores = corerange_to_cores(core_grid, num_cores, true);

    uint32_t max_defer_write_k_block = 0;
    for (const auto& c : cores) {
        uint32_t dwk = std::min(static_cast<uint32_t>(c.y) * k_blocks_per_core, K_blocks - 1);
        max_defer_write_k_block = std::max(max_defer_write_k_block, dwk);
    }

    uint32_t ternary_b_broadcast = 0u;
    if (use_fused_ternary) {
        uint32_t ternary_b_M_tiles = fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
        ternary_b_broadcast = ternary_b_M_tiles == 1 ? 1u : 0u;
    }

    ProgramRunArgs run_args;
    ProgramRunArgs::KernelRunArgs in0_sender_args{.kernel = K_IN0_SENDER};
    ProgramRunArgs::KernelRunArgs in0_receiver_args{.kernel = K_IN0_RECEIVER};
    ProgramRunArgs::KernelRunArgs in1_sender_args{.kernel = K_IN1_SENDER};
    ProgramRunArgs::KernelRunArgs in1_receiver_args{.kernel = K_IN1_RECEIVER};
    ProgramRunArgs::KernelRunArgs compute_args{.kernel = K_COMPUTE};

    // NOTE: Uniform per-core M/N ranges are required for DM forward handshakes to match across links.
    // If neighboring cores along a forwarding chain iterate different (M,N) counts, the sender can wait
    // for requests that the receiver will never issue, leading to deadlock. Keep the original uniform
    // div_up-based ranges for M and N.

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        uint32_t in0_idx = transpose_core_grid ? core.x : core.y;
        uint32_t in1_idx = transpose_core_grid ? core.y : core.x;

        CoreCoord left_core = {(std::size_t)0, (std::size_t)core.y};
        CoreCoord top_core = {(std::size_t)core.x, (std::size_t)0};

        auto [in0_core_order, in0_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            in1_parallel_axis_cores,
            in0_noc,
            /*axis_is_x_when_not_transposed=*/true,
            /*initial_endpoint=*/(transpose_core_grid ? top_core : left_core));

        auto [in1_core_order, in1_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            in0_parallel_axis_cores,
            in1_noc,
            /*axis_is_x_when_not_transposed=*/false,
            /*initial_endpoint=*/(transpose_core_grid ? left_core : top_core));

        auto in0_prev_core = clamped_prev(in0_core_order, in0_core_order_index);
        auto in0_next_core = clamped_next(in0_core_order, in0_core_order_index);
        auto in1_prev_core = clamped_prev(in1_core_order, in1_core_order_index);
        auto in1_next_core = clamped_next(in1_core_order, in1_core_order_index);

        auto in0_prev_core_physical = device->worker_core_from_logical_core(in0_prev_core);
        auto in0_next_core_physical = device->worker_core_from_logical_core(in0_next_core);
        auto in1_prev_core_physical = device->worker_core_from_logical_core(in1_prev_core);
        auto in1_next_core_physical = device->worker_core_from_logical_core(in1_next_core);

        /**
         * NOTE: Some cores are doing unnecessary work, on blocks which are processed just to make
         * the total number of blocks divisible by the number of cores.
         * We can't yet get rid of these blocks, since the receiver cores must ack
         * all blocks that sender cores are expected to send.
         */
        uint32_t M_start_tile = M_tiles_per_core * in0_idx;
        uint32_t M_end_tile = M_tiles_per_core * (in0_idx + 1);
        uint32_t N_start_tile = N_tiles_per_core * in1_idx;
        uint32_t N_end_tile = N_tiles_per_core * (in1_idx + 1);

        // Defer write to K block with same coordinate as core
        // The writer receiver cores always have core.x > 0
        uint32_t defer_write_k_block = std::min(static_cast<uint32_t>(core.y) * k_blocks_per_core, K_blocks - 1);

        bool is_in0_sink = core == in0_core_order.back();
        bool is_in1_sink = core == in1_core_order.back();

        // AddRuntimeArgsForNode takes an initializer_list, so the always-present args go in one
        // call and the conditional ternary arg in a second.
        auto& in0_rt = (in1_idx == 0 ? in0_sender_args : in0_receiver_args).runtime_arg_values;
        AddRuntimeArgsForNode(
            in0_rt,
            core,
            {{"is_sink_core", static_cast<uint32_t>(is_in0_sink)},
             {"in0_dest_noc_x", (std::uint32_t)in0_next_core_physical.x},
             {"in0_dest_noc_y", (std::uint32_t)in0_next_core_physical.y},
             {"in0_sender_noc_x", (std::uint32_t)in0_prev_core_physical.x},
             {"in0_sender_noc_y", (std::uint32_t)in0_prev_core_physical.y},
             {"M_start_tile", M_start_tile},
             {"M_end_tile", M_end_tile},
             {"N_start_tile", N_start_tile},
             {"N_end_tile", N_end_tile},
             {"defer_write_k_block", defer_write_k_block},
             {"max_defer_write_k_block", max_defer_write_k_block}});

        auto& in1_rt = (in0_idx == 0 ? in1_sender_args : in1_receiver_args).runtime_arg_values;
        AddRuntimeArgsForNode(
            in1_rt,
            core,
            {{"is_sink_core", static_cast<uint32_t>(is_in1_sink)},
             {"in1_dest_noc_x", (std::uint32_t)in1_next_core_physical.x},
             {"in1_dest_noc_y", (std::uint32_t)in1_next_core_physical.y},
             {"in1_sender_noc_x", (std::uint32_t)in1_prev_core_physical.x},
             {"in1_sender_noc_y", (std::uint32_t)in1_prev_core_physical.y},
             {"M_start_tile", M_start_tile},
             {"M_end_tile", M_end_tile},
             {"N_start_tile", N_start_tile},
             {"N_end_tile", N_end_tile},
             {"defer_write_k_block", defer_write_k_block},
             {"max_defer_write_k_block", max_defer_write_k_block}});

        AddRuntimeArgsForNode(
            compute_args.runtime_arg_values,
            core,
            {{"M_start_tile", M_start_tile},
             {"M_end_tile", M_end_tile},
             {"N_start_tile", N_start_tile},
             {"N_end_tile", N_end_tile}});

        if (use_fused_ternary) {
            AddRuntimeArgsForNode(in0_rt, core, {{"broadcast_ternary_b", ternary_b_broadcast}});
            AddRuntimeArgsForNode(in1_rt, core, {{"broadcast_ternary_b", ternary_b_broadcast}});
            // fused_ternary_scalar is part of the program hash, so a cache hit guarantees this
            // value is unchanged; it does not need re-applying per dispatch.
            AddRuntimeArgsForNode(
                compute_args.runtime_arg_values,
                core,
                {{"fused_ternary_scalar",
                  *reinterpret_cast<const uint32_t*>(&operation_attributes.fused_ternary_scalar.value())},
                 {"broadcast_ternary_b", ternary_b_broadcast}});
        }
    }

    run_args.kernel_run_args.push_back(std::move(in0_sender_args));
    run_args.kernel_run_args.push_back(std::move(in0_receiver_args));
    run_args.kernel_run_args.push_back(std::move(in1_sender_args));
    run_args.kernel_run_args.push_back(std::move(in1_receiver_args));
    run_args.kernel_run_args.push_back(std::move(compute_args));

    // Tensor arguments. The framework re-applies these on every program-cache hit, which is what
    // the legacy override_runtime_arguments did by hand.
    run_args.tensor_args.insert({TP_IN0, input_tensor.mesh_tensor()});
    run_args.tensor_args.insert({TP_IN1, weight_tensor.mesh_tensor()});
    if (use_bias) {
        run_args.tensor_args.insert({TP_IN2, bias_tensor.value().mesh_tensor()});
    }
    if (two_input_split) {
        run_args.tensor_args.insert({TP_IN3, optional_input_tensor.value().mesh_tensor()});
    }
    if (use_fused_ternary) {
        run_args.tensor_args.insert({TP_TERNARY_A, fused_ternary_input_a.value().mesh_tensor()});
        run_args.tensor_args.insert({TP_TERNARY_B, fused_ternary_input_b.value().mesh_tensor()});
    }
    for (uint32_t chunk = 0; chunk < output_tensors.size(); ++chunk) {
        run_args.tensor_args.insert({TensorParamName{output_param_name(chunk)}, output_tensors[chunk].mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
