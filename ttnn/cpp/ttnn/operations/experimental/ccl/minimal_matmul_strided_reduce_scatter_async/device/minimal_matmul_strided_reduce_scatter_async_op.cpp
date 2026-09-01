// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/device/minimal_matmul_strided_reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

using matmul_device_operation_t = ttnn::experimental::prim::MinimalMatmulDeviceOperation;

namespace ttnn::experimental::prim {

MinimalMatmulStridedReduceScatterAsync::program_factory_t
MinimalMatmulStridedReduceScatterAsync::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return MinimalMatmulStridedReduceScatterAsyncProgramFactory{};
}

void MinimalMatmulStridedReduceScatterAsync::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

void MinimalMatmulStridedReduceScatterAsync::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        attributes.dim == 3, "MinimalMatmulStridedReduceScatterAsync requires dim=3 for the ReduceScatter operation.");
    TT_FATAL(
        tensor_args.input_tensor.padded_shape()[0] == 1 && tensor_args.input_tensor.padded_shape()[1] == 1,
        "MinimalMatmulStridedReduceScatterAsync requires input tensor to have batch size of 1.");
    TT_FATAL(
        attributes.topology == ttnn::ccl::Topology::Ring,
        "MinimalMatmulStridedReduceScatterAsync only supports Ring topology.");

    // Delegate matmul validation (dtype, layout, shape, tile alignment, config/subblock
    // constraints, etc.) — but NOT the ternary checks, because the ternary inputs belong to
    // the RS output, not the matmul output.  The addcmul is fused at the RS final write step,
    // so ternary_a/b are shaped [M, N/ring_size], not [M, N].  Passing them here would cause
    // the matmul validator to reject the correct shape.
    auto to_mutable_opt = [](const std::optional<const Tensor>& opt) -> std::optional<Tensor> {
        return opt.has_value() ? std::optional<Tensor>(opt.value()) : std::nullopt;
    };

    // Delegate to the matmul validator. The fused-concat checks (two sources concatenable on K,
    // K's sum to the weight K) are driven by the presence of optional_input_tensor below.
    matmul_device_operation_t::validate_on_program_cache_miss(
        attributes.matmul_struct,
        matmul_device_operation_t::tensor_args_t{
            .input_tensor = tensor_args.input_tensor,
            .weight_tensor = tensor_args.weight_tensor,
            .bias_tensor = to_mutable_opt(tensor_args.bias),
            .optional_input_tensor = to_mutable_opt(tensor_args.mm_optional_input_tensor),
            .fused_ternary_input_a = std::nullopt,
            .fused_ternary_input_b = std::nullopt,
        });

    // Validate ternary (addcmul) inputs against the RS output shape [M, N/ring_size].
    // The addcmul is applied at the RS final write step, so the reference N is
    // N_mm / ring_size, not the full matmul output N.
    if (tensor_args.addcmul_input_tensor1.has_value() || tensor_args.addcmul_input_tensor2.has_value()) {
        TT_FATAL(
            tensor_args.addcmul_input_tensor1.has_value() && tensor_args.addcmul_input_tensor2.has_value(),
            "Both addcmul_input_tensor1 and addcmul_input_tensor2 must be provided together.");
        TT_FATAL(
            attributes.fused_ternary_scalar.has_value(),
            "fused_ternary_scalar must be set when addcmul inputs are provided.");

        const auto& ta = tensor_args.addcmul_input_tensor1.value();
        const auto& tb = tensor_args.addcmul_input_tensor2.value();

        auto dtype_supported = [](tt::tt_metal::DataType dt) {
            return dt == DataType::BFLOAT16 || dt == DataType::BFLOAT8_B || dt == DataType::BFLOAT4_B ||
                   dt == DataType::FLOAT32;
        };

        TT_FATAL(ta.storage_type() == StorageType::DEVICE, "addcmul_input_tensor1 must be on device");
        TT_FATAL(tb.storage_type() == StorageType::DEVICE, "addcmul_input_tensor2 must be on device");
        TT_FATAL(
            ta.device() == tensor_args.input_tensor.device(), "addcmul_input_tensor1 must be on same device as input");
        TT_FATAL(
            tb.device() == tensor_args.input_tensor.device(), "addcmul_input_tensor2 must be on same device as input");
        TT_FATAL(ta.buffer() != nullptr, "addcmul_input_tensor1 must be allocated");
        TT_FATAL(tb.buffer() != nullptr, "addcmul_input_tensor2 must be allocated");
        TT_FATAL(ta.layout() == Layout::TILE, "addcmul_input_tensor1 must be TILE layout");
        TT_FATAL(tb.layout() == Layout::TILE, "addcmul_input_tensor2 must be TILE layout");
        TT_FATAL(
            dtype_supported(ta.dtype()) && dtype_supported(tb.dtype()),
            "addcmul tensors must have supported dtypes (BFLOAT16, BFLOAT8_B, BFLOAT4_B, FLOAT32)");

        const uint32_t M = tensor_args.input_tensor.padded_shape()[-2];
        const uint32_t N_rs = tensor_args.weight_tensor.padded_shape()[-1] / attributes.ring_size;

        const auto& ta_shape = ta.logical_shape();
        const auto& tb_shape = tb.logical_shape();

        TT_FATAL(
            ta_shape[-2] == M && ta_shape[-1] == N_rs,
            "addcmul_input_tensor1 shape must match RS output [M={}, N/ring_size={}], got [{}, {}]",
            M,
            N_rs,
            ta_shape[-2],
            ta_shape[-1]);
        TT_FATAL(
            (tb_shape[-2] == 1 || tb_shape[-2] == M) && tb_shape[-1] == N_rs,
            "addcmul_input_tensor2 shape must be broadcast [1, N/ring_size={}] or full [M={}, N/ring_size={}], got "
            "[{}, {}]",
            N_rs,
            M,
            N_rs,
            tb_shape[-2],
            tb_shape[-1]);
    }

    // An L1 MM output opts into the L1 handoff (see compute_output_specs). Unwindowed, the
    // resident shard is Mt_per_core * Nt_per_core tiles on every matmul core for the life of the
    // tensor: past a point it fails allocation outright, and well before that it lowers the L1
    // floor enough that a LATER program's static circular buffers clash with it — a failure this
    // op cannot see and the caller cannot easily attribute (issue #52863). Require the window so
    // the footprint is bounded and explicit. W = M_blocks_per_core reproduces the full-residency
    // layout exactly for callers that genuinely want it.
    //
    // The minimal matmul resolves an omitted memory_config_mm as the INPUT's memory config
    // (minimal_matmul_device_operation.cpp compute_output_specs), so an L1 input with no explicit
    // request still produces an L1 MM output — resolve the effective config the same way, or that
    // path would silently bypass the window requirement.
    const bool l1_mm_output =
        attributes.matmul_struct.output_mem_config.value_or(tensor_args.input_tensor.memory_config()).buffer_type() ==
        tt::tt_metal::BufferType::L1;
    TT_FATAL(
        !l1_mm_output || attributes.mm_window_blocks.has_value(),
        "An L1 MM output (explicit memory_config_mm, or inherited from an L1 input when "
        "memory_config_mm is omitted) requires mm_window_blocks: the L1 handoff must bound its "
        "resident shard. Pass mm_window_blocks=2 (measured perf-neutral vs full residency), or "
        "mm_window_blocks=ceil(Mt_per_core / M_block_size) to keep the whole MM output resident.");
    TT_FATAL(
        !attributes.mm_window_blocks.has_value() || *attributes.mm_window_blocks >= 1,
        "mm_window_blocks must be >= 1, got {} — a zero-height window has no valid shard geometry "
        "(2 is the measured perf-neutral default).",
        *attributes.mm_window_blocks);

    // The windowed handoff also requires the caller-owned counter arrays. Without them the op
    // falls back to a private per-program L1 allocation, retained for the program's cached life —
    // and because L1 is handed out top-down, each such block permanently lowers the L1 floor:
    // the slow-motion version of the failure the window prevents. Detailed shape/layout checks
    // (L1 sharded, grid coverage, row width) happen at program build; this only requires presence.
    // Sizing: both are uint32, HEIGHT_SHARDED in L1 over the full compute grid (grid.x * grid.y
    // cores). mm_progress_counters rows need one slot per matmul core; mm_credit_counters rows
    // need one slot per RS reader (2 * num_links * num_workers_per_link). A square
    // [num_cores, num_cores] allocation covers both — see CCLManager.get_mm_progress_counters_buffer
    // / get_mm_credit_counters_buffer.
    if (attributes.mm_window_blocks.has_value()) {
        TT_FATAL(
            tensor_args.mm_progress_counters.has_value(),
            "mm_window_blocks requires a caller-owned mm_progress_counters tensor (uint32, L1 "
            "HEIGHT_SHARDED over the compute grid, one slot per matmul core per row); see "
            "CCLManager.get_mm_progress_counters_buffer.");
        TT_FATAL(
            tensor_args.mm_credit_counters.has_value(),
            "mm_window_blocks requires a caller-owned mm_credit_counters tensor (uint32, L1 "
            "HEIGHT_SHARDED over the compute grid, one slot per RS reader per row, i.e. at least "
            "2 * num_links * num_workers_per_link slots); see "
            "CCLManager.get_mm_credit_counters_buffer.");

        // Layout/size checks, on every call rather than only at program build: on a cached run the
        // factory's checks never rerun, and the address-stability guard alone would miss a
        // same-address reallocation of a smaller tensor. The factory keeps the exact checks that
        // need build-time information (resolved worker count, chosen core placement); these cover
        // what validation can know.
        const auto validate_counter_array = [&tensor_args](const Tensor& t, const char* name, uint32_t min_row_slots) {
            // The factory embeds t.buffer()->address() into kernels running on the input's device;
            // a counter allocated on a different device would pass the layout checks below while
            // pointing the kernels at an unrelated local L1 address.
            TT_FATAL(
                t.device() == tensor_args.input_tensor.device(),
                "{} must be allocated on the same device as the input tensor",
                name);
            TT_FATAL(
                t.memory_config().buffer_type() == tt::tt_metal::BufferType::L1 &&
                    t.memory_config().shard_spec().has_value(),
                "{} must be an L1 sharded tensor so that its row lands at the same local address on "
                "every core that reads it",
                name);
            TT_FATAL(t.dtype() == DataType::UINT32, "{} must be uint32, got {}", name, t.dtype());
            const uint32_t row_slots = t.memory_config().shard_spec()->shape[1];
            TT_FATAL(
                row_slots >= min_row_slots,
                "{} provides {} uint32 slots per row but at least {} are needed; allocate a "
                "[num_cores, num_cores] square over the full compute grid (see CCLManager)",
                name,
                row_slots,
                min_row_slots);
        };
        const auto device_grid = tensor_args.input_tensor.device()->compute_with_storage_grid_size();
        const uint32_t full_grid_slots = device_grid.x * device_grid.y;
        // Progress rows are indexed by MM core id over the full device grid (see the RS factory).
        validate_counter_array(*tensor_args.mm_progress_counters, "mm_progress_counters", full_grid_slots);
        // Credit rows need one slot per RS reader: 2 directions * num_links * workers-per-direction.
        // When num_workers_per_link is defaulted its value is resolved at build, so check the
        // lower bound here; the factory validates the exact count.
        const uint32_t min_rs_readers = 2 * attributes.num_links * attributes.num_workers_per_link.value_or(1);
        validate_counter_array(*tensor_args.mm_credit_counters, "mm_credit_counters", min_rs_readers);
    }

    // RS validation: checks we can perform without the (not-yet-created) MM output tensor.
    TT_FATAL(attributes.num_links > 0, "num_links must be greater than 0.");

    constexpr uint32_t expected_semaphores = 3;
    TT_FATAL(
        attributes.semaphore.size() == expected_semaphores,
        "Expected {} semaphores but got {}.",
        expected_semaphores,
        attributes.semaphore.size());

    // MM output N = weight last dim; its tile count must divide evenly across ring devices.
    const uint32_t N_tiles = tensor_args.weight_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(
        N_tiles % attributes.ring_size == 0,
        "MM output N_tiles ({}) must be divisible by ring_size ({}).",
        N_tiles,
        attributes.ring_size);

    // RS output memory layout must be one of the supported types.
    const auto rs_out_layout = attributes.rs_output_mem_config.memory_layout();
    TT_FATAL(
        rs_out_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED ||
            rs_out_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED ||
            rs_out_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
            rs_out_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        "Unsupported RS output memory layout.");
    if (rs_out_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(
            attributes.rs_output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1,
            "DRAM block sharding is not supported for RS output.");
    }
}

MinimalMatmulStridedReduceScatterAsync::spec_return_value_t
MinimalMatmulStridedReduceScatterAsync::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Output tensor[0]: MM output spec (= RS input)
    tt::tt_metal::TensorSpec mm_output_spec = matmul_device_operation_t::compute_output_specs(
        attributes.matmul_struct, {tensor_args.input_tensor, tensor_args.weight_tensor})[0];

    // Derive RS intermediate and output specs from the MM output shape
    auto mm_output_shape = mm_output_spec.logical_shape();

    // RS intermediate shape: same as MM output for Ring topology.
    // The default is DRAM, NOT the MM output's memory config: the intermediate is a full-size
    // [M, N] tensor, so inheriting an L1 MM config would silently place ~M*N bytes in L1
    // interleaved — far more than the handoff shard the caller opted into. A caller that wants
    // an L1 intermediate must say so explicitly.
    MemoryConfig rs_intermediate_mem_config =
        attributes.rs_intermediate_mem_config.value_or(MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});

    tt::tt_metal::TensorSpec rs_intermediate_spec(
        mm_output_shape,
        tt::tt_metal::TensorLayout(
            mm_output_spec.data_type(), mm_output_spec.page_config(), rs_intermediate_mem_config));

    // RS output shape: scatter dim divided by ring_size
    auto rs_output_shape = mm_output_shape;
    rs_output_shape[attributes.dim] /= attributes.ring_size;

    tt::tt_metal::TensorSpec rs_output_spec(
        rs_output_shape,
        tt::tt_metal::TensorLayout(
            mm_output_spec.data_type(), mm_output_spec.page_config(), attributes.rs_output_mem_config));

    // --- L1 handoff (step 1): block-shard the MM output across the matmul core grid so the RS reader
    // consumes it straight out of L1 instead of round-tripping through DRAM.
    //
    // Opt-in only. Without a window the resident shard is Mt_per_core * Nt_per_core tiles on every
    // matmul core, which for a large M crowds out the L1 that later programs need for their own
    // circular buffers — and past roughly Mt/gy * Nt/gx > bank capacity does not fit at all. A caller
    // that asked for a DRAM MM output therefore keeps getting one. Opting in means either requesting
    // an L1 MM output outright, or setting mm_window_blocks, which bounds the shard to W M blocks and
    // is only meaningful in L1. Tested against the RESOLVED matmul output spec rather than the raw
    // attribute: an omitted memory_config_mm inherits the input's memory config inside the matmul,
    // so an L1 input lands here too (validation has already required its window).
    const bool l1_mm_output = mm_output_spec.memory_config().buffer_type() == BufferType::L1;
    const bool use_l1_handoff = attributes.mm_window_blocks.has_value() || l1_mm_output;
    if (use_l1_handoff && attributes.matmul_struct.config.has_value() &&
        attributes.matmul_struct.config->compute_with_storage_grid_size.x > 0) {
        const auto grid = attributes.matmul_struct.config->compute_with_storage_grid_size;
        const uint32_t gx = grid.x;
        const uint32_t gy = grid.y;
        const uint32_t Mt = mm_output_shape[-2] / tt::constants::TILE_HEIGHT;
        const uint32_t Nt = mm_output_shape[-1] / tt::constants::TILE_WIDTH;
        const uint32_t Mt_per_core = (Mt + gy - 1) / gy;
        const uint32_t Nt_per_core = (Nt + gx - 1) / gx;

        // With mm_window_blocks=W the shard holds only W M blocks instead of all Mt_per_core rows,
        // and the matmul recycles slot m % W. The tensor then covers gy*W*mm_block_ht rows rather
        // than M, so it is no longer the full matmul result — see the attribute's doc comment.
        uint32_t shard_ht = Mt_per_core;
        auto windowed_shape = mm_output_shape;
        if (attributes.mm_window_blocks.has_value()) {
            // Checked here as well as in validate: create_output_tensors (and therefore this
            // function) runs BEFORE validate_on_program_cache_miss in the launch path, and a zero
            // window would otherwise reach TensorSpec's shard-grid check as a zero-height shard —
            // a SIGFPE, not an actionable error.
            TT_FATAL(
                *attributes.mm_window_blocks >= 1,
                "mm_window_blocks must be >= 1, got {} — a zero-height window has no valid shard "
                "geometry (2 is the measured perf-neutral default).",
                *attributes.mm_window_blocks);
            const uint32_t mm_block_ht = attributes.matmul_struct.config->M_block_size;
            shard_ht = attributes.mm_window_blocks.value() * mm_block_ht;
            windowed_shape[-2] = gy * shard_ht * tt::constants::TILE_HEIGHT;
        }

        const auto mm_shard_spec = tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1))),
            {shard_ht * tt::constants::TILE_HEIGHT, Nt_per_core * tt::constants::TILE_WIDTH});
        const auto mm_l1_sharded = MemoryConfig{TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, mm_shard_spec};
        mm_output_spec = tt::tt_metal::TensorSpec(
            windowed_shape,
            tt::tt_metal::TensorLayout(mm_output_spec.data_type(), mm_output_spec.page_config(), mm_l1_sharded));
    }

    return {mm_output_spec, rs_intermediate_spec, rs_output_spec};
}

MinimalMatmulStridedReduceScatterAsync::tensor_return_value_t
MinimalMatmulStridedReduceScatterAsync::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto tensor_specs = compute_output_specs(attributes, tensor_args);

    // MM output tensor
    ttnn::Tensor mm_output_tensor = create_device_tensor(tensor_specs[0], tensor_args.input_tensor.device());

    // RS intermediate tensor (use provided or create new)
    ttnn::Tensor rs_intermediate_tensor =
        tensor_args.optional_rs_intermediate_tensor.has_value()
            ? tensor_args.optional_rs_intermediate_tensor.value()
            : create_device_tensor(tensor_specs[1], tensor_args.input_tensor.device());

    // RS output tensor (use provided or create new)
    ttnn::Tensor rs_output_tensor = tensor_args.optional_rs_output_tensor.has_value()
                                        ? tensor_args.optional_rs_output_tensor.value()
                                        : create_device_tensor(tensor_specs[2], tensor_args.input_tensor.device());

    return {mm_output_tensor, rs_intermediate_tensor, rs_output_tensor};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> minimal_matmul_strided_reduce_scatter_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord reduce_scatter_core_grid_offset,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_mm,
    const MemoryConfig& rs_output_mem_config,
    const std::optional<MemoryConfig>& rs_intermediate_mem_config,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    const std::optional<const Tensor>& bias,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    std::optional<const ttnn::experimental::prim::MinimalMatmulConfig> config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<uint32_t> chunk_width_in_mm_blocks,
    const std::optional<Tensor>& optional_rs_intermediate_tensor,
    const std::optional<Tensor>& optional_rs_output_tensor,
    const std::optional<float> fused_ternary_scalar,
    const std::optional<const Tensor>& addcmul_input_tensor1,
    const std::optional<const Tensor>& addcmul_input_tensor2,
    std::optional<tt::tt_metal::DataType> dtype,
    const std::optional<const Tensor>& mm_progress_counters,
    std::optional<uint32_t> mm_window_blocks,
    const std::optional<const Tensor>& mm_credit_counters,
    const std::optional<const Tensor>& mm_optional_input_tensor) {
    using OperationType = ttnn::experimental::prim::MinimalMatmulStridedReduceScatterAsync;

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    const auto resolved_sub_device_id =
        sub_device_id.has_value()
            ? sub_device_id
            : std::optional<tt::tt_metal::SubDeviceId>(input_tensor.device()->get_sub_device_ids().at(0));

    /* Matmul setup */
    auto matmul_struct =
        decltype(ttnn::experimental::prim::MinimalMatmulStridedReduceScatterAsyncParams::matmul_struct){
            .config = config,
            .fused_activation = std::move(fused_activation),
            .output_mem_config = memory_config_mm,
            .output_dtype = dtype,
            .compute_kernel_config = compute_kernel_config};

    auto operation_attributes = OperationType::operation_attributes_t{
        /* matmul_struct */ matmul_struct,
        /* fused_ternary_scalar */ fused_ternary_scalar,
        /* dim */ dim,
        /* num_links */ num_links,
        /* ring_size */ num_devices,
        /* rs_output_mem_config */ rs_output_mem_config,
        /* rs_intermediate_mem_config */ rs_intermediate_mem_config,
        /* topology */ topology,
        /* semaphore */ multi_device_global_semaphore,
        /* barrier_semaphore */ barrier_semaphore,
        /* using_persistent_buffers */ using_persistent_buffers,
        /* sub_device_id */ resolved_sub_device_id,
        /* cluster_axis */ cluster_axis,
        /* num_workers_per_link */ num_workers_per_link,
        /* num_buffers_per_channel */ num_buffers_per_channel,
        /* chunk_width_in_mm_blocks */ chunk_width_in_mm_blocks,
        /* mm_window_blocks */ mm_window_blocks,
        /* reduce_scatter_core_grid_offset */ reduce_scatter_core_grid_offset};

    auto tensor_args = OperationType::tensor_args_t{
        input_tensor,
        weight_tensor,
        optional_rs_intermediate_tensor,
        optional_rs_output_tensor,
        bias,
        addcmul_input_tensor1,
        addcmul_input_tensor2,
        mm_progress_counters,
        mm_credit_counters,
        mm_optional_input_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
