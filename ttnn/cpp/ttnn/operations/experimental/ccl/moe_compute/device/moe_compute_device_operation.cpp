// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/moe_compute/moe_core_placement.hpp"
#include "kernels/moe_ring_common.h"
#include "moe_compute_device_operation.hpp"
#include "moe_compute_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include "ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_align.hpp>

#include <umd/device/types/arch.hpp>

#include <limits>

namespace ttnn::experimental::prim {
namespace detail {

constexpr auto TOKEN_SIZE = 32;  // This does not mean we only support 32 tokens, just hardcoding the shared buffer size
constexpr auto DOUBLE_BUFFER_SIZE = 2;

uint32_t get_total_tokens(const ttnn::Shape& input_shape) {
    TT_FATAL(
        input_shape.rank() >= 3 && input_shape.rank() <= 4,
        "moe_compute: tilize_input_tensor must be rank 3 or 4 [...,tokens,hidden]; got {}",
        input_shape);

    const uint64_t hidden_size = input_shape[-1];
    TT_FATAL(
        hidden_size > 0, "moe_compute: tilize_input_tensor hidden dimension must be positive; got {}", input_shape);
    const uint64_t volume = input_shape.volume();
    TT_FATAL(
        volume % hidden_size == 0,
        "moe_compute: tilize_input_tensor volume {} is not divisible by hidden dimension {}; got {}",
        volume,
        hidden_size,
        input_shape);
    const uint64_t physical_rows = volume / hidden_size;
    const uint64_t total_tokens = static_cast<uint64_t>(input_shape[0]) * input_shape[1];
    TT_FATAL(
        physical_rows == total_tokens,
        "moe_compute: tilize_input_tensor {} has {} physical rows but dimensions 0*1 specify {} tokens; the "
        "optional rank-4 dimension must be 1",
        input_shape,
        physical_rows,
        total_tokens);
    TT_FATAL(
        total_tokens > 0 && total_tokens <= std::numeric_limits<uint32_t>::max(),
        "moe_compute: total token count must fit uint32_t and be positive; got {}",
        total_tokens);
    return static_cast<uint32_t>(total_tokens);
}

MoEScoreInputOrganization derive_score_input_organization(
    const ttnn::Shape& indices_shape, const ttnn::Shape& scores_shape) {
    const auto indices_rank = indices_shape.rank();
    const auto scores_rank = scores_shape.rank();
    TT_FATAL(
        indices_rank >= 2 && indices_rank <= 4,
        "moe_compute: tilize_expert_indices_tensor must be rank 2, 3, or 4 with trailing [tokens,K]; got {}",
        indices_shape);
    TT_FATAL(
        scores_rank >= 2 && scores_rank <= 5,
        "moe_compute: tilize_expert_scores_tensor must be rank 2 through 5 and match indices exactly or add one "
        "trailing 1; got {}",
        scores_shape);

    const bool scalar_page = scores_rank == indices_rank + 1;
    TT_FATAL(
        scores_rank == indices_rank || scalar_page,
        "moe_compute: unsupported or ambiguous routing-score organization: scores must equal indices {} or add one "
        "trailing singleton; got {}",
        indices_shape,
        scores_shape);
    TT_FATAL(
        !scalar_page || scores_shape[-1] == 1,
        "moe_compute: ScalarPageK routing scores require one trailing singleton after K; got scores {} for indices "
        "{}",
        scores_shape,
        indices_shape);
    const auto organization =
        scalar_page ? MoEScoreInputOrganization::ScalarPageK : MoEScoreInputOrganization::ContiguousK;
    const uint32_t organization_trailing_dims = scalar_page ? 1 : 0;
    const char* organization_name = scalar_page ? "ScalarPageK" : "ContiguousK";

    TT_FATAL(
        scores_shape[scores_rank - 2 - organization_trailing_dims] == indices_shape[-2],
        "moe_compute: {} routing-score token dimension must match indices; got scores shape {} and indices shape {}",
        organization_name,
        scores_shape,
        indices_shape);
    TT_FATAL(
        scores_shape[scores_rank - 1 - organization_trailing_dims] == indices_shape[-1],
        "moe_compute: {} routing-score K dimension must match indices; got scores shape {} and indices shape {}",
        organization_name,
        scores_shape,
        indices_shape);
    for (uint32_t dim = 0; dim + 2 < indices_rank; ++dim) {
        TT_FATAL(
            scores_shape[dim] == indices_shape[dim],
            "moe_compute: {} scores must match all indices dimensions before the optional trailing singleton; "
            "scores {} and indices {} differ at dimension {}",
            organization_name,
            scores_shape,
            indices_shape,
            dim);
    }
    return organization;
}

void validate_device_row_major_tensor(
    const ttnn::Tensor& tensor,
    const char* tensor_name,
    tt::tt_metal::DataType expected_dtype,
    const char* expected_dtype_name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "moe_compute: {} must be a device tensor", tensor_name);
    TT_FATAL(tensor.buffer() != nullptr, "moe_compute: {} must have an allocated device buffer", tensor_name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "moe_compute: {} must be ROW_MAJOR; got {}",
        tensor_name,
        tensor.layout());
    TT_FATAL(
        tensor.dtype() == expected_dtype,
        "moe_compute: {} must be {}; got {}",
        tensor_name,
        expected_dtype_name,
        tensor.dtype());
}

void validate_sparse_tensor_placement(
    const ttnn::Tensor& tensor,
    const char* tensor_name,
    ttnn::MeshDevice* expected_device,
    const CoreRangeSet& expected_drain_grid,
    uint32_t expected_shard_height,
    uint32_t expected_shard_width,
    uint32_t element_size) {
    TT_FATAL(tensor.device() == expected_device, "moe_compute: {} must share tilize_input_tensor's mesh", tensor_name);

    const auto& memory_config = tensor.memory_config();
    TT_FATAL(
        memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        "moe_compute: {} must use HEIGHT_SHARDED memory layout; got {}",
        tensor_name,
        memory_config.memory_layout());
    TT_FATAL(
        memory_config.buffer_type() == tt::tt_metal::BufferType::L1,
        "moe_compute: {} must reside in L1; got buffer type {}",
        tensor_name,
        memory_config.buffer_type());
    TT_FATAL(memory_config.shard_spec().has_value(), "moe_compute: {} needs an explicit shard spec", tensor_name);

    const auto& shard_spec = memory_config.shard_spec().value();
    TT_FATAL(
        shard_spec.grid.num_cores() == 1,
        "moe_compute: {} must be height-sharded on one core; got {}",
        tensor_name,
        shard_spec.grid.num_cores());
    TT_FATAL(
        shard_spec.grid == expected_drain_grid,
        "moe_compute: {} must be placed on the selected tilize drain core {}; got {}",
        tensor_name,
        expected_drain_grid,
        shard_spec.grid);
    TT_FATAL(
        shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR,
        "moe_compute: {} shard orientation must be ROW_MAJOR; got {}",
        tensor_name,
        shard_spec.orientation);
    TT_FATAL(
        shard_spec.shape[0] == expected_shard_height && shard_spec.shape[1] == expected_shard_width,
        "moe_compute: {} shard shape must be [{}, {}]; got [{}, {}]",
        tensor_name,
        expected_shard_height,
        expected_shard_width,
        shard_spec.shape[0],
        shard_spec.shape[1]);

    const auto& buffer = *tensor.buffer();
    const uint64_t expected_page_size = static_cast<uint64_t>(expected_shard_width) * element_size;
    TT_FATAL(
        expected_page_size <= std::numeric_limits<uint32_t>::max(),
        "moe_compute: {} row byte size ({}) exceeds uint32_t",
        tensor_name,
        expected_page_size);
    TT_FATAL(
        buffer.num_pages() == expected_shard_height,
        "moe_compute: {} needs {} row pages for shard [{},{}]; got {}",
        tensor_name,
        expected_shard_height,
        expected_shard_height,
        expected_shard_width,
        buffer.num_pages());
    TT_FATAL(
        buffer.page_size() == expected_page_size,
        "moe_compute: {} page size must be {} bytes for width {}; got {} (unsupported stride)",
        tensor_name,
        expected_page_size,
        expected_shard_width,
        buffer.page_size());
    const uint64_t expected_aligned_page_size_64 =
        tt::align(expected_page_size, static_cast<uint64_t>(tt::tt_metal::hal::get_l1_alignment()));
    TT_FATAL(
        expected_aligned_page_size_64 <= std::numeric_limits<uint32_t>::max(),
        "moe_compute: {} aligned row stride ({}) exceeds uint32_t",
        tensor_name,
        expected_aligned_page_size_64);
    const uint32_t expected_aligned_page_size = static_cast<uint32_t>(expected_aligned_page_size_64);
    TT_FATAL(
        buffer.aligned_page_size() == expected_aligned_page_size,
        "moe_compute: {} aligned row stride must be {} bytes; got {} bytes",
        tensor_name,
        expected_aligned_page_size,
        buffer.aligned_page_size());
    const uint64_t expected_logical_size = static_cast<uint64_t>(expected_shard_height) * expected_page_size;
    TT_FATAL(
        expected_logical_size <= std::numeric_limits<uint32_t>::max(),
        "moe_compute: {} logical buffer size ({}) exceeds uint32_t",
        tensor_name,
        expected_logical_size);
    TT_FATAL(
        buffer.size() == expected_logical_size,
        "moe_compute: {} logical size must be {} bytes ({} rows of {}); got {} (unsupported padding/width)",
        tensor_name,
        expected_logical_size,
        expected_shard_height,
        expected_page_size,
        buffer.size());

    // Buffer::size() is logical bytes; L1 allocation consumes aligned_page_size() per row.
    const uint64_t expected_physical_span = static_cast<uint64_t>(expected_shard_height) * expected_aligned_page_size;
    TT_FATAL(
        expected_physical_span <= std::numeric_limits<uint32_t>::max(),
        "moe_compute: {} aligned physical span ({}) exceeds the 32-bit L1 address contract",
        tensor_name,
        expected_physical_span);
    TT_FATAL(
        buffer.aligned_size() == expected_physical_span,
        "moe_compute: {} aligned span must be {} bytes; got {} (unsupported allocation/stride)",
        tensor_name,
        expected_physical_span,
        buffer.aligned_size());
}

}  // namespace detail
MoEComputeDeviceOperation::program_factory_t MoEComputeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoEComputeMeshWorkloadFactory{};
}

void MoEComputeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEComputeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Tilize
    detail::validate_device_row_major_tensor(
        tensor_args.tilize_input_tensor, "tilize_input_tensor", tt::tt_metal::DataType::BFLOAT16, "BFLOAT16");
    detail::validate_device_row_major_tensor(
        tensor_args.tilize_expert_indices_tensor,
        "tilize_expert_indices_tensor",
        tt::tt_metal::DataType::UINT16,
        "UINT16");
    detail::validate_device_row_major_tensor(
        tensor_args.tilize_expert_scores_tensor,
        "tilize_expert_scores_tensor",
        tt::tt_metal::DataType::BFLOAT16,
        "BFLOAT16");

    // Input tensor rank and score-organization guards. Indices legitimately arrive as rank 2,
    // 3, or 4. Scores must be either the exact same shape (ContiguousK) or that shape plus one
    // trailing singleton dimension (ScalarPageK); matching relative shapes makes inference
    // unambiguous even when K itself is one.
    const auto rank_of = [](const ttnn::Tensor& t) { return t.logical_shape().rank(); };
    const auto& tilize_input_shape = tensor_args.tilize_input_tensor.logical_shape();
    const auto& indices_shape = tensor_args.tilize_expert_indices_tensor.logical_shape();
    const auto& scores_shape = tensor_args.tilize_expert_scores_tensor.logical_shape();
    const uint32_t total_tokens = detail::get_total_tokens(tilize_input_shape);
    const auto score_input_organization = detail::derive_score_input_organization(indices_shape, scores_shape);
    TT_FATAL(
        score_input_organization == args.score_input_organization,
        "moe_compute: derived routing-score organization ({}) does not match the cached operation attribute ({}); "
        "this indicates an invalid program-cache key",
        static_cast<uint32_t>(score_input_organization),
        static_cast<uint32_t>(args.score_input_organization));

    const uint32_t indices_tokens = indices_shape[-2];
    const uint32_t selected_experts_k = indices_shape[-1];
    TT_FATAL(
        indices_tokens > 0,
        "moe_compute: tilize_expert_indices_tensor trailing token dimension must be positive; got shape {}",
        indices_shape);
    TT_FATAL(
        selected_experts_k > 0,
        "moe_compute: tilize_expert_indices_tensor K must be positive; got shape {}",
        indices_shape);
    TT_FATAL(
        indices_tokens == total_tokens,
        "moe_compute: tilize_expert_indices_tensor trailing token dimension must match the activation token count; "
        "got indices shape {} (tokens={}) and activation shape {} (tokens={})",
        indices_shape,
        indices_tokens,
        tilize_input_shape,
        total_tokens);
    const uint64_t expected_sparse_volume = static_cast<uint64_t>(total_tokens) * selected_experts_k;
    TT_FATAL(
        indices_shape.volume() == expected_sparse_volume,
        "moe_compute: tilize_expert_indices_tensor must contain exactly one K-wide row per activation token; got "
        "shape {} (trailing tokens={}, volume={}) for activation shape {} (flattened tokens={}) and K={} "
        "(expected volume={})",
        indices_shape,
        indices_tokens,
        indices_shape.volume(),
        tilize_input_shape,
        total_tokens,
        selected_experts_k,
        expected_sparse_volume);
    TT_FATAL(
        scores_shape.volume() == expected_sparse_volume,
        "moe_compute: routing scores must contain exactly tokens*K logical elements; got scores shape {} (volume={}) "
        "and indices shape {} for tokens={} K={} (expected volume={})",
        scores_shape,
        scores_shape.volume(),
        indices_shape,
        total_tokens,
        selected_experts_k,
        expected_sparse_volume);

    TT_FATAL(
        rank_of(tensor_args.tilize_expert_mapping_tensor) == 2,
        "moe_compute: tilize_expert_mapping_tensor must be rank 2 ([num_devices, experts]); got rank {}",
        rank_of(tensor_args.tilize_expert_mapping_tensor));
    TT_FATAL(
        rank_of(tensor_args.matmul_w0_w1_tensor) == 6,
        "moe_compute: matmul_w0_w1_tensor must be rank 6 ([num_cores, L, E, groups_per_core, K, "
        "4*TILE_SIZE]); got rank {}",
        rank_of(tensor_args.matmul_w0_w1_tensor));
    TT_FATAL(
        rank_of(tensor_args.matmul_w2_tensor) == 6,
        "moe_compute: matmul_w2_tensor must be rank 6 ([num_cores, L, E, groups_per_core, N, 4*TILE_SIZE]); got "
        "rank {}",
        rank_of(tensor_args.matmul_w2_tensor));

    // When has_bias=True, dm0 derives per-expert byte strides using ceil((K+1)/W0W1_TXN)*W0W1_TXN and
    // ceil((N+1)/W2_TXN)*W2_TXN. The physical tensors must be padded to those tile counts; if not,
    // dm0 silently reads from wrong expert boundaries after the first expert.
    if (args.has_bias) {
        constexpr uint32_t tile_h = tt::constants::TILE_HEIGHT;
        constexpr uint32_t w0w1_txn = moe_ring::W0_W1_BLOCK_TILES_H * tile_h;  // bytes per transaction row
        constexpr uint32_t w2_txn = moe_ring::W2_TILES_PER_A2A_ITER_H * tile_h;

        const auto& w0_w1_shape = tensor_args.matmul_w0_w1_tensor.tensor_spec().logical_shape();
        const uint32_t w0_w1_k = w0_w1_shape[-2];
        TT_FATAL(
            w0_w1_k % w0w1_txn == 0,
            "matmul_w0_w1_tensor K-dimension ({}) must be a multiple of {} elements ({} tiles * {} rows/tile) "
            "when has_bias=True. Use moe_compute_utils.prepare_w0_w1_tensor_with_bias() to prepare the tensor.",
            w0_w1_k,
            w0w1_txn,
            moe_ring::W0_W1_TILES_PER_TXN,
            tile_h);

        const auto& w2_shape = tensor_args.matmul_w2_tensor.tensor_spec().logical_shape();
        const uint32_t w2_n = w2_shape[-2];
        TT_FATAL(
            w2_n % w2_txn == 0,
            "matmul_w2_tensor N-dimension ({}) must be a multiple of {} elements ({} tiles * {} rows/tile) "
            "when has_bias=True. Use moe_compute_utils.prepare_w2_tensor_with_bias() to prepare the tensor.",
            w2_n,
            w2_txn,
            moe_ring::W2_TILES_PER_TXN,
            tile_h);
    }

    // validate that 32 (token dim) * output_shard_width * output_shard_height >= total tokens
    const auto combine_token_parallel_cores = args.num_token_parallel_cores;
    const auto combine_data_parallel_cores = args.num_data_parallel_cores;

    // make sure the shared L1 buffer is sufficiently large enough to contain all output tokens
    const auto max_tokens = detail::TOKEN_SIZE * combine_data_parallel_cores * combine_token_parallel_cores;
    TT_FATAL(
        max_tokens >= total_tokens, "Too many tokens in input, got: {} but expected max: {}", total_tokens, max_tokens);

    // Mode-specific validation of combine_params and optional_output_tensor.
    // - ComputeOnly: no combine_params, no optional_output_tensor (5 outputs).
    // - FullLocal: combine_params must be set with local_combine=true; optional_output_tensor
    //   is allowed as the combine output sink (6 outputs, no CCL).
    // - FullCcl: combine_params must be set with local_combine=false (6 outputs, CCL path).
    if (args.path == MoEComputePath::ComputeOnly) {
        TT_FATAL(!args.combine_params.has_value(), "path=ComputeOnly requires combine_params to be std::nullopt");
        TT_FATAL(
            !tensor_args.optional_output_tensor.has_value(),
            "path=ComputeOnly requires optional_output_tensor to be std::nullopt (no combine output is produced)");
    } else {
        TT_FATAL(args.combine_params.has_value(), "path=Full requires combine_params to be set");
        if (args.path == MoEComputePath::FullLocal) {
            TT_FATAL(
                args.combine_params->local_combine, "path=FullLocal requires combine_params->local_combine to be true");
        } else {
            TT_FATAL(
                !args.combine_params->local_combine, "path=FullCcl requires combine_params->local_combine to be false");
            TT_FATAL(args.combine_params->num_links > 0, "num_links must be greater than 0");
            TT_FATAL(args.combine_params->axis < 2, "cluster_axis must be 0 or 1");
        }
    }

    // Validate hidden_size
    const uint32_t hidden_size = tilize_input_shape[-1];
    TT_FATAL(
        hidden_size > 0 && hidden_size % 32 == 0,
        "hidden_size ({}) must be a positive multiple of 32 (TILE_SIZE)",
        hidden_size);

    // Validate intermediate_size
    const uint32_t intermediate_size = args.intermediate_size;
    TT_FATAL(
        intermediate_size > 0 && intermediate_size % 32 == 0,
        "intermediate_size ({}) must be a positive multiple of 32 (TILE_SIZE)",
        intermediate_size);

    // Validate intermediate_tiles >= matmul_num_cores (at least 1 tile per ring core).
    // Both Full and ComputeOnly paths use the same matmul ring kernels, so this applies in both modes.
    //
    // matmul_num_cores must match the actual matmul ring size produced by program_factory:
    //   - WH: ring is always 12 (no DRAM-bank harvesting).
    //   - BH: ring = live DRAM-bank count (7 or 8). args.bh_ring_size is resolved by invoke()
    //     to this value before validate runs.
    auto* mesh_device = tensor_args.tilize_input_tensor.device();
    const uint32_t matmul_num_cores = args.bh_ring_size;
    const uint32_t intermediate_tiles = intermediate_size / 32;
    TT_FATAL(
        intermediate_tiles >= matmul_num_cores,
        "intermediate_size ({}) must yield at least 1 tile per ring core ({} tiles < {} cores)",
        intermediate_size,
        intermediate_tiles,
        matmul_num_cores);

    TT_FATAL(
        matmul_num_cores % combine_data_parallel_cores == 0,
        "matmul_num_cores ({}) must be divisible by num_data_parallel_cores ({}) "
        "so RING_CORES_PER_COMBINE_COL is integral",
        matmul_num_cores,
        combine_data_parallel_cores);
    const uint32_t hidden_tiles = hidden_size / 32;
    TT_FATAL(
        hidden_tiles % combine_data_parallel_cores == 0,
        "hidden_tiles ({}) must be divisible by num_data_parallel_cores ({}) "
        "so output width shards are tile-aligned",
        hidden_tiles,
        combine_data_parallel_cores);

    // dm1 auto-splits each ring A2A transfer into enough noc_async_write_one_packet calls
    // to fit within NOC_MAX_BURST_SIZE (arch-dependent). Validate tiles_per_step matches
    // the round-up formula used in MoeRingConfig::in2_tiles_per_step.
    const uint32_t tiles_per_step_raw = (intermediate_tiles + matmul_num_cores - 1) / matmul_num_cores;
    const uint32_t tiles_per_step = moe_ring::even_stride_at_least_a2a_width(tiles_per_step_raw);
    TT_FATAL(
        tiles_per_step >= moe_ring::W2_TILES_PER_A2A_ITER_W && tiles_per_step % 2 == 0,
        "tiles_per_step ({}) must be even and >= W2_TILES_PER_A2A_ITER_W ({})",
        tiles_per_step,
        moe_ring::W2_TILES_PER_A2A_ITER_W);

    const uint32_t experts_per_device = tensor_args.matmul_w0_w1_tensor.logical_shape()[2];
    TT_FATAL(
        args.num_shared_experts_per_device <= experts_per_device,
        "num_shared_experts_per_device ({}) must be <= experts_per_device ({})",
        args.num_shared_experts_per_device,
        experts_per_device);

    // Validate that dynamic core placement succeeds and bind the sparse tensors to the selected
    // drain core before any program is created. mux_core_range_set comes from combine_params when
    // in Full mode; ComputeOnly uses an empty set.
    const CoreRangeSet validate_mux_cores =
        args.combine_params.has_value() ? args.combine_params->mux_core_range_set : CoreRangeSet{};
    const auto core_selection = ttnn::operations::ccl::common::select_moe_compute_cores(
        mesh_device,
        combine_token_parallel_cores,
        combine_data_parallel_cores,
        hidden_size,
        validate_mux_cores,
        args.bh_ring_size);
    TT_FATAL(
        !core_selection.tilize_cores.empty(),
        "moe_compute: core placement returned no tilize drain core for the requested configuration");
    const CoreCoord drain_core = core_selection.tilize_cores.front();
    const CoreCoord worker_grid_size = mesh_device->compute_with_storage_grid_size();
    TT_FATAL(
        drain_core.x < worker_grid_size.x && drain_core.y < worker_grid_size.y,
        "moe_compute: selected tilize drain core {} is outside worker grid {}",
        drain_core,
        worker_grid_size);
    const CoreRangeSet drain_grid = CoreRangeSet({CoreRange(drain_core, drain_core)});

    detail::validate_sparse_tensor_placement(
        tensor_args.tilize_expert_indices_tensor,
        "tilize_expert_indices_tensor",
        mesh_device,
        drain_grid,
        total_tokens,
        selected_experts_k,
        sizeof(uint16_t));

    const uint64_t score_shard_height_64 = args.score_input_organization == MoEScoreInputOrganization::ScalarPageK
                                               ? static_cast<uint64_t>(total_tokens) * selected_experts_k
                                               : total_tokens;
    TT_FATAL(
        score_shard_height_64 <= std::numeric_limits<uint32_t>::max(),
        "moe_compute: routing-score shard height ({}) exceeds uint32_t for tokens={} K={} organization={}",
        score_shard_height_64,
        total_tokens,
        selected_experts_k,
        static_cast<uint32_t>(args.score_input_organization));
    const uint32_t score_shard_width =
        args.score_input_organization == MoEScoreInputOrganization::ScalarPageK ? 1u : selected_experts_k;
    detail::validate_sparse_tensor_placement(
        tensor_args.tilize_expert_scores_tensor,
        "tilize_expert_scores_tensor",
        mesh_device,
        drain_grid,
        static_cast<uint32_t>(score_shard_height_64),
        score_shard_width,
        sizeof(bfloat16));
}

MoEComputeDeviceOperation::spec_return_value_t MoEComputeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    const ttnn::Tensor& tilize_input_tensor = tensor_args.tilize_input_tensor;
    const auto& tilize_input_shape = tilize_input_tensor.tensor_spec().logical_shape();
    auto* mesh_device = tilize_input_tensor.device();

    uint32_t experts_per_device = tensor_args.matmul_w0_w1_tensor.logical_shape()[2];
    const uint32_t total_tokens = detail::get_total_tokens(tilize_input_shape);

    const uint32_t hidden_size = tilize_input_shape[-1];

    const CoreCoord worker_grid_size = mesh_device->compute_with_storage_grid_size();
    const CoreRangeSet shard_cores =
        CoreRangeSet({CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1})});
    const auto num_cores = shard_cores.num_cores();

    //-------------------------------------------------------------------------
    // Tilize outputs
    //-------------------------------------------------------------------------
    // Output 0: Per expert total tokens tensor
    // This data will be replicated on all cores
    auto per_expert_total_tokens_row_bytes = tt::align(experts_per_device * sizeof(uint32_t), l1_alignment);
    auto per_expert_total_tokens_row_elements = tt::div_up(per_expert_total_tokens_row_bytes, sizeof(uint32_t));
    auto tilize_per_expert_total_tokens_shape = ttnn::Shape({num_cores, per_expert_total_tokens_row_elements});

    const ttnn::MemoryConfig tilize_per_expert_total_tokens_sharded_memory_config = ttnn::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(
            shard_cores, {1, per_expert_total_tokens_row_elements}, tt::tt_metal::ShardOrientation::ROW_MAJOR),
    };

    auto tilize_per_expert_total_tokens_spec = tt::tt_metal::TensorSpec(
        tilize_per_expert_total_tokens_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tilize_per_expert_total_tokens_sharded_memory_config));

    // Output 1: Expert activation tensor
    // Each row: [token_id, k_indices[experts_per_device], scores[experts_per_device]]
    // Row size in uint32_t elements: 2 * experts_per_device + 1
    // Total size: total_tokens * aligned_row_bytes, stored as a single DRAM page
    uint32_t activation_row_elements = (2 * experts_per_device) + 1;
    uint32_t activation_row_bytes = tt::align(activation_row_elements * sizeof(uint32_t), l1_alignment);
    uint32_t activation_total_bytes = total_tokens * activation_row_bytes;
    auto tilize_expert_activation_shape = ttnn::Shape({1, activation_total_bytes / sizeof(uint32_t)});
    auto tilize_expert_activation_spec = tt::tt_metal::TensorSpec(
        tilize_expert_activation_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    // Output 2: Token indices tensor
    // 1 page per expert per device
    // Each index is at a 16B offset due to NoC DMA restrictions
    // (tokens + 1) -> 1 extra element per page for -1 terminator
    uint32_t e_t_row_bytes = (total_tokens + 1) * tt::align(sizeof(uint32_t), l1_alignment);
    uint32_t e_t_row_elements = e_t_row_bytes / sizeof(uint32_t);
    auto tilize_e_t_shape = ttnn::Shape({experts_per_device, e_t_row_elements});
    auto tilize_e_t_spec = tt::tt_metal::TensorSpec(
        Shape(tilize_e_t_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    //-------------------------------------------------------------------------
    // Shared tilize output (sharded)
    //-------------------------------------------------------------------------
    /*
     * Tilize: Used as output CB of tilize operation
     * MM: Used as input CB (where tilized chunks arrive)
     * Combine: Stores output of MM, for input to combine
     */
    ttnn::MemoryConfig output_sharded_memory_config = ttnn::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(
            shard_cores,
            {detail::DOUBLE_BUFFER_SIZE * detail::TOKEN_SIZE, hidden_size},
            tt::tt_metal::ShardOrientation::ROW_MAJOR),
    };

    auto tilize_output_shape =
        ttnn::Shape({shard_cores.num_cores(), detail::DOUBLE_BUFFER_SIZE, detail::TOKEN_SIZE, hidden_size});
    auto tilize_output_spec = tt::tt_metal::TensorSpec(
        Shape(tilize_output_shape),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            output_sharded_memory_config));

    //-------------------------------------------------------------------------
    // Shared output (sharded)
    //-------------------------------------------------------------------------
    /*
     * This will be an alias to the buffer used by Shared tilize output.
     * But re-perceived as RM. This is not strictly necessary but facilitates
     * torch interop and unit testing
     */

    const auto& tilize_output_layout = tilize_output_spec.tensor_layout();
    const tt::tt_metal::TensorLayout matmul_output_layout(
        tilize_output_layout.get_data_type(), ROW_MAJOR_LAYOUT, tilize_output_layout.get_memory_config());
    const auto matmul_output_spec = tt::tt_metal::TensorSpec(tilize_output_shape, matmul_output_layout);

    //-------------------------------------------------------------------------
    // a2a combine output
    //-------------------------------------------------------------------------
    using namespace tt::tt_metal;

    if (args.path == MoEComputePath::ComputeOnly) {
        // No combine output in ComputeOnly mode; matmul_output_spec is the final output (slot 4).
        return {
            tilize_per_expert_total_tokens_spec,
            tilize_expert_activation_spec,
            tilize_e_t_spec,
            tilize_output_spec,
            matmul_output_spec};
    }

    TT_FATAL(args.combine_params.has_value(), "combine_params required when path is not ComputeOnly");

    ttnn::experimental::prim::SelectiveReduceCombineTensors combine_tensor_args{
        .dense_input_tensor = tilize_input_tensor,
        .dense_activations_tensor = tilize_input_tensor,
        .dense_token_maps_tensor = tilize_input_tensor,
        .dense_token_counts_tensor = tilize_input_tensor,
        .optional_output_tensor = std::nullopt,
    };
    const auto output_spec = ttnn::experimental::prim::SelectiveReduceCombineDeviceOperation::compute_output_specs(
        args.combine_params.value(), combine_tensor_args);

    return {
        tilize_per_expert_total_tokens_spec,
        tilize_expert_activation_spec,
        tilize_e_t_spec,
        tilize_output_spec,
        matmul_output_spec,
        output_spec};
}

MoEComputeDeviceOperation::tensor_return_value_t MoEComputeDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const std::vector<tt::tt_metal::TensorSpec>& output_specs = compute_output_specs(args, tensor_args);

    const auto tilize_output_tensor = create_device_tensor(output_specs[3], tensor_args.tilize_input_tensor.device());

    // re-percieve tilize output tensor as RM for output
    const auto matmul_output_tensor =
        ttnn::unchecked_reinterpret_layout(tilize_output_tensor, tt::tt_metal::Layout::ROW_MAJOR);
    TT_FATAL(
        matmul_output_tensor.tensor_spec() == output_specs[4],
        "Reinterpreted tensor spec does not match expected output_specs[4]");

    if (args.path == MoEComputePath::ComputeOnly) {
        // 5-tensor return: matmul_output is the final output (no combine output produced).
        return {
            create_device_tensor(output_specs[0], tensor_args.tilize_input_tensor.device()),
            create_device_tensor(output_specs[1], tensor_args.tilize_input_tensor.device()),
            create_device_tensor(output_specs[2], tensor_args.tilize_input_tensor.device()),
            tilize_output_tensor,
            matmul_output_tensor};
    }

    const auto& combine_output_tensor = tensor_args.optional_output_tensor.value_or(
        create_device_tensor(output_specs[5], tensor_args.tilize_input_tensor.device()));

    return {
        create_device_tensor(output_specs[0], tensor_args.tilize_input_tensor.device()),
        create_device_tensor(output_specs[1], tensor_args.tilize_input_tensor.device()),
        create_device_tensor(output_specs[2], tensor_args.tilize_input_tensor.device()),
        tilize_output_tensor,
        matmul_output_tensor,
        combine_output_tensor};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> moe_compute(
    const ttnn::Tensor& tilize_input_tensor,
    const ttnn::Tensor& tilize_expert_indices_tensor,
    const ttnn::Tensor& tilize_expert_scores_tensor,
    const ttnn::Tensor& tilize_expert_mapping_tensor,
    const ttnn::Tensor& matmul_w0_w1_tensor,
    const ttnn::Tensor& matmul_w2_tensor,
    const uint32_t layer_id,
    const uint32_t output_height_shard_dim,
    const uint32_t intermediate_size,
    const bool has_bias,
    const std::optional<uint32_t>& cluster_axis,
    const std::optional<tt::tt_fabric::Topology>& topology,
    const std::optional<uint32_t>& num_links,
    const std::optional<CoreRangeSet>& mux_core_range_set,
    const std::optional<ttnn::MemoryConfig>& output_memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<GlobalSemaphore>& optional_cross_device_semaphore,
    const std::optional<ttnn::experimental::prim::detail::MoEActivationFunction>& activation_type,
    const bool compute_only,
    const std::optional<uint32_t>& bh_ring_size,
    const std::optional<uint32_t>& num_shared_experts_per_device) {
    using OperationType = ttnn::experimental::prim::MoEComputeDeviceOperation;

    const auto& input_shape = tilize_input_tensor.tensor_spec().logical_shape();
    const auto& indices_shape = tilize_expert_indices_tensor.tensor_spec().logical_shape();
    const auto& scores_shape = tilize_expert_scores_tensor.tensor_spec().logical_shape();
    const uint32_t total_tokens = ttnn::experimental::prim::detail::get_total_tokens(input_shape);
    const auto score_input_organization =
        ttnn::experimental::prim::detail::derive_score_input_organization(indices_shape, scores_shape);
    const uint32_t hidden_size = input_shape[-1];
    const uint32_t select_experts_k = indices_shape[-1];

    const auto& num_token_parallel_cores = output_height_shard_dim;

    auto* mesh_device = tilize_input_tensor.device();

    // Ring size is 12 on Wormhole (no DRAM-bank harvesting). On Blackhole it is the live
    // DRAM-bank count (7 or 8). Resolved by the public API before invocation, but keep a
    // fallback to the live bank count for direct prim callers.
    const uint32_t ring_n = bh_ring_size.value_or(
        mesh_device->arch() == tt::ARCH::BLACKHOLE
            ? mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default).size()
            : 12u);
    // NOTE: the public API auto-detects the ring from the device and does not expose it as a knob.

    // Auto-compute num_data_parallel_cores: largest divisor d of hidden_tiles with d <= 4
    // AND ring_n % d == 0. dm1 maps ring cores to combine columns via
    // RING_CORES_PER_COMBINE_COL = num_cores / width_shard_dim, so both must divide evenly.
    // E.g. GPT-OSS (Ht=90) picks d=3 on WH (N=12) but falls back to d=2 on BH (N=8/7).
    const uint32_t hidden_tiles = hidden_size / 32;
    uint32_t num_data_parallel_cores = 1;
    for (uint32_t d = 4; d >= 1; --d) {
        if (hidden_tiles % d == 0 && ring_n % d == 0) {
            num_data_parallel_cores = d;
            break;
        }
    }

    // Determine the MoE compute path from compute_only and cluster_axis.
    // - ComputeOnly: compute_only=true, cluster_axis must be None, no CCL options.
    // - FullLocal: compute_only=false, cluster_axis=None, only valid on a 1x1 mesh. No CCL
    //   options; combine runs as a local reduction with no fabric.
    // - FullCcl: compute_only=false, cluster_axis must be provided. CCL options required.
    const uint32_t num_devices = mesh_device->num_devices();
    const bool full_local = !compute_only && !cluster_axis.has_value();
    if (full_local) {
        TT_FATAL(
            num_devices == 1,
            "moe_compute(compute_only=false, cluster_axis=None) is only supported on a 1x1 mesh, "
            "got num_devices={}. Pass cluster_axis for multi-device fused compute+combine.",
            num_devices);
    }

    if (compute_only) {
        TT_FATAL(!cluster_axis.has_value(), "moe_compute(compute_only=true) requires cluster_axis to be std::nullopt");
        TT_FATAL(!topology.has_value(), "moe_compute(compute_only=true) requires topology to be std::nullopt");
        TT_FATAL(!num_links.has_value(), "moe_compute(compute_only=true) requires num_links to be std::nullopt");
        TT_FATAL(
            !mux_core_range_set.has_value(),
            "moe_compute(compute_only=true) requires mux_core_range_set to be std::nullopt");
        TT_FATAL(
            !optional_cross_device_semaphore.has_value(),
            "moe_compute(compute_only=true) requires optional_cross_device_semaphore to be std::nullopt");
        TT_FATAL(
            !optional_output_tensor.has_value(),
            "moe_compute(compute_only=true) requires optional_output_tensor to be std::nullopt");
    } else if (full_local) {
        TT_FATAL(!topology.has_value(), "moe_compute(cluster_axis=None) requires topology to be std::nullopt");
        TT_FATAL(!num_links.has_value(), "moe_compute(cluster_axis=None) requires num_links to be std::nullopt");
        TT_FATAL(
            !mux_core_range_set.has_value(),
            "moe_compute(cluster_axis=None) requires mux_core_range_set to be std::nullopt");
        TT_FATAL(
            !optional_cross_device_semaphore.has_value(),
            "moe_compute(cluster_axis=None) requires optional_cross_device_semaphore to be std::nullopt");
    } else {
        TT_FATAL(cluster_axis.has_value(), "moe_compute(compute_only=false) requires cluster_axis to be provided");
    }

    const auto& combine_cores = get_moe_combine_cores(
        mesh_device,
        num_token_parallel_cores,
        num_data_parallel_cores,
        hidden_size,
        mux_core_range_set.value_or(CoreRangeSet{}),
        ring_n);

    std::optional<ttnn::experimental::prim::SelectiveReduceCombineParams> combine_params;
    if (full_local) {
        // Local combine: no fabric, no mux, no cross-device semaphore. axis=0 is a dummy
        // (mesh is 1x1 so mesh_shape[1-axis]=1 for shared_expert_tp_factor).
        combine_params = ttnn::experimental::prim::SelectiveReduceCombineParams{
            .hidden_size = hidden_size,
            .batch_size = 1,
            .seq_size = total_tokens,
            .select_experts_k = select_experts_k,
            .num_links = 1,
            .axis = 0,
            .topology = tt::tt_fabric::Topology::Linear,
            .num_token_parallel_cores = num_token_parallel_cores,
            .num_data_parallel_cores = num_data_parallel_cores,
            .worker_cores = combine_cores,
            .mux_core_range_set = CoreRangeSet{},
            .output_memory_config = output_memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
            .optional_cross_device_semaphore = std::nullopt,
            .local_combine = true};
    } else if (!compute_only) {
        // see #27196 for potential limitations
        const uint32_t resolved_num_links =
            num_links.value_or(ttnn::operations::ccl::common::get_num_links(*mesh_device, *cluster_axis));
        // Resolve `topology` via the shared CCL helper. This (a) substitutes the fabric
        // default when `topology` is nullopt, (b) maps Torus → Mesh when the tensor doesn't
        // span a wrap edge so the TT_FATAL below can reject it, and (c) downgrades Ring → Linear
        // for the trivial `mesh_shape[cluster_axis] == 2` case. Notably, it does NOT detect
        // physically-LINE meshes whose tensor still spans the full cluster axis (e.g. BH single
        // Loudbox 2x4 LINE/LINE with cluster_axis=1) — that case still resolves to Ring here.
        // BH LB callers must pass topology=Linear explicitly; the kernel-side `Topology` template
        // guard in fabric_multicast_bidirectional_atomic_inc_1d (moe_utils.hpp) then routes the
        // multicast through the line-aware code path. (Fixing get_usable_topology() to consult
        // physical mesh wrap capability is a separate follow-up that affects all CCL ops.)
        const auto resolved_topology = ttnn::ccl::get_usable_topology(tilize_input_tensor, topology, cluster_axis);
        // Mirror the kernel-side static_assert in fabric_multicast_bidirectional_atomic_inc_1d
        // (moe_utils.hpp). `get_usable_topology` can return Mesh when the fabric default is Torus
        // and the tensor doesn't span a wrap edge; the combine kernel only handles Ring/Linear and
        // would silently produce wrong wait counts → on-device hang. Reject at the host boundary
        // with a clear message instead of waiting for a JIT compile failure or a hang.
        TT_FATAL(
            resolved_topology == tt::tt_fabric::Topology::Linear || resolved_topology == tt::tt_fabric::Topology::Ring,
            "moe_compute: combine kernel only supports Topology::Linear or Topology::Ring, got {}. "
            "If the fabric default is Torus/Mesh, pass topology=ttnn.Topology.Linear or "
            "ttnn.Topology.Ring explicitly to ttnn.experimental.moe_compute.",
            resolved_topology);
        combine_params = ttnn::experimental::prim::SelectiveReduceCombineParams{
            .hidden_size = hidden_size,
            .batch_size = 1,
            .seq_size = total_tokens,
            .select_experts_k = select_experts_k,
            .num_links = resolved_num_links,
            .axis = cluster_axis.value(),
            .topology = resolved_topology,
            .num_token_parallel_cores = num_token_parallel_cores,
            .num_data_parallel_cores = num_data_parallel_cores,
            .worker_cores = combine_cores,
            .mux_core_range_set = mux_core_range_set.value_or(CoreRangeSet{}),
            .output_memory_config = output_memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
            .optional_cross_device_semaphore = optional_cross_device_semaphore};
    }

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .layer_id = layer_id,
            .output_height_shard_dim = output_height_shard_dim,
            .intermediate_size = intermediate_size,
            .num_shared_experts_per_device = num_shared_experts_per_device,
            .has_bias = has_bias,
            .num_token_parallel_cores = num_token_parallel_cores,
            .num_data_parallel_cores = num_data_parallel_cores,
            .path = compute_only ? experimental::prim::MoEComputePath::ComputeOnly
                                 : (full_local ? experimental::prim::MoEComputePath::FullLocal
                                               : experimental::prim::MoEComputePath::FullCcl),
            .score_input_organization = score_input_organization,
            .bh_ring_size = ring_n,
            .combine_params = combine_params,
            .activation_type = activation_type.value_or(experimental::prim::detail::MoEActivationFunction::SILU)},
        OperationType::tensor_args_t{
            .tilize_input_tensor = tilize_input_tensor,
            .tilize_expert_indices_tensor = tilize_expert_indices_tensor,
            .tilize_expert_scores_tensor = tilize_expert_scores_tensor,
            .tilize_expert_mapping_tensor = tilize_expert_mapping_tensor,
            .matmul_w0_w1_tensor = matmul_w0_w1_tensor,
            .matmul_w2_tensor = matmul_w2_tensor,
            .optional_output_tensor = optional_output_tensor});
}

}  // namespace ttnn::prim
