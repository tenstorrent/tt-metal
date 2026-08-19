// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_derived_slots.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <cmath>
#include <string>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <hostdevcommon/common_values.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Appends 5 compile-time args needed for a fabric MUX client worker kernel.
void fabric_mux_connection_ct_args(
    const uint32_t num_workers_per_link,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& worker_ct_args) {
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    worker_ct_args.push_back(mux_kernel_config.get_num_buffers(channel_type));
    worker_ct_args.push_back(mux_kernel_config.get_buffer_size_bytes(channel_type));
    worker_ct_args.push_back(mux_kernel_config.get_status_address());
    worker_ct_args.push_back(mux_kernel_config.get_termination_signal_address());
    worker_ct_args.push_back(num_workers_per_link);  // num_mux_clients
}

// Allocate a per-core semaphore on `worker_logical_core` by appending a
// SemaphoreDescriptor to `desc.semaphores`. The semaphore ID is the first
// available ID on that core (so this is the descriptor-pattern equivalent of
// CreateSemaphore(program, {worker_logical_core}, 0)). Returns the assigned ID.
uint32_t allocate_per_core_semaphore(
    tt::tt_metal::ProgramDescriptor& desc, const CoreCoord& worker_logical_core, uint32_t initial_value = 0) {
    const auto sem_id_opt = desc.find_available_semaphore_id(worker_logical_core, tt::CoreType::WORKER);
    TT_FATAL(
        sem_id_opt.has_value(),
        "Ran out of semaphore IDs on core ({}, {}) — exceeded NUM_SEMAPHORES per core",
        worker_logical_core.x,
        worker_logical_core.y);
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_id_opt.value(),
        .core_type = tt::CoreType::WORKER,
        .core_ranges = CoreRangeSet(CoreRange{worker_logical_core, worker_logical_core}),
        .initial_value = initial_value,
    });
    return sem_id_opt.value();
}

// Number of runtime args appended per fabric MUX client worker. Kept in sync with the
// pushes in fabric_mux_connection_rt_args() and its disconnected-connection counterpart.
constexpr uint32_t kFabricMuxConnectionRtArgCount = 17;

// Appends kFabricMuxConnectionRtArgCount runtime args for a fabric MUX client worker.
// Allocates 5 per-core semaphores on worker_logical_core for the connection state.
void fabric_mux_connection_rt_args(
    const bool mux_connection_valid,
    const bool is_termination_master,
    const CoreCoord& mux_logical_core,
    const uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    tt::tt_metal::ProgramDescriptor& desc,
    const CoreCoord& termination_master_logical_core,
    tt::tt_metal::IDevice* device,
    std::vector<uint32_t>& worker_rt_args) {
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    const CoreCoord mux_virtual_core = device->worker_core_from_logical_core(mux_logical_core);
    const CoreCoord termination_master_virtual_core =
        device->worker_core_from_logical_core(termination_master_logical_core);

    worker_rt_args.push_back(static_cast<uint32_t>(mux_connection_valid));
    worker_rt_args.push_back(static_cast<uint32_t>(is_termination_master));
    worker_rt_args.push_back(mux_virtual_core.x);
    worker_rt_args.push_back(mux_virtual_core.y);
    const auto ch_id = static_cast<uint8_t>(worker_id);
    worker_rt_args.push_back(mux_kernel_config.get_channel_base_address(channel_type, ch_id));
    worker_rt_args.push_back(mux_kernel_config.get_connection_info_address(channel_type, ch_id));
    worker_rt_args.push_back(mux_kernel_config.get_connection_handshake_address(channel_type, ch_id));
    worker_rt_args.push_back(mux_kernel_config.get_flow_control_address(channel_type, ch_id));
    worker_rt_args.push_back(mux_kernel_config.get_buffer_index_address(channel_type, ch_id));
    worker_rt_args.push_back(mux_kernel_config.get_channel_credits_stream_id(channel_type, ch_id));
    worker_rt_args.push_back(allocate_per_core_semaphore(desc, worker_logical_core));  // termination_sync_address
    worker_rt_args.push_back(
        allocate_per_core_semaphore(desc, worker_logical_core));  // local_fabric_mux_status_address
    worker_rt_args.push_back(allocate_per_core_semaphore(desc, worker_logical_core));  // local_flow_control_address
    worker_rt_args.push_back(allocate_per_core_semaphore(desc, worker_logical_core));  // local_teardown_address
    worker_rt_args.push_back(allocate_per_core_semaphore(desc, worker_logical_core));  // local_buffer_index_address
    worker_rt_args.push_back(termination_master_virtual_core.x);
    worker_rt_args.push_back(termination_master_virtual_core.y);
}

// Per-coord ProgramDescriptor build. Kept inside the anonymous namespace so
// create_workload_descriptor() below can loop coords and reuse this body verbatim.
// Op-specific name suffix avoids Unity-build collisions with sibling ring sdpa factories.
tt::tt_metal::ProgramDescriptor build_exp_ring_joint_sdpa_program_descriptor(
    const ExpRingJointSDPAParams& operation_attributes,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& args = operation_attributes;
    auto& output_tensors = tensor_return_value;
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "build_exp_ring_joint_sdpa_program_descriptor requires mesh_dispatch_coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();
    /*
    The QKV inputs are fractured on the sequence dimension across ring_size.
    The sequence length comes in padded such that it is divisible by `TILE_HEIGHT * ring_size`.
    Therefore each device has `padded_N / ring_size` local tokens.

    Naming:
        - padded_N: the global, padded sequence length
        - local_padded_N: the local shard of the padded sequence length. local_padded_N = padded_N / ring_size
        - logical_n: the logical global sequence length. logical_n <= padded_N.
        - L: the logical joint sequence length

    input_tensor_q: B x NH x local_padded_N x DH
    input_tensor_k: B x NH x local_padded_N x DH
    input_tensor_v: B x NH x local_padded_N x DH

    gathered_input_tensor_k: B x NH x padded_N x DH
    gathered_input_tensor_v: B x NH x padded_N x DH

    joint_tensor_q: B x NH x L x DH
    joint_tensor_k: B x NH x L x DH
    joint_tensor_v: B x NH x L x DH

    output_tensor: B x NH x local_padded_N x DH
    joint_output_tensor: B x NH x L x DH


    The algorithm is roughly described below.
    - for each ring iteration:
        - read a Q chunk from input_tensor_q
        - for each KV chunk in local_padded_N:
            - on the first ring iteration, read from local input_tensor_k and input_tensor_v
            - otherwise, read from gathered_input_tensor_k and gathered_input_tensor_v
            - on the last ring iteration, also read from joint_tensor_k and joint_tensor_v
            - if the KV chunk is from the non-joint input and contains the global token index (logical_n - 1), generate
    a mask
            - else if the KV chunk is from non-joint input and contains the local token index (local_padded_N - 1),
    generate an attention mask
            - else if the KV chunk is from the joint input and contains the local token index (L - 1), generate a mask
            - compute attention
        - write the output Q chunk
        - if this is not the first ring iteration, do the LSE update.
    */

    log_debug(tt::LogOp, "DEBUG: create_descriptor is called");

    const auto& input_tensor_q = tensor_args.input_q;
    const auto& input_tensor_k = tensor_args.input_k;
    const auto& input_tensor_v = tensor_args.input_v;

    // Joint inputs are optional. The kernel's joint accessor CT args + address RT args sit at fixed
    // positions (and the semaphore CT-arg offset derives from joint_v_args), so the arg layout must
    // stay identical whether or not joints are present. When absent, source the joint placeholder
    // from input_q; L is forced to 0 below, so zero joint chunks run and the placeholder is never read.
    const bool has_joint = tensor_args.joint_q.has_value();
    const auto& joint_tensor_q = has_joint ? tensor_args.joint_q.value() : input_tensor_q;
    const auto& joint_tensor_k = has_joint ? tensor_args.joint_k.value() : input_tensor_q;
    const auto& joint_tensor_v = has_joint ? tensor_args.joint_v.value() : input_tensor_q;

    const auto& gathered_input_tensor_k = tensor_args.gathered_k;
    const auto& gathered_input_tensor_v = tensor_args.gathered_v;

    auto& output_tensor = output_tensors[EXP_RING_JOINT_SDPA_OUTPUT_IDX];
    auto& joint_output_tensor = output_tensors[EXP_RING_JOINT_SDPA_JOINT_OUTPUT_IDX];
    auto& stats_output_tensor = output_tensors[EXP_RING_JOINT_SDPA_STATS_OUTPUT_IDX];

    std::size_t q_chunk_size = args.get_q_chunk_size();
    std::size_t k_chunk_size = args.get_k_chunk_size();

    tt::tt_metal::ProgramDescriptor desc;

    auto* mesh_device = input_tensor_q.device();
    uint32_t device_index = ccl::get_linearized_index_from_physical_coord(
        input_tensor_q, coord, args.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ccl::get_physical_neighbor_from_physical_coord(
        input_tensor_q,
        coord,
        1,
        args.topology,
        args.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ccl::get_physical_neighbor_from_physical_coord(
        input_tensor_q,
        coord,
        -1,
        args.topology,
        args.cluster_axis);

    auto scale = args.scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1]));
    }

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = gathered_input_tensor_k.logical_shape();
    const uint32_t B = q_shape[0], NH = q_shape[1], local_padded_N = q_shape[2], DH = q_shape[3];
    const uint32_t padded_N = k_shape[2];
    // Zero joint sequence length when there are no joint inputs (placeholder joint == input_q).
    const uint32_t L = has_joint ? joint_tensor_q.logical_shape()[2] : 0;

    const uint32_t local_padded_Nt = local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t padded_Nt = padded_N / tt::constants::TILE_HEIGHT;
    // Find unpadded sequence lengths in tiles
    const uint32_t Lt = tt::div_up(L, tt::constants::TILE_HEIGHT);
    const uint32_t DHt = DH / tt::constants::TILE_WIDTH;
    const uint32_t logical_nt = tt::div_up(static_cast<uint32_t>(args.logical_n), tt::constants::TILE_HEIGHT);

    /*
    For non-causal case we must provide a padded mask if the K sequence length has been padded
    Note that we dont have this issue in non-causal case if Q is padded, since those pad tokens
    don't affect attention of unpadded tokens.
    In causal case, the causal mask takes care of masking K pad tokens.
    */

    const uint32_t Sq_chunk_t = q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / tt::constants::TILE_HEIGHT;

    // Trace-safe logical_n: args.logical_n is the worst-case placeholder (padded_N) and the kernels read
    // the live value. The placeholder already worst-cases every size derivation below EXCEPT the two mask
    // derivations, which a chunk-aligned placeholder would resolve to "no mask" — CB geometry is fixed at
    // program creation, so both are forced whenever the tensor is present.
    const bool has_logical_n_tensor = tensor_args.has_logical_n_tensor();

    // Lightweight mask: only needed when any K/joint dimension has padding that doesn't fill a chunk.
    const bool local_n_has_padding = (local_padded_Nt % Sk_chunk_t) != 0;
    const bool global_n_has_padding =
        has_logical_n_tensor || (args.logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT)) != 0;
    const bool joint_has_padding = L > 0 && (L % (Sk_chunk_t * tt::constants::TILE_HEIGHT)) != 0;
    const bool needs_lightweight_mask = local_n_has_padding || global_n_has_padding || joint_has_padding;

    // Partial tile support when padding boundary falls inside a tile. The kernels stamp the live column
    // into the forced tile and gate the stamp off when that column is 0 (see partial_tile_present).
    const uint32_t global_n_partial_col = args.logical_n % tt::constants::TILE_HEIGHT;
    const uint32_t joint_l_partial_col = L % tt::constants::TILE_HEIGHT;
    const bool has_global_n_partial_tile = ttnn::operations::transformer::sdpa::ring_joint::partial_tile_present(
        global_n_partial_col, has_logical_n_tensor);
    const uint32_t partial_mask_tiles = (has_global_n_partial_tile ? 1 : 0) + (joint_l_partial_col != 0 ? 1 : 0);
    // Single CB holds: 1 neginf tile + up to 2 partial mask tiles
    const uint32_t total_lightweight_mask_tiles = 1 + partial_mask_tiles;

    const uint32_t num_local_q_chunks = tt::div_up(local_padded_N, q_chunk_size);
    const uint32_t num_joint_q_chunks = tt::div_up(L, q_chunk_size);
    const uint32_t num_q_chunks = num_local_q_chunks + num_joint_q_chunks;
    const uint32_t num_local_k_chunks = tt::div_up(local_padded_N, k_chunk_size);
    const uint32_t num_joint_k_chunks = tt::div_up(L, k_chunk_size);

    log_debug(tt::LogOp, "B: {}", B);
    log_debug(tt::LogOp, "NH: {}", NH);
    log_debug(tt::LogOp, "L: {}", L);
    log_debug(tt::LogOp, "DH: {}", DH);

    // Log padded dimensions
    log_debug(tt::LogOp, "local_padded_N: {}", local_padded_N);
    log_debug(tt::LogOp, "padded_N: {}", padded_N);
    log_debug(tt::LogOp, "L: {}", L);

    // Log tile dimensions
    log_debug(tt::LogOp, "DHt: {}", DHt);
    log_debug(tt::LogOp, "local_padded_Nt: {}", local_padded_Nt);
    log_debug(tt::LogOp, "padded_Nt: {}", padded_Nt);
    log_debug(tt::LogOp, "Lt: {}", Lt);

    // Log chunking parameters
    log_debug(tt::LogOp, "Sq_chunk_t: {}", Sq_chunk_t);
    log_debug(tt::LogOp, "Sk_chunk_t: {}", Sk_chunk_t);
    log_debug(tt::LogOp, "num_local_q_chunks: {}", num_local_q_chunks);
    log_debug(tt::LogOp, "num_joint_q_chunks: {}", num_joint_q_chunks);
    log_debug(tt::LogOp, "q_chunk_size: {}", q_chunk_size);
    log_debug(tt::LogOp, "k_chunk_size: {}", k_chunk_size);
    log_debug(tt::LogOp, "num_q_chunks: {}", num_q_chunks);
    log_debug(tt::LogOp, "num_local_k_chunks: {}", num_local_k_chunks);
    log_debug(tt::LogOp, "num_joint_k_chunks: {}", num_joint_k_chunks);

    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(mesh_device->arch(), args.compute_kernel_config);

    // Grid layout:
    //   user_grid:        Full grid from program_config (or device default). Contains SDPA workers + fabric MUX.
    //   sdpa_grid:        user_grid[:,:-1] — all columns except the last. SDPA workers (some are MUX writers).
    //   fabric_mux_col:   user_grid.x - 1 — last column reserved for fabric MUX kernel.
    //   sdpa_writer_range:  sdpa_grid columns 0..(sdpa_grid.x-3) — non-MUX SDPA writer cores.
    //   mux_writer_range:   sdpa_grid columns (sdpa_grid.x-2)..(sdpa_grid.x-1) — MUX fabric writer cores (2 links).
    const auto device_grid = mesh_device->compute_with_storage_grid_size();
    CoreCoord user_grid =
        args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size : device_grid;
    const bool mux_on_bottom_row = exp_sdpa_mux_on_bottom_row();
    // Bottom-row experiment: SDPA keeps every column but gives up the two bottom rows (row y-1
    // hosts the MUX kernels, row y-2 idles so the SDPA row count stays even for the direction
    // split). Default: SDPA keeps every row but gives up the last column to the MUX kernels.
    CoreCoord sdpa_grid =
        mux_on_bottom_row ? CoreCoord{user_grid.x, user_grid.y - 2} : CoreCoord{user_grid.x - 1, user_grid.y};

    TT_FATAL(
        user_grid.x <= device_grid.x && user_grid.y <= device_grid.y,
        "user_grid ({}x{}) exceeds device grid ({}x{}).",
        user_grid.x,
        user_grid.y,
        device_grid.x,
        device_grid.y);
    TT_FATAL(
        sdpa_grid.x >= 3,
        "SDPA grid must have at least 3 columns (1+ pure SDPA + 2 MUX writers). "
        "user_grid has {} cols, sdpa_grid has {} cols.",
        user_grid.x,
        sdpa_grid.x);
    TT_FATAL(
        sdpa_grid.y % 2 == 0,
        "SDPA grid rows ({}) must be even so the backward/forward MUX client halves match.",
        sdpa_grid.y);
    // The row-half split determines the per-link worker count; derive it from the grid so the
    // bottom-row experiment (fewer rows) keeps the direction and termination groups uniform.
    const uint32_t num_workers_per_link = sdpa_grid.y / 2;
    TT_FATAL(
        num_workers_per_link == args.num_workers_per_link || mux_on_bottom_row,
        "num_workers_per_link ({}) must equal sdpa_grid.y / 2 ({}).",
        args.num_workers_per_link,
        num_workers_per_link);

    bool exp_approx_mode =
        args.program_config.has_value()
            ? (args.program_config->exp_approx_mode.has_value() ? args.program_config->exp_approx_mode.value() : true)
            : true;

    auto sdpa_grid_range = CoreRange({0, 0}, {sdpa_grid.x - 1, sdpa_grid.y - 1});
    uint32_t num_sdpa_cores = sdpa_grid.x * sdpa_grid.y;

    log_debug(tt::LogOp, "user_grid: {}", user_grid);
    log_debug(tt::LogOp, "sdpa_grid: {}", sdpa_grid);
    log_debug(tt::LogOp, "num_sdpa_cores: {}", num_sdpa_cores);

    /**
     * Head-serial passes over row-sized SEGMENTS. A head's num_q_chunks Q chunks are split into
     * segs_per_head = num_q_chunks / sdpa_grid.x segments of one row each; segment s (flat over
     * batch x head x segment) runs as pass p = s / rows on row y = s % rows, and core (x, y) owns
     * that segment's Q chunk x. segs_per_head == 1 is the original one-row-per-head layout.
     *
     * Per pass the pipeline is exactly the single-head pipeline: each core holds one Q chunk per
     * pass (all resident, read once) and one flash-attention accumulator state per pass (kept in
     * the L1 state FIFO, see cb_prev_out). A split head's K/V shard is streamed — and all-gather
     * forwarded — once per segment: the duplicate fabric writes carry identical bytes to identical
     * addresses, and each row's chunk-ready signals come from the same row on the previous device,
     * so every row's pipeline stays self-contained exactly as in the one-row-per-head layout.
     *
     * Splitting heads balances the grid when B*NH does not divide the row count: e.g. 14 heads on
     * 10 rows is 2 passes of 10-tile chunks (20 Q tile-rows on the bottleneck cores, 6 rows idle
     * on the second pass), while segs_per_head=2 makes it 3 passes of 5-tile chunks (15 Q
     * tile-rows on every core) — the per-core matmul work drops by a quarter.
     */
    const uint32_t rows = sdpa_grid.y;
    // Every segment must fill its row exactly: fewer chunks than columns would idle the trailing
    // columns, and the last two SDPA columns are the fabric MUX clients that perform the K/V
    // all-gather — an idle MUX column means that link never forwards its shard.
    TT_FATAL(
        num_q_chunks % sdpa_grid.x == 0,
        "Exp ring joint SDPA requires a head's Q chunks to fill a whole number of rows. Got "
        "num_q_chunks={} with {} columns. Adjust q_chunk_size so ceil(local_padded_N / "
        "q_chunk_size) is a multiple of {}.",
        num_q_chunks,
        sdpa_grid.x,
        sdpa_grid.x);
    const uint32_t segs_per_head = num_q_chunks / sdpa_grid.x;
    const uint32_t total_segments = B * NH * segs_per_head;
    const uint32_t num_passes = tt::div_up(total_segments, rows);

    // L1-bound: the CB budget must hold num_passes resident Q chunks + state-FIFO entries. The
    // caller's program config is responsible for picking (q_chunk, k_chunk, segs) that fit; an
    // oversized combination fails CB allocation at program build.
    constexpr uint32_t kMaxPasses = 3;
    TT_FATAL(
        num_passes <= kMaxPasses,
        "Exp ring joint SDPA supports at most {} head-segments per core row. "
        "Got B*NH={} x segs_per_head={} with {} rows (P={}).",
        kMaxPasses,
        B * NH,
        segs_per_head,
        rows,
        num_passes);

    log_debug(tt::LogOp, "num_passes (heads per row): {}", num_passes);

    // Depth of the L1 state FIFO (c_6 max / c_11 sum / c_7 partial-out), in entries.
    //
    // A pass pops its own entry at its FIRST K chunk (the merge consumes `prev`) and pushes the
    // updated entry at its LAST K chunk, so num_passes entries are enough as long as a pass has at
    // least two K chunks: the pop has already freed the slot the push needs. A pass with exactly one
    // K chunk is the only case that reserves `cur` before popping `prev` (sdpa_inner_loop_step
    // reserves the new entry up front), which would deadlock at full depth — so give that case one
    // spare entry. The last active ring iteration is the only one that can be partial, and it
    // normalizes into cb_out instead of pushing to the FIFO, so it never needs the spare.
    // A pass that processes exactly ONE K/V chunk runs the state-FIFO entry (read + POST-step pop
    // of this pass's previous state) and the FIFO exit (PRE-step reserve + push of its new state)
    // around the SAME sdpa_inner_loop_step call — so the exit's reserve precedes the pop it needs.
    // Rows whose pass count fills the FIFO deadlock on that reserve. Give the FIFO one spare entry
    // whenever any ring iteration can process a single chunk: a one-chunk shard, or the
    // beyond-logical_n whole-chunk skip shaving the last shard down to one chunk (a pad tail that
    // covers at least one full K chunk — first hit by the fl2va 1024x768 canvas at 4x32).
    bool some_iter_processes_one_chunk = (num_local_k_chunks <= 1);
    for (uint32_t rid = 0; rid < args.ring_size && !some_iter_processes_one_chunk; ++rid) {
        uint32_t iter_chunks = 0;
        for (uint32_t kc = 0; kc < num_local_k_chunks; ++kc) {
            // Mirrors the kernels' kv_chunk_is_beyond_logical_n skip (joint chunks never skip).
            if (local_padded_Nt * rid + kc * Sk_chunk_t < logical_nt) {
                iter_chunks++;
            }
        }
        if (rid == args.ring_size - 1) {
            iter_chunks += num_joint_k_chunks;
        }
        some_iter_processes_one_chunk = (iter_chunks == 1);
    }
    const uint32_t state_fifo_entries = num_passes + (some_iter_processes_one_chunk ? 1 : 0);
    log_debug(tt::LogOp, "state_fifo_entries: {}", state_fifo_entries);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    // Q holds one chunk per pass; all of them stay resident for the whole op (read once).
    uint32_t q_tiles = num_passes * Sq_chunk_t * DHt;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t v_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * DHt;
    uint32_t out0_t = Sq_chunk_t * DHt;  // finalized below once out_out_subblock_h is known
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration

    // log all values
    log_debug(tt::LogOp, "q_tiles: {}", q_tiles);
    log_debug(tt::LogOp, "k_tiles: {}", k_tiles);
    log_debug(tt::LogOp, "v_tiles: {}", v_tiles);
    log_debug(tt::LogOp, "qk_tiles: {}", qk_tiles);
    log_debug(tt::LogOp, "statistics_tiles: {}", statistics_tiles);

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = ttnn::get_dest_reg_count(args.compute_kernel_config);
    const uint32_t qk_in0_block_w = DHt;
    auto [qk_out_subblock_h, qk_out_subblock_w] =
        detail::determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size);

    TT_FATAL(
        Sq_chunk_t % qk_out_subblock_h == 0,
        "Sq_chunk_t ({}) must be divisible by qk_out_subblock_h ({})",
        Sq_chunk_t,
        qk_out_subblock_h);
    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;

    // Streaming compute v2: eliminates row buffers via cb_push_back_hold_wr_ptr.
    // Ring joint has no causal/mask/sink/sliding/chunked flags — gating is simpler.
    // Streaming v2 requires q_num_subblocks > 1 (Sq_chunk_t > subblock_h) because the Phase 2
    // pipeline assumes at least one q_subblock iteration for correct softmax drain + SALAD overlap.
    const bool use_streaming_compute = !fp32_dest_acc_en && qk_out_subblock_h <= 2 &&
                                       Sk_chunk_t % (dst_size / qk_out_subblock_h) == 0 && qk_in0_num_subblocks > 1;

    auto [out_out_subblock_h, out_out_subblock_w] =
        detail::determine_largest_subblock_size(Sq_chunk_t, DHt, dst_size, use_streaming_compute ? 2 : UINT32_MAX);

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = DHt / out_out_subblock_w;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    // Streaming: shrink cb_out to a 2-slot ping-pong (see sdpa_subblock_utils.hpp). Safe here
    // because every pass runs with q_per_core == 1 (one Q chunk per pass, head-serial), so
    // Phase-2's save_to_staging branch (pack at offset qktv_h*vDHt into a 2*qktv_h*vDHt buffer)
    // never fires — cross-ring-iteration state lives in the L1 state FIFO instead.
    if (use_streaming_compute) {
        out0_t = detail::streaming_cb_out_tiles(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t, DHt);
        TT_FATAL(
            Sq_chunk_t % out_out_subblock_h == 0,
            "Streaming cb_out drain requires Sq_chunk_t ({}) divisible by out_out_subblock_h ({})",
            Sq_chunk_t,
            out_out_subblock_h);
    }
    log_debug(tt::LogOp, "out0_t: {}", out0_t);
    log_debug(tt::LogOp, "use_streaming_compute: {}", use_streaming_compute);

    // log all values
    log_debug(tt::LogOp, "dst_size: {}", dst_size);
    log_debug(tt::LogOp, "qk_in0_block_w: {}", qk_in0_block_w);
    log_debug(tt::LogOp, "qk_out_subblock_w: {}", qk_out_subblock_w);
    log_debug(tt::LogOp, "qk_out_subblock_h: {}", qk_out_subblock_h);
    log_debug(tt::LogOp, "qk_in0_num_subblocks: {}", qk_in0_num_subblocks);
    log_debug(tt::LogOp, "qk_in1_num_subblocks: {}", qk_in1_num_subblocks);
    log_debug(tt::LogOp, "qk_num_blocks: {}", qk_num_blocks);
    log_debug(tt::LogOp, "out_in0_block_w: {}", out_in0_block_w);
    log_debug(tt::LogOp, "out_out_subblock_w: {}", out_out_subblock_w);
    log_debug(tt::LogOp, "out_out_subblock_h: {}", out_out_subblock_h);
    log_debug(tt::LogOp, "out_in0_num_subblocks: {}", out_in0_num_subblocks);
    log_debug(tt::LogOp, "out_in1_num_subblocks: {}", out_in1_num_subblocks);
    log_debug(tt::LogOp, "out_num_blocks: {}", out_num_blocks);

    // Determine granularity for statistics computation
    // Each granularity must evenly divide its tile count to avoid dropping tiles
    const uint32_t stats_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size);
    const uint32_t sub_exp_granularity = detail::find_valid_granularity(Sk_chunk_t, dst_size);
    const uint32_t mul_bcast_granularity = detail::find_valid_granularity(Sq_chunk_t * Sk_chunk_t, dst_size);
    const uint32_t dht_granularity = detail::find_valid_granularity(DHt, dst_size);
    const uint32_t reduce_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size / 2);

    // Log these
    log_debug(tt::LogOp, "stats_granularity: {}", stats_granularity);
    log_debug(tt::LogOp, "sub_exp_granularity: {}", sub_exp_granularity);
    log_debug(tt::LogOp, "mul_bcast_granularity: {}", mul_bcast_granularity);
    log_debug(tt::LogOp, "dht_granularity: {}", dht_granularity);
    log_debug(tt::LogOp, "reduce_granularity: {}", reduce_granularity);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    const float scale_value = scale.value_or(1.0f);
    const uint32_t scale_packed = std::bit_cast<uint32_t>(scale_value);

    // log scale
    log_debug(tt::LogOp, "scale: {}", scale_value);

    std::vector<uint32_t> reader_compile_time_args = {
        B,
        NH,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        local_padded_Nt,
        padded_Nt,
        static_cast<uint32_t>(args.logical_n),
        logical_nt,
        Lt,
        L,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        args.ring_size,
        qk_out_subblock_h};

    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_v.buffer()).append_to(reader_compile_time_args);

    /**
     * Create semaphores used for L1-L1 store-and-forward of KV between cores.
     * In the descriptor pattern, semaphore IDs are explicit sequential integers
     * matching the order they are pushed into desc.semaphores below. Since these
     * three cover the same sdpa_grid_range, they receive distinct IDs 0, 1, 2.
     */
    const uint32_t sender_semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sender_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = CoreRangeSet(sdpa_grid_range),
        .initial_value = INVALID,
    });
    const uint32_t receiver_semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = receiver_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = CoreRangeSet(sdpa_grid_range),
        .initial_value = INVALID,
    });
    const uint32_t valid_semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = valid_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = CoreRangeSet(sdpa_grid_range),
        .initial_value = VALID,
    });
    // Second receiver valid flag for the mcast ping-pong: K chunks land on receiver_semaphore_id,
    // V chunks on receiver_semaphore_b_id. With separate flags a receiver can post the
    // (reserve + flag-reset + ack) credit for chunk n+1 of a channel right after consuming chunk n,
    // so two row-broadcasts (V of chunk n, K of chunk n+1) can be in flight at once instead of the
    // injector paying a full receiver round trip between every mcast.
    const uint32_t receiver_semaphore_b_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = receiver_semaphore_b_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = CoreRangeSet(sdpa_grid_range),
        .initial_value = INVALID,
    });
    // Separate V-channel ack counter. The K/V ack totals must be counted independently: a
    // receiver's K credits run one chunk ahead of its V credits, so a single combined counter
    // would let the fast receivers' K(n+1) credits stand in for a slow receiver's missing V(n)
    // credit — the injector would multicast V(n) before that receiver reset its V flag, and the
    // relayed VALID would be erased by the late reset (observed deadlock). Counted per channel,
    // a receiver can be at most one credit ahead, and that credit is exactly the event the
    // injector waits on, so total >= 10*event_index implies every receiver posted this event.
    const uint32_t sender_semaphore_v_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sender_semaphore_v_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = CoreRangeSet(sdpa_grid_range),
        .initial_value = INVALID,
    });
    // Split-head forwarding dedup buddy gate. With segs_per_head == 2 the two rows of an adjacent
    // pair (2i, 2i+1) process the SAME head against the SAME deterministic ring sequence in every
    // pass (when both rows are in the same fabric direction half), so their mux clients forward
    // byte-identical packets to identical remote addresses. The follower (odd) row's clients skip
    // forwarding entirely; the leader (even) row's injector — after its own per-link gate passes —
    // relays one on-chip semaphore inc per chunk to the follower's injector, which gates on this
    // semaphore instead of the (now silent) per-link semaphores.
    const uint32_t buddy_gate_semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = buddy_gate_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = CoreRangeSet(sdpa_grid_range),
        .initial_value = 0,
    });

    // Append semaphore ids to reader compile-time args (must match reader kernel expectations)
    const auto sem_args_offset = reader_compile_time_args.size();
    reader_compile_time_args.push_back(sender_semaphore_id);
    reader_compile_time_args.push_back(receiver_semaphore_id);
    reader_compile_time_args.push_back(valid_semaphore_id);
    reader_compile_time_args.push_back(0);  // mcast_enabled placeholder (patched after chain construction)
    reader_compile_time_args.push_back(0);  // stream_q placeholder (patched after CB sizing)
    reader_compile_time_args.push_back(receiver_semaphore_b_id);
    reader_compile_time_args.push_back(sender_semaphore_v_id);
    reader_compile_time_args.push_back(buddy_gate_semaphore_id);
    // Trace-safe logical_n: presence flag, then the accessor args for the 1-element tensor. The accessor
    // block is emitted unconditionally (nullptr when absent) so the CT-arg layout after it never shifts.
    reader_compile_time_args.push_back(static_cast<uint32_t>(has_logical_n_tensor));
    TensorAccessorArgs(has_logical_n_tensor ? tensor_args.logical_n_tensor->buffer() : nullptr)
        .append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NH,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        local_padded_N,
        local_padded_Nt,
        args.logical_n,
        logical_nt,
        Lt,
        L,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        packed_identity_scalar,
        scale_packed,
        args.ring_size,
        global_n_partial_col,
        joint_l_partial_col,
        static_cast<std::uint32_t>(use_streaming_compute),
        static_cast<std::uint32_t>(out_out_subblock_h),
        static_cast<std::uint32_t>(has_logical_n_tensor),
    };

    // Trace-safe logical_n accessor block sits ahead of the output accessors so every downstream offset
    // (out / joint_out / stats, and the MUX + AG block that chains off stats) keeps deriving normally.
    // Emitted unconditionally (nullptr when absent) to keep the layout stable across both modes.
    TensorAccessorArgs(has_logical_n_tensor ? tensor_args.logical_n_tensor->buffer() : nullptr)
        .append_to(writer_compile_time_args);
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(joint_output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(stats_output_tensor.buffer()).append_to(writer_compile_time_args);

    // Streaming-only compute kernel: NH and the classic matmul block params (in0_block_w,
    // num_subblocks, num_blocks) are not consumed — only the subblock shapes are.
    std::vector<uint32_t> compute_compile_time_args = {
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        local_padded_N,
        local_padded_Nt,
        args.logical_n,
        logical_nt,
        Lt,
        L,
        num_local_k_chunks,
        num_joint_k_chunks,
        args.ring_size,
        qk_out_subblock_w,
        qk_out_subblock_h,
        out_out_subblock_w,
        out_out_subblock_h,
        scale_packed,
        static_cast<std::uint32_t>(use_streaming_compute),
        global_n_partial_col,
        joint_l_partial_col,
        0,  // stream_q placeholder (patched after CB sizing)
        // Single-pass programs keep the original persist-in-scratch accumulator path: the L1
        // state FIFO only earns its per-iteration entry/exit cost (redirected packs + a dual
        // max write) when several head-passes share a core. The 1-pass shapes are gated by
        // utilization-band perf tests calibrated on the scratch path.
        (num_passes > 1) ? 1u : 0u,  // use_l1_state_fifo
        // Trace-safe logical_n: compute reads the live values from the reader's derived CB.
        static_cast<uint32_t>(has_logical_n_tensor),
    };

    std::map<std::string, std::string> defines;
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["REDUCE_GRANULARITY"] = std::to_string(reduce_granularity);
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);

    // NOTE: KernelDescriptor construction is deferred until after chain construction
    // so that the mcast_enabled compile-time arg can be determined first.

    // Create circular buffers

    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_k.dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_v.dtype());
    // Lightweight mask: both streaming and non-streaming paths use Float16_b
    // to support L1-accumulation and avoid Bfp4_b precision loss.
    tt::DataFormat mask_df = tt::DataFormat::Float16_b;
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;  // need to disable fp32 cbs (Issue #13364) fp32_dest_acc_en ?
                                                       // tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = im_df;

    uint32_t q_tile_size = tt::tile_size(q_df);
    uint32_t k_tile_size = tt::tile_size(k_df);
    uint32_t v_tile_size = tt::tile_size(v_df);
    uint32_t mask_tile_size = tt::tile_size(mask_df);
    uint32_t out_tile_size = tt::tile_size(out_df);
    uint32_t scalar_tile_size = tt::tile_size(scalar_df);
    uint32_t im_tile_size = tt::tile_size(im_df);
    uint32_t stats_tile_size = tt::tile_size(stats_df);

    log_debug(tt::LogOp, "q_data_format: {}", q_df);
    log_debug(tt::LogOp, "k_data_format: {}", k_df);
    log_debug(tt::LogOp, "v_data_format: {}", v_df);
    log_debug(tt::LogOp, "mask_data_format: {}", mask_df);
    log_debug(tt::LogOp, "out_data_format: {}", out_df);
    log_debug(tt::LogOp, "scalar_data_format: {}", scalar_df);
    log_debug(tt::LogOp, "intermediate_data_format: {}", im_df);
    log_debug(tt::LogOp, "statistics_data_format: {}", stats_df);

    const auto sdpa_grid_set = CoreRangeSet(sdpa_grid_range);

    // Q input. NOTE: sized for resident Q here; the streamed-Q fallback below (search stream_q)
    // patches desc.cbs[0].total_size down to one chunk when the resident total does not fit L1.
    desc.cbs.push_back(CBDescriptor{
        .total_size = q_tiles * q_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = q_df,
            .page_size = q_tile_size,
        }}},
    });
    // K and V input CBs with overlapping handles (c_1+c_14 for K, c_2+c_15 for V) so both
    // compute and MUX writer can pop independently from the same L1 address space.
    desc.cbs.push_back(CBDescriptor{
        .total_size = k_tiles * k_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors =
            {{CBFormatDescriptor{
                  .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
                  .data_format = k_df,
                  .page_size = k_tile_size,
              },
              CBFormatDescriptor{
                  .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_14),
                  .data_format = k_df,
                  .page_size = k_tile_size,
              }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = v_tiles * v_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors =
            {{CBFormatDescriptor{
                  .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
                  .data_format = v_df,
                  .page_size = v_tile_size,
              },
              CBFormatDescriptor{
                  .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_15),
                  .data_format = v_df,
                  .page_size = v_tile_size,
              }}},
    });

    // Lightweight mask: single CB holds 1 neginf tile + up to 2 partial mask tiles
    if (needs_lightweight_mask) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = total_lightweight_mask_tiles * mask_tile_size,
            .core_ranges = sdpa_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = mask_df,
                .page_size = mask_tile_size,
            }}},
        });
    }

    // scale input
    desc.cbs.push_back(CBDescriptor{
        .total_size = scale_tiles * scalar_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
            .data_format = scalar_df,
            .page_size = scalar_tile_size,
        }}},
    });

    // identity scale input
    desc.cbs.push_back(CBDescriptor{
        .total_size = scale_tiles * scalar_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
            .data_format = scalar_df,
            .page_size = scalar_tile_size,
        }}},
    });

    // Running-max half of the L1 state FIFO (num_passes + 1 entries; see cb_prev_out below).
    desc.cbs.push_back(CBDescriptor{
        .total_size = state_fifo_entries * statistics_tiles * im_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
            .data_format = im_df,
            .page_size = im_tile_size,
        }}},
    });

    // Partial-output half of the L1 state FIFO.
    //
    // Head-serial passes keep one flash-attention accumulator state (max + sum + partial out) per
    // pass live across all ring iterations. The three state CBs (c_6 max, c_11 sum, c_7 out) are
    // used as FIFOs: a pass pops its own state at its first K chunk and pushes the updated state at
    // its last K chunk, so the fixed cyclic pass order keeps each pass's state at the front when it
    // runs. See state_fifo_entries above for the depth derivation. Sizing stays an exact multiple of
    // the per-entry tile count so no entry straddles the ring-buffer wrap — intra-entry reads index
    // tiles relative to the CB front.
    //
    // Format is im_df (not out_df): these tiles are accumulator intermediates shared with the
    // c_25/c_26 scratch halves, not DRAM-formatted output.
    desc.cbs.push_back(CBDescriptor{
        .total_size = state_fifo_entries * out_im_tiles * im_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_7),
            .data_format = im_df,
            .page_size = im_tile_size,
        }}},
    });

    // column identity input
    desc.cbs.push_back(CBDescriptor{
        .total_size = scale_tiles * scalar_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_8),
            .data_format = scalar_df,
            .page_size = scalar_tile_size,
        }}},
    });

    // cb_qk_im
    desc.cbs.push_back(CBDescriptor{
        .total_size = qk_tiles * im_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_24),
            .data_format = im_df,
            .page_size = im_tile_size,
        }}},
    });

    // cb_out_im
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_im_tiles * im_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_25),
            .data_format = im_df,
            .page_size = im_tile_size,
        }}},
    });

    // cb_out_accumulate_im
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_im_tiles * im_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_26),
            .data_format = im_df,
            .page_size = im_tile_size,
        }}},
    });

    // cb_cur_max
    desc.cbs.push_back(CBDescriptor{
        .total_size = statistics_tiles * stats_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_27),
            .data_format = stats_df,
            .page_size = stats_tile_size,
        }}},
    });

    // cb_prev_max
    desc.cbs.push_back(CBDescriptor{
        .total_size = statistics_tiles * stats_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_28),
            .data_format = stats_df,
            .page_size = stats_tile_size,
        }}},
    });

    // cb_cur_sum
    desc.cbs.push_back(CBDescriptor{
        .total_size = statistics_tiles * stats_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_29),
            .data_format = stats_df,
            .page_size = stats_tile_size,
        }}},
    });

    // cb_prev_sum
    desc.cbs.push_back(CBDescriptor{
        .total_size = statistics_tiles * stats_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_30),
            .data_format = stats_df,
            .page_size = stats_tile_size,
        }}},
    });

    // cb_exp_max_diff
    desc.cbs.push_back(CBDescriptor{
        .total_size = statistics_tiles * stats_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_31),
            .data_format = stats_df,
            .page_size = stats_tile_size,
        }}},
    });

    // Output
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * out_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
            .data_format = out_df,
            .page_size = out_tile_size,
        }}},
    });

    // stats output
    desc.cbs.push_back(CBDescriptor{
        .total_size = statistics_tiles * im_tile_size,
        .core_ranges = sdpa_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_17),
            .data_format = im_df,
            .page_size = im_tile_size,
        }}},
    });

    // Streaming compute v2: 1-tile recip scratch CB (c_9) for normalize_row_streaming.
    // c_4 is used by cb_scale_in in ring joint, so we use c_9 instead.
    if (use_streaming_compute) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = 1 * im_tile_size,
            .core_ranges = sdpa_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_9),
                .data_format = im_df,
                .page_size = im_tile_size,
            }}},
        });
    }

    // Compute RISCs cannot NoC-read DRAM, so the reader publishes derived values here (slots per
    // ring_joint_derived_slots.hpp) and compute reads them with read_tile_value. Must be UInt32:
    // read_tile_value's indexing follows the CB format, and a float format misreads these raw words.
    if (has_logical_n_tensor) {
        constexpr uint32_t kDerivedPageBytes = 64;
        static_assert(
            ttnn::operations::transformer::sdpa::ring_joint::kDerivedSlotCount * sizeof(uint32_t) <= kDerivedPageBytes,
            "derived slots must fit the page");
        desc.cbs.push_back(CBDescriptor{
            .total_size = kDerivedPageBytes,
            .core_ranges = sdpa_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_13),
                .data_format = tt::DataFormat::UInt32,
                .page_size = kDerivedPageBytes,
            }}},
        });
    }

    // c_10 (cb_sum_out) belongs to the multi-Q DRAM round-trip, which head-serial passes never
    // take (every pass runs with q_per_core == 1); it is NOT allocated — the kernel's cb_sum_out
    // index is only touched in the staging branch, which is dead at q_per_core == 1.
    // c_11 (cb_sum_in) is the running-sum half of the L1 state FIFO — see cb_prev_out (c_7).
    if (use_streaming_compute) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = state_fifo_entries * statistics_tiles * stats_tile_size,
            .core_ranges = sdpa_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_11),
                .data_format = stats_df,
                .page_size = stats_tile_size,
            }}},
        });
    }

    // Streamed-Q fallback: if the CBs do not fit L1 with all num_passes Q chunks resident, keep
    // only one chunk resident; the reader then re-reads each pass's Q every ring iteration and
    // compute pops it at the end of the pass. Buys back (num_passes - 1) * Sq_chunk_t * DHt tiles,
    // which at H3 15s (q=320, k=384, P=2) is the difference between fitting and not.
    // See exp_more_heads_per_row.md §9.
    // CBs must end below the lowest live L1 buffer (validate_circular_buffer_region enforces
    // exactly this). In the pipeline, global semaphores and persistent buffers occupy the top of
    // L1, so budgeting against the raw L1 size over-promises and the program clashes at allocate.
    const auto lowest_l1_buffer = mesh_device->lowest_occupied_compute_l1_address();
    const uint32_t cb_space_top = lowest_l1_buffer.has_value() ? static_cast<uint32_t>(lowest_l1_buffer.value())
                                                               : mesh_device->l1_size_per_core();
    const uint32_t usable_l1 =
        cb_space_top - mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    uint64_t total_cb_bytes = 0;
    for (const auto& cb : desc.cbs) {
        total_cb_bytes += cb.total_size;
    }
    const bool stream_q = (num_passes > 1) && (total_cb_bytes > usable_l1);
    if (stream_q) {
        total_cb_bytes -= desc.cbs[0].total_size;
        desc.cbs[0].total_size = Sq_chunk_t * DHt * q_tile_size;  // c_0 is the first CB pushed
        total_cb_bytes += desc.cbs[0].total_size;
    }
    TT_FATAL(
        total_cb_bytes <= usable_l1,
        "Exp ring joint SDPA CBs need {} B but only {} B of L1 are usable at this chunk shape "
        "(q_chunk={}, k_chunk={}, passes={}); reduce k_chunk_size.",
        total_cb_bytes,
        usable_l1,
        q_chunk_size,
        k_chunk_size,
        num_passes);
    log_debug(tt::LogOp, "stream_q: {}", stream_q);
    reader_compile_time_args[sem_args_offset + 4] = stream_q ? 1 : 0;
    compute_compile_time_args[20] = stream_q ? 1 : 0;

    auto* const q_buf = input_tensor_q.buffer();
    auto* const k_buf = input_tensor_k.buffer();
    auto* const v_buf = input_tensor_v.buffer();
    auto* const gathered_k_buf = gathered_input_tensor_k.buffer();
    auto* const gathered_v_buf = gathered_input_tensor_v.buffer();
    auto* const joint_q_buf = joint_tensor_q.buffer();
    auto* const joint_k_buf = joint_tensor_k.buffer();
    auto* const joint_v_buf = joint_tensor_v.buffer();
    auto* const out_buf = output_tensor.buffer();
    auto* const joint_out_buf = joint_output_tensor.buffer();
    auto* const stats_buf = stats_output_tensor.buffer();

    /**
     * Build per-row store-and-forward chains.
     *
     * Head-serial passes place num_passes heads on every row, and every pass of a row runs on the
     * same cores (logical x in [0, num_q_chunks)), so a row has ONE chain shared by all its passes.
     * One injector per row is required, not merely convenient: each MUX writer pre-configures a
     * single fabric atomic-inc destination for the whole op (see exp_ring_joint_writer.cpp), so
     * every pass of a row must signal the same injector core.
     */
    struct CoreHeadWork {
        uint32_t batch = 0;
        uint32_t head = 0;
        uint32_t q_chunk_start = 0;
        uint32_t q_chunk_count = 0;
    };

    struct CoreWork {
        CoreCoord logical_core;
        CoreCoord physical_core;
        // Q work descriptor: this core owns flat chunks q_base + p * q_stride for p in [0, q_count).
        uint32_t q_base = 0;
        uint32_t q_stride = 0;
        uint32_t q_count = 0;
        std::vector<CoreHeadWork> head_work;  // one entry per pass, in pass order
    };

    struct CoreChainInfo {
        bool participates = false;
        bool is_injector = false;
        bool is_sink = false;
        // Pass-0 (batch, head) and chunk range. The kernels no longer compare chunks against these:
        // row-aligned scheduling makes every chunk a core owns part of its row's chain. Kept so the
        // reader RT-arg layout is unchanged.
        uint32_t batch = 0;
        uint32_t head = 0;
        uint32_t q_chunk_start = 0;
        uint32_t q_chunk_count = 0;
        CoreCoord prev_physical = CoreCoord{0, 0};
        CoreCoord next_physical = CoreCoord{0, 0};
        uint32_t next_core_q_chunks = 0;
        uint32_t mcast_num_dests = 0;
        uint32_t mcast_sender_wait = 0;
    };

    std::vector<CoreWork> core_work(num_sdpa_cores);
    std::vector<CoreChainInfo> core_chain_info(num_sdpa_cores);

    /**
     * Row-aligned work assignment over head-segments. The flat chunk encoding is unchanged:
     *     flat = (batch * NH + head) * num_q_chunks + q_chunk
     * Pass p of core (x, y) owns segment s = p * rows + y, chunk x within it. Since
     * num_q_chunks = segs_per_head * sdpa_grid.x, the flat id collapses to the same affine form
     * the kernels already consume:
     *     flat = (s / segs_per_head) * num_q_chunks + (s % segs_per_head) * sdpa_grid.x + x
     *          = s * sdpa_grid.x + x = q_base + p * q_stride
     * with q_base = y * sdpa_grid.x + x and q_stride = rows * sdpa_grid.x.
     *
     * Two properties the kernels rely on:
     *   - every kernel derives (batch, head, q_chunk) from the flat id per pass, so a
     *     pass-dependent q_chunk (segments) needs no kernel change;
     *   - when total_segments % rows != 0 an entire row idles the final pass, so all cores of a
     *     row always run the same number of passes (required for K/V CB pointer lockstep under
     *     mcast).
     */
    const uint32_t q_stride = rows * sdpa_grid.x;
    for (uint32_t i = 0; i < num_sdpa_cores; ++i) {
        const CoreCoord core = {i % sdpa_grid.x, i / sdpa_grid.x};
        auto& work = core_work.at(i);
        work.logical_core = core;
        work.physical_core = device->worker_core_from_logical_core(core);
        work.q_base = core.y * sdpa_grid.x + core.x;
        work.q_stride = q_stride;
        work.q_count = 0;
        for (uint32_t p = 0; p < num_passes; ++p) {
            const uint32_t seg_id = p * rows + core.y;
            if (seg_id >= total_segments) {
                break;
            }
            const uint32_t head_id = seg_id / segs_per_head;
            work.q_count++;
            work.head_work.push_back(CoreHeadWork{
                .batch = head_id / NH,
                .head = head_id % NH,
                .q_chunk_start = (seg_id % segs_per_head) * sdpa_grid.x + core.x,
                .q_chunk_count = 1,
            });
        }
    }

    log_debug(
        tt::LogOp,
        "[ExpRingJointSDPA] grid={}x{}={} cores, B={}, NH={}, num_q_chunks={}({} local+{} joint), "
        "passes_per_row={}, q_stride={}",
        sdpa_grid.x,
        sdpa_grid.y,
        num_sdpa_cores,
        B,
        NH,
        num_q_chunks,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_passes,
        q_stride);

    // One chain per row, over that row's participating cores in ascending logical x (which is
    // ascending physical x within a row). Injector is refined below for DRAM channel spreading.
    std::vector<std::vector<uint32_t>> row_chain_cores(rows);
    for (uint32_t y = 0; y < rows; ++y) {
        auto& chain_cores = row_chain_cores[y];
        for (uint32_t x = 0; x < sdpa_grid.x; ++x) {
            const uint32_t core_idx = y * sdpa_grid.x + x;
            if (core_work.at(core_idx).q_count == 0) {
                continue;
            }
            chain_cores.push_back(core_idx);
        }

        if (chain_cores.size() < 2) {
            continue;  // a single-core row needs no store-and-forward
        }

        for (std::size_t idx = 0; idx < chain_cores.size(); ++idx) {
            const uint32_t core_idx = chain_cores[idx];
            const auto& hw = core_work.at(core_idx).head_work.at(0);
            auto& chain = core_chain_info.at(core_idx);

            chain.participates = true;
            chain.batch = hw.batch;
            chain.head = hw.head;
            chain.q_chunk_start = hw.q_chunk_start;
            chain.q_chunk_count = hw.q_chunk_count;
            chain.is_injector = (idx == 0);
            chain.is_sink = (idx + 1 == chain_cores.size());

            if (idx > 0) {
                chain.prev_physical = core_work.at(chain_cores[idx - 1]).physical_core;
            }
            if (idx + 1 < chain_cores.size()) {
                const uint32_t next_core_idx = chain_cores[idx + 1];
                chain.next_physical = core_work.at(next_core_idx).physical_core;
                chain.next_core_q_chunks = core_work.at(next_core_idx).head_work.at(0).q_chunk_count;
            }
        }
    }

    {
        uint32_t num_chains = 0;
        std::string hist_str;
        for (uint32_t y = 0; y < rows; ++y) {
            if (row_chain_cores[y].size() >= 2) {
                num_chains++;
                hist_str += std::to_string(row_chain_cores[y].size()) + " ";
            }
        }
        log_debug(
            tt::LogOp,
            "[ExpRingJointSDPA] {} row chains (lengths: {})",
            num_chains,
            hist_str.empty() ? "none" : hist_str);
    }

    // Multicast eligibility, evaluated per row. All-or-nothing: mcast is mandatory for this op, so
    // a single ineligible row is a hard error below rather than a per-row fallback.
    uint32_t mcast_chains = 0;
    {
        struct McastCandidate {
            std::vector<uint32_t> core_indices;
            uint32_t ref_q_chunks;
        };
        std::vector<McastCandidate> candidates;
        candidates.reserve(rows);
        bool all_eligible = true;

        for (uint32_t y = 0; y < rows && all_eligible; ++y) {
            const auto& chain_cores = row_chain_cores[y];
            if (chain_cores.size() < 2) {
                continue;
            }

            // Condition 1: all physical cores share the same Y coordinate.
            const uint32_t ref_y = core_work[chain_cores[0]].physical_core.y;
            bool same_row = true;
            for (std::size_t ci = 1; ci < chain_cores.size(); ++ci) {
                if (core_work[chain_cores[ci]].physical_core.y != ref_y) {
                    same_row = false;
                    break;
                }
            }
            if (!same_row) {
                all_eligible = false;
                log_debug(tt::LogOp, "Row {}: mcast ineligible - cores span multiple physical rows", y);
                break;
            }

            // Condition 2: no non-chain worker core inside the mcast rectangle.
            uint32_t min_x = core_work[chain_cores[0]].physical_core.x;
            uint32_t max_x = min_x;
            for (const auto& ci : chain_cores) {
                const uint32_t x = core_work[ci].physical_core.x;
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
            }

            bool has_gap = false;
            for (uint32_t ci = 0; ci < num_sdpa_cores; ++ci) {
                const auto& phys = core_work[ci].physical_core;
                if (phys.y != ref_y || phys.x < min_x || phys.x > max_x) {
                    continue;
                }
                if (std::find(chain_cores.begin(), chain_cores.end(), ci) == chain_cores.end()) {
                    has_gap = true;
                    break;
                }
            }
            if (has_gap) {
                all_eligible = false;
                log_debug(tt::LogOp, "Row {}: mcast ineligible - non-chain worker core inside mcast rectangle", y);
                break;
            }

            // Condition 3: uniform per-core chunk count (one chunk per pass, so always uniform).
            const uint32_t ref_q_chunks = core_chain_info[chain_cores[0]].q_chunk_count;
            bool uniform_q_mcast = true;
            for (std::size_t ci = 1; ci < chain_cores.size(); ++ci) {
                if (core_chain_info[chain_cores[ci]].q_chunk_count != ref_q_chunks) {
                    uniform_q_mcast = false;
                    break;
                }
            }
            if (!uniform_q_mcast) {
                all_eligible = false;
                log_debug(tt::LogOp, "Row {}: mcast ineligible - mixed q_chunk_counts", y);
                break;
            }

            candidates.push_back(McastCandidate{chain_cores, ref_q_chunks});
        }

        if (all_eligible && !candidates.empty()) {
            mcast_chains = candidates.size();
            // Track injector physical X columns for DRAM channel spreading across rows.
            std::vector<uint32_t> injector_phys_x;
            injector_phys_x.reserve(candidates.size());
            for (const auto& cand : candidates) {
                const uint32_t chain_size = cand.core_indices.size();
                const uint32_t num_receivers = chain_size - 1;

                uint32_t injector_idx = cand.core_indices[0];
                for (const auto& ci : cand.core_indices) {
                    if (core_chain_info[ci].is_injector) {
                        injector_idx = ci;
                        break;
                    }
                }

                // Reselect injector for DRAM channel spreading: pick the core whose physical X is
                // furthest from all previously chosen injectors.
                {
                    uint32_t best_idx = injector_idx;
                    uint32_t best_dist = 0;
                    for (const auto& ci : cand.core_indices) {
                        const uint32_t phys_x = core_work[ci].physical_core.x;
                        uint32_t min_dist = UINT32_MAX;
                        for (uint32_t ix : injector_phys_x) {
                            const uint32_t d = (phys_x > ix) ? (phys_x - ix) : (ix - phys_x);
                            min_dist = std::min(min_dist, d);
                        }
                        if (min_dist > best_dist) {
                            best_dist = min_dist;
                            best_idx = ci;
                        }
                    }
                    if (best_idx != injector_idx) {
                        core_chain_info[injector_idx].is_injector = false;
                        core_chain_info[injector_idx].is_sink = true;
                        core_chain_info[best_idx].is_injector = true;
                        core_chain_info[best_idx].is_sink = false;
                        injector_idx = best_idx;
                    }
                }
                injector_phys_x.push_back(core_work[injector_idx].physical_core.x);

                uint32_t min_x = core_work[cand.core_indices[0]].physical_core.x;
                uint32_t max_x = min_x;
                for (std::size_t ci = 1; ci < cand.core_indices.size(); ++ci) {
                    const uint32_t x = core_work[cand.core_indices[ci]].physical_core.x;
                    min_x = std::min(min_x, x);
                    max_x = std::max(max_x, x);
                }
                const uint32_t injector_y = core_work[injector_idx].physical_core.y;
                const CoreCoord rect_start = CoreCoord{min_x, injector_y};
                const CoreCoord rect_end = CoreCoord{max_x, injector_y};

                // noc_async_write_multicast (non-loopback) never writes to self, so
                // num_dests must be num_receivers (chain_size - 1), not chain_size.
                const uint32_t mcast_num_dests = num_receivers;

                auto& injector_chain = core_chain_info[injector_idx];
                injector_chain.prev_physical = rect_start;
                injector_chain.next_physical = rect_end;
                injector_chain.mcast_num_dests = mcast_num_dests;
                injector_chain.mcast_sender_wait = num_receivers;
                injector_chain.next_core_q_chunks = cand.ref_q_chunks;

                for (const auto& ci : cand.core_indices) {
                    if (ci == injector_idx) {
                        continue;
                    }
                    auto& receiver_chain = core_chain_info[ci];
                    receiver_chain.prev_physical = core_work[injector_idx].physical_core;
                    receiver_chain.next_physical = CoreCoord{0, 0};
                    receiver_chain.next_core_q_chunks = 0;
                    receiver_chain.is_sink = true;
                }

                log_debug(
                    tt::LogOp,
                    "Row mcast enabled - {} receivers, injector core {} (phys_x={}), num_dests={} -> rect "
                    "({},{}) to ({},{})",
                    num_receivers,
                    injector_idx,
                    core_work[injector_idx].physical_core.x,
                    mcast_num_dests,
                    rect_start.x,
                    rect_start.y,
                    rect_end.x,
                    rect_end.y);
            }
        }

        log_debug(
            tt::LogOp,
            "Multicast eligibility: {}/{} row chains using mcast (all-or-nothing)",
            mcast_chains,
            static_cast<uint32_t>(candidates.size()));
    }

    log_debug(
        tt::LogOp,
        "[ExpRingJointSDPA] mcast: {} ({}/{} chains)",
        mcast_chains > 0 ? "ENABLED" : "DISABLED",
        mcast_chains,
        mcast_chains > 0 ? mcast_chains
                         : static_cast<uint32_t>(std::count_if(
                               core_chain_info.begin(), core_chain_info.end(), [](const CoreChainInfo& c) {
                                   return c.is_injector;
                               })));

    // Validate that mcast is enabled on all chains
    {
        const uint32_t total_chains = std::count_if(
            core_chain_info.begin(), core_chain_info.end(), [](const CoreChainInfo& c) { return c.is_injector; });
        TT_FATAL(
            mcast_chains == total_chains,
            "Exp ring joint SDPA requires mcast on all chains. Got {}/{} chains using mcast.",
            mcast_chains,
            total_chains);
    }

    // Update mcast_enabled compile-time arg now that chain construction is complete
    reader_compile_time_args[sem_args_offset + 3] = (mcast_chains > 0) ? 1 : 0;

    // Map core row -> injector physical coordinates for MUX writer signaling. Per row (not per
    // head) because all passes of a row share one injector — the MUX writer bakes a single
    // atomic-inc destination for the whole op.
    std::vector<std::optional<CoreCoord>> injector_physical_by_row(rows);
    for (uint32_t ci = 0; ci < num_sdpa_cores; ++ci) {
        if (core_chain_info[ci].is_injector) {
            injector_physical_by_row.at(core_work[ci].logical_core.y) = core_work[ci].physical_core;
        }
    }

    // Split-head forwarding dedup roles (see the buddy_gate semaphore comment). Enabled per row
    // pair (2i, 2i+1), which shares one head in EVERY pass when segs_per_head == 2 and the row
    // count is even (head = (p*rows + y) / 2, and p*rows is even). Requirements per pair: same
    // fabric direction half (identical deterministic ring sequences and identical remote
    // destination device — the cross-direction middle pair keeps duplicate forwarding), equal
    // q_count (identical gate schedules), and both injectors known. Roles: 0 = none (forward and
    // gate as before), 1 = leader (forward; relay gate to buddy), 2 = follower (skip forwarding;
    // gate on the leader's relay).
    std::vector<uint32_t> row_dedup_role(sdpa_grid.y, 0);
    std::vector<CoreCoord> row_buddy_injector(sdpa_grid.y, CoreCoord{0, 0});
    if (mcast_chains > 0 && segs_per_head == 2 && (sdpa_grid.y % 2 == 0)) {
        for (uint32_t y = 0; y + 1 < sdpa_grid.y; y += 2) {
            const uint32_t yf = y + 1;
            const bool same_direction = (y < num_workers_per_link) == (yf < num_workers_per_link);
            const uint32_t qc_leader = core_work.at(y * sdpa_grid.x).q_count;
            const uint32_t qc_follower = core_work.at(yf * sdpa_grid.x).q_count;
            const auto& inj_leader = injector_physical_by_row.at(y);
            const auto& inj_follower = injector_physical_by_row.at(yf);
            if (same_direction && qc_leader == qc_follower && qc_leader > 0 && inj_leader.has_value() &&
                inj_follower.has_value()) {
                row_dedup_role[y] = 1;
                row_dedup_role[yf] = 2;
                row_buddy_injector[y] = inj_follower.value();
                row_buddy_injector[yf] = inj_leader.value();
            }
        }
    }

    // Follower rows send nothing over fabric, so they do not connect to the MUX at all: the MUX
    // config allocates channels only for FORWARDING rows (compact channel ids per direction
    // half), and the freed L1 goes into deeper per-channel buffering (num_buffers_per_channel is
    // scaled up by the client reduction). With dedup off every row forwards and this reduces to
    // the original 1:1 layout.
    uint32_t fwd_rows_dir0 = 0;
    uint32_t fwd_rows_dir1 = 0;
    std::vector<uint32_t> row_mux_channel(sdpa_grid.y, 0);
    for (uint32_t y = 0; y < sdpa_grid.y; ++y) {
        if (row_dedup_role[y] == 2) {
            continue;
        }
        if (y < num_workers_per_link) {
            row_mux_channel[y] = fwd_rows_dir0++;
        } else {
            row_mux_channel[y] = fwd_rows_dir1++;
        }
    }
    TT_FATAL(
        fwd_rows_dir0 == fwd_rows_dir1 && fwd_rows_dir0 > 0,
        "Split-head dedup produced asymmetric forwarding groups ({} backward vs {} forward): the "
        "shared MUX config and termination count require equal per-direction client counts.",
        fwd_rows_dir0,
        fwd_rows_dir1);
    const uint32_t num_mux_clients_per_group = fwd_rows_dir0;

    // ---- Fabric MUX config (needed for writer kernel CT args below) ----
    // Default: MUX cores are placed at the last x coordinate of the user grid.
    //   Backward MUX: first y (0) and last y (sdpa_grid.y - 1). Forward MUX: middle y - 1 and middle y.
    // Bottom-row experiment: MUX cores sit on the last ROW of the user grid, transposing the same
    //   arrangement — backward at the first and last x, forward at the middle pair.
    const uint32_t mid = mux_on_bottom_row ? sdpa_grid.x / 2 : sdpa_grid.y / 2;
    const uint32_t fabric_mux_col = user_grid.x - 1;  // Last column of user_grid (default placement)
    const uint32_t fabric_mux_row = user_grid.y - 1;  // Last row of user_grid (bottom-row placement)
    const bool mux_top_cluster = exp_sdpa_mux_top_cluster();
    const std::vector<CoreCoord> mux_backward_logical_cores =
        mux_on_bottom_row ? std::vector<CoreCoord>{{0, fabric_mux_row}, {sdpa_grid.x - 1, fabric_mux_row}}
        : mux_top_cluster ? std::vector<CoreCoord>{{fabric_mux_col, 0}, {fabric_mux_col, 1}}
                          : std::vector<CoreCoord>{{fabric_mux_col, 0}, {fabric_mux_col, sdpa_grid.y - 1}};
    const std::vector<CoreCoord> mux_forward_logical_cores =
        mux_on_bottom_row ? std::vector<CoreCoord>{{mid - 1, fabric_mux_row}, {mid, fabric_mux_row}}
        : mux_top_cluster ? std::vector<CoreCoord>{{fabric_mux_col, 2}, {fabric_mux_col, 3}}
                          : std::vector<CoreCoord>{{fabric_mux_col, mid - 1}, {fabric_mux_col, mid}};

    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t num_mux_full_size_channels = num_mux_clients_per_group;
    const uint32_t num_mux_header_only_channels = 0;
    // The caller's num_buffers_per_channel is calibrated for one channel per row; with follower
    // rows disconnected, redistribute the same L1 across the remaining channels.
    const uint32_t num_mux_buffers_per_channel =
        args.num_buffers_per_channel * num_workers_per_link / num_mux_clients_per_group;
    const size_t mux_buffer_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_mux_full_size_channels,
        num_mux_header_only_channels,
        num_mux_buffers_per_channel,
        0,
        mux_buffer_size_bytes,
        l1_unreserved_base_address);

    // Convert std::map<string,string> defines to KernelDescriptor::Defines vector form.
    KernelDescriptor::Defines kernel_defines(defines.begin(), defines.end());

    // Build kernel descriptors locally so we can append per-core runtime args
    // before pushing them into desc.kernels at the end. KernelDescriptor creation
    // is deferred (just like the original CreateKernel calls were) until after chain
    // construction, since the mcast_enabled compile-time arg is patched above.
    KernelDescriptor reader_kernel{};
    reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_reader.cpp";
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = CoreRangeSet(sdpa_grid_range);
    reader_kernel.compile_time_args = reader_compile_time_args;
    reader_kernel.defines = kernel_defines;
    reader_kernel.config = ReaderConfigDescriptor{};

    // Non-fabric writer: columns 0..(sdpa_grid.x-3)
    // sdpa_grid.x-2 and sdpa_grid.x-1 are fabric MUX client columns
    CoreRange sdpa_writer_range({0, 0}, {sdpa_grid.x - 3, sdpa_grid.y - 1});
    KernelDescriptor writer_kernel{};
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_writer.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = CoreRangeSet(sdpa_writer_range);
    writer_kernel.compile_time_args = writer_compile_time_args;
    writer_kernel.defines = kernel_defines;
    writer_kernel.config = WriterConfigDescriptor{};

    // Fabric writer: columns sdpa_grid.x-2 and sdpa_grid.x-1 (backward and forward MUX clients)
    CoreRange mux_writer_range({sdpa_grid.x - 2, 0}, {sdpa_grid.x - 1, sdpa_grid.y - 1});
    auto writer_fabric_compile_time_args = writer_compile_time_args;
    fabric_mux_connection_ct_args(num_mux_clients_per_group, mux_kernel_config, writer_fabric_compile_time_args);

    // All-gather CT args for the fabric writer (integrated K/V all-gather on MUX client columns)
    const uint32_t ag_page_size = input_tensor_k.buffer()->page_size();
    const size_t ag_packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    constexpr uint32_t max_scatter_addresses = 4;
    const uint32_t ag_packet_size_in_pages =
        std::min(static_cast<uint32_t>(ag_packet_size_bytes / ag_page_size), max_scatter_addresses);

    writer_fabric_compile_time_args.push_back(ag_packet_size_in_pages);
    writer_fabric_compile_time_args.push_back(ag_page_size);
    TensorAccessorArgs(gathered_input_tensor_k.buffer()).append_to(writer_fabric_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_v.buffer()).append_to(writer_fabric_compile_time_args);

    auto writer_fabric_defines = defines;
    writer_fabric_defines["USE_MUX"] = "1";
    KernelDescriptor::Defines writer_fabric_kernel_defines(writer_fabric_defines.begin(), writer_fabric_defines.end());
    KernelDescriptor writer_fabric_kernel{};
    writer_fabric_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_writer.cpp";
    writer_fabric_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_fabric_kernel.core_ranges = CoreRangeSet(mux_writer_range);
    writer_fabric_kernel.compile_time_args = writer_fabric_compile_time_args;
    writer_fabric_kernel.defines = writer_fabric_kernel_defines;
    writer_fabric_kernel.config = WriterConfigDescriptor{};

    KernelDescriptor compute_kernel{};
    compute_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/exp_ring_joint_sdpa.cpp";
    compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = CoreRangeSet(sdpa_grid_range);
    compute_kernel.compile_time_args = compute_compile_time_args;
    compute_kernel.defines = kernel_defines;
    compute_kernel.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    // Live-length tensor address for all three kernels -- each derives chunk-skip counts from it and
    // the credit/gate protocol requires they agree. Buffer* (not ->address()) so a cache hit re-patches it.
    if (has_logical_n_tensor) {
        auto* logical_n_buffer = tensor_args.logical_n_tensor->buffer();
        const auto logical_n_common_args = [&]() {
            KernelDescriptor::RTArgList args_list;
            args_list.push_back(logical_n_buffer);
            return args_list;
        };
        reader_kernel.emplace_common_runtime_args(logical_n_common_args());
        writer_kernel.emplace_common_runtime_args(logical_n_common_args());
        writer_fabric_kernel.emplace_common_runtime_args(logical_n_common_args());
    }

    // Build backward and forward termination master core sets (1 per link per direction)
    // Backward masters: row 0 of both MUX client columns (top half = backward direction).
    // Forward masters:  row num_workers_per_link of both MUX client columns (bottom half = forward).
    // Link is determined by column: col sdpa_grid.x-2 = link 0, col sdpa_grid.x-1 = link 1.
    std::vector<CoreCoord> ag_backward_master_cores, ag_forward_master_cores;
    std::set<CoreRange> ag_backward_master_ranges, ag_forward_master_ranges;
    for (uint32_t col_offset = 0; col_offset < 2; ++col_offset) {
        CoreCoord bwd_master = {sdpa_grid.x - 2 + col_offset, 0};
        CoreCoord fwd_master = {sdpa_grid.x - 2 + col_offset, num_workers_per_link};
        ag_backward_master_cores.push_back(bwd_master);
        ag_forward_master_cores.push_back(fwd_master);
        ag_backward_master_ranges.insert(CoreRange(bwd_master));
        ag_forward_master_ranges.insert(CoreRange(fwd_master));
    }

    // Pass the full direction-half range across both MUX client columns so that
    // any AG sync semaphores would be allocated on ALL workers in each direction group.
    // This ensures every core (both term-masters and non-masters) has the same number of
    // semaphores allocated before fabric_mux_connection_rt_args runs, keeping
    // termination_sync IDs consistent.
    CoreRange all_backward_clients({sdpa_grid.x - 2, 0}, {sdpa_grid.x - 1, num_workers_per_link - 1});
    CoreRange all_forward_clients({sdpa_grid.x - 2, num_workers_per_link}, {sdpa_grid.x - 1, sdpa_grid.y - 1});
    // K/V tensor shape info for all-gather RT args
    const auto& ag_input_shape = input_tensor_k.padded_shape();
    const auto& ag_output_shape = gathered_input_tensor_k.padded_shape();
    TT_ASSERT(!(ag_input_shape[3] % tt::constants::TILE_WIDTH));
    TT_ASSERT(!(ag_output_shape[3] % tt::constants::TILE_WIDTH));
    const uint32_t ag_output_Wt = ag_output_shape[3] / tt::constants::TILE_WIDTH;
    const uint32_t ag_output_Ht = ag_output_shape[2] / tt::constants::TILE_HEIGHT;

    // Set reader rt args
    for (uint32_t i = 0; i < num_sdpa_cores; ++i) {
        CoreCoord core = {i % sdpa_grid.x, i / sdpa_grid.x};

        // Row-aligned head-serial work descriptor: chunks q_base + p*q_stride, p in [0, q_count).
        const auto& work = core_work.at(i);

        // Direction: top half of rows = backward (0), bottom half = forward (1)
        const uint32_t direction = (core.y < num_workers_per_link) ? 0 : 1;

        // log the above
        log_debug(tt::LogOp, "core: {}", i);
        log_debug(tt::LogOp, "x={},y={}", core.x, core.y);
        log_debug(tt::LogOp, "q_base={}, q_stride={}, q_count={}", work.q_base, work.q_stride, work.q_count);

        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(q_buf);
        reader_args.push_back(k_buf);
        reader_args.push_back(v_buf);
        reader_args.push_back(gathered_k_buf);
        reader_args.push_back(gathered_v_buf);
        reader_args.push_back(joint_q_buf);
        reader_args.push_back(joint_k_buf);
        reader_args.push_back(joint_v_buf);
        reader_args.push_back(work.q_base);
        reader_args.push_back(work.q_stride);
        reader_args.push_back(work.q_count);
        // Append chain runtime args for store-and-forward
        const auto& chain = core_chain_info.at(i);

        log_debug(
            tt::LogOp,
            "core logical=({},{})->phys=({},{}), q_base={} stride={} count={}, chain={{part:{}, inj:{}, sink:{}, "
            "b:{}, h:{}, q_start:{}, q_cnt:{}, next_cnt:{}}}",
            core.x,
            core.y,
            core_work.at(i).physical_core.x,
            core_work.at(i).physical_core.y,
            work.q_base,
            work.q_stride,
            work.q_count,
            chain.participates,
            chain.is_injector,
            chain.is_sink,
            chain.batch,
            chain.head,
            chain.q_chunk_start,
            chain.q_chunk_count,
            chain.next_core_q_chunks);

        reader_args.push_back(static_cast<uint32_t>(chain.participates));
        reader_args.push_back(static_cast<uint32_t>(chain.is_injector));
        reader_args.push_back(static_cast<uint32_t>(chain.is_sink));
        reader_args.push_back(chain.batch);
        reader_args.push_back(chain.head);
        reader_args.push_back(chain.q_chunk_start);
        reader_args.push_back(chain.q_chunk_count);
        reader_args.push_back(static_cast<uint32_t>(chain.prev_physical.x));
        reader_args.push_back(static_cast<uint32_t>(chain.prev_physical.y));
        reader_args.push_back(static_cast<uint32_t>(chain.next_physical.x));
        reader_args.push_back(static_cast<uint32_t>(chain.next_physical.y));
        reader_args.push_back(chain.next_core_q_chunks);
        reader_args.push_back(chain.mcast_num_dests);
        reader_args.push_back(chain.mcast_sender_wait);

        // Determine if this core's writer has a valid MUX connection (for reader-side forwarding).
        // Split-head dedup follower rows forward nothing, so their clients do not connect and
        // their readers do not feed c_14/c_15 at all.
        const bool row_forwards = row_dedup_role.at(core.y) != 2;
        const bool is_mux_writer = (core.x >= sdpa_grid.x - 2);
        bool is_mux_writer_valid = false;
        if (is_mux_writer && row_forwards) {
            const uint32_t half_within_col = core.y / num_workers_per_link;
            const bool is_backward = (half_within_col == 0);
            const uint32_t link = (core.x == sdpa_grid.x - 1) ? 1 : 0;
            const bool link_in_range = (link < args.num_links) && (link < mux_backward_logical_cores.size()) &&
                                       (link < mux_forward_logical_cores.size());
            if (link_in_range) {
                const bool valid = is_backward ? backward_coord.has_value() : forward_coord.has_value();
                is_mux_writer_valid = valid;
            }
        }
        reader_args.push_back(static_cast<uint32_t>(is_mux_writer_valid));

        // Per-link semaphore addresses for chunk-level sync. These occupy per-core reader slots
        // exp_ring_joint_sdpa_dynamic::kReaderSemaphoreArgBase .. +num_links-1. They are hash-excluded, so
        // they are baked here for the cache-miss build and re-applied every dispatch by
        // ExpRingJointSDPAMeshWorkloadFactory::override_runtime_arguments(), which asserts the total
        // per-core count: adding or removing an arg fails loudly, a count-preserving reorder does not.
        reader_args.push_back(args.num_links);
        for (uint32_t lnk = 0; lnk < args.num_links; ++lnk) {
            reader_args.push_back(static_cast<uint32_t>(
                args.semaphore[lnk]
                    .address()));  // smuggled-rta-ok: hash-excluded global-semaphore address, re-applied every dispatch
                                   // via ExpRingJointSDPAMeshWorkloadFactory::override_runtime_arguments
        }

        // Inject fused-op synchronization RT args: ring_size, ring_index, direction (3 values)
        reader_args.push_back(static_cast<uint32_t>(args.ring_size));
        reader_args.push_back(device_index);
        reader_args.push_back(direction);

        // Split-head forwarding dedup descriptor (meaningful on injector cores only).
        reader_args.push_back(row_dedup_role.at(core.y));
        reader_args.push_back(static_cast<uint32_t>(row_buddy_injector.at(core.y).x));
        reader_args.push_back(static_cast<uint32_t>(row_buddy_injector.at(core.y).y));

        reader_kernel.emplace_runtime_args(core, reader_args);

        // Writer args
        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(out_buf);
        // Zero-seq when no joint; address unused at runtime (L=0 => no joint writes).
        writer_args.push_back(joint_out_buf);
        writer_args.push_back(stats_buf);
        writer_args.push_back(work.q_base);
        writer_args.push_back(work.q_stride);
        writer_args.push_back(work.q_count);
        writer_args.push_back(static_cast<uint32_t>(args.ring_size));
        writer_args.push_back(device_index);
        writer_args.push_back(direction);

        if (is_mux_writer) {
            // Direction is determined by row half: top half = backward, bottom half = forward.
            // Link is determined by column: col sdpa_grid.x-2 = link 0, col sdpa_grid.x-1 = link 1.
            const uint32_t half_within_col = core.y / num_workers_per_link;
            const bool is_backward = (half_within_col == 0);
            const uint32_t link = (core.x == sdpa_grid.x - 1) ? 1 : 0;
            // Compact channel id among the direction half's FORWARDING rows (followers do not
            // connect and hold no channel). Term master = channel 0 of the group; with dedup off
            // this is the original worker_idx layout (row 0 of each half).
            const uint32_t worker_idx = row_mux_channel.at(core.y);
            const bool is_term_master = row_forwards && (worker_idx == 0);
            // First forwarding row of this direction half hosts the termination master.
            uint32_t term_master_row = half_within_col * num_workers_per_link;
            while (term_master_row + 1 < (half_within_col + 1) * num_workers_per_link &&
                   row_dedup_role.at(term_master_row) == 2) {
                term_master_row++;
            }
            const CoreCoord termination_master_logical = {core.x, term_master_row};

            const bool link_in_range = (link < args.num_links) && (link < mux_backward_logical_cores.size()) &&
                                       (link < mux_forward_logical_cores.size());
            // fabric_mux_connection_rt_args appends to a std::vector<uint32_t>; collect mux args
            // separately and then merge into the RTArgList so BufferBinding entries above are preserved.
            std::vector<uint32_t> mux_writer_args;
            mux_writer_args.reserve(kFabricMuxConnectionRtArgCount);
            if (link_in_range) {
                const CoreCoord& mux_core =
                    is_backward ? mux_backward_logical_cores[link] : mux_forward_logical_cores[link];
                const bool valid =
                    row_forwards && (is_backward ? backward_coord.has_value() : forward_coord.has_value());
                fabric_mux_connection_rt_args(
                    valid,
                    is_term_master,
                    mux_core,
                    worker_idx,
                    core,
                    mux_kernel_config,
                    desc,
                    termination_master_logical,
                    device,
                    mux_writer_args);
            } else {
                // link index out of range or invalid direction — append a disconnected MUX connection
                // Still need valid semaphore IDs for the 5 semaphore fields
                mux_writer_args.push_back(0);                                        // mux_connection_valid = false
                mux_writer_args.push_back(0);                                        // is_termination_master
                mux_writer_args.push_back(0);                                        // mux_x
                mux_writer_args.push_back(0);                                        // mux_y
                mux_writer_args.push_back(0);                                        // channel_base_address
                mux_writer_args.push_back(0);                                        // connection_info_address
                mux_writer_args.push_back(0);                                        // connection_handshake_address
                mux_writer_args.push_back(0);                                        // flow_control_address
                mux_writer_args.push_back(0);                                        // buffer_index_address
                mux_writer_args.push_back(0);                                        // channel_credits_stream_id
                mux_writer_args.push_back(allocate_per_core_semaphore(desc, core));  // termination_sync
                mux_writer_args.push_back(allocate_per_core_semaphore(desc, core));  // local_fabric_mux_status
                mux_writer_args.push_back(allocate_per_core_semaphore(desc, core));  // local_flow_control
                mux_writer_args.push_back(allocate_per_core_semaphore(desc, core));  // local_teardown
                mux_writer_args.push_back(allocate_per_core_semaphore(desc, core));  // local_buffer_index
                mux_writer_args.push_back(0);                                        // termination_master_noc_x
                mux_writer_args.push_back(0);                                        // termination_master_noc_y
            }
            writer_args.append(mux_writer_args);

            // MUX writer RT args: out_ready_sem, injector coords, AG params, op signaler.
            if (link_in_range) {
                // out_ready_sem_addr occupies per-core fabric-writer slot
                // exp_ring_joint_sdpa_dynamic::kWriterFabricOutReadySemArg. It is a hash-excluded
                // global-semaphore address, so it is baked here for the cache-miss build and re-applied
                // every dispatch by ExpRingJointSDPAMeshWorkloadFactory::override_runtime_arguments(),
                // which asserts the per-core count: an added/removed arg fails loudly, a reorder does not.
                const uint32_t out_ready_sem_addr = args.semaphore[link].address();
                writer_args.push_back(
                    out_ready_sem_addr);  // smuggled-rta-ok: hash-excluded global-semaphore address, re-applied every
                                          // dispatch via
                                          // ExpRingJointSDPAMeshWorkloadFactory::override_runtime_arguments

                // The injector this MUX writer signals is its own row's injector: all passes of a
                // row share one injector, and this atomic-inc destination is baked once for the
                // whole op. A participating row without an injector is a host bug, not a runtime
                // condition, so fail loudly instead of signaling core (0,0).
                const auto& row_injector = injector_physical_by_row.at(core.y);
                TT_FATAL(
                    row_injector.has_value(),
                    "MUX writer core ({},{}) has no injector for row {}: chain construction is inconsistent.",
                    core.x,
                    core.y,
                    core.y);
                const CoreCoord injector_physical = row_injector.value();
                writer_args.push_back(static_cast<uint32_t>(injector_physical.x));
                writer_args.push_back(static_cast<uint32_t>(injector_physical.y));
                writer_args.push_back(args.num_links);  // num_muxes_in_direction
                writer_args.push_back(link);            // my_mux_index
                writer_args.push_back(ag_output_Wt);
                writer_args.push_back(ag_output_Ht);
                writer_args.push_back(gathered_k_buf);
                writer_args.push_back(gathered_v_buf);
                // Split-head forwarding dedup: follower rows' clients skip fabric data + semincs
                // (the leader row of the pair sends byte-identical packets).
                writer_args.push_back(row_dedup_role.at(core.y) == 2 ? 1u : 0u);
            }
            writer_fabric_kernel.emplace_runtime_args(core, writer_args);
        } else {
            writer_kernel.emplace_runtime_args(core, writer_args);
        }

        // Compute args
        KernelDescriptor::RTArgList compute_args;
        compute_args.push_back(work.q_base);
        compute_args.push_back(work.q_stride);
        compute_args.push_back(work.q_count);
        compute_args.push_back(static_cast<uint32_t>(args.ring_size));
        compute_args.push_back(device_index);
        compute_args.push_back(direction);
        compute_kernel.emplace_runtime_args(core, compute_args);
    }

    // Push the SDPA kernels into desc before building the fabric MUX kernel so its
    // index is deterministic (kernels 0/1/2/3 = reader/writer/writer_fabric/compute).
    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    desc.kernels.push_back(std::move(writer_fabric_kernel));
    desc.kernels.push_back(std::move(compute_kernel));

    // ---- Fabric MUX cores ----
    std::vector<CoreRange> mux_core_ranges;
    mux_core_ranges.reserve(2 * args.num_links);
    for (uint32_t link = 0; link < args.num_links; ++link) {
        if (backward_coord.has_value()) {
            mux_core_ranges.emplace_back(mux_backward_logical_cores[link]);
        }
        if (forward_coord.has_value()) {
            mux_core_ranges.emplace_back(mux_forward_logical_cores[link]);
        }
    }
    CoreRangeSet mux_core_range_set(mux_core_ranges);

    if (!mux_core_ranges.empty()) {
        KernelDescriptor mux_kernel{};
        mux_kernel.kernel_source = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
        mux_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        mux_kernel.core_ranges = mux_core_range_set;
        mux_kernel.compile_time_args = mux_kernel_config.get_fabric_mux_compile_time_args();
        mux_kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
        mux_kernel.config = DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
        };

        const auto src_node_id = mesh_device->get_fabric_node_id(coord);
        for (uint32_t link = 0; link < args.num_links; ++link) {
            if (backward_coord.has_value()) {
                const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                auto mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, desc, mux_backward_logical_cores[link]);
                KernelDescriptor::RTArgList mux_args;
                mux_args.append(mux_rt_args);
                mux_kernel.emplace_runtime_args(mux_backward_logical_cores[link], mux_args);
            }
            if (forward_coord.has_value()) {
                const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                auto mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, desc, mux_forward_logical_cores[link]);
                KernelDescriptor::RTArgList mux_args;
                mux_args.append(mux_rt_args);
                mux_kernel.emplace_runtime_args(mux_forward_logical_cores[link], mux_args);
            }
        }

        desc.kernels.push_back(std::move(mux_kernel));
    }

    return desc;
}

}  // namespace

// Exp ring-joint SDPA returns a WorkloadDescriptor with one ProgramDescriptor per coord:
// device_index / forward_coord / backward_coord / DEST_CHIP_ID-style fabric routing all
// depend on the mesh coordinate, so descriptors cannot be shared across coords.
tt::tt_metal::WorkloadDescriptor ExpRingJointSDPAProgramFactory::create_workload_descriptor(
    const ExpRingJointSDPAParams& operation_attributes,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc =
            build_exp_ring_joint_sdpa_program_descriptor(operation_attributes, tensor_args, tensor_return_value, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

ExpRingJointSDPAMeshWorkloadFactory::cached_mesh_workload_t ExpRingJointSDPAMeshWorkloadFactory::create_mesh_workload(
    const ExpRingJointSDPAParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& tensor_return_value) {
    return descriptor_adapter_t::create_mesh_workload(
        operation_attributes, tensor_coords, tensor_args, tensor_return_value);
}

void ExpRingJointSDPAMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ExpRingJointSDPAParams& operation_attributes,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& tensor_return_value) {
    // apply_descriptor re-points the Buffer* runtime args; the hash-excluded per-link GlobalSemaphore
    // addresses are all that is left to patch.
    descriptor_adapter_t::apply_descriptor(cached_workload, operation_attributes, tensor_args, tensor_return_value);

    namespace dyn = exp_ring_joint_sdpa_dynamic;
    const auto& args = operation_attributes;

    // Recompute the SDPA worker grid exactly as build_exp_ring_joint_sdpa_program_descriptor() does.
    auto* mesh_device = tensor_args.input_q.device();
    const CoreCoord user_grid = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                                : mesh_device->compute_with_storage_grid_size();
    const CoreCoord sdpa_grid = exp_sdpa_mux_on_bottom_row() ? CoreCoord{user_grid.x, user_grid.y - 2}
                                                             : CoreCoord{user_grid.x - 1, user_grid.y};
    const uint32_t num_sdpa_cores = sdpa_grid.x * sdpa_grid.y;
    const uint32_t expected_reader_args = dyn::reader_arg_count(args.num_links);

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        // Hoisted out of the per-core loop, and by reference: a copy would clone the whole arg grid.
        auto& reader_grid = GetRuntimeArgs(program, dyn::kReaderKernelIdx);
        auto& writer_fabric_grid = GetRuntimeArgs(program, dyn::kWriterFabricKernelIdx);

        for (uint32_t i = 0; i < num_sdpa_cores; ++i) {
            const CoreCoord core = {i % sdpa_grid.x, i / sdpa_grid.x};

            auto& reader_args = reader_grid[core.x][core.y];
            TT_FATAL(
                reader_args.size() == expected_reader_args,
                "Exp ring joint SDPA reader expected {} runtime args on core ({},{}), cached program has {}",
                expected_reader_args,
                core.x,
                core.y,
                reader_args.size());
            for (uint32_t lnk = 0; lnk < args.num_links; ++lnk) {
                reader_args[dyn::kReaderSemaphoreArgBase + lnk] = static_cast<uint32_t>(args.semaphore[lnk].address());
            }

            // out_ready_sem_addr lives only on the two MUX-writer columns. num_links is TT_FATAL-fixed
            // to 2, so link_in_range always holds in the factory and the slot is always present.
            if (core.x >= sdpa_grid.x - 2) {
                const uint32_t link = (core.x == sdpa_grid.x - 1) ? 1u : 0u;
                auto& writer_args = writer_fabric_grid[core.x][core.y];
                TT_FATAL(
                    writer_args.size() == dyn::kWriterFabricArgCount,
                    "Exp ring joint SDPA fabric writer expected {} runtime args on core ({},{}), cached program has {}",
                    dyn::kWriterFabricArgCount,
                    core.x,
                    core.y,
                    writer_args.size());
                writer_args[dyn::kWriterFabricOutReadySemArg] = static_cast<uint32_t>(args.semaphore[link].address());
            }
        }
    }
}

}  // namespace ttnn::prim
