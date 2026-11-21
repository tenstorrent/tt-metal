// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/recipe_ring_cbs.hpp"
#include "ttnn/operations/transformer/sdpa/device/recipe/sdpa_recipe_program.hpp"

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
    // A mux in program-local L1 publishes its map's address in a semaphore, which its clients wait on in place of the
    // status word; every other mux address is then an offset from that address.
    worker_ct_args.push_back(
        mux_kernel_config.is_in_program_local_l1() ? mux_kernel_config.get_l1_base_semaphore_id()
                                                   : mux_kernel_config.get_status_address());
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
    const MeshDevice& device,
    std::vector<uint32_t>& worker_rt_args) {
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    const CoreCoord mux_virtual_core = device.worker_core_from_logical_core(mux_logical_core);
    const CoreCoord termination_master_virtual_core =
        device.worker_core_from_logical_core(termination_master_logical_core);

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

    const uint32_t Sq_chunk_t = q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / tt::constants::TILE_HEIGHT;

    // Trace-safe logical_n: args.logical_n is the worst-case placeholder (padded_N) and the kernels read the live value
    // from the tensor. Key tails are masked per chunk in the pack thread from the live length, so no CB geometry
    // depends on it.
    const bool has_logical_n_tensor = tensor_args.has_logical_n_tensor();

    const uint32_t num_local_q_chunks = tt::div_up(local_padded_N, q_chunk_size);
    const uint32_t num_joint_q_chunks = tt::div_up(L, q_chunk_size);
    const uint32_t num_q_chunks = num_local_q_chunks + num_joint_q_chunks;
    const uint32_t num_local_k_chunks = tt::div_up(local_padded_N, k_chunk_size);
    const uint32_t num_joint_k_chunks = tt::div_up(L, k_chunk_size);

    log_debug(
        tt::LogOp,
        "ExpRingJointSDPA: B={} NH={} DH={} local_padded_N={} padded_N={} L={} Q/K chunk tiles {}/{} Q chunks {}+{} K "
        "chunks {}+{}",
        B,
        NH,
        DH,
        local_padded_N,
        padded_N,
        L,
        Sq_chunk_t,
        Sk_chunk_t,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks);

    MeshDevice* device = input_tensor_q.device();

    // The precision recipe owns the numerics (device/recipe/sdpa_precision_policy.hpp).
    namespace recipes = ttnn::operations::transformer::sdpa::detail;
    const recipes::PrecisionPolicy policy = recipes::resolve_recipe_policy(
        input_tensor_q, gathered_input_tensor_k, args.precision, scale, args.program_config);

    // Grid layout:
    //   user_grid:        Full grid from program_config (or device default). Contains SDPA workers + fabric MUX.
    //   sdpa_grid:        user_grid[:,:-1] — all columns except the last. SDPA workers (some are MUX writers).
    //   fabric_mux_col:   user_grid.x - 1 — last column reserved for fabric MUX kernel.
    //   sdpa_writer_range:  sdpa_grid columns 0..(sdpa_grid.x-3) — non-MUX SDPA writer cores.
    //   mux_writer_range:   sdpa_grid columns (sdpa_grid.x-2)..(sdpa_grid.x-1) — MUX fabric writer cores (2 links).
    const auto device_grid = mesh_device->compute_with_storage_grid_size();
    CoreCoord user_grid =
        args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size : device_grid;
    CoreCoord sdpa_grid = exp_ring_sdpa_grid(user_grid);

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
    // The row-half split determines the per-link worker count.
    const uint32_t num_workers_per_link = sdpa_grid.y / 2;
    TT_FATAL(
        num_workers_per_link == args.num_workers_per_link,
        "num_workers_per_link ({}) must equal sdpa_grid.y / 2 ({}).",
        args.num_workers_per_link,
        num_workers_per_link);

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
     * The kernels walk the passes pass-outer, ring-inner: each pass is exactly the single-head
     * pipeline, with one resident Q chunk and one recurrent state, so a core's L1 does not grow with
     * its pass count. A split head's K/V shard is streamed — and all-gather forwarded — once per
     * segment: the duplicate fabric writes carry identical bytes to identical addresses, and each
     * row's chunk-ready signals come from the same row on the previous device, so every row's
     * pipeline stays self-contained exactly as in the one-row-per-head layout.
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
    log_debug(tt::LogOp, "num_passes (heads per row): {}", num_passes);

    // The reader pushes Q in the row groups compute starts its QK on: one row for the FP32 recipes and for a paired
    // recipe's odd chunk (it ends with a one-row group), else two.
    const uint32_t q_push_rows = policy.fp32_destination || Sq_chunk_t % 2 != 0 ? 1 : 2;
    // The writer drains the output in the rows compute packs: pairs of rows when the Q chunk is a whole number of them,
    // so a pair never straddles the output CB's wrap.
    const uint32_t writer_out_row_group_h = Sq_chunk_t % 2 == 0 ? 2 : 1;

    const float scale_value = scale.value_or(1.0f);
    const uint32_t scale_packed = std::bit_cast<uint32_t>(scale_value);
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
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        args.ring_size,
        q_push_rows};

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
    reader_compile_time_args.push_back(receiver_semaphore_b_id);
    reader_compile_time_args.push_back(sender_semaphore_v_id);
    reader_compile_time_args.push_back(buddy_gate_semaphore_id);
    // Trace-safe logical_n: presence flag, then the accessor args for the 1-element tensor. The accessor
    // block is emitted unconditionally (nullptr when absent) so the CT-arg layout after it never shifts.
    reader_compile_time_args.push_back(static_cast<uint32_t>(has_logical_n_tensor));
    TensorAccessorArgs(has_logical_n_tensor ? tensor_args.logical_n_tensor->buffer() : nullptr)
        .append_to(reader_compile_time_args);

    // Both writer kernels (plain and MUX client) take these compile-time args; the MUX and all-gather args follow
    // once the MUX config is known.
    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NH,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        local_padded_Nt,
        static_cast<uint32_t>(args.logical_n),
        logical_nt,
        Lt,
        L,
        num_local_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        args.ring_size,
        writer_out_row_group_h,
        static_cast<std::uint32_t>(has_logical_n_tensor),
    };

    // Trace-safe logical_n accessor block sits ahead of the output accessors so every downstream offset
    // (out / joint_out, and the MUX + AG block that chains off joint_out) keeps deriving normally.
    // Emitted unconditionally (nullptr when absent) to keep the layout stable across both modes.
    TensorAccessorArgs(has_logical_n_tensor ? tensor_args.logical_n_tensor->buffer() : nullptr)
        .append_to(writer_compile_time_args);
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(joint_output_tensor.buffer()).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        local_padded_N,
        local_padded_Nt,
        static_cast<uint32_t>(args.logical_n),
        logical_nt,
        L,
        num_local_k_chunks,
        num_joint_k_chunks,
        args.ring_size,
        scale_packed,
        // Trace-safe logical_n: compute reads the live values from the reader's derived CB.
        static_cast<uint32_t>(has_logical_n_tensor),
    };

    // NOTE: KernelDescriptor construction is deferred until after chain construction
    // so that the mcast_enabled compile-time arg can be determined first.

    // Circular buffers: the recipe compute layout (recipe_compute_program, CBs 0-16 and 29-31;
    // kernels/recipe_ring_cbs.hpp) with the MUX writer's second handles on K and V, then the reader's derived-length
    // mailbox. Every CB spans the SDPA grid.
    const tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    const tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_k.dtype());
    const tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_v.dtype());
    const uint32_t q_tile_size = tt::tile_size(q_df);
    const uint32_t k_tile_size = tt::tile_size(k_df);
    const uint32_t v_tile_size = tt::tile_size(v_df);
    const auto sdpa_grid_set = CoreRangeSet(sdpa_grid_range);
    const auto v_storage = recipes::recipe_kv_storage(gathered_input_tensor_v.dtype());
    tt::tt_metal::ProgramDescriptor recipe_program = recipes::recipe_compute_program(
        policy,
        sdpa_grid_set,
        1,
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        scale,
        0,
        v_storage != policy.selection.kv_storage ? std::optional<recipes::KVStorage>(v_storage) : std::nullopt,
        q_df);
    auto& compute_recipe_kernel = recipe_program.kernels.front();
    desc.cbs = std::move(recipe_program.cbs);
    for (auto& cb : desc.cbs) {
        auto& format = cb.format_descriptors.front();
        if (format.buffer_index == recipe_ring_cb::q) {
            // One Q chunk: a pass's chunk stays resident across its ring iterations and compute pops it before the
            // reader fetches the next pass's.
            cb.total_size = Sq_chunk_t * DHt * q_tile_size;
        } else if (format.buffer_index == recipe_ring_cb::k || format.buffer_index == recipe_ring_cb::v) {
            // K and V double-buffered (L1 fallback below: one entry each), with a second handle each so the MUX writer
            // pops independently of compute.
            const bool is_k = format.buffer_index == recipe_ring_cb::k;
            cb.total_size = Sk_chunk_t * DHt * 2 * (is_k ? k_tile_size : v_tile_size);
            auto alias = format;
            alias.buffer_index = static_cast<uint8_t>(is_k ? recipe_ring_cb::k_writer : recipe_ring_cb::v_writer);
            cb.format_descriptors.push_back(alias);
        } else if (format.buffer_index == recipe_ring_cb::out) {
            // The output tensor is BF16 whatever Q's type.
            const uint32_t tiles = cb.total_size / format.page_size;
            format.data_format = tt::DataFormat::Float16_b;
            format.page_size = tt::tile_size(tt::DataFormat::Float16_b);
            cb.total_size = tiles * format.page_size;
        }
    }
    if (has_logical_n_tensor) {
        // Compute RISCs cannot NoC-read DRAM, so the reader publishes the live length here (slots per
        // ring_joint_derived_slots.hpp) and compute reads it with read_tile_value. UInt32: read_tile_value's indexing
        // follows the CB format.
        constexpr uint32_t kDerivedPageBytes = 64;
        desc.cbs.push_back(CBDescriptor{
            .total_size = kDerivedPageBytes,
            .core_ranges = sdpa_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(recipe_ring_cb::derived),
                .data_format = tt::DataFormat::UInt32,
                .page_size = kDerivedPageBytes,
            }}},
        });
    }

    uint32_t kv_cb_entries = 2;
    // Fit L1 below the lowest live L1 buffer (global semaphores and persistent buffers occupy the top of L1 in a
    // pipeline): first stage the output one row group at a time, then keep one K and one V entry (the reader then
    // fetches a chunk only once compute has released the previous one; the FP32 recipes' QK and state CBs are twice
    // the BF16 ones), then give up the fused chunks' CBs (every K chunk then runs on the reduce path, slower than
    // either).
    const uint64_t l1_available = recipes::recipe_free_l1(*device);
    uint64_t cb_bytes = recipes::recipe_cb_bytes(desc);
    for (const uint32_t out_rows : {2u, 1u}) {
        if (cb_bytes <= l1_available || out_rows < std::min(Sq_chunk_t, 2u)) {
            break;
        }
        for (auto& cb : desc.cbs) {
            if (cb.format_descriptors.front().buffer_index == recipe_ring_cb::out) {
                cb.total_size =
                    std::min<uint32_t>(cb.total_size, out_rows * DHt * cb.format_descriptors.front().page_size);
            }
        }
        cb_bytes = recipes::recipe_cb_bytes(desc);
        log_debug(tt::LogOp, "ExpRingJointSDPA recipe: output staged in {} row(s) to fit L1", out_rows);
    }
    if (cb_bytes > l1_available) {
        for (auto& cb : desc.cbs) {
            const auto index = cb.format_descriptors.front().buffer_index;
            if (index == recipe_ring_cb::k || index == recipe_ring_cb::v) {
                cb.total_size /= 2;
            }
        }
        kv_cb_entries = 1;
        cb_bytes = recipes::recipe_cb_bytes(desc);
        log_debug(tt::LogOp, "ExpRingJointSDPA recipe: single K and V entries to fit L1");
    }
    if (cb_bytes > l1_available) {
        cb_bytes -= recipes::recipe_drop_fused(desc.cbs, compute_recipe_kernel.named_compile_time_args, Sq_chunk_t);
        log_debug(tt::LogOp, "ExpRingJointSDPA recipe: unfused chunks to fit L1");
    }
    TT_FATAL(
        cb_bytes <= l1_available,
        "Exp ring joint SDPA recipe needs {} B of L1 per core at Q{}/K{}, but only {} B are available; use a smaller Q "
        "or K chunk",
        cb_bytes,
        q_chunk_size,
        k_chunk_size,
        l1_available);
    const bool recipe_fused = recipes::named_arg(compute_recipe_kernel.named_compile_time_args, "recipe_fused") != 0;

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

    /**
     * Build per-row store-and-forward chains.
     *
     * Head-serial passes place num_passes heads on every row, and every pass of a row runs on the
     * same cores (logical x in [0, num_q_chunks)), so a row has ONE chain shared by all its passes.
     * One injector per row is required, not merely convenient: each MUX writer pre-configures a
     * single fabric atomic-inc destination for the whole op (see exp_ring_joint_writer.cpp), so
     * every pass of a row must signal the same injector core.
     */
    struct CoreWork {
        CoreCoord logical_core;
        CoreCoord physical_core;
        // Q work descriptor: this core owns flat chunks q_base + p * q_stride for p in [0, q_count).
        uint32_t q_base = 0;
        uint32_t q_stride = 0;
        uint32_t q_count = 0;
    };

    struct CoreChainInfo {
        bool participates = false;
        bool is_injector = false;
        bool is_sink = false;
        CoreCoord prev_physical = CoreCoord{0, 0};
        CoreCoord next_physical = CoreCoord{0, 0};
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
            work.q_count++;
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
            auto& chain = core_chain_info.at(core_idx);

            chain.participates = true;
            chain.is_injector = (idx == 0);
            chain.is_sink = (idx + 1 == chain_cores.size());

            if (idx > 0) {
                chain.prev_physical = core_work.at(chain_cores[idx - 1]).physical_core;
            }
            if (idx + 1 < chain_cores.size()) {
                const uint32_t next_core_idx = chain_cores[idx + 1];
                chain.next_physical = core_work.at(next_core_idx).physical_core;
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

            // Every participating core owns one Q chunk per head-serial pass.
            candidates.push_back(McastCandidate{chain_cores});
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

                for (const auto& ci : cand.core_indices) {
                    if (ci == injector_idx) {
                        continue;
                    }
                    auto& receiver_chain = core_chain_info[ci];
                    receiver_chain.prev_physical = core_work[injector_idx].physical_core;
                    receiver_chain.next_physical = CoreCoord{0, 0};
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
    // MUX cores are placed at the last x coordinate of the user grid.
    //   Backward MUX: first y (0) and last y (sdpa_grid.y - 1). Forward MUX: middle y - 1 and middle y.
    const uint32_t mid = sdpa_grid.y / 2;
    const uint32_t fabric_mux_col = user_grid.x - 1;
    const std::vector<CoreCoord> mux_backward_logical_cores = {{fabric_mux_col, 0}, {fabric_mux_col, sdpa_grid.y - 1}};
    const std::vector<CoreCoord> mux_forward_logical_cores = {{fabric_mux_col, mid - 1}, {fabric_mux_col, mid}};

    const uint32_t num_mux_full_size_channels = num_mux_clients_per_group;
    const uint32_t num_mux_header_only_channels = 0;
    // The caller's num_buffers_per_channel is calibrated for one channel per row; with follower
    // rows disconnected, redistribute the same L1 across the remaining channels.
    const uint32_t num_mux_buffers_per_channel =
        args.num_buffers_per_channel * num_workers_per_link / num_mux_clients_per_group;
    const size_t mux_buffer_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*mesh_device);
    // The mux's map goes in program-local L1 (a circular buffer on the mux cores), so it stays clear of persistent
    // L1 there; its addresses are offsets until the mux publishes the map's base.
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        *mesh_device,
        num_mux_full_size_channels,
        num_mux_header_only_channels,
        num_mux_buffers_per_channel,
        0,
        mux_buffer_size_bytes,
        /*base_l1_address=*/0);

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
        mux_kernel_config.place_in_program_local_l1(desc, mux_core_range_set);
    }

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
    reader_kernel.named_compile_time_args = {{"kv_cb_entries", kv_cb_entries}};
    reader_kernel.config = ReaderConfigDescriptor{};

    // All-gather CT args for the fabric writer (integrated K/V all-gather on MUX client columns); both writer kernels
    // take the same compile-time args (the plain writer ignores the MUX and all-gather ones).
    fabric_mux_connection_ct_args(num_mux_clients_per_group, mux_kernel_config, writer_compile_time_args);
    const uint32_t ag_page_size = input_tensor_k.mesh_buffer().page_size();
    const size_t ag_packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*mesh_device);
    constexpr uint32_t max_scatter_addresses = 4;
    const uint32_t ag_packet_size_in_pages =
        std::min(static_cast<uint32_t>(ag_packet_size_bytes / ag_page_size), max_scatter_addresses);
    writer_compile_time_args.push_back(ag_packet_size_in_pages);
    writer_compile_time_args.push_back(ag_page_size);
    TensorAccessorArgs(gathered_input_tensor_k.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_v.buffer()).append_to(writer_compile_time_args);

    // Non-fabric writer: columns 0..(sdpa_grid.x-3)
    // sdpa_grid.x-2 and sdpa_grid.x-1 are fabric MUX client columns
    CoreRange sdpa_writer_range({0, 0}, {sdpa_grid.x - 3, sdpa_grid.y - 1});
    KernelDescriptor writer_kernel{};
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_writer.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = CoreRangeSet(sdpa_writer_range);
    writer_kernel.compile_time_args = writer_compile_time_args;
    writer_kernel.named_compile_time_args = {{"use_mux", 0}};
    writer_kernel.config = WriterConfigDescriptor{};

    // Fabric writer: columns sdpa_grid.x-2 and sdpa_grid.x-1 (backward and forward MUX clients)
    CoreRange mux_writer_range({sdpa_grid.x - 2, 0}, {sdpa_grid.x - 1, sdpa_grid.y - 1});
    KernelDescriptor writer_fabric_kernel = writer_kernel;
    writer_fabric_kernel.core_ranges = CoreRangeSet(mux_writer_range);
    writer_fabric_kernel.named_compile_time_args = {{"use_mux", 1}};

    // The recipe compute: its named arguments (after any L1 fallback above), compute config (fidelity, FP32 DEST and
    // unpack-to-dest modes) and defines. Exp ring masks only key padding (recipe_ring_mask 0) and keeps each pass's
    // state resident (no checkpoints).
    auto recipe_args = recipes::recipe_named_args(compute_recipe_kernel.named_compile_time_args);
    recipes::set_named_arg(recipe_args, "recipe_ring", 1);
    KernelDescriptor compute_kernel{};
    compute_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/exp_ring_joint_sdpa.cpp";
    compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = CoreRangeSet(sdpa_grid_range);
    compute_kernel.compile_time_args = compute_compile_time_args;
    compute_kernel.named_compile_time_args = recipe_args;
    compute_kernel.defines = compute_recipe_kernel.defines;
    compute_kernel.config = compute_recipe_kernel.config;
    // Math builds at -O2 (exp_ring_joint_sdpa.cpp); with fused chunks or ACCURATE, unpack and pack too (their code at
    // -O3 overflows the kernel config buffer, as on the ring).
    if (recipe_fused || policy.selection.recipe == recipes::Recipe::D) {
        compute_kernel.opt_level = recipes::recipe_opt_level(tt::tt_metal::KernelBuildOptLevel::O2);
    }

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
            "core logical=({},{})->phys=({},{}), q_base={} stride={} count={}, chain={{part:{}, inj:{}, sink:{}}}",
            core.x,
            core.y,
            core_work.at(i).physical_core.x,
            core_work.at(i).physical_core.y,
            work.q_base,
            work.q_stride,
            work.q_count,
            chain.participates,
            chain.is_injector,
            chain.is_sink);

        reader_args.push_back(static_cast<uint32_t>(chain.participates));
        reader_args.push_back(static_cast<uint32_t>(chain.is_injector));
        reader_args.push_back(static_cast<uint32_t>(chain.is_sink));
        reader_args.push_back(static_cast<uint32_t>(chain.prev_physical.x));
        reader_args.push_back(static_cast<uint32_t>(chain.prev_physical.y));
        reader_args.push_back(static_cast<uint32_t>(chain.next_physical.x));
        reader_args.push_back(static_cast<uint32_t>(chain.next_physical.y));
        reader_args.push_back(chain.mcast_num_dests);
        reader_args.push_back(chain.mcast_sender_wait);

        // Determine if this core's writer has a valid MUX connection (for reader-side forwarding).
        // Split-head dedup follower rows forward nothing, so their clients do not connect and
        // their readers do not feed the MUX writer's K/V handles at all.
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
                    *device,
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

            // MUX writer RT args: out_ready_sem, injector coords, AG params, forwarding dedup flag.
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

    // ---- Fabric MUX kernel ----
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
                    *mesh_device, src_node_id, dst_node_id, link, desc, mux_backward_logical_cores[link]);
                KernelDescriptor::RTArgList mux_args;
                mux_args.append(mux_rt_args);
                mux_kernel.emplace_runtime_args(mux_backward_logical_cores[link], mux_args);
            }
            if (forward_coord.has_value()) {
                const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                auto mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    *mesh_device, src_node_id, dst_node_id, link, desc, mux_forward_logical_cores[link]);
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
    const CoreCoord sdpa_grid = exp_ring_sdpa_grid(user_grid);
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
