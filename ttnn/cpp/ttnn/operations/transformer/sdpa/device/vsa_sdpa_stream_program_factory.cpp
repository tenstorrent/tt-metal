// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa streaming (v3) program factory. Cores are grouped per head: the group's first core is a
// fetch-only LEADER that streams the head's KV blocks from DRAM once per pass into its L1 slots;
// the WORKERS hold resident query rows as CONTIGUOUS runs (real VSA index sets are spatially
// correlated and every row lists the exempt prefix, so contiguity maximizes how many resident rows
// each arriving block visits -- the multi-row batching that amortizes per-visit overhead), pull
// blocks from the leader's L1 over the NoC, and run the fused multi-row online-softmax engine. DRAM reads the head's KV exactly n_passes times per group instead
// of once per (row, listed block) -- the redundancy that made v1/v2 DRAM-bound.

#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <bit>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace ttnn::prim {

namespace {
using RtArgs = std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>>;

constexpr uint32_t kRMax = 15;         // resident rows per pass (bounded by L1; <= 32 for the bitmasks)
constexpr uint32_t kRowGroup = 8;      // rows fused per compute phase group
constexpr uint32_t kStreamDepth = 12;  // KV blocks in flight (two window halves of 6)
constexpr uint32_t kLogDepth = 16;     // leader arrival-log ring (> leader depth + sentinel slack)
constexpr uint32_t kMaxWorkers = 16;   // leader runtime-arg array bound
constexpr uint32_t kRowChunk = 4;      // contiguous rows per placement chunk (matches the kernels)

struct StreamSchedule {
    uint32_t head = 0;
    bool is_leader = false;
    uint32_t group_first = 0;  // core id of the group's leader
    uint32_t n_workers = 0;
    uint32_t worker_index = 0;  // valid when !is_leader
};

StreamSchedule core_schedule(uint32_t core, uint32_t num_cores, uint32_t heads) {
    uint32_t head = static_cast<uint32_t>((static_cast<uint64_t>(core) * heads) / num_cores);
    while ((static_cast<uint64_t>(head + 1) * num_cores) / heads <= core) {
        ++head;
    }
    while ((static_cast<uint64_t>(head) * num_cores) / heads > core) {
        --head;
    }
    const uint32_t c0 = static_cast<uint32_t>((static_cast<uint64_t>(head) * num_cores) / heads);
    const uint32_t c1 = static_cast<uint32_t>((static_cast<uint64_t>(head + 1) * num_cores) / heads);
    return StreamSchedule{head, core == c0, c0, c1 - c0 - 1, core - c0 - 1};
}

uint32_t active_workers(uint32_t group_workers, uint32_t n_q_tiles) {
    // The leader tracks per-worker coords in a fixed array, and a head never benefits from more
    // workers than it has rows; surplus cores idle (n_passes = 0, row_count = 0).
    uint32_t n = group_workers;
    if (n > kMaxWorkers) {
        n = kMaxWorkers;
    }
    if (n > n_q_tiles) {
        n = n_q_tiles;
    }
    return n == 0 ? 1 : n;
}
}  // namespace

VsaSdpaOperation::program_factory_t VsaSdpaOperation::select_program_factory(
    const VsaSdpaParams& attrs, const VsaSdpaInputs& /*t*/) {
    if (attrs.streaming) {
        return VsaSdpaStreamProgramFactory{};
    }
    return VsaSdpaProgramFactory{};
}

tt::tt_metal::ProgramDescriptor VsaSdpaOperation::VsaSdpaStreamProgramFactory::create_descriptor(
    const VsaSdpaParams& attrs, const VsaSdpaInputs& t, Tensor& output) {
    // Function-scoped to avoid unity-build collisions with the v1 factory's enum.
    enum StreamCB : uint32_t {
        cb_q_res = 0,      // resident Q rows [R_MAX, Sqt*DHt] (RAM-mode)
        cb_k_stream,       // stream slots [depth, Skt*DHt] (RAM-mode; leader staging on leaders)
        cb_v_stream,       // stream slots [depth, Skt*vDHt]
        cb_o_res,          // resident O accumulators [R_MAX, Sqt*vDHt]
        cb_max_res,        // resident running max, ping-pong [R_MAX, 2, Sqt]
        cb_sum_res,        // resident running sum, ping-pong [R_MAX, 2, Sqt]
        cb_corr,           // per-group rescale factors [G, Sqt]
        cb_qk,             // per-group scores/probs scratch [G, Sqt*Skt]
        cb_scale,          // reduce identity scaler
        cb_col_identity,   // ones-in-col0 for the flush sum reduction
        cb_recip_scratch,  // reciprocal scratch
        cb_neginf,         // persistent -inf tile
        cb_vmask,          // ragged partial-column mask tiles (FIFO)
        cb_ctrl,           // reader -> compute visit/flush messages
        cb_kreq,           // reader -> writer K fetch/pull requests
        cb_kack,           // writer -> reader K completion
        cb_free,           // compute -> reader stream-slot credits
        cb_qdone,          // writer -> compute pass-Q-resident token
        cb_out,            // normalized row outputs (compute -> writer)
        cb_idxrow,         // reader scratch: one index row
        cb_counts,         // counts row, resident
        cb_bitmap,         // reader scratch: R_MAX x ceil(KVB/32) membership words
        cb_log,            // leader arrival log ring (leader stages, workers receive)
        cb_ackbox,         // per-worker progress words (leader-resident; workers post their count)
        cb_count
    };
    constexpr uint32_t sem_arrivals = 0;

    tt::tt_metal::ProgramDescriptor desc;

    const uint32_t H = t.q.logical_shape()[1];
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t d = t.q.logical_shape()[3];
    const uint32_t T = t.k.logical_shape()[2];
    const uint32_t W = t.indices.logical_shape()[3];
    const uint32_t block_size = attrs.block_size;

    constexpr uint32_t q_chunk_tokens = 64;
    const uint32_t DHt = d / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = DHt;
    const uint32_t Sqt = q_chunk_tokens / tt::constants::TILE_HEIGHT;
    const uint32_t Skt = block_size / tt::constants::TILE_WIDTH;
    const uint32_t n_q_tiles = S / q_chunk_tokens;
    const uint32_t n_kv_blocks = T / block_size;
    const uint32_t k_tiles_per_block = Skt * DHt;
    const uint32_t v_tiles_per_block = Skt * vDHt;
    const uint32_t q_tiles_per_row = Sqt * DHt;
    const uint32_t out_tiles_per_row = Sqt * vDHt;
    const uint32_t k_head_stride = (T / tt::constants::TILE_HEIGHT) * DHt;
    const uint32_t v_head_stride = (T / tt::constants::TILE_HEIGHT) * vDHt;
    const uint32_t scale_packed = std::bit_cast<uint32_t>(attrs.scale);
    const uint32_t idx_row_bytes = W * t.indices.element_size();
    const uint32_t counts_row_bytes = W * t.block_counts.element_size();
    const uint32_t bitmap_bytes = ((kRMax * ((n_kv_blocks + 31) / 32) * 4 + 15) / 16) * 16;
    const uint32_t ctrl_page_bytes = ((4 + kRMax) * 4 + 15) / 16 * 16;

    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr uint32_t tile_bytes = tt::tile_size(bf);
    const tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(t.q.dtype());
    const tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(t.k.dtype());
    const tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(t.v.dtype());
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t q_tile_bytes = tt::tile_size(q_df);
    const uint32_t k_tile_bytes = tt::tile_size(k_df);
    const uint32_t v_tile_bytes = tt::tile_size(v_df);
    const uint32_t out_tile_bytes = tt::tile_size(out_df);

    auto* device = t.q.device();
    const tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    TT_FATAL(num_cores >= 2 * H, "vsa_sdpa streaming needs >= 2 cores per head (H {}, cores {})", H, num_cores);
    auto core_grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));

    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = sem_arrivals, .core_type = tt::CoreType::WORKER, .core_ranges = core_grid, .initial_value = 0});

    // ---- CBs (fixed order = StreamCB enum; identical layout on every core is load-bearing:
    // workers address leader slots as their own CB base + slot offset) ----
    const auto cb = [&](uint32_t page_size, uint32_t num_pages, tt::DataFormat df) {
        const uint32_t idx = desc.cbs.size();
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = df, .page_size = page_size}}},
        });
    };
    cb(q_tile_bytes, kRMax * q_tiles_per_row, q_df);           // cb_q_res
    cb(k_tile_bytes, kStreamDepth * k_tiles_per_block, k_df);  // cb_k_stream
    cb(v_tile_bytes, kStreamDepth * v_tiles_per_block, v_df);  // cb_v_stream
    cb(tile_bytes, kRMax * out_tiles_per_row, bf);             // cb_o_res
    cb(tile_bytes, kRMax * 2 * Sqt, bf);                       // cb_max_res
    cb(tile_bytes, kRMax * 2 * Sqt, bf);                       // cb_sum_res
    cb(tile_bytes, kRowGroup * Sqt, bf);                       // cb_corr
    cb(tile_bytes, kRowGroup * Sqt * Skt, bf);                 // cb_qk
    cb(tile_bytes, 1, bf);                                     // cb_scale
    cb(tile_bytes, 1, bf);                                     // cb_col_identity
    cb(tile_bytes, 1, bf);                                     // cb_recip_scratch
    cb(tile_bytes, 1, bf);                                     // cb_neginf
    cb(tile_bytes, kStreamDepth, bf);                          // cb_vmask (slot-indexed RAM)
    cb(ctrl_page_bytes, 8, bf);                                // cb_ctrl
    cb(16, kStreamDepth, bf);                                  // cb_kreq
    cb(16, kStreamDepth, bf);                                  // cb_kack
    cb(16, kStreamDepth + 2, bf);                              // cb_free
    cb(16, 2, bf);                                             // cb_qdone
    cb(out_tile_bytes, 2 * out_tiles_per_row, out_df);         // cb_out
    cb(idx_row_bytes, 1, bf);                                  // cb_idxrow
    cb(counts_row_bytes, 1, bf);                               // cb_counts
    cb(bitmap_bytes, 1, bf);                                   // cb_bitmap
    cb(16, kLogDepth, bf);                                     // cb_log
    cb(4 * kMaxWorkers, 1, bf);                                // cb_ackbox

    // ---- compile-time args ----
    std::vector<uint32_t> reader_ct = {
        W,
        n_kv_blocks,
        n_q_tiles,
        block_size,
        kRMax,
        kStreamDepth,
        kLogDepth,
        k_tiles_per_block,
        v_tiles_per_block,
        v_head_stride,
        idx_row_bytes,
        counts_row_bytes,
        k_tile_bytes,
        v_tile_bytes};
    for (uint32_t id :
         {cb_k_stream, cb_v_stream, cb_idxrow, cb_counts, cb_bitmap, cb_log, cb_ctrl, cb_kreq, cb_kack, cb_free,
          cb_vmask, cb_ackbox}) {
        reader_ct.push_back(id);
    }
    reader_ct.push_back(sem_arrivals);
    std::vector<uint32_t> reader_crt;
    tt::tt_metal::TensorAccessorArgs(t.v.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.indices.buffer()).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(t.block_counts.buffer()).append_to(reader_ct, reader_crt);

    std::vector<uint32_t> writer_ct = {
        n_q_tiles,
        kRMax,
        q_tiles_per_row,
        out_tiles_per_row,
        k_tiles_per_block,
        k_head_stride,
        q_tile_bytes,
        k_tile_bytes,
        out_tile_bytes};
    for (uint32_t id : {cb_q_res, cb_k_stream, cb_scale, cb_col_identity, cb_neginf, cb_kreq, cb_kack, cb_qdone, cb_out}) {
        writer_ct.push_back(id);
    }
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.k.buffer()).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(t.q.buffer()).append_to(writer_ct, writer_crt);

    std::vector<uint32_t> compute_ct = {
        DHt, vDHt, Skt, Sqt, kRMax, kRowGroup, block_size, scale_packed,
        cb_q_res, cb_k_stream, cb_v_stream, cb_o_res, cb_max_res, cb_sum_res, cb_corr, cb_qk,
        cb_scale, cb_col_identity, cb_recip_scratch, cb_neginf, cb_vmask, cb_ctrl, cb_free,
        cb_qdone, cb_out, kStreamDepth};

    // ---- kernels ----
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source = kdir + "dataflow/vsa_sdpa_stream_reader.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.common_runtime_args = reader_crt;
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source = kdir + "dataflow/vsa_sdpa_stream_writer.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.common_runtime_args = writer_crt;
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    auto [math_fidelity, math_approx, fp32_acc, packer_l1_acc, dst_full_sync] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), attrs.compute_kernel_config);
    (void)packer_l1_acc;
    const uint32_t dst_size = fp32_acc ? 4u : 8u;
    TT_FATAL(Sqt * vDHt <= dst_size, "vsa_sdpa streaming needs the PV subblock ({}) to fit DEST ({})",
             Sqt * vDHt, dst_size);

    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source = kdir + "compute/vsa_sdpa_stream_compute.cpp";
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = compute_ct;
    compute_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_acc,
        .dst_full_sync_en = dst_full_sync,
        .math_approx_mode = math_approx};
    std::map<std::string, std::string> cdefs{
        {"EXP_APPROX_MODE", std::to_string(static_cast<int>(math_approx))},
    };
    compute_desc.defines = tt::tt_metal::KernelDescriptor::Defines(cdefs.begin(), cdefs.end());

    auto* q_buf = t.q.buffer();
    auto* k_buf = t.k.buffer();
    auto* v_buf = t.v.buffer();
    auto* idx_buf = t.indices.buffer();
    auto* counts_buf = t.block_counts.buffer();
    auto* out_buf = output.buffer();
    for (uint32_t i = 0; i < num_cores; ++i) {
        tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        const auto sched = core_schedule(i, num_cores, H);
        const uint32_t n_active = active_workers(sched.n_workers, n_q_tiles);
        const bool is_idle = !sched.is_leader && sched.worker_index >= n_active;

        const tt::tt_metal::CoreCoord leader_logical = {sched.group_first % grid.x, sched.group_first / grid.x};
        const tt::tt_metal::CoreCoord leader_phys = device->worker_core_from_logical_core(leader_logical);
        // worker 0 always holds the most rows under chunk-cyclic dealing
        uint32_t max_rows = 0;
        {
            const uint32_t nc = (n_q_tiles + kRowChunk - 1) / kRowChunk;
            for (uint32_t c = 0; c < nc; c += n_active) {
                const uint32_t c0 = c * kRowChunk;
                max_rows += (n_q_tiles - c0 < kRowChunk) ? (n_q_tiles - c0) : kRowChunk;
            }
        }
        const uint32_t n_passes = is_idle ? 0 : (max_rows + kRMax - 1) / kRMax;
        // Chunked round-robin placement: 4-row contiguous chunks dealt cyclically across the
        // group's workers. Adjacent rows share most of their index sets (spatial correlation plus
        // the exempt prefix), so intra-chunk contiguity feeds the engine's multi-row batching,
        // while dealing chunks cyclically spreads the fully-dense exempt-query rows -- a purely
        // contiguous split piles them all onto worker 0 (measured 3x worst-shard regression).
        const uint32_t n_chunks = (n_q_tiles + kRowChunk - 1) / kRowChunk;
        uint32_t row_count = 0;
        if (!sched.is_leader && !is_idle) {
            for (uint32_t c = sched.worker_index; c < n_chunks; c += n_active) {
                const uint32_t c0 = c * kRowChunk;
                row_count += (n_q_tiles - c0 < kRowChunk) ? (n_q_tiles - c0) : kRowChunk;
            }
        }
        const uint32_t row_start = sched.is_leader ? 0 : sched.worker_index * kRowChunk;

        RtArgs reader_rt;
        reader_rt.reserve(13 + 2 * kMaxWorkers);
        reader_rt.push_back(v_buf);
        reader_rt.push_back(idx_buf);
        reader_rt.push_back(counts_buf);
        reader_rt.push_back(sched.head);
        reader_rt.push_back(sched.is_leader ? 1u : 0u);
        reader_rt.push_back(n_passes);
        reader_rt.push_back(static_cast<uint32_t>(leader_phys.x));
        reader_rt.push_back(static_cast<uint32_t>(leader_phys.y));
        reader_rt.push_back(n_active);
        reader_rt.push_back(sched.is_leader ? 0u : sched.worker_index);
        reader_rt.push_back(row_start);
        reader_rt.push_back(n_active * kRowChunk);  // chunk-cyclic big stride
        reader_rt.push_back(row_count);
        if (sched.is_leader) {
            for (uint32_t w = 0; w < n_active; ++w) {
                const uint32_t wc = sched.group_first + 1 + w;
                const tt::tt_metal::CoreCoord w_logical = {wc % grid.x, wc / grid.x};
                const tt::tt_metal::CoreCoord w_phys = device->worker_core_from_logical_core(w_logical);
                reader_rt.push_back(static_cast<uint32_t>(w_phys.x));
                reader_rt.push_back(static_cast<uint32_t>(w_phys.y));
            }
        }
        reader_desc.emplace_runtime_args(core, reader_rt);

        RtArgs writer_rt(12);
        writer_rt[0] = out_buf;
        writer_rt[1] = k_buf;
        writer_rt[2] = q_buf;
        writer_rt[3] = sched.head;
        writer_rt[4] = sched.is_leader ? 1u : 0u;
        writer_rt[5] = n_passes;
        writer_rt[6] = static_cast<uint32_t>(leader_phys.x);
        writer_rt[7] = static_cast<uint32_t>(leader_phys.y);
        writer_rt[8] = row_start;
        writer_rt[9] = n_active * kRowChunk;  // chunk-cyclic big stride
        writer_rt[10] = row_count;
        writer_rt[11] = 0u;
        writer_desc.emplace_runtime_args(core, writer_rt);

        RtArgs compute_rt(1);
        compute_rt[0] = row_count;
        compute_desc.emplace_runtime_args(core, compute_rt);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

void VsaSdpaOperation::VsaSdpaStreamProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const VsaSdpaParams& attrs,
    const VsaSdpaInputs& t,
    Tensor& tensor_return_value) {
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;
    const tt::tt_metal::CoreCoord grid = t.q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    (void)attrs;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i % grid.x, i / grid.x};
        auto& reader = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx, core);
        reader[0] = t.v.buffer()->address();
        reader[1] = t.indices.buffer()->address();
        reader[2] = t.block_counts.buffer()->address();
        auto& writer = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx, core);
        writer[0] = tensor_return_value.buffer()->address();
        writer[1] = t.k.buffer()->address();
        writer[2] = t.q.buffer()->address();
    }
}

}  // namespace ttnn::prim
