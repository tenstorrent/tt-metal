// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/deepseek_v4_prefill_indexer_select_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_indexer_select_common.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>  // NUM_CIRCULAR_BUFFERS
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <string>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor DeepseekV4PrefillIndexerSelectOperation::PoolProgramFactory::create_descriptor(
    const DeepseekV4PrefillIndexerSelectParams& attrs,
    const DeepseekV4PrefillIndexerSelectInputs& t,
    std::array<Tensor, 4>& outputs) {
    namespace dis = ::deepseek_v4_prefill_indexer_select;
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_HW;
    enum : uint32_t {
        cb_q = 0,   // the query's q as [heads, dims] tiles
        cb_w,       // the query's gates as row 0 of a tile
        cb_k_rm,    // pool key rows of one step (row-major)
        cb_k,       // the step's key tiles [key tile][dim tile]
        cb_qk,      // relu(q . k^T) of the step, [heads, keys] per key tile
        cb_scores,  // a group's score rows (slot 0 of each block's ROWS slots)
        cb_neginf,  // all -inf chunk
        cb_surv_v,  // survivor values (fp32 bits)
        cb_surv_i,  // survivor indices (uint32)
        cb_out,     // one output row (pool positions, mapped to entries by the writer)
        cb_meta,    // reader -> compute work-unit mailbox
        cb_start,   // reader scratch for the start position
        cb_list,    // reader -> writer: the query's pool list
        cb_bounce,  // reader scratch: aligned q / gate row reads
        cb_cands,   // reader scratch: the query's candidate row
        cb_unused,  // ids of the compute kernel's buffers this mode never touches
    };
    tt::tt_metal::ProgramDescriptor desc;

    const uint32_t H = t.q.logical_shape()[1];
    const uint32_t T = t.q.logical_shape()[2];
    const uint32_t D = t.q.logical_shape()[3];
    const uint32_t E = t.idx_keys.logical_shape()[2];
    const uint32_t Dt = D / tt::constants::TILE_WIDTH;
    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr tt::DataFormat f32 = tt::DataFormat::Float32;
    constexpr tt::DataFormat i32 = tt::DataFormat::Int32;
    constexpr tt::DataFormat u32 = tt::DataFormat::UInt32;
    constexpr uint32_t bf_tile = TILE_HW * sizeof(uint16_t);
    constexpr uint32_t f32_tile = TILE_HW * sizeof(uint32_t);
    constexpr uint32_t step_rows = dis::STEP_KEY_TILES * TILE_HEIGHT;

    auto* device = t.q.device();
    const tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t used_cores = std::min<uint32_t>(grid.x * grid.y, T);
    const auto logical = [&](uint32_t i) { return tt::tt_metal::CoreCoord{i % grid.x, i / grid.x}; };
    std::vector<tt::tt_metal::CoreRange> ranges;
    for (uint32_t i = 0; i < used_cores; ++i) {
        ranges.emplace_back(logical(i));
    }
    const auto cores = tt::tt_metal::CoreRangeSet(ranges).merge_ranges();

    const auto cb = [&](uint32_t id, tt::DataFormat fmt, uint32_t page_size, uint32_t num_pages) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = cores,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(id), .data_format = fmt, .page_size = page_size}}},
        });
    };
    cb(cb_q, bf, bf_tile, 2 * Dt);
    cb(cb_w, bf, bf_tile, 2);
    cb(cb_k_rm, bf, D * sizeof(uint16_t), 2 * step_rows);
    cb(cb_k, bf, bf_tile, 2 * dis::STEP_KEY_TILES * Dt);
    cb(cb_qk, bf, bf_tile, dis::STEP_KEY_TILES);
    cb(cb_scores, bf, bf_tile, dis::GROUP_BLOCKS * dis::ROWS);
    cb(cb_neginf, bf, bf_tile, dis::NEGINF_TILES);
    cb(cb_surv_v, i32, f32_tile, 1);
    cb(cb_surv_i, i32, f32_tile, 1);
    cb(cb_meta, u32, dis::META_PAGE_BYTES, 2);
    cb(cb_start, u32, dis::START_PAGE_BYTES, 1);
    cb(cb_list, u32, dis::POOL_LIST_PAGE_BYTES, 2);
    cb(cb_bounce, bf, (H * Dt * 2 + 2) * dis::POOL_PIECE_BYTES, 1);
    cb(cb_cands, u32, dis::CAND_TOPK * sizeof(uint32_t), 1);
    // A K = 512 output row is the top half (two 16-row faces) of an fp32 tile.
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = 2 * dis::TOPK * static_cast<uint32_t>(sizeof(uint32_t)),
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_out),
            .data_format = f32,
            .page_size = dis::TOPK * static_cast<uint32_t>(sizeof(uint32_t)),
            .face_geometry = tt::tt_metal::FaceGeometry{.face_r_dim = tt::constants::FACE_HEIGHT, .num_faces = 2}}}},
    });

    auto* q_buf = t.q.buffer();
    auto* w_buf = t.weights.buffer();
    auto* keys_buf = t.idx_keys.buffer();
    auto* start_buf = t.start.buffer();
    auto* cands_buf = t.candidates->buffer();
    auto* out_buf = outputs[0].buffer();

    std::vector<uint32_t> reader_ct = {
        H, D, T, attrs.compress_ratio, cb_q, cb_w, cb_k_rm, cb_neginf, cb_meta, cb_start, cb_list, cb_bounce, cb_cands};
    TT_FATAL(
        reader_ct.size() == dis::pool_reader_ct_arg::END,
        "deepseek_v4_prefill_indexer_select pool reader compile-time argument layout is out of sync");
    std::vector<uint32_t> reader_crt;
    tt::tt_metal::TensorAccessorArgs(q_buf).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(w_buf).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(keys_buf).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(start_buf).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(cands_buf).append_to(reader_ct, reader_crt);

    std::vector<uint32_t> writer_ct = {cb_out, cb_list};
    TT_FATAL(
        writer_ct.size() == dis::pool_writer_ct_arg::END,
        "deepseek_v4_prefill_indexer_select pool writer compile-time argument layout is out of sync");
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(out_buf).append_to(writer_ct, writer_crt);

    const std::vector<uint32_t> compute_ct = {
        H,         Dt,
        1,  // splits
        cb_q,      cb_w,      cb_k_rm,   cb_k,      cb_qk,     cb_scores, cb_neginf, cb_surv_v, cb_surv_i,
        cb_unused,  // peer
        cb_out,    cb_meta,
        cb_unused,                                                             // done
        0,                                                                     // candidates
        cb_unused, cb_unused, cb_unused, cb_unused, cb_unused, cb_unused, 1};  // pool
    TT_FATAL(
        compute_ct.size() == dis::compute_ct_arg::END,
        "deepseek_v4_prefill_indexer_select pool compute compile-time argument layout is out of sync");

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source = kdir + "dataflow/deepseek_v4_prefill_indexer_select_pool_reader.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = cores;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source = kdir + "dataflow/deepseek_v4_prefill_indexer_select_pool_writer.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = cores;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    auto [math_fidelity, math_approx, fp32_acc, packer_l1_acc, dst_full_sync] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), attrs.compute_kernel_config);
    (void)fp32_acc;
    (void)packer_l1_acc;
    (void)dst_full_sync;
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_to_dest[cb_surv_v] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[cb_surv_i] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source = kdir + "compute/deepseek_v4_prefill_indexer_select_compute.cpp";
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = cores;
    compute_desc.compile_time_args = compute_ct;
    compute_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .unpack_to_dest_mode = unpack_to_dest,
        .math_approx_mode = math_approx};

    reader_desc.emplace_common_runtime_args({q_buf, w_buf, keys_buf, start_buf, cands_buf, E});
    writer_desc.emplace_common_runtime_args({out_buf});
    TT_FATAL(
        reader_desc.common_runtime_args.size() == dis::pool_reader_common_arg::END &&
            writer_desc.common_runtime_args.size() == dis::pool_writer_common_arg::END,
        "deepseek_v4_prefill_indexer_select pool common runtime argument layout is out of sync");
    reader_desc.common_runtime_args.insert(reader_desc.common_runtime_args.end(), reader_crt.begin(), reader_crt.end());
    writer_desc.common_runtime_args.insert(writer_desc.common_runtime_args.end(), writer_crt.begin(), writer_crt.end());

    // Core i takes queries [i * T / used_cores, (i + 1) * T / used_cores).
    for (uint32_t i = 0; i < used_cores; ++i) {
        const uint32_t first = i * T / used_cores;
        const uint32_t count = (i + 1) * T / used_cores - first;
        const std::initializer_list<uint32_t> dm_args = {first, count};
        TT_FATAL(
            dm_args.size() == dis::pool_core_arg::END,
            "deepseek_v4_prefill_indexer_select pool core argument layout is out of sync");
        reader_desc.emplace_runtime_args(logical(i), dm_args);
        writer_desc.emplace_runtime_args(logical(i), dm_args);
        // Compute walks its queries as one-query "tiles" of one split.
        const auto self = device->worker_core_from_logical_core(logical(i));
        const std::initializer_list<uint32_t> compute_args = {
            first, 1, count, 0, static_cast<uint32_t>(self.x), static_cast<uint32_t>(self.y)};
        TT_FATAL(
            compute_args.size() == dis::core_arg::END,
            "deepseek_v4_prefill_indexer_select core argument layout is out of sync");
        compute_desc.emplace_runtime_args(logical(i), compute_args);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

void DeepseekV4PrefillIndexerSelectOperation::PoolProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const DeepseekV4PrefillIndexerSelectParams& /*operation_attributes*/,
    const DeepseekV4PrefillIndexerSelectInputs& t,
    std::array<Tensor, 4>& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    namespace dis = ::deepseek_v4_prefill_indexer_select;
    auto& reader_args = tt::tt_metal::GetCommonRuntimeArgs(program, 0);
    auto& writer_args = tt::tt_metal::GetCommonRuntimeArgs(program, 1);
    reader_args[dis::pool_reader_common_arg::Q_ADDRESS] = t.q.buffer()->address();
    reader_args[dis::pool_reader_common_arg::W_ADDRESS] = t.weights.buffer()->address();
    reader_args[dis::pool_reader_common_arg::KEYS_ADDRESS] = t.idx_keys.buffer()->address();
    reader_args[dis::pool_reader_common_arg::START_ADDRESS] = t.start.buffer()->address();
    reader_args[dis::pool_reader_common_arg::CANDS_ADDRESS] = t.candidates->buffer()->address();
    reader_args[dis::pool_reader_common_arg::NUM_ENTRIES] = t.idx_keys.logical_shape()[2];
    writer_args[dis::pool_writer_common_arg::OUTPUT_ADDRESS] = outputs[0].buffer()->address();
}

}  // namespace ttnn::prim
