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

tt::tt_metal::ProgramDescriptor DeepseekV4PrefillIndexerSelectOperation::ProgramFactory::create_descriptor(
    const DeepseekV4PrefillIndexerSelectParams& attrs,
    const DeepseekV4PrefillIndexerSelectInputs& t,
    std::array<Tensor, 4>& outputs) {
    namespace dis = ::deepseek_v4_prefill_indexer_select;
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_HW;
    enum : uint32_t {
        cb_q = 0,   // q tiles of the work unit, [head][dim tile]
        cb_w,       // one gate tile per head, w[:, h] in column 0
        cb_w_in,    // reader scratch: the unit's [32 queries, H] gate tiles
        cb_k_rm,    // idx_keys rows of one step (row-major)
        cb_k,       // the step's key tiles [key tile][dim tile]
        cb_qk,      // relu(q . k^T) of the step, head-major [head][key tile]
        cb_scores,  // a group's score rows: per block, one bf16 tile slot per query
        cb_neginf,  // all -inf chunk
        cb_surv_v,  // per-query survivor values (fp32 bits), random access
        cb_surv_i,  // per-query survivor indices (uint32), random access
        cb_peer,    // leader: one peer survivor [values, indices]
        cb_out,     // one output row (uint32, K = 512 half tile)
        cb_meta,    // reader -> compute work-unit mailbox
        cb_start,   // reader scratch for the start position
        cb_done,    // split > 0: compute -> writer "survivors final"
        cb_plan,    // reader -> writer: copy of the work-unit mailbox
        // Candidate mode only.
        cb_group,      // compute -> writer: a group's score rows are packed
        cb_group_ack,  // writer -> compute: block maxima taken, the group's scores may be freed
        cb_stage,      // writer -> compute: one candidate chunk of block maxima per query (bf16 rows)
        cb_csurv_in,   // writer -> compute: one candidate survivor
        cb_csurv_out,  // compute -> writer: one candidate survivor
        cb_cand_out,   // leader: one candidate id row
    };
    constexpr uint32_t semaphore_id = 0;
    tt::tt_metal::ProgramDescriptor desc;

    const uint32_t H = t.q.logical_shape()[1];
    const uint32_t T = t.q.logical_shape()[2];
    const uint32_t D = t.q.logical_shape()[3];
    const uint32_t E = t.idx_keys.logical_shape()[2];
    const uint32_t Dt = D / tt::constants::TILE_WIDTH;
    const uint32_t q_tiles = T / TILE_HEIGHT;
    constexpr tt::DataFormat bf = tt::DataFormat::Float16_b;
    constexpr tt::DataFormat f32 = tt::DataFormat::Float32;
    constexpr tt::DataFormat i32 = tt::DataFormat::Int32;
    constexpr uint32_t bf_tile = TILE_HW * sizeof(uint16_t);
    constexpr uint32_t f32_tile = TILE_HW * sizeof(uint32_t);
    constexpr uint32_t step_rows = dis::STEP_KEY_TILES * TILE_HEIGHT;

    auto* device = t.q.device();
    const tt::tt_metal::CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    const uint32_t splits = deepseek_v4_prefill_indexer_select_num_splits(q_tiles, num_cores);
    const uint32_t used_cores = std::min(num_cores, q_tiles * splits);
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
    cb(cb_q, bf, bf_tile, H * Dt);
    cb(cb_w, bf, bf_tile, H);
    cb(cb_w_in, bf, bf_tile, H / tt::constants::TILE_WIDTH);
    cb(cb_k_rm, bf, D * sizeof(uint16_t), 2 * step_rows);
    cb(cb_k, bf, bf_tile, 2 * dis::STEP_KEY_TILES * Dt);
    cb(cb_qk, bf, bf_tile, dis::STEP_KEY_TILES * H);
    cb(cb_scores, bf, bf_tile, dis::GROUP_BLOCKS * dis::ROWS);
    cb(cb_neginf, bf, bf_tile, dis::NEGINF_TILES);
    cb(cb_meta, tt::DataFormat::UInt32, dis::META_PAGE_BYTES, 2);
    cb(cb_start, tt::DataFormat::UInt32, dis::START_PAGE_BYTES, 1);
    cb(cb_plan, tt::DataFormat::UInt32, dis::META_PAGE_BYTES, 2);
    if (attrs.candidates) {
        cb(cb_group, tt::DataFormat::UInt32, dis::TOKEN_PAGE_BYTES, 1);
        cb(cb_group_ack, tt::DataFormat::UInt32, dis::TOKEN_PAGE_BYTES, 1);
        cb(cb_stage, bf, bf_tile, dis::ROWS * dis::CAND_ROW_BYTES / bf_tile);
        cb(cb_csurv_in, i32, f32_tile, dis::CAND_SURV_TILES);
        cb(cb_csurv_out, i32, f32_tile, dis::CAND_SURV_TILES);
        cb(cb_cand_out, i32, f32_tile, 2);
    } else {
        cb(cb_surv_v, i32, f32_tile, dis::ROWS);
        cb(cb_surv_i, i32, f32_tile, dis::ROWS);
        cb(cb_peer, i32, f32_tile, 2);
        cb(cb_done, tt::DataFormat::UInt32, dis::TOKEN_PAGE_BYTES, 1);
        // A K = 512 output row is the top half (two 16-row faces) of an fp32 tile.
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = 2 * dis::TOPK * static_cast<uint32_t>(sizeof(uint32_t)),
            .core_ranges = cores,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_out),
                .data_format = f32,
                .page_size = dis::TOPK * static_cast<uint32_t>(sizeof(uint32_t)),
                .face_geometry =
                    tt::tt_metal::FaceGeometry{.face_r_dim = tt::constants::FACE_HEIGHT, .num_faces = 2}}}},
        });
    }
    desc.semaphores.push_back(
        tt::tt_metal::SemaphoreDescriptor{.id = semaphore_id, .core_ranges = cores, .initial_value = 0});

    auto* q_buf = t.q.buffer();
    auto* w_buf = t.weights.buffer();
    auto* keys_buf = t.idx_keys.buffer();
    auto* start_buf = t.start.buffer();
    auto* out_buf = outputs[0].buffer();
    auto* scratch_buf = outputs[1].buffer();
    auto* cand_out_buf = outputs[2].buffer();
    auto* cand_scratch_buf = outputs[3].buffer();

    std::vector<uint32_t> reader_ct = {
        H, D, T, attrs.compress_ratio, splits, cb_q, cb_w, cb_w_in, cb_k_rm, cb_neginf, cb_meta, cb_start, cb_plan};
    TT_FATAL(
        reader_ct.size() == dis::reader_ct_arg::END,
        "deepseek_v4_prefill_indexer_select reader compile-time argument layout is out of sync");
    std::vector<uint32_t> reader_crt;
    tt::tt_metal::TensorAccessorArgs(q_buf).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(w_buf).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(keys_buf).append_to(reader_ct, reader_crt);
    tt::tt_metal::TensorAccessorArgs(start_buf).append_to(reader_ct, reader_crt);

    const uint32_t candidates = attrs.candidates ? 1 : 0;
    std::vector<uint32_t> writer_ct = {
        T,
        splits,
        cb_surv_v,
        cb_surv_i,
        cb_peer,
        cb_out,
        cb_done,
        cb_plan,
        semaphore_id,
        candidates,
        cb_scores,
        cb_group,
        cb_group_ack,
        cb_stage,
        cb_csurv_in,
        cb_csurv_out,
        cb_cand_out};
    TT_FATAL(
        writer_ct.size() == dis::writer_ct_arg::END,
        "deepseek_v4_prefill_indexer_select writer compile-time argument layout is out of sync");
    std::vector<uint32_t> writer_crt;
    tt::tt_metal::TensorAccessorArgs(out_buf).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(scratch_buf).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(cand_out_buf).append_to(writer_ct, writer_crt);
    tt::tt_metal::TensorAccessorArgs(cand_scratch_buf).append_to(writer_ct, writer_crt);

    const std::vector<uint32_t> compute_ct = {
        H,          Dt,        splits,       cb_q,      cb_w,        cb_k_rm,      cb_k,       cb_qk,
        cb_scores,  cb_neginf, cb_surv_v,    cb_surv_i, cb_peer,     cb_out,       cb_meta,    cb_done,
        candidates, cb_group,  cb_group_ack, cb_stage,  cb_csurv_in, cb_csurv_out, cb_cand_out};
    TT_FATAL(
        compute_ct.size() == dis::compute_ct_arg::END,
        "deepseek_v4_prefill_indexer_select compute compile-time argument layout is out of sync");

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source = kdir + "dataflow/deepseek_v4_prefill_indexer_select_reader.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = cores;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source = kdir + "dataflow/deepseek_v4_prefill_indexer_select_writer.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = cores;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    // TopK XL keeps fp32 values and uint32 indices in DEST and needs all 8 fp32 tiles (full sync). Survivors are
    // unpacked straight to DEST so index bits pass unchanged.
    auto [math_fidelity, math_approx, fp32_acc, packer_l1_acc, dst_full_sync] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), attrs.compute_kernel_config);
    (void)fp32_acc;
    (void)packer_l1_acc;
    (void)dst_full_sync;
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_to_dest[cb_surv_v] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[cb_surv_i] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[cb_peer] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest[cb_csurv_in] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
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

    reader_desc.emplace_common_runtime_args({q_buf, w_buf, keys_buf, start_buf, E});
    writer_desc.emplace_common_runtime_args({out_buf, scratch_buf, cand_out_buf, cand_scratch_buf});
    TT_FATAL(
        reader_desc.common_runtime_args.size() == dis::reader_common_arg::END &&
            writer_desc.common_runtime_args.size() == dis::writer_common_arg::END,
        "deepseek_v4_prefill_indexer_select common runtime argument layout is out of sync");
    reader_desc.common_runtime_args.insert(reader_desc.common_runtime_args.end(), reader_crt.begin(), reader_crt.end());
    writer_desc.common_runtime_args.insert(writer_desc.common_runtime_args.end(), writer_crt.begin(), writer_crt.end());

    for (uint32_t i = 0; i < used_cores; ++i) {
        // splits > 1: one unit per core, the splits of a query tile on consecutive cores. Otherwise a core walks
        // query tiles i, i + used_cores, ...
        const uint32_t first_q_tile = splits > 1 ? i / splits : i;
        const uint32_t split = splits > 1 ? i % splits : 0;
        const uint32_t num_units = splits > 1 ? 1 : (q_tiles - i + used_cores - 1) / used_cores;
        const auto leader = device->worker_core_from_logical_core(logical(i - split));
        const std::initializer_list<uint32_t> args = {
            first_q_tile,
            used_cores,
            num_units,
            split,
            static_cast<uint32_t>(leader.x),
            static_cast<uint32_t>(leader.y)};
        TT_FATAL(
            args.size() == dis::core_arg::END,
            "deepseek_v4_prefill_indexer_select core argument layout is out of sync");
        reader_desc.emplace_runtime_args(logical(i), args);
        writer_desc.emplace_runtime_args(logical(i), args);
        compute_desc.emplace_runtime_args(logical(i), args);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

void DeepseekV4PrefillIndexerSelectOperation::ProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const DeepseekV4PrefillIndexerSelectParams& /*operation_attributes*/,
    const DeepseekV4PrefillIndexerSelectInputs& t,
    std::array<Tensor, 4>& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    namespace dis = ::deepseek_v4_prefill_indexer_select;
    auto& reader_args = tt::tt_metal::GetCommonRuntimeArgs(program, 0);
    auto& writer_args = tt::tt_metal::GetCommonRuntimeArgs(program, 1);
    reader_args[dis::reader_common_arg::Q_ADDRESS] = t.q.buffer()->address();
    reader_args[dis::reader_common_arg::W_ADDRESS] = t.weights.buffer()->address();
    reader_args[dis::reader_common_arg::KEYS_ADDRESS] = t.idx_keys.buffer()->address();
    reader_args[dis::reader_common_arg::START_ADDRESS] = t.start.buffer()->address();
    reader_args[dis::reader_common_arg::NUM_ENTRIES] = t.idx_keys.logical_shape()[2];
    writer_args[dis::writer_common_arg::OUTPUT_ADDRESS] = outputs[0].buffer()->address();
    writer_args[dis::writer_common_arg::SCRATCH_ADDRESS] = outputs[1].buffer()->address();
    writer_args[dis::writer_common_arg::CAND_OUTPUT_ADDRESS] = outputs[2].buffer()->address();
    writer_args[dis::writer_common_arg::CAND_SCRATCH_ADDRESS] = outputs[3].buffer()->address();
}

}  // namespace ttnn::prim
