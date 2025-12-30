// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_sharded_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>

#include <map>
#include <set>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {
struct AddressPair {
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t src_offset = 0;
    uint32_t dst_offset = 0;
};
}  // namespace

TransposeHCShardedTiledProgramFactory::cached_program_t TransposeHCShardedTiledProgramFactory::create(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    Program program = CreateProgram();

    uint32_t N = input_tensor.logical_shape()[0];
    uint32_t C = input_tensor.logical_shape()[1];
    uint32_t H = input_tensor.logical_shape()[2];
    uint32_t W = input_tensor.logical_shape()[3];

    bool is_row_major = input_tensor.layout() == Layout::ROW_MAJOR;
    uint32_t page_shape[2] = {1, 1};
    auto tile = input_tensor.tensor_spec().tile();
    auto face_shape = tile.get_face_shape();
    if (is_row_major) {
        page_shape[1] = W;
    } else {
        page_shape[0] = tile.get_height();
        page_shape[1] = tile.get_width();
    }
    uint32_t Ct = tt::div_up(C, page_shape[0]);
    uint32_t Ht = tt::div_up(H, page_shape[0]);
    uint32_t Wt = tt::div_up(W, page_shape[1]);

    auto input_worker_cores = input_tensor.buffer()->buffer_distribution_spec()->cores_with_data();
    auto output_worker_cores = output_tensor.buffer()->buffer_distribution_spec()->cores_with_data();
    const auto output_page_mapping = output_tensor.buffer()->buffer_distribution_spec()->compute_page_mapping();
    const auto input_page_mapping = input_tensor.buffer()->buffer_distribution_spec()->compute_page_mapping();
    const auto& input_mapped_cores = input_page_mapping.all_cores;
    const auto& output_mapped_cores = output_page_mapping.all_cores;
    bool input_cores_are_workers = true;
    const auto& worker_cores = (input_cores_are_workers) ? input_worker_cores : output_worker_cores;

    std::vector<CoreRange> worker_core_ranges;
    worker_core_ranges.reserve(worker_cores.size());
    for (const auto& coord : worker_cores) {
        worker_core_ranges.emplace_back(coord, coord);
    }
    auto worker_core_rangeset = CoreRangeSet(worker_core_ranges);

    auto reader_kernel_config0 = tt::tt_metal::ReaderDataMovementConfig{};
    reader_kernel_config0.defines["IS_ROW_MAJOR"] = is_row_major ? "1" : "0";
    reader_kernel_config0.defines["IS_READER"] = input_cores_are_workers ? "0" : "1";
    reader_kernel_config0.compile_args = {
        input_tensor.buffer()->aligned_page_size(), output_tensor.buffer()->aligned_page_size(), N, C, H, W};

    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_kernel_config0.compile_args);

    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_kernel_config0.compile_args);

    KernelHandle reader_kernel_id0 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_sharded_tile.cpp",
        worker_core_rangeset,
        reader_kernel_config0);

    auto reader_kernel_config1 = tt::tt_metal::WriterDataMovementConfig{};
    reader_kernel_config1.defines["IS_ROW_MAJOR"] = is_row_major ? "1" : "0";
    reader_kernel_config1.defines["IS_READER"] = input_cores_are_workers ? "0" : "1";
    reader_kernel_config1.compile_args = {
        input_tensor.buffer()->aligned_page_size(), output_tensor.buffer()->aligned_page_size(), N, C, H, W};

    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_kernel_config1.compile_args);

    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_kernel_config1.compile_args);

    KernelHandle reader_kernel_id1 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_sharded_tile.cpp",
        worker_core_rangeset,
        reader_kernel_config1);

    auto& worker_mapped_cores = (input_cores_are_workers) ? input_mapped_cores : output_mapped_cores;
    auto& remote_mapped_cores = (input_cores_are_workers) ? output_mapped_cores : input_mapped_cores;
    auto& worker_page_mapping = (input_cores_are_workers) ? input_page_mapping : output_page_mapping;
    auto& remote_page_mapping = (input_cores_are_workers) ? output_page_mapping : input_page_mapping;
    auto remote_num_pages =
        (input_cores_are_workers) ? output_tensor.buffer()->num_pages() : input_tensor.buffer()->num_pages();
    if (input_cores_are_workers) {
        std::swap(C, H);
        std::swap(Ct, Ht);
    }

    std::vector<uint32_t> pages_to_core_map(remote_num_pages);
    for (size_t core_idx = 0; core_idx < remote_mapped_cores.size(); ++core_idx) {
        const auto& host_page_indices = remote_page_mapping.core_host_page_indices[core_idx];
        for (const auto& page_index : host_page_indices) {
            if (page_index != UncompressedBufferPageMapping::PADDING) {
                pages_to_core_map[page_index] = core_idx;
            }
        }
    }

    for (uint32_t core_idx = 0; core_idx < worker_mapped_cores.size(); ++core_idx) {
        const auto& core = worker_mapped_cores[core_idx];
        const auto& host_page_indices = worker_page_mapping.core_host_page_indices[core_idx];
        const uint32_t num_pages = host_page_indices.size();
        std::vector<std::vector<AddressPair>> cores_send(remote_mapped_cores.size());
        bool has_data = false;
        for (const auto& page_index : host_page_indices) {
            if (page_index != UncompressedBufferPageMapping::PADDING) {
                has_data = true;
                uint32_t n = page_index / (H * Ct * Wt);
                uint32_t h = page_index / (Ct * Wt) % H;
                uint32_t c = page_index / Wt % Ct;
                uint32_t w = page_index % Wt;
                if (is_row_major) {
                    uint32_t transposed_tile_id = n * C * Ht * Wt + c * Ht * Wt + h * Wt + w;
                    cores_send[pages_to_core_map[transposed_tile_id]].push_back({page_index, transposed_tile_id});
                } else {
                    uint32_t h_tile = h / page_shape[0];
                    uint32_t h_row = h % page_shape[0];
                    uint32_t tranpose_offset =
                        h_row * face_shape[1] + (h_row / face_shape[0]) * face_shape[0] * face_shape[1];
                    c *= page_shape[0];
                    uint32_t transposed_tile_id = n * C * Ht * Wt + c * Ht * Wt + h_tile * Wt + w;
                    for (uint32_t c_row = 0; c_row < std::min(C - c, page_shape[0]); ++c_row) {
                        uint32_t current_offset =
                            c_row * face_shape[1] + (c_row / face_shape[0]) * face_shape[0] * face_shape[1];
                        if (w == 0) {
                            cores_send[pages_to_core_map[transposed_tile_id]].push_back(
                                {page_index, transposed_tile_id, current_offset * 2, tranpose_offset * 2});
                        }
                        transposed_tile_id += Ht * Wt;
                    }
                }
            }
        }

        if (has_data) {
            std::vector<uint32_t> cores_with_needed_data;
            for (size_t i = 0; i < cores_send.size(); ++i) {
                if (!cores_send[i].empty()) {
                    cores_with_needed_data.push_back(i);
                }
            }

            std::vector<uint32_t> reader_rt_args0 = {
                input_tensor.buffer()->address(), output_tensor.buffer()->address(), core_idx, 0, num_pages / 2, 0};

            reader_rt_args0.push_back(cores_with_needed_data.size() / 2);
            for (uint32_t core_d = 0; core_d < cores_with_needed_data.size() / 2; ++core_d) {
                uint32_t core_idx = cores_with_needed_data[core_d];
                const auto& pairs = cores_send[core_idx];
                reader_rt_args0.push_back(pairs.size());
                for (const auto& page_pair : pairs) {
                    reader_rt_args0.push_back(page_pair.src_addr);
                    reader_rt_args0.push_back(page_pair.dst_addr);
                    if (!is_row_major) {
                        reader_rt_args0.push_back(page_pair.src_offset);
                        reader_rt_args0.push_back(page_pair.dst_offset);
                    }
                }
            }

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id0, {core}, reader_rt_args0);

            std::vector<uint32_t> reader_rt_args1 = {
                input_tensor.buffer()->address(),
                output_tensor.buffer()->address(),
                core_idx,
                num_pages / 2,
                num_pages,
                0};

            reader_rt_args1.push_back(cores_with_needed_data.size() - cores_with_needed_data.size() / 2);
            for (uint32_t core_d = cores_with_needed_data.size() / 2; core_d < cores_with_needed_data.size();
                 ++core_d) {
                uint32_t core_idx = cores_with_needed_data[core_d];
                const auto& pairs = cores_send[core_idx];
                reader_rt_args1.push_back(pairs.size());
                for (const auto& page_pair : pairs) {
                    reader_rt_args1.push_back(page_pair.src_addr);
                    reader_rt_args1.push_back(page_pair.dst_addr);
                    if (!is_row_major) {
                        reader_rt_args1.push_back(page_pair.src_offset);
                        reader_rt_args1.push_back(page_pair.dst_offset);
                    }
                }
            }

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id1, {core}, reader_rt_args1);
        }
    }

    return {
        std::move(program),
        {.reader_kernel_id0 = reader_kernel_id0, .reader_kernel_id1 = reader_kernel_id1, .worker_cores = worker_cores}};
}

void TransposeHCShardedTiledProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TransposeParams& /*operation_attributes*/,
    const TransposeInputs& tensor_args,
    Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    auto* const src_buffer = tensor_args.input.buffer();
    auto* const dst_buffer = output_tensor.buffer();

    auto& reader_runtime_args0 = GetRuntimeArgs(program, shared_variables.reader_kernel_id0);
    auto& reader_runtime_args1 = GetRuntimeArgs(program, shared_variables.reader_kernel_id1);
    for (const auto& core : shared_variables.worker_cores) {
        auto& worker_reader_runtime_args0 = reader_runtime_args0[core.x][core.y];
        auto& worker_reader_runtime_args1 = reader_runtime_args1[core.x][core.y];
        worker_reader_runtime_args0[0] = src_buffer->address();
        worker_reader_runtime_args0[1] = dst_buffer->address();
        worker_reader_runtime_args1[0] = src_buffer->address();
        worker_reader_runtime_args1[1] = dst_buffer->address();
    }
}

}  // namespace ttnn::prim
