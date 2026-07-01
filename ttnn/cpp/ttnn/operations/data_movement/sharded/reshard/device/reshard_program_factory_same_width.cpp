// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_width.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Anonymous-namespace helper unique to reshard same-width to avoid unity-build collisions.
void push_reshard_same_width_cb_pair(
    ProgramDescriptor& desc,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t total_size,
    uint32_t page_size,
    const CoreRangeSet& core_ranges,
    Buffer* bound_buffer) {
    CBDescriptor cb;
    cb.total_size = total_size;
    cb.core_ranges = core_ranges;
    cb.format_descriptors.push_back(CBFormatDescriptor{
        .buffer_index = static_cast<uint8_t>(cb_index),
        .data_format = data_format,
        .page_size = page_size,
    });
    cb.buffer = bound_buffer;
    desc.cbs.push_back(std::move(cb));
}

}  // namespace

template <bool local_is_output>
ProgramDescriptor ReshardSameWidthFactory<local_is_output>::create_descriptor(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    const auto& output = output_tensor;
    const auto& local_tensor = local_is_output ? output : input;
    const auto& remote_tensor = local_is_output ? input : output;

    auto* device = input.device();

    const auto local_shard_spec = local_tensor.shard_spec().value();
    const auto remote_shard_spec = remote_tensor.shard_spec().value();

    auto remote_core_type = remote_tensor.buffer()->core_type();
    constexpr uint32_t cb_index = tt::CBIndex::c_0;
    constexpr uint32_t cb_scratch_index = tt::CBIndex::c_1;
    auto local_cores = get_optimal_worker_cores_for_sharded_tensor(local_tensor);
    auto all_cores = CoreRangeSet(ttsl::Span<const CoreCoord>(local_cores));
    auto remote_cores = remote_tensor.buffer()->buffer_distribution_spec().value().cores_with_data();

    uint32_t unit_size = 0;
    uint32_t local_units_per_shard = 0;
    uint32_t remote_units_per_shard = 0;
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(local_tensor.dtype());

    uint32_t num_units = local_tensor.buffer()->num_pages();
    if (local_tensor.layout() == Layout::TILE) {
        unit_size = tt::tile_size(data_format);
        local_units_per_shard = local_shard_spec.numel() / TILE_HW;
        remote_units_per_shard = remote_shard_spec.numel() / TILE_HW;
    } else {
        unit_size = static_cast<uint32_t>(local_shard_spec.shape[1] * local_tensor.element_size());
        local_units_per_shard = local_shard_spec.shape[0];
        remote_units_per_shard = remote_shard_spec.shape[0];
    }
    uint32_t local_unit_size_padded = tt::align(unit_size, local_tensor.buffer()->alignment());
    uint32_t remote_unit_size_padded = tt::align(unit_size, remote_tensor.buffer()->alignment());
    bool unaligned = false;
    if (remote_unit_size_padded != unit_size || local_unit_size_padded != unit_size) {
        unaligned = true;
    }
    const uint32_t total_size = local_units_per_shard * unit_size;
    const std::string kernel_name =
        local_is_output
            ? "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_width_reader.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_width_writer.cpp";

    bool interface_with_dram = (remote_core_type == tt::CoreType::DRAM);
    auto* local_buffer = local_tensor.buffer();
    auto* remote_buffer = remote_tensor.buffer();
    auto remote_buffer_type = remote_buffer->buffer_type();

    ProgramDescriptor desc;

    // Local sharded CB. Bind to local buffer for dynamic-CB rebinding on cache hits via cb.buffer.
    push_reshard_same_width_cb_pair(
        desc, cb_index, data_format, total_size, unit_size, all_cores, /*bound_buffer=*/local_buffer);

    if (unaligned) {
        // Scratch CB used by kernels when local/remote alignments differ.
        push_reshard_same_width_cb_pair(
            desc,
            cb_scratch_index,
            data_format,
            remote_units_per_shard * remote_unit_size_padded,
            unit_size,
            all_cores,
            /*bound_buffer=*/nullptr);
    }

    // Reader/writer kernels share the same source and compile-time args.
    std::vector<uint32_t> compile_args = {
        cb_index,
        static_cast<uint32_t>(interface_with_dram),
        static_cast<uint32_t>(unaligned),
        unit_size,
        local_unit_size_padded,
        remote_unit_size_padded,
        cb_scratch_index};

    KernelDescriptor reader_desc;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.kernel_source = kernel_name;
    reader_desc.core_ranges = all_cores;
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.compile_time_args = compile_args;

    KernelDescriptor writer_desc;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.kernel_source = kernel_name;
    writer_desc.core_ranges = all_cores;
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.compile_time_args = std::move(compile_args);

    uint32_t remote_core_idx = 0;
    uint32_t remote_core_units_rem = remote_units_per_shard;
    auto bank_id =
        device->allocator()->get_bank_ids_from_logical_core(remote_buffer_type, remote_cores[remote_core_idx])[0];

    std::array<KernelDescriptor*, 2> kernels = {&reader_desc, &writer_desc};
    uint32_t local_units_left = num_units;
    for (const auto& core : local_cores) {
        uint32_t local_units_per_core = std::min(local_units_left, local_units_per_shard);
        local_units_left -= local_units_per_core;
        uint32_t local_units_per_kernel = tt::div_up(local_units_per_core, static_cast<uint32_t>(kernels.size()));
        uint32_t local_start_offset = 0;
        for (auto* kernel : kernels) {
            // arg 0 is remote-buffer base address (binding via Buffer*).
            // RTArgList doesn't expose operator[] for back-patching, so we build
            // a std::vector<variant> here and pass via the vector overload of
            // emplace_runtime_args.
            std::vector<std::variant<uint32_t, Buffer*>> kernel_args;
            kernel_args.emplace_back(remote_buffer);
            kernel_args.emplace_back(uint32_t{0});
            kernel_args.emplace_back(uint32_t{0});
            uint32_t local_units_to_transfer = std::min(local_units_per_core, local_units_per_kernel);
            if (local_units_to_transfer != 0) {
                uint32_t num_transfers = 0;
                kernel_args[1] = local_start_offset;
                local_start_offset += local_units_to_transfer * unit_size;
                while (local_units_to_transfer > 0) {
                    if (remote_core_units_rem == 0) {
                        remote_core_idx++;
                        remote_core_units_rem = remote_units_per_shard;
                        bank_id = device->allocator()->get_bank_ids_from_logical_core(
                            remote_buffer_type, remote_cores[remote_core_idx])[0];
                    }
                    uint32_t units_to_transfer = std::min(remote_core_units_rem, local_units_to_transfer);
                    bank_id = device->allocator()->get_bank_ids_from_logical_core(
                        remote_buffer_type, remote_cores[remote_core_idx])[0];
                    kernel_args.emplace_back(bank_id);
                    kernel_args.emplace_back(
                        (remote_units_per_shard - remote_core_units_rem) * remote_unit_size_padded);
                    kernel_args.emplace_back(units_to_transfer);
                    local_units_per_core -= units_to_transfer;
                    local_units_to_transfer -= units_to_transfer;
                    remote_core_units_rem -= units_to_transfer;
                    num_transfers++;
                }
                kernel_args[2] = num_transfers;
            }
            kernel->emplace_runtime_args(core, kernel_args);
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

// Explicit template instantiations
template struct ReshardSameWidthFactory<true>;
template struct ReshardSameWidthFactory<false>;

}  // namespace ttnn::prim
