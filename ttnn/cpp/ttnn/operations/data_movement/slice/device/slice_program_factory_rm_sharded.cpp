// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"

#include <map>
#include <optional>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

namespace {

inline std::vector<std::vector<uint32_t>> group_contiguous_values(std::vector<uint32_t>& values) {
    std::vector<std::vector<uint32_t>> chunks;
    if (values.empty()) {
        return chunks;
    }

    // Contiguous values coalesce into far fewer chunks than there are values, so count the
    // runs up front instead of reserving the worst case
    size_t num_chunks = 1;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] != values[i - 1] + 1) {
            ++num_chunks;
        }
    }
    chunks.reserve(num_chunks);

    // Initialize the first chunk
    std::vector<uint32_t> current_chunk;
    current_chunk.reserve(values.size());
    current_chunk.push_back(values[0]);

    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] == values[i - 1] + 1) {
            current_chunk.push_back(values[i]);
        } else {
            chunks.push_back(current_chunk);
            current_chunk.clear();
            current_chunk.push_back(values[i]);
        }
    }
    // Add the last chunk
    chunks.push_back(std::move(current_chunk));
    return chunks;
}

inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_slice_runtime_args_rm_sharded(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    uint32_t num_cores_unpadded,
    bool row_major,
    uint32_t num_cores_x_unpadded,
    uint32_t num_cores_y_unpadded,
    uint32_t shard_height_unpadded,
    uint32_t shard_height_padded,
    uint32_t num_cores_x_padded,
    uint32_t num_cores_y_padded) {
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_unpadded);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, output_tensor_start);
    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_unpadded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        uint32_t num_sticks_per_core_unpadded = shard_height_unpadded;
        uint32_t num_sticks_per_core_padded = shard_height_padded;

        // figure out the start read stick id for each core, and the start id for each dim
        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        num_sticks_written += num_sticks_per_core_unpadded;

        // stores all sticks id for a core
        std::vector<uint32_t> stick_ids_per_core;
        stick_ids_per_core.reserve(num_sticks_per_core_unpadded);
        uint32_t src_stick_id = start_id;
        for (uint32_t i = 0; i < num_sticks_per_core_unpadded; ++i) {
            stick_ids_per_core.push_back(src_stick_id);
            src_stick_id++;
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks_per_dim[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks_per_dim[j];
                } else {
                    break;
                }
            }
        }

        // figure out the stick id in a shard, and the core id for the stick.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> core_stick_map;
        for (uint32_t i = 0; i < num_sticks_per_core_unpadded; ++i) {
            uint32_t stick_id = stick_ids_per_core[i];
            uint32_t shard_id = stick_id / num_sticks_per_core_padded;
            uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core_padded);

            uint32_t shard_grid_inner_dim = row_major ? num_cores_x_padded : num_cores_y_padded;
            uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
            uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

            uint32_t worker_y_logical = row_major ? shard_grid_outer_dim_id : shard_grid_inner_dim_id;
            uint32_t worker_x_logical = row_major ? shard_grid_inner_dim_id : shard_grid_outer_dim_id;

            if (worker_x_logical < num_cores_x_padded and worker_y_logical < num_cores_y_padded) {
                auto core_physical =
                    device->worker_core_from_logical_core(CoreCoord{worker_x_logical, worker_y_logical});
                // save stick id in a shard, and core coord into a map
                std::pair<uint32_t, uint32_t> xy_pair = row_major ? std::make_pair(core_physical.y, core_physical.x)
                                                                  : std::make_pair(core_physical.x, core_physical.y);
                core_stick_map[xy_pair].push_back(stick_id_in_shard);
            }
        }

        // reader rt args
        std::vector<uint32_t> reader_kernel_args;
        reader_kernel_args.reserve(1 + 3 * core_stick_map.size() + 2 * num_sticks_per_core_unpadded);
        reader_kernel_args.push_back(core_stick_map.size());  // num_cores

        for (const auto& core_stick_pair : core_stick_map) {
            auto xy_pair = core_stick_pair.first;
            if (row_major) {
                reader_kernel_args.push_back(xy_pair.second);  // noc x
                reader_kernel_args.push_back(xy_pair.first);   // noc y
            } else {
                reader_kernel_args.push_back(xy_pair.first);   // noc x
                reader_kernel_args.push_back(xy_pair.second);  // noc y
            }
        }

        // coalesce the sticks into chunks
        std::vector<std::vector<std::vector<uint32_t>>> stick_chunks_per_core;
        stick_chunks_per_core.reserve(core_stick_map.size());
        for (auto core_stick_pair : core_stick_map) {
            auto stick_chunks = group_contiguous_values(core_stick_pair.second);
            reader_kernel_args.push_back(stick_chunks.size());  // num_chunks for current core

            stick_chunks_per_core.push_back(std::move(stick_chunks));
        }
        for (const auto& stick_chunks : stick_chunks_per_core) {
            for (auto chunk : stick_chunks) {
                reader_kernel_args.push_back(chunk[0]);      // start id of a chunk
                reader_kernel_args.push_back(chunk.size());  // length of a chunk
            }
        }

        std::vector<uint32_t> writer_kernel_args;
        ret_val[i] = {std::move(reader_kernel_args), std::move(writer_kernel_args)};
    }

    return ret_val;
}

}  // namespace

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SliceRmShardedProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    ProgramDescriptor desc;

    [[maybe_unused]] uint32_t num_padded_sticks = input.physical_volume() / input.padded_shape()[-1];
    [[maybe_unused]] uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    uint32_t W_unpadded = output.logical_shape()[-1];
    auto stick_size_unpadded = W_unpadded * output.element_size();

    // input shard spec
    auto shard_spec_padded = input.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];

    [[maybe_unused]] auto& all_cores_padded = shard_spec_padded.grid;
    [[maybe_unused]] uint32_t num_cores_padded = shard_spec_padded.num_cores();
    auto bbox_padded = shard_spec_padded.grid.bounding_box();
    CoreCoord grid_size_padded = {bbox_padded.end_coord.x + 1, bbox_padded.end_coord.y + 1};
    uint32_t num_cores_x_padded = grid_size_padded.x;
    uint32_t num_cores_y_padded = grid_size_padded.y;

    if (args.sub_core_grids.has_value()) {
        log_warning(tt::LogOp, "sub_core_grids is not used when input tensor is sharded");
    }

    log_debug(tt::LogOp, "num_padded_sticks: {}", num_padded_sticks);
    log_debug(tt::LogOp, "shard_height_padded: {}", shard_height_padded);
    log_debug(tt::LogOp, "all_cores_padded: {}", all_cores_padded);
    log_debug(tt::LogOp, "num_cores_padded: {}", num_cores_padded);

    // output shard spec
    auto shard_spec_unpadded = output.shard_spec().value();
    uint32_t shard_height_unpadded = shard_spec_unpadded.shape[0];
    bool row_major = shard_spec_unpadded.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores_unpadded = shard_spec_unpadded.grid;
    uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
    auto bbox_unpadded = all_cores_unpadded.bounding_box();
    CoreCoord grid_size_unpadded = {bbox_unpadded.end_coord.x + 1, bbox_unpadded.end_coord.y + 1};
    uint32_t num_cores_x_unpadded = grid_size_unpadded.x;
    uint32_t num_cores_y_unpadded = grid_size_unpadded.y;

    log_debug(tt::LogOp, "num_unpadded_sticks: {}", num_unpadded_sticks);
    log_debug(tt::LogOp, "shard_height_unpadded: {}", shard_height_unpadded);
    log_debug(tt::LogOp, "all_cores_unpadded: {}", all_cores_unpadded);
    log_debug(tt::LogOp, "num_cores_unpadded: {}", num_cores_unpadded);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // Real per-row L1 stride is aligned_page_size(), not the compact payload (differs when W·E % 16 != 0).
    const uint32_t src_stride_bytes = input.buffer()->aligned_page_size();
    const uint32_t dst_stride_bytes = output.buffer()->aligned_page_size();
    const uint32_t begins_bytes = args.slice_start[-1] * input.element_size();
    TT_FATAL(
        begins_bytes % ::hal::get_l1_alignment() == 0,
        "SliceRmShardedProgramFactory: width-begin ({} bytes) must be L1-aligned.",
        begins_bytes);

    // Sharded CBs: total_size and page_size vary with shard shape / element size,
    // so padded_shape is folded into compute_program_hash() to keep each unique
    // sizing in its own cache entry.  On cache hit, the framework copies runtime
    // args and patches dynamic CB addresses (.buffer is set below); CB sizing
    // itself is not re-applied — it is carried by the cached descriptor.
    // CB order here (src0, then c_16) is mirrored positionally by override_runtime_arguments; keep in sync.
    constexpr uint8_t src0_cb_index = 0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = shard_height_padded * src_stride_bytes,
        .core_ranges = all_cores_unpadded,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = src_stride_bytes,
        }}},
        .buffer = input.buffer(),
    });

    constexpr uint8_t output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = shard_height_unpadded * dst_stride_bytes,
        .core_ranges = all_cores_unpadded,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = dst_cb_data_format,
            .page_size = dst_stride_bytes,
        }}},
        .buffer = output.buffer(),
    });

    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(stick_size_unpadded),
        static_cast<uint32_t>(shard_height_unpadded),
        src_stride_bytes,
        dst_stride_bytes,
        begins_bytes};

    auto all_runtime_args = ttnn::operations::data_movement::get_slice_runtime_args_rm_sharded(
        input,
        output,
        args.slice_start,
        num_cores_unpadded,
        row_major,
        num_cores_x_unpadded,
        num_cores_y_unpadded,
        shard_height_unpadded,
        shard_height_padded,
        num_cores_x_padded,
        num_cores_y_padded);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_reader_unary_unpad_dims_rm_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores_unpadded;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    reader_desc.runtime_args.reserve(num_cores_unpadded);
    for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        reader_desc.runtime_args.emplace_back(core, std::move(all_runtime_args[i].first));
    }

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

// Re-point every per-dispatch address in a cached slice program, for the factory that built it.
// Shared with MeshPartition, which drives these same factories directly, so the slot layout has one
// home. Every shape-derived arg is keyed (both tensor specs, the slice params and factory.index() are
// folded into compute_program_hash), so addresses are all that move on a hit.
void patch_slice_program_addresses(
    tt::tt_metal::Program& program,
    const SliceDeviceOperation::program_factory_t& factory,
    const SliceParams& operation_attributes,
    const SliceInputs& tensor_args,
    Tensor& output) {
    // Height-sharded RM is CB-bound: the reader args are all keyed, so only the two sharded CB
    // addresses move. CBs are matched positionally -- src0, then c_16.
    if (std::holds_alternative<SliceRmShardedProgramFactory>(factory)) {
        tt::tt_metal::ProgramDescriptor cb_addr_only;
        cb_addr_only.cbs.push_back(tt::tt_metal::CBDescriptor{.buffer = tensor_args.input.buffer()});
        cb_addr_only.cbs.push_back(tt::tt_metal::CBDescriptor{.buffer = output.buffer()});
        tt::tt_metal::apply_descriptor_runtime_args(program, cb_addr_only);
        return;
    }

    // A slot holding 0 belongs to a core create_descriptor left zero-filled; leave those alone.
    constexpr uint32_t kReaderKernelIdx = 0, kWriterKernelIdx = 1;
    const auto patch_slot0 = [&program](uint32_t kernel_idx, uint32_t addr) {
        for (auto& col : tt::tt_metal::GetRuntimeArgs(program, kernel_idx)) {
            for (auto& a : col) {
                if (a.size() > 0 && a[0] != 0) {
                    a[0] = addr;
                }
            }
        }
    };
    patch_slot0(kWriterKernelIdx, output.buffer()->address());

    std::visit(
        [&](auto&& f) {
            using Factory = std::decay_t<decltype(f)>;
            if constexpr (std::is_same_v<Factory, SliceRmProgramFactory>) {
                // The reader reads from base+offset, which no Buffer* binding can express; reuse the
                // helper create_descriptor calls so the emitted value cannot drift.
                const auto dynamic_args = slice_rm_reader_dynamic_args(operation_attributes, tensor_args, output);
                tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
            } else if constexpr (std::is_same_v<Factory, SliceRmStrideProgramFactory>) {
                patch_slot0(kReaderKernelIdx, tensor_args.input.buffer()->address());
            } else if constexpr (
                std::is_same_v<Factory, SliceTileProgramFactory> ||
                std::is_same_v<Factory, SliceTileTensorArgsProgramFactory>) {
                // Divergent-partition hit leaves writer num_pages=0 -> all-zero output (#52651).
                std::vector<tt::tt_metal::DynamicRuntimeArg> dyn{
                    {kReaderKernelIdx, {}, 0, tensor_args.input.buffer()->address(), true}};
                if constexpr (std::is_same_v<Factory, SliceTileTensorArgsProgramFactory>) {
                    dyn.push_back(
                        {kReaderKernelIdx, {}, 1, tensor_args.start_tensor.value().buffer()->address(), true});
                    dyn.push_back({kReaderKernelIdx, {}, 2, tensor_args.end_tensor.value().buffer()->address(), true});
                }
                tt::tt_metal::apply_dynamic_runtime_args(program, dyn);

                const uint32_t start_offset = std::is_same_v<Factory, SliceTileProgramFactory>
                                                  ? ttnn::operations::data_movement::get_tiled_start_offset(
                                                        tensor_args.input, operation_attributes.slice_start)
                                                  : 0u;
                const auto per_core = slice_tile_dynamic_args(
                    operation_attributes, tensor_args, output, start_offset, kReaderKernelIdx, kWriterKernelIdx);
                tt::tt_metal::apply_dynamic_runtime_args(program, per_core);
            }
        },
        factory);
}

void SliceRmShardedProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SliceParams& args,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    patch_slice_program_addresses(program, SliceRmShardedProgramFactory{}, args, tensor_args, output);
}

}  // namespace ttnn::prim
