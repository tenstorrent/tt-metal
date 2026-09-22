// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_backward_program_factory.hpp"

#include <map>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "unary_backward_device_operation_types.hpp"
#include "unary_backward_op_utils.hpp"

namespace ttnn::operations::unary_backward {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

ProgramDescriptor UnaryBackwardProgramFactory::create_descriptor(
    const UnaryBackwardParams& args, const UnaryBackwardInputs& tensor_args, Tensor& output) {
    const auto& grad_output = tensor_args.grad_output;
    const auto& input = tensor_args.input;

    const UnaryBackwardKernelSpec& spec = get_kernel_spec(args.op_type);

    // c_0 carries grad_output and c_1 carries input, in both the CB formats and the buffers
    // bound to them. Kernels index the operands by these buffer indices, so the pairing is
    // part of the contract in UnaryBackwardKernelSpec, not a detail of one op.
    const DataFormat grad_output_cb_data_format = datatype_to_dataformat_converter(grad_output.dtype());
    const uint32_t grad_output_single_tile_size = tile_size(grad_output_cb_data_format);
    const DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tile_size(input_cb_data_format);
    const DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tile_size(output_cb_data_format);

    const uint32_t num_tiles = input.physical_volume() / TILE_HW;

    // Two modes, following eltwise unary (see is_native_L1_sharding / get_shard_specs in
    // unary/common/unary_utils.cpp):
    //
    //  1. ALIAS. Every operand and the output are L1-sharded on the same grid with the same
    //     shard shape, so each circular buffer can be bound to its tensor's own buffer and the
    //     dataflow kernels copy nothing -- the shard already is the CB.
    //  2. ADDRESS. Anything else: bind nothing and let TensorAccessor address every operand by
    //     logical page. A sharded buffer is fully addressable that way, because the accessor
    //     encodes the shard mapping, so this covers DRAM-sharded operands and any mix of
    //     sharded and interleaved.
    //
    // Mixing the two is what must not happen. A bound CB presents the shard in PHYSICAL order
    // while an addressed operand is read in LOGICAL order; for a height shard those coincide,
    // but for a width or block shard they do not, and the operands end up misaligned -- silently,
    // measured at PCC 0.25 and 0.50 before this rule existed.
    const auto shard_of = [](const Tensor& t) { return t.memory_config().shard_spec(); };
    const auto is_uneven_shard = [](const Tensor& t) {
        if (!t.is_sharded()) {
            return false;
        }
        const auto& shape = t.padded_shape();
        const auto& shard = t.memory_config().shard_spec()->shape;
        uint64_t volume_except_last = 1;
        for (int i = 0; i < static_cast<int>(shape.rank()) - 1; ++i) {
            volume_except_last *= shape[i];
        }
        return (volume_except_last % shard[0]) != 0 || (shape[-1] % shard[1]) != 0;
    };
    const auto is_l1_sharded = [](const Tensor& t) {
        return t.is_sharded() && t.memory_config().buffer_type() == BufferType::L1;
    };

    const bool can_alias_shards = [&]() {
        if (!(is_l1_sharded(grad_output) && is_l1_sharded(input) && is_l1_sharded(output))) {
            return false;
        }
        if (is_uneven_shard(grad_output) || is_uneven_shard(input) || is_uneven_shard(output)) {
            return false;
        }
        // By value: memory_config() returns a temporary, so a reference into its shard_spec
        // would dangle.
        const auto a = *shard_of(grad_output);
        const auto b = *shard_of(input);
        const auto c = *shard_of(output);
        return a.grid == b.grid && b.grid == c.grid && a.shape == b.shape && b.shape == c.shape;
    }();

    const bool grad_output_sharded = can_alias_shards;
    const bool input_sharded = can_alias_shards;
    const bool output_sharded = can_alias_shards;
    const bool any_sharded = can_alias_shards;

    // Tiles in one shard, for whichever operand is sharded. All sharded operands share a shape
    // and a grid here (validation pins the padded shapes equal), so one figure serves them all.
    const auto shard_tiles_of = [](const Tensor& t) -> uint32_t {
        const auto& shard_shape = t.memory_config().shard_spec()->shape;
        return (shard_shape[0] / TILE_HEIGHT) * (shard_shape[1] / TILE_WIDTH);
    };
    const uint32_t shard_tiles = grad_output_sharded ? shard_tiles_of(grad_output)
                                 : input_sharded     ? shard_tiles_of(input)
                                 : output_sharded    ? shard_tiles_of(output)
                                                     : 0;

    IDevice* device = input.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Work split: for a sharded program the cores are dictated by the shard grid -- each core
    // owns exactly its own shard -- so split_work_to_cores (which picks its own grid from a tile
    // count) must not be used.
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1, core_group_2;
    uint32_t num_cores = 0, num_tiles_per_core_group_1 = 0, num_tiles_per_core_group_2 = 0;
    if (any_sharded) {
        const auto& shard_grid = grad_output_sharded ? grad_output.memory_config().shard_spec()->grid
                                 : input_sharded     ? input.memory_config().shard_spec()->grid
                                                     : output.memory_config().shard_spec()->grid;
        all_cores = shard_grid;
        core_group_1 = shard_grid;
        num_cores = shard_grid.num_cores();
        num_tiles_per_core_group_1 = shard_tiles;
    } else {
        auto split = split_work_to_cores(compute_with_storage_grid_size, num_tiles);
        num_cores = std::get<0>(split);
        all_cores = std::get<1>(split);
        core_group_1 = std::get<2>(split);
        core_group_2 = std::get<3>(split);
        num_tiles_per_core_group_1 = std::get<4>(split);
        num_tiles_per_core_group_2 = std::get<5>(split);
    }

    // A globally allocated CB must hold a whole shard; an interleaved one only needs its double
    // buffer. Applied per operand at the push_cb calls below.
    constexpr uint32_t grad_output_cb_index = CBIndex::c_0;
    constexpr uint32_t input_cb_index = CBIndex::c_1;
    constexpr uint32_t grad_input_cb_index = CBIndex::c_2;

    ProgramDescriptor desc;

    // ---- Circular buffers ----

    auto* grad_output_buffer = grad_output.buffer();
    auto* input_buffer = input.buffer();
    auto* grad_input_buffer = output.buffer();

    // `buffer` set means a globally allocated CB: the CB is the tensor's own shard in L1 rather
    // than a staging area, so the dataflow kernel copies nothing for that operand. Left null for
    // an interleaved operand, which keeps its double buffer.
    const auto push_cb =
        [&desc, &all_cores](
            uint32_t index, DataFormat data_format, uint32_t tile_size, uint32_t tiles, Buffer* shard_buffer) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = tiles * tile_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(index),
                    .data_format = data_format,
                    .page_size = tile_size,
                }}},
                .buffer = shard_buffer,
            });
        };

    push_cb(
        grad_output_cb_index,
        grad_output_cb_data_format,
        grad_output_single_tile_size,
        grad_output_sharded ? shard_tiles : 2,
        grad_output_sharded ? grad_output_buffer : nullptr);
    push_cb(
        input_cb_index,
        input_cb_data_format,
        input_single_tile_size,
        input_sharded ? shard_tiles : 2,
        input_sharded ? input_buffer : nullptr);
    push_cb(
        grad_input_cb_index,
        output_cb_data_format,
        output_single_tile_size,
        output_sharded ? shard_tiles : 2,
        output_sharded ? grad_input_buffer : nullptr);

    // ---- Reader / writer kernels ----
    //
    // Both are the generic interleaved dataflow kernels already used by the forward ops: a
    // unary gradient reads two same-shaped operands and writes one, which is exactly the
    // binary reader's and the unary writer's contract.

    // The reader declares a TensorAccessor only for an operand it actually reads, so the
    // compile-time arg list must carry exactly the accessors the defines leave enabled --
    // appending one for a sharded operand would shift the other's offset.
    std::vector<uint32_t> reader_compile_time_args = {0};
    if (!grad_output_sharded) {
        TensorAccessorArgs(*grad_output_buffer).append_to(reader_compile_time_args);
    }
    if (!input_sharded) {
        TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
    }

    std::map<std::string, std::string> reader_defines;
    if (grad_output_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    }
    if (input_sharded) {
        reader_defines["IN1_SHARDED"] = "1";
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reader_defines.begin(), reader_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    // The writer declares its accessor unconditionally, so it is always appended; under
    // OUT_SHARDED the kernel just waits on the CB and never uses it.
    std::vector<uint32_t> writer_compile_time_args = {grad_input_cb_index};
    TensorAccessorArgs(*grad_input_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> writer_defines;
    if (output_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = {writer_defines.begin(), writer_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};

    // ---- Compute kernel ----
    //
    // A gradient chains several SFPU steps in DEST, so a float32 operand has to stay float32
    // all the way through: accumulate in fp32 if any operand or the output is fp32, and only
    // then ask the unpacker for fp32 DEST values, per operand that actually is fp32.
    const bool fp32_dest_acc_en = spec.force_fp32_dest_acc || (grad_output_cb_data_format == DataFormat::Float32) ||
                                  (input_cb_data_format == DataFormat::Float32) ||
                                  (output_cb_data_format == DataFormat::Float32);

    // Ask the unpacker for float32 DEST values only where they can actually be held: when DEST
    // is accumulating in float32. Requesting it otherwise would widen values into a DEST that
    // cannot represent them.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[grad_output_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[input_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    // Supplied to every kernel whether or not it copies between DEST slots or packs bfloat16:
    // both are properties of the program the factory built, so a kernel that needs them must
    // not have to re-derive them, and one that does not simply leaves the define unused.
    std::map<std::string, std::string> compute_defines;
    // A kernel only needs the unpacker to switch format between the two operand buffers when the
    // operands actually carry different formats; otherwise the single configuration
    // compute_kernel_hw_startup() installs covers both, and reconfiguring per tile transition is
    // measurable overhead on the common same-dtype path (~2% for bfloat16). Both operand dtypes
    // are part of compute_program_hash, so a program built for one pairing is never replayed for
    // another and the kernel can make this a compile-time decision.
    if (grad_output_cb_data_format != input_cb_data_format) {
        compute_defines["MIXED_OPERAND_DATA_FORMATS"] = "1";
    }
    compute_defines["COPY_DEST_DATA_FORMAT"] = fp32_dest_acc_en ? "DataFormat::Float32" : "DataFormat::Float16_b";
    compute_defines["BF16_ROUNDING_MODE"] = (output.dtype() == DataType::BFLOAT16)
                                                ? "ckernel::DstRoundingMode::NearestEven"
                                                : "ckernel::DstRoundingMode::Default";

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(spec.compute_kernel_path);
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = spec.math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
    };
    compute_desc.defines = {compute_defines.begin(), compute_defines.end()};

    // ---- Per-core runtime args ----

    // Core enumeration. For an interleaved program the linear-index-to-coordinate mapping below
    // has to agree with split_work_to_cores, which walks columns of the compute grid. For a
    // sharded program it instead has to agree with the SHARD's own core order, which follows the
    // shard spec's orientation -- a linear mapping through the compute grid's height silently
    // hands each core another core's tile range. That produced correct results only where the
    // shard grid happened to be one compute-grid column; a full 8x8 grid came out at PCC 0.13.
    std::vector<CoreCoord> cores;
    if (any_sharded) {
        const auto& shard_spec = grad_output_sharded ? grad_output.memory_config().shard_spec()
                                 : input_sharded     ? input.memory_config().shard_spec()
                                                     : output.memory_config().shard_spec();
        cores = corerange_to_cores(all_cores, num_cores, shard_spec->orientation == ShardOrientation::ROW_MAJOR);
    } else {
        cores.reserve(num_cores);
        for (uint32_t i = 0; i < num_cores; i++) {
            cores.push_back(CoreCoord{i / num_cores_y, i % num_cores_y});
        }
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = cores[i];
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(
            core, {grad_output_buffer, input_buffer, num_tiles_per_core, num_tiles_written, 0u, 0u, num_cores_y});
        compute_desc.emplace_runtime_args(core, {num_tiles_per_core});
        writer_desc.emplace_runtime_args(core, {grad_input_buffer, num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::unary_backward
