// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_backward_program_factory.hpp"

#include <algorithm>
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
    // is_sharded() is also true for ND_SHARDED, and an ND distribution that has no legacy
    // equivalent (CONTIGUOUS_1D) deliberately carries no legacy shard_spec at all -- see
    // TensorSpec::populate_legacy_shard_spec_from_nd. So every use of the legacy spec has to be
    // guarded: a tensor without one can only take the addressing path.
    const auto shard_of = [](const Tensor& t) { return t.memory_config().shard_spec(); };
    const auto has_legacy_shard = [&](const Tensor& t) { return t.is_sharded() && shard_of(t).has_value(); };
    const auto is_uneven_shard = [&](const Tensor& t) {
        if (!has_legacy_shard(t)) {
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
    const auto is_l1_sharded = [&](const Tensor& t) {
        return has_legacy_shard(t) && t.memory_config().buffer_type() == BufferType::L1;
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
        // The WHOLE ShardSpec must match, orientation included. Orientation decides which
        // logical shard lands on which core, and the runtime-argument enumeration below reads
        // one operand's orientation for all of them -- so aliasing specs that agree on grid and
        // shape but differ in orientation pairs up different logical shards and computes
        // finite, misordered gradients.
        if (!(a == b && b == c)) {
            return false;
        }
        // A row-major shard is consumed as tile-sized CB pages, so it has to be a whole number of
        // them, and dense: a row padded out to the buffer alignment leaves gaps in the shard, and
        // those gaps sit at different element positions for operands of different dtypes.
        if (input.layout() == Layout::ROW_MAJOR) {
            if ((static_cast<uint64_t>(a.shape[0]) * a.shape[1]) % TILE_HW != 0) {
                return false;
            }
            for (const Tensor* t : std::initializer_list<const Tensor*>{&grad_output, &input, &output}) {
                if (t->buffer()->page_size() != t->buffer()->aligned_page_size()) {
                    return false;
                }
            }
        }
        return true;
    }();

    // Aliasing is all-or-nothing across the three tensors -- see the rule above -- so a single
    // flag describes every operand; there is no per-operand sharded state.
    const bool alias = can_alias_shards;

    // ROW_MAJOR, following eltwise unary's RM_INTERLEAVED path. When the shards cannot be aliased
    // a row-major tensor is read and written in BLOCKS rather than tiles: rows_per_tile narrow rows
    // packed into one tile-sized CB page, or a row wider than a tile split into chunks_per_row
    // tile-sized chunks. Eltwise compute is position-independent and a page round-trips through
    // the same face layout on unpack and pack, so the compute kernel needs no change. An aliased
    // row-major shard needs none of this: its pages are consumed in place like a tiled shard's.
    const bool row_major = input.layout() == Layout::ROW_MAJOR;
    const bool rm_blocked = row_major && !alias;

    // Row geometry. The element geometry is shared by all three tensors because validation pins
    // their shapes equal; the paging is per tensor, since a width- or block-sharded row-major buffer
    // pages each row by shard width while an interleaved one pages it whole, and the dtypes (hence
    // element sizes) of grad_output and input may differ.
    struct RmPaging {
        uint32_t page_elements = 0;
        uint32_t pages_per_row = 0;
        uint32_t element_bytes = 0;
        uint32_t alignment = 0;
    };
    uint32_t row_elements = 0;
    uint32_t chunks_per_row = 1;
    uint32_t rows_per_tile = 1;
    uint32_t total_rows = 0;
    RmPaging grad_paging, input_paging, output_paging;
    if (rm_blocked) {
        row_elements = input.padded_shape()[-1];
        total_rows = static_cast<uint32_t>(input.physical_volume() / row_elements);
        chunks_per_row = (row_elements + TILE_HW - 1) / TILE_HW;

        const auto paging_of = [&](const Tensor& t, DataFormat df) {
            const uint32_t element_bytes = datum_size(df);
            const uint32_t page_elements = static_cast<uint32_t>(t.buffer()->page_size()) / element_bytes;
            return RmPaging{
                .page_elements = page_elements,
                .pages_per_row = (row_elements + page_elements - 1) / page_elements,
                .element_bytes = element_bytes,
                .alignment = t.buffer()->alignment()};
        };
        grad_paging = paging_of(grad_output, grad_output_cb_data_format);
        input_paging = paging_of(input, input_cb_data_format);
        output_paging = paging_of(output, output_cb_data_format);

        // Several rows share one CB page only when a row is narrower than half a tile and every
        // row lands at a NoC-aligned CB offset. Unary requires the latter as page_size ==
        // aligned_page_size on its one input; here every tensor has to satisfy it, for its whole
        // row and for each of its pages, since a piece may start at either boundary.
        const auto packs_rows = [&](const Tensor& t, const RmPaging& paging) {
            const uint32_t alignment = t.buffer()->alignment();
            return (row_elements * paging.element_bytes) % alignment == 0 &&
                   (paging.page_elements * paging.element_bytes) % alignment == 0;
        };
        if (row_elements * 2 <= TILE_HW && packs_rows(grad_output, grad_paging) && packs_rows(input, input_paging) &&
            packs_rows(output, output_paging)) {
            rows_per_tile = TILE_HW / row_elements;
        }
    }

    // Units the reader and writer walk: blocks for row-major, tiles otherwise. The compute kernel
    // sees chunks_per_row pages per block.
    const uint32_t num_units = rm_blocked ? (total_rows + rows_per_tile - 1) / rows_per_tile : num_tiles;

    // Pages in one shard. Counted by elements rather than by tile rows and columns so that it holds
    // for a row-major shard too, whose shape need not be a multiple of the tile in either dimension
    // -- only its element count must be (enforced by can_alias_shards).
    const uint32_t shard_tiles = alias ? static_cast<uint32_t>(shard_of(grad_output)->numel() / TILE_HW) : 0;

    IDevice* device = input.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Work split: for an aliased program the cores are dictated by the shard grid -- each core
    // owns exactly its own shard -- so split_work_to_cores (which picks its own grid from a unit
    // count) must not be used.
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1, core_group_2;
    uint32_t num_cores = 0, num_units_per_core_group_1 = 0, num_units_per_core_group_2 = 0;
    if (alias) {
        const auto shard_grid = shard_of(grad_output)->grid;
        all_cores = shard_grid;
        core_group_1 = shard_grid;
        num_cores = shard_grid.num_cores();
        num_units_per_core_group_1 = shard_tiles;
    } else {
        auto split = split_work_to_cores(compute_with_storage_grid_size, num_units);
        num_cores = std::get<0>(split);
        all_cores = std::get<1>(split);
        core_group_1 = std::get<2>(split);
        core_group_2 = std::get<3>(split);
        num_units_per_core_group_1 = std::get<4>(split);
        num_units_per_core_group_2 = std::get<5>(split);
    }

    constexpr uint32_t grad_output_cb_index = CBIndex::c_0;
    constexpr uint32_t input_cb_index = CBIndex::c_1;
    constexpr uint32_t grad_input_cb_index = CBIndex::c_2;

    // An addressed CB is a staging area: two pages, so the reader can fill one while compute
    // drains the other. An aliased CB is the whole shard in place.
    constexpr uint32_t kDoubleBufferPages = 2;
    const uint32_t cb_pages = alias ? shard_tiles : kDoubleBufferPages;

    ProgramDescriptor desc;

    // ---- Circular buffers ----

    auto* grad_output_buffer = grad_output.buffer();
    auto* input_buffer = input.buffer();
    auto* grad_input_buffer = output.buffer();

    // `buffer` set means a globally allocated CB: the CB is the tensor's own shard in L1 rather
    // than a staging area, so the dataflow kernel copies nothing for it.
    const auto push_cb = [&desc, &all_cores, cb_pages](
                             uint32_t index, DataFormat data_format, uint32_t tile_size, Buffer* shard_buffer) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb_pages * tile_size,
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
        alias ? grad_output_buffer : nullptr);
    push_cb(input_cb_index, input_cb_data_format, input_single_tile_size, alias ? input_buffer : nullptr);
    push_cb(grad_input_cb_index, output_cb_data_format, output_single_tile_size, alias ? grad_input_buffer : nullptr);

    // Scratch for the row-major kernels' unaligned pieces (see kernels/dataflow/row_major_pages.hpp):
    // one staging window of at most a chunk plus two alignment units, plus slack for rounding the
    // window's start up to the widest alignment. One each, since the reader and writer run
    // concurrently.
    if (rm_blocked) {
        constexpr uint32_t kMaxAlignment = 64;
        const uint32_t scratch_bytes =
            std::max({grad_output_single_tile_size, input_single_tile_size, output_single_tile_size}) +
            (3 * kMaxAlignment);
        for (uint32_t index : {static_cast<uint32_t>(CBIndex::c_3), static_cast<uint32_t>(CBIndex::c_4)}) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = scratch_bytes,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(index),
                    .data_format = output_cb_data_format,
                    .page_size = scratch_bytes,
                }}},
            });
        }
    }

    // ---- Reader / writer kernels ----

    KernelDescriptor reader_desc;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.config = WriterConfigDescriptor{};

    if (rm_blocked) {
        // Unary's row-major block scheme, with page addressing generalized per tensor -- see
        // kernels/dataflow/row_major_pages.hpp.
        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*grad_output_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/dataflow/"
            "reader_unary_backward_row_major.cpp";
        reader_desc.compile_time_args = reader_compile_time_args;

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*grad_input_buffer).append_to(writer_compile_time_args);
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/dataflow/"
            "writer_unary_backward_row_major.cpp";
        writer_desc.compile_time_args = writer_compile_time_args;
    } else {
        // The reader declares a TensorAccessor only for an operand it actually reads, so the
        // compile-time arg list must carry exactly the accessors the defines leave enabled --
        // appending one for an aliased operand would shift the other's offset.
        std::vector<uint32_t> reader_compile_time_args = {0};
        std::map<std::string, std::string> reader_defines;
        if (alias) {
            reader_defines["IN0_SHARDED"] = "1";
            reader_defines["IN1_SHARDED"] = "1";
        } else {
            TensorAccessorArgs(*grad_output_buffer).append_to(reader_compile_time_args);
            TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
        }
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp";
        reader_desc.compile_time_args = reader_compile_time_args;
        reader_desc.defines = {reader_defines.begin(), reader_defines.end()};

        // The writer declares its accessor unconditionally, so it is always appended; under
        // OUT_SHARDED the kernel just waits on the CB and never uses it.
        std::vector<uint32_t> writer_compile_time_args = {grad_input_cb_index};
        TensorAccessorArgs(*grad_input_buffer).append_to(writer_compile_time_args);
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
        writer_desc.compile_time_args = writer_compile_time_args;
        if (alias) {
            writer_desc.defines = {{"OUT_SHARDED", "1"}};
        }
    }

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

    // Core enumeration. For an addressed program the linear-index-to-coordinate mapping below has
    // to agree with split_work_to_cores, which walks columns of the compute grid. For an aliased
    // program it instead has to agree with the SHARD's own core order, which follows the shard
    // spec's orientation -- a linear mapping through the compute grid's height silently hands
    // each core another core's tile range. That produced correct results only where the shard
    // grid happened to be one compute-grid column; a full 8x8 grid came out at PCC 0.13.
    std::vector<CoreCoord> cores;
    if (alias) {
        cores =
            corerange_to_cores(all_cores, num_cores, shard_of(grad_output)->orientation == ShardOrientation::ROW_MAJOR);
    } else {
        cores.reserve(num_cores);
        for (uint32_t i = 0; i < num_cores; i++) {
            cores.push_back(CoreCoord{i / num_cores_y, i % num_cores_y});
        }
    }

    for (uint32_t i = 0, units_done = 0; i < num_cores; i++) {
        CoreCoord core = cores[i];
        uint32_t units = 0;
        if (core_group_1.contains(core)) {
            units = num_units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units = num_units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        if (rm_blocked) {
            reader_desc.emplace_runtime_args(
                core,
                {units,
                 units_done,
                 row_elements,
                 TILE_HW,
                 chunks_per_row,
                 rows_per_tile,
                 total_rows,
                 grad_output_buffer,
                 grad_paging.page_elements,
                 grad_paging.pages_per_row,
                 grad_paging.element_bytes,
                 grad_paging.alignment,
                 input_buffer,
                 input_paging.page_elements,
                 input_paging.pages_per_row,
                 input_paging.element_bytes,
                 input_paging.alignment});
            writer_desc.emplace_runtime_args(
                core,
                {units,
                 units_done,
                 row_elements,
                 TILE_HW,
                 chunks_per_row,
                 rows_per_tile,
                 total_rows,
                 grad_input_buffer,
                 output_paging.page_elements,
                 output_paging.pages_per_row,
                 output_paging.element_bytes,
                 output_paging.alignment});
            // Each block is chunks_per_row CB pages.
            compute_desc.emplace_runtime_args(core, {units * chunks_per_row});
        } else {
            reader_desc.emplace_runtime_args(
                core, {grad_output_buffer, input_buffer, units, units_done, 0u, 0u, num_cores_y});
            compute_desc.emplace_runtime_args(core, {units});
            writer_desc.emplace_runtime_args(core, {grad_input_buffer, units, units_done});
        }

        units_done += units;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::unary_backward
