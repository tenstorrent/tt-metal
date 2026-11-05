// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_backward_program_factory.hpp"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include "unary_backward_device_operation_types.hpp"
#include "unary_backward_op_utils.hpp"

namespace ttnn::operations::unary_backward {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts UnaryBackwardProgramFactory::create_program_artifacts(
    const UnaryBackwardParams& args, const UnaryBackwardInputs& tensor_args, Tensor& output) {
    const auto& grad_output = tensor_args.grad_output;
    const auto& input = tensor_args.input;

    // The tensor parameters and their run-time arguments are built from the MeshTensors. The
    // geometry below is still read from the ttnn::Tensor wrappers, as before.
    const auto& grad_output_mesh_tensor = grad_output.mesh_tensor();
    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    const UnaryBackwardKernelSpec& spec = get_kernel_spec(args.op_type);

    // The grad_output DFB carries grad_output and the input DFB carries input, in both the DFB
    // formats and the tensors bound to them. Kernels name the operands by these DFB bindings, so
    // the pairing is part of the contract in UnaryBackwardKernelSpec, not a detail of one op.
    const DataFormat grad_output_dfb_data_format = datatype_to_dataformat_converter(grad_output.dtype());
    const uint32_t grad_output_single_tile_size = tile_size(grad_output_dfb_data_format);
    const DataFormat input_dfb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tile_size(input_dfb_data_format);
    const DataFormat output_dfb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tile_size(output_dfb_data_format);

    const uint32_t num_tiles = input.physical_volume() / TILE_HW;

    // Two modes, following eltwise unary (see is_native_L1_sharding / get_shard_specs in
    // unary/common/unary_utils.cpp):
    //
    //  1. ALIAS. Every operand and the output are L1-sharded on the same grid with the same
    //     shard shape, so each dataflow buffer can be bound to its tensor's own buffer and the
    //     dataflow kernels copy nothing -- the shard already is the DFB.
    //  2. ADDRESS. Anything else: bind nothing and let TensorAccessor address every operand by
    //     logical page. A sharded buffer is fully addressable that way, because the accessor
    //     encodes the shard mapping, so this covers DRAM-sharded operands and any mix of
    //     sharded and interleaved.
    //
    // Mixing the two is what must not happen. A bound DFB presents the shard in PHYSICAL order
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
        // ShardSpec equality leaves out the memory layout, which picks the distribution (2D grid
        // for block, round-robin otherwise), so equal specs could still place shards differently.
        const auto layout = grad_output.memory_config().memory_layout();
        if (input.memory_config().memory_layout() != layout || output.memory_config().memory_layout() != layout) {
            return false;
        }
        // Each core gets exactly one shard's work below, so the grid must hold exactly one shard per
        // core. A grid larger than the tensor needs would hand the spare cores work on an L1 region
        // that holds no data.
        const uint64_t shard_elements = static_cast<uint64_t>(a.shape[0]) * a.shape[1];
        if (shard_elements * a.grid.num_cores() != input.physical_volume()) {
            return false;
        }
        // A row-major shard is consumed as tile-sized DFB entries, so it has to be a whole number of
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

    // ROW_MAJOR, following eltwise unary's RM_INTERLEAVED path. When the shards cannot be aliased
    // a row-major tensor is read and written in BLOCKS rather than tiles: rows_per_tile narrow rows
    // packed into one tile-sized DFB entry, or a row wider than a tile split into chunks_per_row
    // tile-sized chunks. Eltwise compute is position-independent and a page round-trips through
    // the same face layout on unpack and pack, so the compute kernel needs no change. An aliased
    // row-major shard needs none of this: its pages are consumed in place like a tiled shard's.
    const bool row_major = input.layout() == Layout::ROW_MAJOR;
    const bool rm_blocked = row_major && !can_alias_shards;

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
        grad_paging = paging_of(grad_output, grad_output_dfb_data_format);
        input_paging = paging_of(input, input_dfb_data_format);
        output_paging = paging_of(output, output_dfb_data_format);

        // Several rows share one DFB entry only when a row is narrower than half a tile and every
        // row lands at a NoC-aligned DFB offset. Unary requires the latter as page_size ==
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
    const uint32_t shard_tiles = can_alias_shards ? static_cast<uint32_t>(shard_of(grad_output)->numel() / TILE_HW) : 0;

    auto* device = input.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Work split: for an aliased program the cores are dictated by the shard grid -- each core
    // owns exactly its own shard -- so split_work_to_cores (which picks its own grid from a unit
    // count) must not be used.
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1, core_group_2;
    uint32_t num_cores = 0, num_units_per_core_group_1 = 0, num_units_per_core_group_2 = 0;
    if (can_alias_shards) {
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

    // Spec names. They are declared inside this function because a unity build compiles several
    // factories in one translation unit, where names at namespace scope could collide.
    const DFBSpecName GRAD_OUTPUT_DFB{"grad_output"};
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName GRAD_INPUT_DFB{"grad_input"};
    const DFBSpecName READER_SCRATCH_DFB{"reader_scratch"};
    const DFBSpecName WRITER_SCRATCH_DFB{"writer_scratch"};
    const TensorParamName GRAD_OUTPUT{"grad_output"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // An addressed DFB is a staging area: two pages, so the reader can fill one while compute
    // drains the other. An aliased DFB is the whole shard in place.
    constexpr uint32_t kDoubleBufferPages = 2;
    const uint32_t dfb_entries = can_alias_shards ? shard_tiles : kDoubleBufferPages;

    // ---- Dataflow buffers ----

    // `borrowed_from` set means the DFB is built on the tensor's own shard in L1 rather than on a
    // staging area. The dataflow kernel then copies nothing for it.
    using BorrowedFrom = std::optional<TensorParamName>;
    const auto make_dfb =
        [dfb_entries](
            const DFBSpecName& name, DataFormat data_format, uint32_t tile_size, const BorrowedFrom& shard_tensor) {
            return DataflowBufferSpec{
                .unique_id = name,
                .entry_size = tile_size,
                .num_entries = dfb_entries,
                .data_format_metadata = data_format,
                .borrowed_from = shard_tensor,
            };
        };

    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(make_dfb(
        GRAD_OUTPUT_DFB,
        grad_output_dfb_data_format,
        grad_output_single_tile_size,
        can_alias_shards ? BorrowedFrom{GRAD_OUTPUT} : std::nullopt));
    dataflow_buffers.push_back(make_dfb(
        INPUT_DFB,
        input_dfb_data_format,
        input_single_tile_size,
        can_alias_shards ? BorrowedFrom{INPUT} : std::nullopt));
    dataflow_buffers.push_back(make_dfb(
        GRAD_INPUT_DFB,
        output_dfb_data_format,
        output_single_tile_size,
        can_alias_shards ? BorrowedFrom{OUTPUT} : std::nullopt));

    // Scratch for the row-major kernels' unaligned pieces (see kernels/dataflow/row_major_pages.hpp):
    // one staging window of at most a chunk plus two alignment units, plus slack for rounding the
    // scratch start up to the widest buffer alignment. One each, since the reader and writer run
    // concurrently.
    if (rm_blocked) {
        const uint32_t max_alignment =
            std::max({grad_paging.alignment, input_paging.alignment, output_paging.alignment});
        const uint32_t scratch_bytes =
            std::max({grad_output_single_tile_size, input_single_tile_size, output_single_tile_size}) +
            (3 * max_alignment);
        for (const DFBSpecName& name : {READER_SCRATCH_DFB, WRITER_SCRATCH_DFB}) {
            dataflow_buffers.push_back(DataflowBufferSpec{
                .unique_id = name,
                .entry_size = scratch_bytes,
                .num_entries = 1,
                .data_format_metadata = output_dfb_data_format,
            });
        }
    }

    // ---- Tensor parameters ----

    // Exact match. compute_program_hash keys on each bound tensor's whole TensorSpec.
    // A cache hit therefore always sees the same specs as the miss that built the program.
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{
            .unique_id = GRAD_OUTPUT,
            .spec = grad_output_mesh_tensor.tensor_spec(),
            .relaxations = {},
        },
        TensorParameter{
            .unique_id = INPUT,
            .spec = input_mesh_tensor.tensor_spec(),
            .relaxations = {},
        },
        TensorParameter{
            .unique_id = OUTPUT,
            .spec = output_mesh_tensor.tensor_spec(),
            .relaxations = {},
        },
    };

    // ---- Reader / writer kernels ----

    KernelSpec reader;
    KernelSpec writer;

    if (rm_blocked) {
        // Unary's row-major block scheme, with page addressing generalized per tensor -- see
        // kernels/dataflow/row_major_pages.hpp.
        //
        // Each scratch DFB is the staging window of one kernel. Only that kernel touches it, and only
        // through get_write_ptr(), never through the FIFO. So the kernel is bound as both its
        // producer and its consumer.
        reader = KernelSpec{
            .unique_id = READER,
            .source =
                "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/dataflow/"
                "reader_unary_backward_row_major.cpp",
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = GRAD_OUTPUT_DFB,
                        .accessor_name = "grad",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = INPUT_DFB,
                        .accessor_name = "input",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = READER_SCRATCH_DFB,
                        .accessor_name = "scratch",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = READER_SCRATCH_DFB,
                        .accessor_name = "scratch",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .tensor_bindings =
                {
                    TensorBinding{
                        .tensor_parameter_name = GRAD_OUTPUT,
                        .accessor_name = "grad",
                    },
                    TensorBinding{
                        .tensor_parameter_name = INPUT,
                        .accessor_name = "input",
                    },
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names =
                        {
                            "num_blocks",
                            "start_block",
                            "row_elements",
                            "chunk_elements",
                            "chunks_per_row",
                            "rows_per_tile",
                            "total_rows",
                            "grad_page_elements",
                            "grad_pages_per_row",
                            "grad_element_bytes",
                            "grad_alignment",
                            "input_page_elements",
                            "input_pages_per_row",
                            "input_element_bytes",
                            "input_alignment",
                        },
                },
            .hw_config = ttnn::create_reader_datamovement_config(),
        };

        writer = KernelSpec{
            .unique_id = WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/eltwise/unary_backward/device/kernels/dataflow/"
                "writer_unary_backward_row_major.cpp",
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = GRAD_INPUT_DFB,
                        .accessor_name = "dst",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = WRITER_SCRATCH_DFB,
                        .accessor_name = "scratch",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = WRITER_SCRATCH_DFB,
                        .accessor_name = "scratch",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .tensor_bindings =
                {
                    TensorBinding{
                        .tensor_parameter_name = OUTPUT,
                        .accessor_name = "dst",
                    },
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names =
                        {
                            "num_blocks",
                            "start_block",
                            "row_elements",
                            "chunk_elements",
                            "chunks_per_row",
                            "rows_per_tile",
                            "total_rows",
                            "dst_page_elements",
                            "dst_pages_per_row",
                            "dst_element_bytes",
                            "dst_alignment",
                        },
                },
            .hw_config = ttnn::create_writer_datamovement_config(),
        };
    } else {
        // The reader declares a TensorAccessor only for an operand it actually reads. It therefore
        // binds exactly the tensors the defines leave enabled. An aliased operand is read in place,
        // through the DFB that borrows its shard.
        Table<std::string, std::string> reader_defines;
        Group<TensorBinding> reader_tensor_bindings;
        if (can_alias_shards) {
            reader_defines["IN0_SHARDED"] = "1";
            reader_defines["IN1_SHARDED"] = "1";
        } else {
            reader_tensor_bindings = {
                TensorBinding{
                    .tensor_parameter_name = GRAD_OUTPUT,
                    .accessor_name = "src0",
                },
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "src1",
                },
            };
        }
        reader = KernelSpec{
            .unique_id = READER,
            .source =
                "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/"
                "reader_binary_interleaved_start_id_metal2.cpp",
            .compiler_options = {.defines = reader_defines},
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = GRAD_OUTPUT_DFB,
                        .accessor_name = "in0",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = INPUT_DFB,
                        .accessor_name = "in1",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            .tensor_bindings = reader_tensor_bindings,
            .compile_time_args = {{"block_or_width_sharded", 0u}},
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"num_tiles", "start_id", "block_height", "block_width", "num_cores_y"},
                },
            .hw_config = ttnn::create_reader_datamovement_config(can_alias_shards),
        };

        // The writer uses its accessor only when OUT_SHARDED is not defined. The output is therefore
        // bound only then. Under OUT_SHARDED the kernel just waits on the DFB and never uses it.
        Table<std::string, std::string> writer_defines;
        Group<TensorBinding> writer_tensor_bindings;
        if (can_alias_shards) {
            writer_defines["OUT_SHARDED"] = "1";
        } else {
            writer_tensor_bindings = {
                TensorBinding{
                    .tensor_parameter_name = OUTPUT,
                    .accessor_name = "dst",
                },
            };
        }
        writer = KernelSpec{
            .unique_id = WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                "writer_unary_interleaved_start_id_metal2.cpp",
            .compiler_options = {.defines = writer_defines},
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = GRAD_INPUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .tensor_bindings = writer_tensor_bindings,
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"num_pages", "start_id"},
                },
            .hw_config = ttnn::create_writer_datamovement_config(can_alias_shards),
        };
    }

    // ---- Compute kernel ----
    //
    // A gradient chains several SFPU steps in DEST, so a float32 operand has to stay float32
    // all the way through: accumulate in fp32 if any operand or the output is fp32, and only
    // then ask the unpacker for fp32 DEST values, per operand that actually is fp32.
    const bool fp32_dest_acc_en = spec.force_fp32_dest_acc || (grad_output_dfb_data_format == DataFormat::Float32) ||
                                  (input_dfb_data_format == DataFormat::Float32) ||
                                  (output_dfb_data_format == DataFormat::Float32);

    // Every field not set here keeps its default, which matches the legacy compute config this
    // replaces: precise SFPU math, double-buffered DEST and approximate BFP packing.
    ComputeHardwareConfig compute_hw_config{
        .fpu_math_fidelity = spec.math_fidelity,
        .enable_32_bit_dest = fp32_dest_acc_en,
    };

    // Ask the unpacker for float32 DEST values only where they can actually be held: when DEST
    // is accumulating in float32. Requesting it otherwise would widen values into a DEST that
    // cannot represent them.
    if (fp32_dest_acc_en) {
        compute_hw_config.unpack_modes = {
            {GRAD_OUTPUT_DFB, UnpackMode::UnpackToDest},
            {INPUT_DFB, UnpackMode::UnpackToDest},
        };
    }

    // Supplied to every kernel whether or not it copies between DEST slots or packs bfloat16:
    // both are properties of the program the factory built, so a kernel that needs them must
    // not have to re-derive them, and one that does not simply leaves the define unused.
    Table<std::string, std::string> compute_defines;
    // A kernel only needs the unpacker to switch format between the two operand buffers when the
    // operands actually carry different formats; otherwise the single configuration
    // compute_kernel_hw_startup() installs covers both, and reconfiguring per tile transition is
    // measurable overhead on the common same-dtype path (~2% for bfloat16). Both operand dtypes
    // are part of compute_program_hash, so a program built for one pairing is never replayed for
    // another and the kernel can make this a compile-time decision.
    if (grad_output_dfb_data_format != input_dfb_data_format) {
        compute_defines["MIXED_OPERAND_DATA_FORMATS"] = "1";
    }
    compute_defines["COPY_DEST_DATA_FORMAT"] = fp32_dest_acc_en ? "DataFormat::Float32" : "DataFormat::Float16_b";
    compute_defines["BF16_ROUNDING_MODE"] = (output.dtype() == DataType::BFLOAT16)
                                                ? "ckernel::DstRoundingMode::NearestEven"
                                                : "ckernel::DstRoundingMode::Default";

    const std::string_view compute_kernel_path = (fp32_dest_acc_en && !spec.compute_kernel_path_fp32_dest.empty())
                                                     ? spec.compute_kernel_path_fp32_dest
                                                     : spec.compute_kernel_path;
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(compute_kernel_path),
        // A legacy compute kernel builds at O3 by default, but a KernelSpec defaults to O2. Set O3
        // to keep the level the kernel was built at.
        .compiler_options =
            {
                .defines = compute_defines,
                .opt_level = KernelBuildOptLevel::O3,
            },
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = GRAD_OUTPUT_DFB,
                    .accessor_name = "grad_out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INPUT_DFB,
                    .accessor_name = "input",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = GRAD_INPUT_DFB,
                    .accessor_name = "grad_in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"per_core_tile_cnt"},
            },
        .hw_config = compute_hw_config,
    };

    // ---- Per-core runtime args ----

    // Core enumeration. For an addressed program the linear-index-to-coordinate mapping below has
    // to agree with split_work_to_cores, which walks columns of the compute grid. For an aliased
    // program it instead has to agree with the SHARD's own core order, which follows the shard
    // spec's orientation -- a linear mapping through the compute grid's height silently hands
    // each core another core's tile range. That produced correct results only where the shard
    // grid happened to be one compute-grid column; a full 8x8 grid came out at PCC 0.13.
    std::vector<CoreCoord> cores;
    if (can_alias_shards) {
        cores =
            corerange_to_cores(all_cores, num_cores, shard_of(grad_output)->orientation == ShardOrientation::ROW_MAJOR);
    } else {
        cores.reserve(num_cores);
        for (uint32_t i = 0; i < num_cores; i++) {
            cores.push_back(CoreCoord{i / num_cores_y, i % num_cores_y});
        }
    }

    KernelRunArgs reader_args{.kernel = READER};
    KernelRunArgs writer_args{.kernel = WRITER};
    KernelRunArgs compute_args{.kernel = COMPUTE};

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
            AddRuntimeArgsForNode(
                reader_args.runtime_arg_values,
                core,
                {{"num_blocks", units},
                 {"start_block", units_done},
                 {"row_elements", row_elements},
                 {"chunk_elements", TILE_HW},
                 {"chunks_per_row", chunks_per_row},
                 {"rows_per_tile", rows_per_tile},
                 {"total_rows", total_rows},
                 {"grad_page_elements", grad_paging.page_elements},
                 {"grad_pages_per_row", grad_paging.pages_per_row},
                 {"grad_element_bytes", grad_paging.element_bytes},
                 {"grad_alignment", grad_paging.alignment},
                 {"input_page_elements", input_paging.page_elements},
                 {"input_pages_per_row", input_paging.pages_per_row},
                 {"input_element_bytes", input_paging.element_bytes},
                 {"input_alignment", input_paging.alignment}});
            AddRuntimeArgsForNode(
                writer_args.runtime_arg_values,
                core,
                {{"num_blocks", units},
                 {"start_block", units_done},
                 {"row_elements", row_elements},
                 {"chunk_elements", TILE_HW},
                 {"chunks_per_row", chunks_per_row},
                 {"rows_per_tile", rows_per_tile},
                 {"total_rows", total_rows},
                 {"dst_page_elements", output_paging.page_elements},
                 {"dst_pages_per_row", output_paging.pages_per_row},
                 {"dst_element_bytes", output_paging.element_bytes},
                 {"dst_alignment", output_paging.alignment}});
            // Each block is chunks_per_row DFB entries.
            AddRuntimeArgsForNode(
                compute_args.runtime_arg_values, core, {{"per_core_tile_cnt", units * chunks_per_row}});
        } else {
            AddRuntimeArgsForNode(
                reader_args.runtime_arg_values,
                core,
                {{"num_tiles", units},
                 {"start_id", units_done},
                 {"block_height", 0u},
                 {"block_width", 0u},
                 {"num_cores_y", num_cores_y}});
            AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"per_core_tile_cnt", units}});
            AddRuntimeArgsForNode(
                writer_args.runtime_arg_values, core, {{"num_pages", units}, {"start_id", units_done}});
        }

        units_done += units;
    }

    ProgramSpec program_spec{
        .name = "unary_backward",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER, COMPUTE},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_args), std::move(writer_args), std::move(compute_args)},
        .tensor_args =
            {
                {GRAD_OUTPUT, grad_output_mesh_tensor},
                {INPUT, input_mesh_tensor},
                {OUTPUT, output_mesh_tensor},
            },
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(program_spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::operations::unary_backward
