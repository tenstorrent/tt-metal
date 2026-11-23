// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar-native slice program factory (Metal 2.0 ProgramSpec).
//
// One data-movement KernelSpec of num_threads (<= 6) threads per cluster, no DataflowBuffer: a slice only
// relocates bytes, so there is nothing for a compute unit to do and no producer / consumer pair to
// synchronise. Each thread stages its pages in its own region of a per-cluster scratchpad (a NoC transfer
// needs one local end) and owns every num_threads-th item of the cluster's range.
//
// Page model. A page is one row (ROW_MAJOR) or one tile (TILE). Output page ids count the sliced page dims
// in mixed radix; the input page of an output page is base_page + sum_d digit[d] * coef[d], where coef[d]
// is step[d] times the input pages per unit index of dim d and base_page folds the begins. Dims with a
// single output index contribute to base_page only.
//
// Inside a page:
//  - TILE, last two begins tile aligned and unstrided: the whole tile moves.
//  - TILE otherwise: an output tile is assembled in L1 from the elements of the input tiles it covers
//    (kernel slice_tile_gather.cpp; the page dims are then the dims above the last two).
//  - ROW_MAJOR, last-dim step 1: the row's byte range [begin, begin + width) moves in chunks of at most
//    MAX_CHUNK_BYTES, read at its byte offset in the source row.
//  - ROW_MAJOR, last-dim step > 1: a chunk is a run of output elements; its source span is read into L1,
//    compacted in place and written.
//
// Work split: items (page x chunk) are dealt in contiguous ranges to the clusters of the compute grid (or
// of sub_core_grids).

#include "ttnn/operations/experimental/quasar/slice/device/slice_program_factory.hpp"

#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/kernel_spec.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/specification/scratchpad_spec.hpp>
#include <tt-metalium/workload/specification/tensor_parameter.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr const char* KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/slice.cpp";
constexpr const char* TILE_GATHER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/slice_tile_gather.cpp";
// Input tiles a tile-gather thread keeps in flight (SRC_SLOTS of the kernel).
constexpr uint32_t TILE_GATHER_SRC_SLOTS = 4;

// Largest NoC transfer of one item, and so the payload of a staging slot. A multiple of every element size.
constexpr uint32_t MAX_CHUNK_BYTES = 4096;
// Scratchpad budget per cluster, shared by the threads; bounds the items a thread keeps in flight.
constexpr uint32_t STAGE_BUDGET_BYTES = 64 * 1024;
constexpr uint32_t MAX_SLOTS_PER_THREAD = 8;
// The DM harts reach L1 through a cache of 64-byte lines. A slot is a whole number of lines plus one guard
// line, so a thread that locks (invalidates / flushes) its slots never touches a line that holds another
// thread's bytes, whatever the alignment of the scratchpad base.
constexpr uint32_t CACHE_LINE_BYTES = 64;

// One sliced dim in page units.
struct PageDim {
    uint32_t in_size;   // input pages along this dim
    uint32_t out_size;  // output pages along this dim
    uint32_t begin;
    uint32_t step;
};

struct PageMap {
    std::vector<uint32_t> extent;  // output index count of every dim with more than one, outermost first
    std::vector<uint32_t> coef;    // input pages per output index step of that dim
    uint32_t base_page = 0;        // input page of output page 0
    uint64_t num_pages = 1;        // output pages
    std::vector<uint32_t> stride;  // input pages per unit index of every page dim
};

PageMap make_page_map(const std::vector<PageDim>& dims) {
    PageMap map;
    map.stride.assign(dims.size(), 1);
    for (size_t d = dims.size(); d-- > 1;) {
        map.stride[d - 1] = map.stride[d] * dims[d].in_size;
    }
    for (size_t d = 0; d < dims.size(); ++d) {
        map.base_page += dims[d].begin * map.stride[d];
        map.num_pages *= dims[d].out_size;
        if (dims[d].out_size > 1) {
            map.extent.push_back(dims[d].out_size);
            map.coef.push_back(dims[d].step * map.stride[d]);
        }
    }
    if (map.extent.empty()) {
        // A single output page: keep one digit so the kernel arrays are never empty.
        map.extent.push_back(1);
        map.coef.push_back(0);
    }
    return map;
}

// Items dealt in contiguous ranges to the clusters of the compute grid (or of sub_core_grids).
struct WorkSplit {
    CoreRangeSet all_cores;
    std::vector<std::pair<CoreCoord, std::pair<uint32_t, uint32_t>>> ranges;  // cluster -> (start, count)
    uint32_t max_items_per_core = 0;
};

WorkSplit split_items(const SliceParams& args, const Tensor& input, uint32_t num_items) {
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, items_per_core_1, items_per_core_2] =
        args.sub_core_grids.has_value() ? split_work_to_cores(args.sub_core_grids.value(), num_items)
                                        : split_work_to_cores(grid, num_items);
    WorkSplit split{.all_cores = all_cores, .max_items_per_core = std::max(items_per_core_1, items_per_core_2)};
    uint32_t item_start = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t item_count = 0;
        if (core_group_1.contains(core)) {
            item_count = items_per_core_1;
        } else if (core_group_2.contains(core)) {
            item_count = items_per_core_2;
        }
        split.ranges.push_back({core, {item_start, item_count}});
        item_start += item_count;
    }
    TT_FATAL(item_start == num_items, "quasar.slice: the work split covers {} of {} items", item_start, num_items);
    return split;
}

uint32_t checked_num_items(uint64_t num_pages, uint32_t chunks_per_page) {
    const uint64_t num_items = num_pages * chunks_per_page;
    TT_FATAL(
        num_items <= std::numeric_limits<uint32_t>::max(),
        "quasar.slice: {} output pages x {} chunks overflow the 32-bit item index of the kernel",
        num_pages,
        chunks_per_page);
    return static_cast<uint32_t>(num_items);
}

// TILE slice whose last two dims are not tile aligned or are strided: element gather per output tile.
ttnn::device_operation::ProgramArtifacts create_tile_gather_artifacts(
    const SliceParams& args, const Tensor& input, const Tensor& output) {
    using namespace tt::constants;
    const auto& in_shape = input.padded_shape();
    const auto& out_padded = output.padded_shape();
    const auto& out_logical = output.logical_shape();
    const uint32_t rank = in_shape.rank();
    const uint32_t num_threads = args.num_threads;

    const uint32_t in_ht = in_shape[-2] / TILE_HEIGHT;
    const uint32_t in_wt = in_shape[-1] / TILE_WIDTH;
    const uint32_t out_ht = out_padded[-2] / TILE_HEIGHT;
    const uint32_t out_wt = out_padded[-1] / TILE_WIDTH;

    // Page dims: the dims above the last two, then the whole tile plane as the innermost unit.
    std::vector<PageDim> dims;
    for (uint32_t d = 0; d + 2 < rank; ++d) {
        dims.push_back({in_shape[d], out_padded[d], args.slice_start[d], args.step[d]});
    }
    dims.push_back({in_ht * in_wt, 1, 0, 1});
    const PageMap map = make_page_map(dims);
    const uint32_t num_dims = static_cast<uint32_t>(map.extent.size());

    const uint32_t tile_bytes = static_cast<uint32_t>(input.tensor_spec().compute_page_size_bytes());
    const uint32_t num_items = checked_num_items(map.num_pages * out_ht * out_wt, 1);
    const WorkSplit split = split_items(args, input, num_items);

    const uint32_t slot_bytes = tt::align(tile_bytes, CACHE_LINE_BYTES) + CACHE_LINE_BYTES;

    const KernelSpecName SLICE{"slice_tile_gather"};
    const ScratchpadSpecName STAGE{"stage"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    ProgramSpec spec;
    spec.name = "quasar_slice_tile_gather";
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});
    spec.scratchpads.push_back(
        ScratchpadSpec{.unique_id = STAGE, .size_per_node = num_threads * (1 + TILE_GATHER_SRC_SLOTS) * slot_bytes});

    spec.kernels.push_back(KernelSpec{
        .unique_id = SLICE,
        .source = TILE_GATHER_KERNEL_PATH,
        .num_threads = num_threads,
        .compiler_options = {.opt_level = KernelSpec::CompilerOptions::OptLevel::O3},
        .scratchpad_bindings = {KernelSpec::ScratchpadBinding{.scratchpad_spec_name = STAGE, .accessor_name = "stage"}},
        .tensor_bindings =
            {KernelSpec::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
             KernelSpec::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"elem_bytes", input.element_size()}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"item_start", "item_count"},
             .common_runtime_arg_names =
                 {"base_page",
                  "tile_bytes",
                  "slot_bytes",
                  "in_wt",
                  "out_ht",
                  "out_wt",
                  "out_h",
                  "out_w",
                  "begin_h",
                  "begin_w",
                  "step_h",
                  "step_w"},
             .common_runtime_array_args = {{.name = "extent", .size = num_dims}, {.name = "coef", .size = num_dims}}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });
    spec.work_units.push_back(WorkUnitSpec{
        .name = "slice_tile_gather_wu",
        .kernels = {SLICE},
        .target_nodes = split.all_cores,
    });

    KernelRunArgs kernel_run_args{
        .kernel = SLICE,
        .common_runtime_arg_values =
            {{"base_page", map.base_page},
             {"tile_bytes", tile_bytes},
             {"slot_bytes", slot_bytes},
             {"in_wt", in_wt},
             {"out_ht", out_ht},
             {"out_wt", out_wt},
             {"out_h", out_logical[-2]},
             {"out_w", out_logical[-1]},
             {"begin_h", args.slice_start[-2]},
             {"begin_w", args.slice_start[-1]},
             {"step_h", args.step[-2]},
             {"step_w", args.step[-1]}},
        .common_runtime_array_arg_values = {{"extent", map.extent}, {"coef", map.coef}},
    };
    for (const auto& [core, range] : split.ranges) {
        AddRuntimeArgsForNode(
            kernel_run_args.runtime_arg_values, core, {{"item_start", range.first}, {"item_count", range.second}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(kernel_run_args)};
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts SliceProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using namespace tt::constants;

    const Tensor& input = tensor_args.input;
    const auto& in_shape = input.padded_shape();
    const auto& out_shape = output.padded_shape();
    const uint32_t rank = in_shape.rank();
    const bool tile = input.layout() == Layout::TILE;
    const uint32_t num_threads = args.num_threads;

    if (slice_gathers_tile_elements(args, input)) {
        return create_tile_gather_artifacts(args, input, output);
    }

    // ---- Page dims ----
    // The tensor-args arm takes its begins on the device, so the host map starts at page 0.
    auto begin_of = [&](uint32_t d) { return args.use_tensor_args ? 0u : args.slice_start[d]; };
    auto step_of = [&](uint32_t d) { return args.use_tensor_args ? 1u : args.step[d]; };
    const uint32_t num_index_dims = tile ? rank - 2 : rank - 1;  // dims above the page
    std::vector<PageDim> dims;
    dims.reserve(num_index_dims + 2);
    // Per input dim: the divisor that turns a begin index into page units, for the tensor-args arm.
    std::vector<uint32_t> start_div(rank, 1);
    for (uint32_t d = 0; d < num_index_dims; ++d) {
        dims.push_back({in_shape[d], out_shape[d], begin_of(d), step_of(d)});
    }
    if (tile) {
        dims.push_back({in_shape[-2] / TILE_HEIGHT, out_shape[-2] / TILE_HEIGHT, begin_of(rank - 2) / TILE_HEIGHT, 1});
        dims.push_back({in_shape[-1] / TILE_WIDTH, out_shape[-1] / TILE_WIDTH, begin_of(rank - 1) / TILE_WIDTH, 1});
        start_div[rank - 2] = TILE_HEIGHT;
        start_div[rank - 1] = TILE_WIDTH;
    }
    const PageMap map = make_page_map(dims);
    const uint32_t num_dims = static_cast<uint32_t>(map.extent.size());

    // ---- Chunks of a page ----
    const uint32_t elem_bytes = input.element_size();
    const uint32_t last_dim_step = tile ? 1u : step_of(rank - 1);
    const bool strided_last_dim = last_dim_step > 1;
    uint32_t page_bytes = 0;          // bytes of an output page
    uint32_t chunk_bytes = 0;         // output bytes of a full chunk
    uint32_t src_offset_bytes = 0;    // byte offset of the slice inside a source page
    uint32_t slot_payload_bytes = 0;  // most bytes a slot ever holds
    if (tile) {
        page_bytes = static_cast<uint32_t>(input.tensor_spec().compute_page_size_bytes());
        chunk_bytes = page_bytes;
        slot_payload_bytes = page_bytes;
    } else {
        page_bytes = out_shape[-1] * elem_bytes;
        src_offset_bytes = begin_of(rank - 1) * elem_bytes;
        if (strided_last_dim) {
            // elems output elements span (elems - 1) * step + 1 source elements, which must fit a slot.
            const uint32_t max_elems = (MAX_CHUNK_BYTES / elem_bytes - 1) / last_dim_step + 1;
            const uint32_t elems = std::min<uint32_t>(out_shape[-1], max_elems);
            chunk_bytes = elems * elem_bytes;
            slot_payload_bytes = ((elems - 1) * last_dim_step + 1) * elem_bytes;
        } else {
            chunk_bytes = std::min(page_bytes, MAX_CHUNK_BYTES);
            slot_payload_bytes = chunk_bytes;
        }
    }
    const uint32_t chunks_per_page = tt::div_up(page_bytes, chunk_bytes);
    const uint32_t last_chunk_bytes = page_bytes - (chunks_per_page - 1) * chunk_bytes;
    if (args.use_tensor_args) {
        slot_payload_bytes = std::max<uint32_t>(slot_payload_bytes, rank * sizeof(uint32_t));
    }

    // ---- Work split over clusters ----
    const uint32_t num_items = checked_num_items(map.num_pages, chunks_per_page);
    const WorkSplit split = split_items(args, input, num_items);

    // ---- Staging slots ----
    const uint32_t slot_bytes = tt::align(slot_payload_bytes, CACHE_LINE_BYTES) + CACHE_LINE_BYTES;
    const uint32_t max_items_per_thread = tt::div_up(split.max_items_per_core, num_threads);
    const uint32_t slots = std::clamp<uint32_t>(
        std::min(STAGE_BUDGET_BYTES / (num_threads * slot_bytes), max_items_per_thread), 1, MAX_SLOTS_PER_THREAD);

    // ---- ProgramSpec ----
    const KernelSpecName SLICE{"slice"};
    const ScratchpadSpecName STAGE{"stage"};
    const TensorParamName INPUT{"input"};
    const TensorParamName START{"start"};
    const TensorParamName OUTPUT{"output"};

    ProgramSpec spec;
    spec.name = args.use_tensor_args ? "quasar_slice_tensor_args" : "quasar_slice";
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});
    spec.scratchpads.push_back(ScratchpadSpec{.unique_id = STAGE, .size_per_node = num_threads * slots * slot_bytes});

    KernelSpec::CompilerOptions::Defines defines;
    KernelSpec::CompileTimeArgs compile_time_args;
    Group<KernelSpec::TensorBinding> tensor_bindings = {
        KernelSpec::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
        KernelSpec::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}};
    Group<KernelSpec::RuntimeArgSchema::ArrayArgDecl> array_args = {
        {.name = "extent", .size = num_dims}, {.name = "coef", .size = num_dims}};
    KernelRunArgs::CommonRuntimeArrayArgValues array_arg_values = {{"extent", map.extent}, {"coef", map.coef}};
    if (strided_last_dim) {
        defines.emplace("SLICE_STRIDED_LAST_DIM", "1");
        compile_time_args.emplace("elem_bytes", elem_bytes);
    }
    if (args.use_tensor_args) {
        // The begins tensor is one page of `rank` uint32 values; the kernel turns them into pages with a
        // per-dim divisor (elements per page) and page stride.
        const Tensor& start = tensor_args.start_tensor.value();
        spec.tensor_parameters.push_back(TensorParameter{.unique_id = START, .spec = start.tensor_spec()});
        tensor_bindings.push_back(KernelSpec::TensorBinding{.tensor_parameter_name = START, .accessor_name = "start"});
        defines.emplace("SLICE_TENSOR_ARGS", "1");
        // TILE only (validated), so every input dim is a page dim and has a page stride.
        TT_FATAL(
            map.stride.size() == rank, "quasar.slice: {} page dims for a rank {} TILE input", map.stride.size(), rank);
        array_args.push_back({.name = "start_div", .size = rank});
        array_args.push_back({.name = "start_coef", .size = rank});
        array_arg_values.emplace("start_div", start_div);
        array_arg_values.emplace("start_coef", map.stride);
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = SLICE,
        .source = KERNEL_PATH,
        .num_threads = num_threads,
        .compiler_options = {.defines = defines, .opt_level = KernelSpec::CompilerOptions::OptLevel::O3},
        .scratchpad_bindings = {KernelSpec::ScratchpadBinding{.scratchpad_spec_name = STAGE, .accessor_name = "stage"}},
        .tensor_bindings = tensor_bindings,
        .compile_time_args = compile_time_args,
        .runtime_arg_schema =
            {.runtime_arg_names = {"item_start", "item_count"},
             .common_runtime_arg_names =
                 {"base_page",
                  "chunks_per_page",
                  "chunk_bytes",
                  "last_chunk_bytes",
                  "src_offset_bytes",
                  "slots",
                  "slot_bytes",
                  "last_dim_step"},
             .common_runtime_array_args = array_args},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });
    spec.work_units.push_back(WorkUnitSpec{
        .name = "slice_wu",
        .kernels = {SLICE},
        .target_nodes = split.all_cores,
    });

    // ---- ProgramRunArgs ----
    KernelRunArgs kernel_run_args{
        .kernel = SLICE,
        .common_runtime_arg_values =
            {{"base_page", map.base_page},
             {"chunks_per_page", chunks_per_page},
             {"chunk_bytes", chunk_bytes},
             {"last_chunk_bytes", last_chunk_bytes},
             {"src_offset_bytes", src_offset_bytes},
             {"slots", slots},
             {"slot_bytes", slot_bytes},
             {"last_dim_step", last_dim_step}},
        .common_runtime_array_arg_values = std::move(array_arg_values),
    };
    for (const auto& [core, range] : split.ranges) {
        AddRuntimeArgsForNode(
            kernel_run_args.runtime_arg_values, core, {{"item_start", range.first}, {"item_count", range.second}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(kernel_run_args)};
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());
    if (args.use_tensor_args) {
        run_args.tensor_args.emplace(START, tensor_args.start_tensor.value().mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
