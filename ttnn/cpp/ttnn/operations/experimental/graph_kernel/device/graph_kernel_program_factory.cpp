// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_kernel_program_factory.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// The sliding window (pages per node per step) is sized per program to the largest value L1 allows,
// see choose_dfb_length below. Every node's init + data-format reconfig runs once per window, so a
// longer window means fewer inits and longer uninterrupted LLK chains. The window must be a multiple
// of the dest-register count for both the fp32 (4) and non-fp32 (8) cases, which chain_reads /
// chain_llk assert at kernel compile time.
constexpr uint32_t kDfbLengthGranule = 8;

// Largest multiple of kDfbLengthGranule such that num_dfbs buffers of that many pages fit the L1 the
// allocator hands out, capped at the longest per-core stream (a window longer than the stream would
// only waste L1). Every buffer in the program is one window deep.
uint32_t choose_dfb_length(uint64_t l1_budget, uint32_t num_dfbs, uint32_t page_size, uint32_t max_pages_per_core) {
    const uint64_t per_entry = static_cast<uint64_t>(num_dfbs) * page_size;
    const uint64_t fit = l1_budget / per_entry;
    const uint64_t cap = tt::round_up(static_cast<uint64_t>(max_pages_per_core), uint64_t{kDfbLengthGranule});
    const uint64_t length = std::min(fit, cap) / kDfbLengthGranule * kDfbLengthGranule;
    TT_FATAL(
        length >= kDfbLengthGranule,
        "graph_kernel: {} dataflow buffers x {} pages x {} B = {} B does not fit the {} B of L1 allocatable per core; "
        "use fewer inputs or smaller pages",
        num_dfbs,
        kDfbLengthGranule,
        page_size,
        per_entry * kDfbLengthGranule,
        l1_budget);
    return static_cast<uint32_t>(length);
}

constexpr std::string_view kKernelDir = "ttnn/cpp/ttnn/operations/experimental/graph_kernel/device/kernels/";

// ---- Expression -> binary node list -----------------------------------------------------------------
//
//   expr   := term (('+' | '-') term)*
//   term   := factor ('*' factor)*
//   factor := [a-z] | '(' expr ')'
//
// Letter a is input 0, b input 1, ... Node k writes intermediate t<k>; the last node writes the output.
// Operand codes match compute_graph_kernel.cpp: input i -> i, intermediate k -> 128 + k, output -> 255.

enum class BinaryOp : uint32_t { Add = 0, Sub = 1, Mul = 2 };

struct GraphNode {
    BinaryOp op;
    uint32_t a;
    uint32_t b;
};

constexpr uint32_t kIntermBase = 128;
constexpr uint32_t kOutputCode = 255;

class ExpressionParser {
public:
    explicit ExpressionParser(std::string_view text) : text_(text) {}

    std::vector<GraphNode> parse() {
        const uint32_t root = expr();
        TT_FATAL(pos_ == text_.size(), "graph_kernel: unexpected '{}' at position {} in \"{}\"", peek(), pos_, text_);
        TT_FATAL(!nodes_.empty(), "graph_kernel: \"{}\" has no binary operation", text_);
        TT_FATAL(root == kIntermBase + nodes_.size() - 1, "graph_kernel: internal error, root is not the last node");
        return nodes_;
    }

private:
    char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

    uint32_t emit(BinaryOp op, uint32_t a, uint32_t b) {
        nodes_.push_back(GraphNode{op, a, b});
        return kIntermBase + static_cast<uint32_t>(nodes_.size() - 1);
    }

    uint32_t expr() {
        uint32_t lhs = term();
        while (peek() == '+' || peek() == '-') {
            const BinaryOp op = text_[pos_++] == '+' ? BinaryOp::Add : BinaryOp::Sub;
            lhs = emit(op, lhs, term());
        }
        return lhs;
    }

    uint32_t term() {
        uint32_t lhs = factor();
        while (peek() == '*') {
            ++pos_;
            lhs = emit(BinaryOp::Mul, lhs, factor());
        }
        return lhs;
    }

    uint32_t factor() {
        const char c = peek();
        if (c == '(') {
            ++pos_;
            const uint32_t inner = expr();
            TT_FATAL(peek() == ')', "graph_kernel: missing ')' at position {} in \"{}\"", pos_, text_);
            ++pos_;
            return inner;
        }
        TT_FATAL(c >= 'a' && c <= 'z', "graph_kernel: expected an operand a..z at position {} in \"{}\"", pos_, text_);
        ++pos_;
        return static_cast<uint32_t>(c - 'a');
    }

    std::string_view text_;
    size_t pos_ = 0;
    std::vector<GraphNode> nodes_;
};

// GK_GRAPH define value: op_a_b_out per node, joined by '_'. Digits and underscores only, so it is
// safe on the unquoted -D command line.
std::string encode_graph(const std::vector<GraphNode>& nodes) {
    std::string out;
    for (size_t k = 0; k < nodes.size(); ++k) {
        const uint32_t out_code = k + 1 == nodes.size() ? kOutputCode : kIntermBase + static_cast<uint32_t>(k);
        if (k != 0) {
            out += '_';
        }
        out += std::to_string(static_cast<uint32_t>(nodes[k].op)) + '_' + std::to_string(nodes[k].a) + '_' +
               std::to_string(nodes[k].b) + '_' + std::to_string(out_code);
    }
    return out;
}

}  // namespace

// Program: reader streams every input into its own DFB (chain_reads), compute evaluates the parsed
// expression as a chain of binary LLK nodes (chain_llk) through per-node intermediates, and the
// writer drains the final node's DFB into the output tensor.
ttnn::device_operation::ProgramArtifacts GraphKernelProgramFactory::create_program_artifacts(
    const GraphKernelParams& operation_attributes, const GraphKernelInputs& tensor_args, Tensor& output) {
    const auto& inputs = tensor_args.inputs;
    const Tensor& src = inputs.front();
    auto* device = src.device();
    const uint32_t num_inputs = static_cast<uint32_t>(inputs.size());

    // ---- Expression ----
    const std::vector<GraphNode> nodes = ExpressionParser(operation_attributes.text).parse();
    const uint32_t num_interm = static_cast<uint32_t>(nodes.size()) - 1;
    {
        // Each input feeds exactly one node: the reader streams every input once per window, and a
        // DFB consumed twice or never would stall the sliding window.
        std::vector<uint32_t> uses(num_inputs, 0);
        for (const auto& node : nodes) {
            for (uint32_t operand : {node.a, node.b}) {
                if (operand < kIntermBase) {
                    TT_FATAL(
                        operand < num_inputs,
                        "graph_kernel: \"{}\" uses input '{}' but only {} input(s) were passed",
                        operation_attributes.text,
                        static_cast<char>('a' + operand),
                        num_inputs);
                    ++uses[operand];
                }
            }
        }
        for (uint32_t i = 0; i < num_inputs; ++i) {
            TT_FATAL(
                uses[i] == 1,
                "graph_kernel: input '{}' must be used exactly once in \"{}\" (used {} times)",
                static_cast<char>('a' + i),
                operation_attributes.text,
                uses[i]);
        }
    }
    const std::string graph_define = encode_graph(nodes);
    log_debug(tt::LogOp, "graph_kernel: \"{}\" -> GK_GRAPH={}", operation_attributes.text, graph_define);

    // ---- Page geometry (taken from inputs[0]; every input and the output share it) ----
    TT_FATAL(src.layout() == Layout::TILE, "graph_kernel: inputs must be tile layout");
    const auto* src_buffer = src.buffer();
    const uint32_t page_size = static_cast<uint32_t>(src_buffer->page_size());
    const uint32_t num_pages = static_cast<uint32_t>(src_buffer->num_pages());
    const uint32_t aligned_page_size = tt::align(page_size, static_cast<uint32_t>(src_buffer->alignment()));
    const tt::DataFormat data_format = datatype_to_dataformat_converter(src.dtype());
    const bool is_fp_32 = src.dtype() == DataType::FLOAT32;
    for (size_t i = 1; i < inputs.size(); ++i) {
        TT_FATAL(
            inputs[i].layout() == Layout::TILE && inputs[i].dtype() == src.dtype() &&
                inputs[i].buffer()->page_size() == page_size && inputs[i].buffer()->num_pages() == num_pages,
            "graph_kernel: input {} must match input 0 in layout, dtype and page geometry",
            i);
    }

    // ---- Work split ----
    const auto grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, pages_per_core_g1, pages_per_core_g2] =
        split_work_to_cores(grid, num_pages);

    // ---- Names: input i is tensor "in<i>" and DFB "in<i>", intermediate k is DFB "t<k>" ----
    std::vector<std::string> input_names;
    std::vector<std::string> interm_names;
    for (uint32_t i = 0; i < num_inputs; ++i) {
        input_names.push_back("in" + std::to_string(i));
    }
    for (uint32_t k = 0; k < num_interm; ++k) {
        interm_names.push_back("t" + std::to_string(k));
    }
    const std::string out_name = "out";

    ProgramSpec spec;
    spec.name = "graph_kernel";

    // ---- Window size: fill the allocatable L1 with num_dfbs equally deep buffers ----
    const uint32_t num_dfbs = num_inputs + num_interm + 1;
    const uint64_t l1_budget =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t dfb_length =
        choose_dfb_length(l1_budget, num_dfbs, aligned_page_size, std::max(pages_per_core_g1, pages_per_core_g2));
    log_debug(
        tt::LogOp,
        "graph_kernel: window {} pages, {} DFBs x {} B = {} B of {} B L1 per core",
        dfb_length,
        num_dfbs,
        aligned_page_size,
        static_cast<uint64_t>(num_dfbs) * dfb_length * aligned_page_size,
        l1_budget);

    // ---- Dataflow buffers: one window deep each ----
    auto add_dfb = [&](const std::string& name) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = DFBSpecName{name},
            .entry_size = aligned_page_size,
            .num_entries = dfb_length,
            .data_format_metadata = data_format,
        });
    };
    for (const auto& name : input_names) {
        add_dfb(name);
    }
    for (const auto& name : interm_names) {
        add_dfb(name);
    }
    add_dfb(out_name);

    // ---- Tensor parameters ----
    for (uint32_t i = 0; i < num_inputs; ++i) {
        spec.tensor_parameters.push_back(TensorParameter{
            .unique_id = TensorParamName{input_names[i]}, .spec = inputs[i].mesh_tensor().tensor_spec()});
    }
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = TensorParamName{out_name}, .spec = output.mesh_tensor().tensor_spec()});

    // ---- Compute hardware config: fp32 dest for float32, HiFi4, no approx ----
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(
        device->arch(),
        ttnn::init_device_compute_kernel_config(device->arch(), std::nullopt, MathFidelity::HiFi4, false, is_fp_32));
    if (auto* gen1 = std::get_if<ComputeGen1Config>(&compute_hw); gen1 != nullptr && gen1->enable_32_bit_dest) {
        // Every buffer the FPU consumes is Float32 here, so each one takes the SrcA/SrcB unpack path.
        for (const auto& name : input_names) {
            gen1->unpack_modes.emplace(DFBSpecName{name}, UnpackMode::UnpackToSrc);
        }
        for (const auto& name : interm_names) {
            gen1->unpack_modes.emplace(DFBSpecName{name}, UnpackMode::UnpackToSrc);
        }
    }

    // ---- Kernels: reader / compute / writer per core group (pages_per_core is a compile-time arg) ----
    const Group<std::string> rta_names{"start_id"};

    auto make_group = [&](const std::string& suffix, uint32_t pages_per_core, const CoreRangeSet& cores) {
        const KernelSpecName reader_name{"reader_" + suffix};
        const KernelSpecName compute_name{"compute_" + suffix};
        const KernelSpecName writer_name{"writer_" + suffix};

        Group<TensorBinding> reader_tensors;
        Group<DFBBinding> reader_dfbs;
        Group<DFBBinding> compute_dfbs;
        for (uint32_t i = 0; i < num_inputs; ++i) {
            reader_tensors.push_back(TensorBinding{
                .tensor_parameter_name = TensorParamName{input_names[i]}, .accessor_name = input_names[i]});
            reader_dfbs.push_back(DFBBinding{
                .dfb_spec_name = DFBSpecName{input_names[i]},
                .accessor_name = input_names[i],
                .endpoint_type = DFBEndpointType::PRODUCER});
            compute_dfbs.push_back(DFBBinding{
                .dfb_spec_name = DFBSpecName{input_names[i]},
                .accessor_name = input_names[i],
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        // Intermediates are private to the compute kernel: it packs into them and unpacks them back.
        for (const auto& name : interm_names) {
            compute_dfbs.push_back(DFBBinding{
                .dfb_spec_name = DFBSpecName{name}, .accessor_name = name, .endpoint_type = DFBEndpointType::PRODUCER});
            compute_dfbs.push_back(DFBBinding{
                .dfb_spec_name = DFBSpecName{name}, .accessor_name = name, .endpoint_type = DFBEndpointType::CONSUMER});
        }
        compute_dfbs.push_back(DFBBinding{
            .dfb_spec_name = DFBSpecName{out_name},
            .accessor_name = out_name,
            .endpoint_type = DFBEndpointType::PRODUCER});

        const KernelSpec::CompileTimeArgs ctas{
            {"num_inputs", num_inputs},
            {"page_size", page_size},
            {"pages_per_core", pages_per_core},
            {"dfb_length", dfb_length},
            {"is_fp_32", is_fp_32 ? 1u : 0u},
        };

        spec.kernels.push_back(KernelSpec{
            .unique_id = reader_name,
            .source = std::string(kKernelDir) + "reader_graph_kernel.cpp",
            .dfb_bindings = std::move(reader_dfbs),
            .tensor_bindings = std::move(reader_tensors),
            .compile_time_args = ctas,
            .runtime_arg_schema = {.runtime_arg_names = rta_names},
            .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        });
        spec.kernels.push_back(KernelSpec{
            .unique_id = compute_name,
            .source = std::string(kKernelDir) + "compute_graph_kernel.cpp",
            .compiler_options = {.defines = {{"GK_GRAPH", graph_define}}, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(compute_dfbs),
            .compile_time_args = ctas,
            .hw_config = compute_hw,
        });
        spec.kernels.push_back(KernelSpec{
            .unique_id = writer_name,
            .source = std::string(kKernelDir) + "writer_graph_kernel.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = DFBSpecName{out_name},
                .accessor_name = out_name,
                .endpoint_type = DFBEndpointType::CONSUMER}},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = TensorParamName{out_name}, .accessor_name = out_name}},
            .compile_time_args = ctas,
            .runtime_arg_schema = {.runtime_arg_names = rta_names},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        });
        spec.work_units.push_back(WorkUnitSpec{
            .name = "graph_" + suffix, .kernels = {reader_name, compute_name, writer_name}, .target_nodes = cores});
        return std::pair{reader_name, writer_name};
    };

    const auto [reader_g1, writer_g1] = make_group("g1", pages_per_core_g1, core_group_1);
    std::optional<std::pair<KernelSpecName, KernelSpecName>> g2;
    if (core_group_2.num_cores() > 0) {
        g2 = make_group("g2", pages_per_core_g2, core_group_2);
    }

    // ---- Runtime args: each core's first page, on the reader and the writer ----
    ProgramRunArgs run_args;
    std::vector<KernelRunArgs> ras;
    ras.push_back(KernelRunArgs{.kernel = reader_g1});
    ras.push_back(KernelRunArgs{.kernel = writer_g1});
    if (g2.has_value()) {
        ras.push_back(KernelRunArgs{.kernel = g2->first});
        ras.push_back(KernelRunArgs{.kernel = g2->second});
    }

    const uint32_t num_cores_g1 = core_group_1.num_cores();
    const auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    uint32_t start_id = 0;
    for (size_t i = 0; i < cores.size(); ++i) {
        const bool in_g1 = i < num_cores_g1;
        const size_t base = in_g1 ? 0 : 2;
        AddRuntimeArgsForNode(ras[base].runtime_arg_values, cores[i], {{"start_id", start_id}});
        AddRuntimeArgsForNode(ras[base + 1].runtime_arg_values, cores[i], {{"start_id", start_id}});
        start_id += in_g1 ? pages_per_core_g1 : pages_per_core_g2;
    }
    for (auto& ra : ras) {
        run_args.kernel_run_args.push_back(std::move(ra));
    }

    for (uint32_t i = 0; i < num_inputs; ++i) {
        run_args.tensor_args.emplace(TensorParamName{input_names[i]}, TensorArgument{inputs[i].mesh_tensor()});
    }
    run_args.tensor_args.emplace(TensorParamName{out_name}, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
