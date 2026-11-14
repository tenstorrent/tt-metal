// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/arg_value.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/tensor/tensor.hpp"

// Experimental and subject to change. Command-list parameters: what callers bind and what program factories map.
// Program factories include this header, so it must not include tt-metalium/experimental/command_list.hpp: that
// header declares tt::tt_metal::experimental::detail, which changes unqualified `detail::` lookup in unity builds.

namespace ttnn::command_list {

// Python float and int convert to these without ambiguity. Encoders narrow and range-check.
struct CmdListScalar {
    std::variant<double, int64_t> value;
};

using CmdListValue = std::variant<Tensor, CmdListScalar>;

// Encodes a scalar as the word written to the kernel arguments it is bound to.
using CmdListEncodeFn = std::function<tt::tt_metal::experimental::ArgValue(const CmdListScalar&)>;

// Default encoders. Layouts are [high 16 bits | low 16 bits].
tt::tt_metal::experimental::ArgValue encode_uint32(const CmdListScalar&);            // integer -> unchanged
tt::tt_metal::experimental::ArgValue encode_int32_bits(const CmdListScalar&);        // int32 -> two's complement
tt::tt_metal::experimental::ArgValue encode_float32_bits(const CmdListScalar&);      // float -> IEEE-754 bits
tt::tt_metal::experimental::ArgValue encode_bfloat16_pair(const CmdListScalar&);     // float -> [bf16 | bf16]
tt::tt_metal::experimental::ArgValue encode_bfloat16_low(const CmdListScalar&);      // float -> [zero | bf16]
tt::tt_metal::experimental::ArgValue encode_bfloat16_high(const CmdListScalar&);     // float -> [bf16 | zero]
tt::tt_metal::experimental::ArgValue encode_uint16_pair(const CmdListScalar&);       // uint16 -> [uint16 | uint16]
tt::tt_metal::experimental::ArgValue encode_uint16_low(const CmdListScalar&);        // uint16 -> [zero | uint16]
tt::tt_metal::experimental::ArgValue encode_uint16_high(const CmdListScalar&);       // uint16 -> [uint16 | zero]
tt::tt_metal::experimental::ArgValue encode_float8_e4m3_quad(const CmdListScalar&);  // float -> four FP8 E4M3 bytes

// One op argument recorded as a TTNN command-list parameter.
struct CmdListBind {
    std::string name;         // the key for TTNN CommandList::update_args
    std::string invoke_name;  // the op's keyword ("input_tensor", "scale")
    CmdListValue value;       // the value the op is recorded with
};

// Python's push/pop pair, on the TTNN CommandListBuilder recording on the calling thread. Binds are matched against
// the device operations it records while they are pushed. pop_cmd_list_binds(true) throws if a bind of the popped
// scope reached no recorded program.
void push_cmd_list_binds(std::string op_name, std::vector<CmdListBind> binds);
void pop_cmd_list_binds(bool validate);

// RAII push_cmd_list_binds / pop_cmd_list_binds. Scopes nest. Needs a recording TTNN CommandListBuilder.
class CmdListBindScope {
public:
    CmdListBindScope(std::string op_name, std::vector<CmdListBind> binds);
    ~CmdListBindScope();  // pops without validating unless finish() ran

    CmdListBindScope(const CmdListBindScope&) = delete;
    CmdListBindScope& operator=(const CmdListBindScope&) = delete;
    CmdListBindScope(CmdListBindScope&&) = delete;
    CmdListBindScope& operator=(CmdListBindScope&&) = delete;

    // Pops, then throws if a bind reached no recorded program, naming the bind and the op.
    void finish();

private:
    bool popped_ = false;
};

// What a program factory supports for one program, independent of any call's binds. Built by CmdListBinder on each
// program-cache miss and matched against the active binds when the program is recorded.
struct CmdListTensorBinding {
    std::string invoke_name;
    tt::tt_metal::experimental::TensorParamName program_spec_param_name;
};

struct CmdListRuntimeArgBinding {
    tt::tt_metal::experimental::KernelSpecName kernel_name;
    std::string arg_name;
    std::vector<tt::tt_metal::experimental::NodeCoord> nodes;
};

struct CmdListCommonRuntimeArgBinding {
    tt::tt_metal::experimental::KernelSpecName kernel_name;
    std::string arg_name;
};

struct CmdListScalarBinding {
    std::string invoke_name;
    std::string metal_cmd_list_param_name;  // unqualified; defaults to invoke_name
    CmdListEncodeFn encode;
    std::vector<CmdListRuntimeArgBinding> runtime_args;
    std::vector<CmdListCommonRuntimeArgBinding> common_runtime_args;
};

struct CmdListBindings {
    std::string factory_name;  // set by the adapter; qualifies Metal param names
    std::vector<CmdListTensorBinding> tensors;
    std::vector<CmdListScalarBinding> scalars;
};

// Builds a program factory's CmdListBindings in map_cmd_list_parameters, checking each binding against the program:
//   ttnn::command_list::CmdListBinder binder{spec, run_params};
//   binder.tensor(/*invoke_name=*/"input_tensor", /*program_spec_param_name=*/TensorParamName{"src"});
//   binder.param(/*invoke_name=*/"scale", /*encode=*/encode_float32_bits)
//       .runtime(/*kernel_name=*/KernelSpecName{"reader"}, /*arg_name=*/"pre_scale");
//   return binder.finish();
class CmdListBinder {
public:
    CmdListBinder(
        const tt::tt_metal::experimental::ProgramSpec& spec,
        const tt::tt_metal::experimental::ProgramRunArgs& run_params);

    // Binds op keyword `invoke_name` to TensorParameter `program_spec_param_name`. A keyword may bind several.
    void tensor(
        std::string_view invoke_name, const tt::tt_metal::experimental::TensorParamName& program_spec_param_name);

    // One Metal parameter: one encoding, bound to one or more kernel arguments of either kind.
    class Param {
    public:
        // Binds runtime argument `arg_name` of `kernel_name`. nodes defaults to every node run_params sets it on.
        Param& runtime(
            const tt::tt_metal::experimental::KernelSpecName& kernel_name,
            std::string arg_name,
            std::optional<std::vector<tt::tt_metal::experimental::NodeCoord>> nodes = std::nullopt);

        // Binds common runtime argument `arg_name` of `kernel_name`.
        Param& common_runtime(const tt::tt_metal::experimental::KernelSpecName& kernel_name, std::string arg_name);

    private:
        friend class CmdListBinder;
        Param(CmdListBinder& binder, std::size_t index) : binder_(binder), index_(index) {}
        CmdListBinder& binder_;
        std::size_t index_;  // into bindings_.scalars
    };

    // Declares a Metal parameter fed by op keyword `invoke_name` through `encode`. metal_cmd_list_param_name defaults
    // to invoke_name; name it only when one keyword needs a second encoding.
    Param param(
        std::string_view invoke_name,
        CmdListEncodeFn encode = encode_uint32,
        std::optional<std::string> metal_cmd_list_param_name = std::nullopt);

    // Returns the bindings. Call it last: it moves them out of the binder.
    CmdListBindings finish();

private:
    const tt::tt_metal::experimental::KernelSpec& kernel_spec(
        const tt::tt_metal::experimental::KernelSpecName& kernel_name) const;
    void claim_arg(bool common, const tt::tt_metal::experimental::KernelSpecName& kernel_name, const std::string& arg);

    const tt::tt_metal::experimental::ProgramSpec& spec_;
    const tt::tt_metal::experimental::ProgramRunArgs& run_params_;
    CmdListBindings bindings_;
    std::unordered_set<std::string> bound_tensors_;
    std::unordered_set<std::string> declared_params_;
    std::set<std::tuple<bool, std::string, std::string>> bound_args_;  // (common, kernel, argument)
};

}  // namespace ttnn::command_list
