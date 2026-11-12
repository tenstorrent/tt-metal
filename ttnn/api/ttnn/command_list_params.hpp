// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/arg_value.hpp>

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

}  // namespace ttnn::command_list
