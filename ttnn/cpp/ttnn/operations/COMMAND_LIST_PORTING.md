# Porting an op factory to TTNN command-list parameters

Experimental. Reference ports: `softmax_program_factory_attention_optimized.cpp` (tensors + a scalar),
`ema_program_factory.cpp` and `accumulation_program_factory.cpp` (tensors only).
API: `ttnn/api/ttnn/command_list_params.hpp`. Tests: `tests/ttnn/unit_tests/base_functionality/test_command_list*.py`.

## What a port is

One static function on a spec factory (one that has `create_program_artifacts`), declared next to it:

```cpp
static ttnn::command_list::CmdListBindings map_cmd_list_parameters(
    const operation_attributes_t&, const tensor_args_t&,
    const tt::tt_metal::ProgramSpec&, const tt::tt_metal::ProgramRunArgs&);
```

It builds the bindings with `CmdListBinder` and returns `binder.finish()`. The adapter calls it on a program-cache
miss. A factory without it still records, with every `CmdListParam` at its default and a warning.
Factories that return a `ProgramDescriptor` or use `cached_program_t` have no `ProgramSpec` and cannot be ported.

## Checklist, per factory

1. **Keyword names are the Python names.** `invoke_name` is the keyword of the public op in its `*_nanobind.cpp`
   (`input_tensor`, `out`, `output_tensor`), not the C++ parameter name. One op may name the same tensor
   differently per variant (`cumsum` takes `input`, `cumprod` takes `input_tensor`): branch on the attributes.
2. **Tensors.** `binder.tensor(keyword, TensorParamName{"<spec param>"})` for every `TensorParameter` the caller
   can pass. Bind an optional tensor only when the factory declares its parameter (`if (tensor_args.x.has_value())`),
   because the binder rejects a name the spec lacks. An output the caller never passes is harmless to bind.
3. **Scalars** bind only when the value reaches the kernel as a runtime or common runtime argument:
   `binder.param(keyword, encode_*).runtime(KernelSpecName, "arg")` or `.common_runtime(...)`. Pick the encoder from
   how the kernel reads the word (`encode_float32_bits`, `encode_bfloat16_pair`, ...).
   - A value that is a **compile-time argument or a define** is part of the program hash. Do not bind it. A new value
     is a different program, so it cannot be updated in a recorded list. Leave a comment saying so.
   - Do not bind attributes that change shapes, grids or memory configs.
4. **Tuples and lists cannot be bound.** `CmdListParam` is a direct keyword argument holding a Tensor or a number.
   Outputs returned as a tuple (`topk`) are not bindable.
5. **Wrappers.** The bound tensor must be the one the factory reads. If the public op permutes, pads, tilizes or
   typecasts first, recording fails with "is not the tensor the device operation reads" (or "through a view with a
   different TensorSpec"). That is the intended safe behaviour. Test the direct path and say in the PR which
   inputs take it.
6. **The wrapper must hand the caller's tensors to the primitive, in and out.** Two cases that look bindable are not:
   - A wrapper that passes `nullopt` for an optional output and swaps the caller's storage afterwards
     (`cumsum`/`cumprod` `out`, TODO #37807) gives the program its own output, so binding `out` fails the allocation check.
   - A wrapper that reads the result on the host after the primitive (`prod` over all elements, `prod_nc_op`/
     `prod_result_computation_WH_B0`) does work that is not recorded and is not redone on replay. Do not port it.
   Arch-dependent preprocessing counts too: on Quasar cumprod always permutes, so skip such tests there.
7. **In-place:** if an output aliases an input, bind the same keyword to both parameters (see softmax).

## Done means

- Declared in the header, defined beside `create_program_artifacts`, clang-formatted.
- A Python test records with `CmdListParam`, calls `update_args`, replays, and checks against torch. Cover a cache hit.
- Scalars: a test updates the value and checks the output changes accordingly.
- No "unsupported parameter" warning for the arguments you bound.

## Status

| Directory | Ported | Notes |
|---|---|---|
| normalization/softmax | attention_optimized | other 6 factories pending |
| reduction | argmax (single, multi), prod_nc, accumulation (cumsum, cumprod), ema, moe, sampling, topk single-core | tensor-only. `sampling` seed (runtime arg) pending. `ema` alpha is compile-time, not bindable. `prod_all` is not ported (host finalization). cumsum/cumprod bind the input only. |

## Known limitations

A command list records Metal 2.0 programs only. A factory that still returns a `ProgramDescriptor` (or uses
`cached_program_t`) fails recording with "Command lists only support Metal 2.0 programs", with or without a
`CmdListParam`. Which factory runs can depend on the input, so one op may record for some inputs and not others:

| Op | Records | Fails to record |
|---|---|---|
| `topk` | single-core factory | multi-core factory (wide inputs: width >= 8192, or width >= 1024 with at most 2 tile rows, and `k` <= 64; never selected on Quasar) |
| `argmax` | reductions over H or W, and reduce-all | reductions over a non-HW dim (`argmax_nc`, a descriptor factory) |

These close as the descriptor factories move to Metal 2.0.
