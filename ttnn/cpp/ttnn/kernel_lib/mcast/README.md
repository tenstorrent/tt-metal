<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Multicast helper library

This library builds multicast topologies on the host and exposes matching sender and receiver pipes to data-movement
kernels. The host serializes the topology, resource, and protocol metadata; the device entry point decodes it and
selects hardware multicast or chain unicast without requiring separate kernel implementations.

## Start here

| Side | Entry point | Use |
| --- | --- | --- |
| Host | [`host/mcast.hpp`](host/mcast.hpp) | Configure an `Mcast`, attach it to a program-construction API, and append its resources and arguments. |
| Device, `ProgramDescriptor` or direct `Program` | [`kernel/mcast_args.hpp`](kernel/mcast_args.hpp) | Construct `McastArgs<CT_OFFSET, RT_OFFSET>()` from the offsets supplied by the host. |
| Device, Metal 2.0 `ProgramSpec` | [`kernel/mcast_args_metal2.hpp`](kernel/mcast_args_metal2.hpp) | Construct the channel with `MCAST_ARGS(prefix)`, where `prefix` matches the host attachment prefix. |

The host and device entry-point headers contain usage examples. Both device entry points produce the same channel
interface: kernels obtain a sender with `sender(noc)` and a receiver with `receiver(noc)`. Role-polymorphic kernels can
use `optional_sender(noc)`, `optional_receiver(noc)`, `can_send()`, and `can_receive()`.

For example, a Metal 2.0 host attachment and device declaration share the `channel` prefix:

```cpp
// Host
mcast.attach(spec, run_args, "channel", target_kernels);

// Device
constexpr auto channel = MCAST_ARGS(channel);
```

The decoded channel selects the pipe implementation from the host-configured `TransferMode`. A regular or irregular
hardware multicast uses `SenderPipe` and `ReceiverPipe`. `TransferMode::ChainUnicast` uses `ChainSenderPipe` and
`ChainReceiverPipe`; a chain receiver calls `receive_and_forward()` so intermediate receivers relay the payload.

## File map

### Host

| Path | Role |
| --- | --- |
| [`host/mcast.hpp`](host/mcast.hpp) | Public host API and usage examples. Defines `Mcast`, topology and sender configuration types, attachment overloads, and argument appenders. |
| [`host/mcast.cpp`](host/mcast.cpp) | Public `Mcast` implementation and receiver-group construction. |
| [`host/mcast_impl.hpp`](host/mcast_impl.hpp) | Internal `McastImpl` lowering interface and prepared topology state. |
| [`host/mcast_impl.cpp`](host/mcast_impl.cpp) | Topology preparation, metadata construction, and compile-time/runtime argument lowering. |
| [`host/program_adapters/mcast_descriptor_adapter.cpp`](host/program_adapters/mcast_descriptor_adapter.cpp) | Adapts `Mcast` attachment to `ProgramDescriptor` and `KernelDescriptor`. |
| [`host/program_adapters/mcast_legacy_program_adapter.cpp`](host/program_adapters/mcast_legacy_program_adapter.cpp) | Allocates multicast resources for the direct, legacy `Program` construction path. |
| [`host/program_adapters/mcast_spec_adapter.cpp`](host/program_adapters/mcast_spec_adapter.cpp) | Adapts `Mcast` to Metal 2.0 `ProgramSpec` and `ProgramRunArgs`, including named resource bindings. |

### Device

| Path | Role |
| --- | --- |
| [`kernel/mcast_args.hpp`](kernel/mcast_args.hpp) | Main device channel interface, positional argument decoder, transport selection, and kernel usage examples. |
| [`kernel/mcast_args.inl`](kernel/mcast_args.inl) | Out-of-line definitions for the device channel interface. Included by `mcast_args.hpp`. |
| [`kernel/mcast_args_metal2.hpp`](kernel/mcast_args_metal2.hpp) | Metal 2.0 named-argument and named-resource adapter. Defines `MCAST_ARGS(prefix)`. |
| [`kernel/mcast_semaphore.hpp`](kernel/mcast_semaphore.hpp) | Normalizes legacy numeric semaphore IDs and Metal 2.0 semaphore binding tokens. |
| [`kernel/pipes/pipe_common.hpp`](kernel/pipes/pipe_common.hpp) | Device-only policy shared by the hardware-multicast and chain-unicast pipes. |
| [`kernel/pipes/mcast_pipe.hpp`](kernel/pipes/mcast_pipe.hpp) | Hardware-multicast `SenderPipe` and `ReceiverPipe` declarations. |
| [`kernel/pipes/mcast_pipe.inl`](kernel/pipes/mcast_pipe.inl) | Hardware-multicast pipe definitions. Included by `mcast_pipe.hpp`. |
| [`kernel/pipes/chain_pipe.hpp`](kernel/pipes/chain_pipe.hpp) | Chain-unicast `ChainSenderPipe` and `ChainReceiverPipe` declarations. |
| [`kernel/pipes/chain_pipe.inl`](kernel/pipes/chain_pipe.inl) | Chain-unicast pipe definitions. Included by `chain_pipe.hpp`. |

Most kernels should include one of the `mcast_args` headers and obtain their pipe through the decoded channel. Include
the pipe headers directly only when constructing the lower-level transport primitives explicitly.

### Shared host/device format

| Path | Role |
| --- | --- |
| [`mcast_protocol.hpp`](mcast_protocol.hpp) | Shared host/device multicast protocol types, metadata, constants, and serialized argument layouts. |
| [`mcast_protocol.inl`](mcast_protocol.inl) | Constexpr protocol layout and encoding definitions included by `mcast_protocol.hpp`. |
| [`mcast_common_metal2.hpp`](mcast_common_metal2.hpp) | Shared Metal 2.0 generated-name spellings used by the host adapter and `MCAST_ARGS(prefix)`. |
