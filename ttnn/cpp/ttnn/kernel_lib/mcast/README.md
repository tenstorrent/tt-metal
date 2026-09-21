# Multicast API v27 / wire v3

Decoder semaphore bindings are internal: kernels use pipe operations rather than
accessing `data_ready`, `consumer_ready`, or `signal_source` on `McastArgs` or
`MCAST_SPEC_ARGS`. Separate operation phases own their synchronization rather
than borrowing the channel's internal bindings.

Present blocks start with a packed control word whose low four bits are tag `3`;
word `0` remains the one-word absent block with no runtime payload. Both positional
and native-spec decoders reject previous tags `1` and `2`. The control word and
metadata participate in kernel specialization/cache keys; `MCAST_PIPE_API_VERSION` is documentation, not a
runtime compatibility check. Rebuild host code when updating the wire format.

## Compact compile-time arguments

`mcast_compile_time_args.hpp` defines the shared host/device CT codec. The control
word packs the version, protocol flags, receiver presence, multicast mode,
rectangle capacity, roles/capabilities, coordinate encoding, and optional-field
selectors. The remaining words are, in order:

1. Data-ready ID, consumer-ready ID only with handshaking, and signal-source ID only for chain.
2. Known remote count, only for a sending placement with one rectangle.
3. A distinct constant ACK count, only when required. Equal constant ACK/fanout
   counts share one word; dynamic ACKs keep their existing RT field.
4. Rotating span, only for rotating placements.
5. Columns, rows, X-range count and Y-range count, only when the receiving
   placement uses compressed coordinates.

Semaphore IDs, counts and dimensions remain full-width `uint32_t`; only bounded
flags/enums are bit-packed. Unused decoded metadata has canonical default values.
The operation-facing API and RT layouts/payloads are unchanged.

Counts below include both attachment offsets, per kernel/channel, not per program:

| Situation | Before | Now |
| --- | ---: | ---: |
| Fixed single-rectangle sender, handshake ACK equals uniform fanout | 20 | 6 |
| Same sender, separate constant ACK count | 20 | 7 |
| Fixed receiver, handshake enabled | 20 | 5 |
| Fixed receiver, handshake disabled | 20 | 4 |
| Rotating mixed role, range-compressed, uniform fanout and equal ACK | 20 | 11 |
| Chain | 21 | 6 |
| Absent | 3 | 3 |

ProgramSpec uses the same compact metadata in native `compile_time_varargs`, with
named CT/RT offsets. It keeps native semaphore bindings and emits no numeric IDs.
For a fixed uniform single-rectangle channel, its sender uses 4 scalar CT entries
and its receiver 3, down from 17 each (bindings are counted separately). Existing
CT/RT prefixes, multiple channels and absent channels compose without caller changes.

## Runtime arguments

`mcast_common.hpp::RuntimeLayout` defines the fixed runtime layout of each
kernel/channel. Descriptor and ProgramSpec attachment infer roles over the
complete placement, including inactive cores. Direct Program callers can use
`append_kernel_args_to(ct, per_core_rt, placement)` after binding semaphores;
it emits matching CT/RT transactionally and returns their starting offsets.
The separate family-level append methods remain paired, conservative APIs.
Do not combine their runtime blocks with placement-specific compile-time blocks.

The compact order is optional roles, rotating sender phase, rectangle count,
nonuniform handshake ACK, receiver sender-coordinates, and fixed-capacity
rectangle records. Rectangle records contain conditional bounds, remote count,
and mode. Known zero fanout is explicit; loopback fanout is remote count plus one.
LocalCopy omits unused bounds. Chain runtime bytes are unchanged; its three
semaphore IDs immediately follow the compact CT control word.

Regular rotating schedules may encode ordered inclusive X/Y coordinate ranges,
including senders outside the receiver set. Preparation validates every
reconstructed mapped worker coordinate and phase; incompatible groups and
schedules that cannot be represented exactly use explicit pairs.
Compression is selected only when it saves runtime words. Sender-only kernels
store no coordinates. Receivers retain coordinates even without helper ACKs,
because callers may use `sender_x()`/`sender_y()` for independent signaling.
Calling these accessors on a placement without coordinates is a compile error.

Construct pipes once outside transfer loops. A compressed receiver owns its
expanded lookup table, constructed directly in the returned pipe or optional;
raw pointer/view callers retain their existing lifetime requirements. The table
costs `2 * sender_count * sizeof(uint32_t)` bytes, so smaller transmitted
arguments do not imply lower local-memory usage or improved execution time.
Unavailable optional faces are well-formed empty optionals; unavailable direct
faces are compile errors. `can_send()` and `can_receive()` remain member queries.

## Unified host frontend

Include `host/mcast_host_unified.hpp` for the additive `Mcast` frontend. Existing
`McastFamily`, `Mcast1D`, `Mcast2D`, and `McastConfig` remain available. The new
constructor prepares an owned family immediately and delegates
all descriptor, ProgramSpec, direct-Program, and topology interfaces to it.

```cpp
Mcast channel(
    device,
    McastUnifiedConfig{.noc = noc},
    receivers,
    4,  // Four consecutive receivers per independent group.
    McastRotatingSenderConfig{});
channel.attach(descriptor, "input", std::array{std::ref(kernel)});
```

The optional final `receiver_order` argument defaults to `McastCoreOrder::RowMajor`.
Receiver sets are sorted row-major (y,x) or column-major (x,y), then split into
equal consecutive groups. Empty receivers, zero group size, and incomplete final
groups are rejected. Custom receiver partitions remain available through
`McastFamily::add_group()`; the unified frontend has no ordered-vector overload.

Fixed senders use a group-local index. Uniform placement requires that index
to be in range; Staggered uses `(sender_index + group_index) % group_size`, with
overflow-safe addition. `McastRotatingSenderConfig{}` selects all group receivers
in their order. Its optional `sender_grid` selects a separate grid, sorted in
receiver order and divided evenly among groups; this does not infer spatial
alignment. There is no separate sender-order option. `McastExplicitSenderConfig`
supplies one ordered list per group, including singleton lists for external fixed
senders or custom rotating schedules. All lists must have the same nonzero length.
Independent group footprints may not overlap, including external sender cores.

`McastUnifiedConfig::handshake_cores` is copied during construction. With
handshaking enabled, null means all receivers and an explicitly empty set means
zero ACKs. A nonempty set must be a subset of receivers. Each sender expects the
number of participating handshake cores in its group, minus itself if included.
For receivers `{0,1,2,3}`, handshake cores `{0,1}`, and senders `{0,3}`, counts
are `{1,2}` while the full destination set is retained. Legacy scalar ACK
overrides are unaffected; the private subset bridge emits the existing wire.

This is a sender expectation, not an automatic kernel ACK filter. Kernels must
ensure excluded cores do not acknowledge, and keep passive landing storage live.
With handshaking disabled the pointer must be null; the same early-arrival and
landing-region ownership requirements as `McastConfig` apply. Native ProgramSpec
continues to reject numeric semaphore configuration and accepts named adoption.

The existing three-rectangle and chain limits remain. An explicit partial/empty
subset is rejected when chain forwarding is actually selected; null or the full
receiver set is accepted. A rectangular family still selects multicast even when
the irregular-set policy requests chain. Unequal receiver group sizes and rotating
chains are not supported by this frontend; use existing supported family patterns
instead of expanding destinations or dropping forwarding participants.

## Unified frontend feasibility coverage

Native fixtures exercise communication semantics, not operation migrations or
historical argument-list identity. Matmul fixed/rotating groups, DRAM storage and
compute overlap, Conv2D rotating activation with passive landing, Conv3D weight
strips, LayerNorm readiness/final-statistics policies, GroupNorm row/column and wrapped
groups, attention's external rotating senders, and TopK's external readiness
sender are represented with independent membership, sender-order, and ACK checks.
Supported patterns are checked on both NoCs; default cases also match explicitly
constructed legacy families. Tests retain the full requested receiver membership.

Sharded GroupNorm requires each channel shard to contain whole normalization
groups. Its block-sharded reductions therefore run along the spatial-sharding
axis, not across arbitrary two-dimensional blocks. These groups and the existing
consecutive wrapped groups use ordinary row/column ordering.

The mixed-ACK native device fixture uses the four-core `{1,2}` example, both NoCs,
repeated sender turns, and copied/rebound ProgramSpec invocations. It uses Counter
publication and a distinct landing slot per round, so non-acknowledging receivers
may lag safely. Operation migrations, distributed DiT fabric behavior, unequal
receiver groups, more than three rectangles, partial-handshake chains, and rotating
chains are not claimed as supported by these fixtures.
