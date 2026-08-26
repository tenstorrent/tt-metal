# AGMM generated table contract

This is the only supported integration boundary between the offline 32-chip
predictor pipeline and TT-metal. The exporter replaces
`agmm_registry_data.hpp` with deterministic C++ constants. TT-metal compiles
those constants into the native AGMM operation. Runtime Python, JSON parsing,
model inference, wrapper dispatch, environment-selected sidecars, and network
or filesystem lookup are not part of this contract.

## Required C++ surface

The generated header must define exactly these objects and accessors in
`ttnn::experimental::all_gather_minimal_matmul_registry::generated`:

```cpp
inline constexpr compact::TableLock kLock{/* explicit fields */};
inline constexpr std::array<compact::EntryDescriptor, N> kEntries{/* entries */};

static_assert(
    compact::validate_table_lock(kLock, kEntries) ==
    (N == 0 ? compact::TableValidationStatus::Empty
            : compact::TableValidationStatus::Valid));

inline constexpr const compact::TableLock& lock() noexcept { return kLock; }
inline constexpr std::span<const compact::EntryDescriptor> entries() noexcept { return kEntries; }
```

The checked-in empty fixture uses `N == 0`. A production export must have
`N > 0`, set `entry_count == N`, use the schema/ABI constants declared in
`agmm_registry_descriptor.hpp`, and explicitly initialize every field. It must
not depend on aggregate defaults for a populated lock or entry.

All entries must:

- have a nonzero `entry_id`;
- use the lock's key, replay, and codegen ABI versions;
- have the exact lock `runtime_capability_sha256` in their device key;
- be strictly increasing by `KeyDescriptor::operator<=>`; and
- therefore contain no duplicate exact key.

The constexpr validator makes violations compilation errors in the generated
header. The runtime validates the same typed contract before exact lookup and
fails closed on malformed or incompatible tables.

## Digest rules

Every digest in a populated lock is required and is a 32-byte SHA-256 value.
The exporter writes bytes as explicit unsigned hexadecimal literals in digest
order. The following domain-separated byte encodings define the two generated
identities:

- `entry_id = SHA256("ttnn-agmm-entry-v1\0" || encode(key) || encode(replay))`
- `content_sha256 = SHA256("ttnn-agmm-table-v1\0" || u64(N) ||
  entry_id[0] || ... || entry_id[N-1])`

`encode` visits fields in their declaration order in
`agmm_registry_descriptor.hpp`. Unsigned and signed integers use fixed-width
little-endian two's-complement bytes, booleans use one byte (`0` or `1`), enums
use their explicitly stored integer field, arrays encode every element without
a length prefix, and structs recursively encode their fields without C++
padding. Floats never enter the ABI directly; stored IEEE-754 values use their
exact `uint32_t` bits. No text rendering, locale, native struct bytes, generic
C++ hash, or process-local identifier is permitted.

The remaining lock digests bind independent inputs:

- `semantic_source_sha256`: the canonical `semantic_dependencies.txt` path and
  file-content manifest used for certification;
- `build_identity_sha256`: the exact reviewed TT-metal build identity;
- `runtime_capability_sha256`: the canonical 32-chip capability and fabric
  attestation used by every entry;
- `evidence_manifest_sha256`: the immutable silicon evidence manifest;
- `predictor_sha256`: the immutable predictor artifact and feature schema; and
- `exporter_sha256`: the exporter implementation plus its schema/configuration.

The codegen repository must regenerate the header byte-for-byte and test both
digests independently before proposing it to TT-metal. Native code does not
claim to recompute SHA-256 at dispatch time; it binds the reviewed values and
compares semantic, build, and runtime capability identities to independently
obtained runtime/build attestations.

## Promotion gates

A nonempty generated table is necessary but not sufficient to select a recipe.
Production selection remains fail closed until TT-metal can build the complete
exact request and independently provide nonzero semantic, build, and runtime
capability digests. Shadow must demonstrate exact hits first. Only certified
keys may be exported; predictor rankings for unseen keys are candidates for
silicon validation, never executable recipes.
