# Review Comments from tenstorrent/tt-llk#1574

Migrated from: https://github.com/tenstorrent/tt-llk/pull/1574

## Unresolved / Carry-forward Comments

### [fvranicTT] on `tt_llk_quasar/llk_lib/experimental/ckernel_sfpu_fill.h` (magic numbers)
> There are sfpi constants that we could use instead of the magic numbers `sfpi::SFPLOADI_MOD0_LOWER` (for 10) and `sfpi::SFPLOADI_MOD0_UPPER` (8). It'd be nice to teach the AI to use them.

Additional comments from fvranicTT on same file:
- Use `sfpi::SFPLOADI_MOD0_USHORT` (2) and `static_cast<std::uint16_t>(value)`.
- Use `sfpi::SFPLOADI_MOD0_FLOATB` instead of magic numbers.

**⚠️ ACTION NEEDED:** Replace magic number constants in fill kernel with named sfpi constants.

---

### [fvranicTT] on `tests/sources/quasar/sfpu_fill_quasar_test.cpp:113`
> Since this test should be testing SFPU kernel, I think it's perfectly OK to just use unpack to dest and there's no need to use datacopy. This would simplify test logic in both Python and C++.

**Response (rtawfik01):** sounds good to me!

**⚠️ ACTION NEEDED:** Simplify test to use unpack-to-dest instead of datacopy.

---

### [Copilot] on `tests/python_tests/quasar/test_sfpu_fill_quasar.py` (artifact reference)
> The comment references `codegen/artifacts/fill_phase1_spec.md`, but that file/path doesn't exist in this repo. Please update or remove the reference.

---

### [Copilot] on `tests/python_tests/quasar/test_sfpu_fill_quasar.py` (int/bitcast coverage)
> Test is limited to "Phase 1: Float fill only", but the kernel also has integer-fill and bitcast-fill paths. Add basic functional coverage for those modes or clarify they're out of scope.

---

### [Copilot] on `tt_llk_quasar/llk_lib/experimental/ckernel_sfpu_fill.h` (bitcast comment accuracy)
> The comment says the bitcast path "stores with implied format, preserving all 32 bits", but `p_sfpu::sfpmem::DEFAULT` may truncate/round when the implied format isn't FP32. Adjust the comment to match actual instruction behavior.

---

### [nvelickovicTT] on `tt_llk_quasar/llk_lib/experimental/ckernel_sfpu_fill.h:19`
> This function seems like an overkill. It's a one-liner.

**⚠️ ACTION NEEDED:** Consider inlining the helper function if it's genuinely a one-liner.
