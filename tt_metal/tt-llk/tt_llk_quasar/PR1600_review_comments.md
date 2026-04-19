# Review Comments from tenstorrent/tt-llk#1600

Migrated from: https://github.com/tenstorrent/tt-llk/pull/1600

## Unresolved / Carry-forward Comments

### [Copilot] on `ckernel_sfpu_trigonometry.h:121` (_calculate_sine_ ITERATIONS)
> `_calculate_sine_` declares a template parameter `ITERATIONS` but the function uses the runtime `iterations` argument instead, leaving `ITERATIONS` unused. Either remove `ITERATIONS` from the template parameters or switch the loop bound to `ITERATIONS` (and drop the runtime argument).

**⚠️ ACTION NEEDED:** Resolve unused template parameter `ITERATIONS` vs runtime `iterations`.

---

### [Copilot / nvelickovicTT] on `ckernel_sfpu_trigonometry.h:224` (_calculate_cosine_ ITERATIONS)
> Same issue as sine — `ITERATIONS` template parameter is declared but `iterations` runtime argument is used.

**Response (nvelickovicTT):** "I guess there is no need for both ITERATIONS and iterations. That is something the AI reviewer could check easily."

**⚠️ ACTION NEEDED:** Same fix as sine — unify to one iteration control.

---

### [Copilot] on `ckernel_sfpu_trigonometry.h:340` (_calculate_acosh_ runtime iterations)
> `_calculate_acosh_` hard-codes iteration count via `ITERATIONS` template parameter (default 8) without a runtime `iterations` argument. If `TEST_FACE_R_DIM` ever changes from default (16), this will process wrong number of rows. Add runtime `iterations` parameter or static assertion.

---

### [Copilot] on `ckernel_sfpu_trigonometry.h:383` (_calculate_asinh_ runtime iterations)
> Same as acosh — inconsistent with other Quasar SFPU entrypoints that take a runtime `iterations` argument.

---

### [nvelickovicTT] on `ckernel_sfpu_trigonometry.h:470` (_calculate_atanh_ ITERATIONS)
> Not consistent where it places iteration parameter.

**⚠️ ACTION NEEDED:** Make all 5 trig functions consistent in how they handle iteration count (either all use runtime `iterations` or all use compile-time `ITERATIONS`).

---

### [Copilot] on `tests/python_tests/quasar/test_sfpu_trigonometry_quasar.py:133` (boundary coverage)
> Test input generation intentionally clamps values to "safe" domains (acosh to [1,10], atanh to [-0.95, 0.95]), meaning domain-handling branches are never exercised. Add targeted test cases for boundary and out-of-domain values.

**⚠️ ACTION NEEDED:** Add boundary/out-of-domain test cases for acosh and atanh.

---

### [nvelickovicTT] on `ckernel_sfpu_trigonometry.h:475` (llk_defs.h entry)
> Should this also make changes in llk_defs.h file? Or is it not needed?

**⚠️ ACTION NEEDED:** Clarify whether trig ops need a `SfpuType` enum entry. If yes, add them to `llk_defs.h`.
