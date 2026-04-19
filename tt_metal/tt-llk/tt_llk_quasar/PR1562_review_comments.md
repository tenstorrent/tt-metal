# Review Comments from tenstorrent/tt-llk#1562

Migrated from: https://github.com/tenstorrent/tt-llk/pull/1562

## Unresolved / Carry-forward Comments

### [ldjurovicTT] on `tests/python_tests/quasar/test_sfpu_abs_quasar.py:5`
> Since we now have few tests that have _sfpu_ in their name for quasar i suppose they can all be combined in one bigger sfpu test file that would just sweep through all operations we have available right now. Should be easy task for AI to combine it the way we did in WH/BH.

**Response (vvukomanovicTT):** Noted, I will create a separate PR once I add more SFPU kernels.

---

### [nvelickovicTT] on `tests/sources/quasar/sfpu_abs_quasar_test.cpp:36`
> I would expect more comments here. It is not obvious to the reader why when unpacking to dest they need an additional hw configure for MATH.

---

### [nvelickovicTT] on `tests/sources/quasar/sfpu_abs_quasar_test.cpp:33`
> Can we get AI to generate more comments. Here I would prefer it to explain why it is calling this function and why in this way. Something like: `Setting up data valid scheme for Dest which has UNPACK, SFPU, and PACK as clients when unpacking is done directly to Dest.`

**Response (vvukomanovicTT):** Proposed two comment styles — technical rationale vs. historical context (BH ELWADD workaround). See original PR thread.

---

### [nvelickovicTT] on `tests/sources/quasar/sfpu_abs_quasar_test.cpp` (buffer descriptor dims)
> A comment summarizing values of x, y and z dim would help the reviewer. For example: `In the buffer descriptor X dim represents columns, Y dim rows in a face, where Z dim is number of faces.`

---

### [nvelickovicTT] on `tests/sources/quasar/sfpu_abs_quasar_test.cpp` (MOV2D abbreviation)
> There is no MOV2D. It probably means either MOVA2D or MOVB2D. Avoid abbreviations. Also, ELWADD was used in WH/BH due to a bug (8 row move wasn't working properly), which is not there any more in Quasar, so probably no need to do it like that.

---

### [nvelickovicTT] on `tests/sources/quasar/sfpu_abs_quasar_test.cpp` (wait comment)
> This comment could be better. Reader can't determine easily which are all operations that need to complete, so it is harder to be sure why these 3 exactly are waited for. And not for example, replay?

---

### [nvelickovicTT] on `tt_llk_quasar/llk_lib/experimental/ckernel_sfpu_abs.h:1`
> Am I missing something or it created 2 exactly same files in 2 different locations? Do we need this? Seems wrong.
> (Files: `tt_llk_quasar/common/inc/experimental/ckernel_sfpu_abs.h` and `tt_llk_quasar/llk_lib/experimental/ckernel_sfpu_abs.h`)

**⚠️ ACTION NEEDED:** Investigate whether both file locations are needed or if one should be removed.

---

### [nvelickovicTT] on `tests/sources/quasar/sfpu_abs_quasar_test.cpp:121`
> So, this should be number of needed iterations per face, right? But, in the math call below it iterates on TILE_CNT. It looks like it will do TILE_CNT faces, and not TILE_CNT tiles, no?

**Resolution (nvelickovicTT):** Actually fine — on Quasar it only operates in VectorMode::RC, which is the default and only option, so iterating over 1 face automatically iterates over the whole tile.

---

### [nvelickovicTT] on `tests/sources/quasar/sfpu_abs_quasar_test.cpp:141`
> `num_faces` is not used anywhere.

**Resolution (nvelickovicTT):** Fine, although not immediately clear to the reader. Would be good to prompt AI to include the iterating strategy in the comment.
