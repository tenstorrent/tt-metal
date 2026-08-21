# A dataflow buffer defect on Quasar, seen in simulation

Companion to the [Quasar uplift report](ttnn/cpp/ttnn/operations/embedding/QUASAR_UPLIFT_REPORT.md),
which covers porting `ttnn.embedding` to Quasar. This document explains only the platform defect that
port ran into, and is written to stand on its own for a reader who has not seen the operation or the
port.

## In one paragraph

A dataflow buffer is a queue
that one small program fills and another drains, and it promises the draining program never sees an
entry before the filling program finished writing it. On Quasar that promise breaks, and the data that
comes out is wrong. Filling one entry with two copy operations instead of one is enough to trigger it,
which is an ordinary thing to write. We found it while porting `ttnn.embedding`, an operation in ttnn
that looks up rows of a table by index, and we then reproduced it with none of that library involved.

Why it matters beyond one operation: filling an entry with more than one copy is normal. A program that
fetches a row in pieces because it does not fit, one that reads a partial width, or one that fetches
some control information before streaming the real data all do it.

The rest of this document explains the machinery, then the smallest reproduction, then the boundaries of
the defect. The precise statement of the condition is at the end of the background section, once the
words in it mean something.

Quasar silicon does not exist yet, so all Quasar results here come from craq-sim, Tenstorrent's Quasar
simulator. Wormhole results are from a real chip. That limit is discussed under
[Silicon](#what-is-still-open).

## Background: the queue and its promise

A **dataflow buffer** hands data from one kernel to another. It is a ring of a fixed number of
**entries**, and the two kernels run at the same time and coordinate through four calls
([dataflow_buffer.h:180-183](tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L180-L183)):

- The **producer** calls `reserve_back(n)` to claim `n` free entries, writes data into them, then calls
  `push_back(n)` to announce "these are full."
- The **consumer** calls `wait_front(n)`, which blocks until `n` entries have been announced, reads
  them, then calls `pop_front(n)` to announce "done, you can reuse them."

All four take a count, so `push_back(2)` announces two entries at once. Under the hood the counters
these calls move are called **credits**, which is where the word in this document's filename and in the
test names comes from. This document says "announce" and "grant" instead, because they are easier to
follow: a **grant** is one return of `wait_front`, the moment the consumer is handed an entry.

`wait_front` is the promise: the consumer never reads an entry the producer has not finished writing.
That promise is what these tests check.

A **scratchpad** is the other thing a kernel can reserve in SRAM: a plain private region, not a ring,
with no announcements attached. It appears here only because an extra transfer into one was our first
suspect, and it turned out to be a red herring.

Two more pieces of vocabulary this document needs:

- **Surplus** means total transfers minus total announcements, counted over a whole run. A producer that
  fills four entries with two transfers each performs eight transfers and four announcements, so its
  surplus is four. Every surplus number below is a run total, never a per-entry figure. Note that the
  test sources say it the other way round, per entry, so a figure quoted there will not match one here.
- A producer calls `async_read_barrier()` after issuing its transfers. That blocks until they have all
  landed, so the entry really is complete before `push_back` announces it. This matters because the
  first thing to suspect in the code below is that the producer announced too early, and the barrier is
  what rules that out.

### The condition, precisely

On Quasar a dataflow buffer misbehaves when **both** of these hold:

1. the producer has a **surplus**, meaning it performs more transfers than it announces, and
2. the **consumer is a data movement kernel** rather than a compute kernel.

The two are not symmetric, and it helps to see why. Condition 1 is what makes the count wrong.
Condition 2 is what makes the wrong count do damage, because `wait_front` has two entirely separate
implementations underneath. A data movement consumer runs a small loop on its RISC-V core, reading the
credit counter until it is high enough. A compute consumer never runs that loop; the unpacker performs
the wait. Only the first is affected.

Everything we measured is consistent with one model: announcements are counted correctly, and each
surplus transfer adds one grant on top of them. What the hardware actually derives that count from is
still open.

## The smallest reproduction

One producer, one consumer, one ring, one round trip through DRAM. No compute kernel, no scratchpad,
nothing from ttnn. The whole thing is
[test_dfb_gen2_split_read_repro_hw.cpp:69-168](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_split_read_repro_hw.cpp#L69-L168).

The producer's entire body is this
([dfb_split_read_producer.cpp:34-41](tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_split_read_producer.cpp#L34-L41)):

```cpp
for (uint32_t i = 0; i < num_entries; i++) {
    buf.reserve_back(1);
    noc.async_read(dram, buf, half, {.bank_id = 0, .addr = src_addr}, {.offset_bytes = 0});
    noc.async_read(dram, buf, half, {.bank_id = 0, .addr = src_addr + half}, {.offset_bytes = half});
    noc.async_read_barrier();
    buf.push_back(1);
    src_addr += entry_size;
}
```

The consumer is the textbook drain loop: `wait_front(1)`, write the entry out to DRAM, `pop_front(1)`
([dfb_split_read_consumer.cpp:29-35](tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_split_read_consumer.cpp#L29-L35)).

The only thing that differs from a program that works is one line. The correct version reads the whole
entry in one transfer; this one reads it as two halves, into the first and second half of the same
entry. Both land inside the entry it just reserved, the barrier waits for both, and then it announces
one entry. Four entries go through a ring two entries deep, so the surplus is four.

Both versions put the same bytes in the same place before announcing, so splitting the read should make
no difference at all. On Wormhole it makes none. On Quasar:

```
entry 1 of 4 came back wrong: expected 4352, got 0
entry 2 of 4 came back wrong: expected 4608, got 4096
entry 3 of 4 came back wrong: expected 4864, got 4352
```

Each entry is filled with a distinct value derived from its index, spaced 256 apart, so entries 0 to 3
begin with 4096, 4352, 4608 and 4864. Reading the failure that way: entry 1 came back as zero, meaning
the consumer was handed an entry nothing had written yet. Entries 2 and 3 came back holding the values
belonging to entries 0 and 1, meaning the consumer was reading entries the producer had not caught up
to. Entry 0 happened to be correct. This is identical across repeated runs.

**One note on direction, so the rest of the document reads consistently.** We describe this throughout
as the consumer being released early. The evidence cannot actually distinguish that from the producer
running ahead and overwriting entries, because free space and occupancy are two readings of the same
pair of counters. See [What is still open](#what-is-still-open).

## The fuller test suite

The minimal case above is deliberately one scenario. A second file,
[test_dfb_gen2_credits_hw.cpp:1-94](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L1-L94),
explores the space around it, which is what turns "this one program misbehaves" into a statement about
the condition. Every test that varies the transfer count uses a ring two entries deep; the controls
deliberately sweep depth 2 and 4, since depth is one of the things they rule out.

**How its tests decide pass or fail.** Two independent checks, and both are needed.

- The **per-grant check**: the consumer records the first word of each entry the instant its grant
  arrives. Entries carry distinct values, so grant *k* must see entry *k*'s value.
- The **end-to-end check**: the data delivered to DRAM must equal the data that went in.

The per-grant check is the more sensitive of the two, because a consumer released early often reads the
entry after the producer's transfer happens to land anyway, and then the output looks correct. The
end-to-end check is kept because it is the one that shows the defect actually damages a program rather
than only tripping a probe, and it is the only check the minimal reproduction above uses.

### The surplus is the trigger

The table below has three rows. Each pushes four entries' worth of data, so the announcement count is
four every time and only the transfer count changes. Every transfer targets the ring itself. One
producer kernel covers all three rows, varying only how it splits its reads
([dfb_ratio_probe_producer.cpp:48-87](tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_ratio_probe_producer.cpp#L48-L87)).

| Producer's pattern | transfers | announcements | surplus | Wormhole | Quasar |
|---|---|---|---|---|---|
| four reads, one per entry | 4 | 4 | 0 | pass | pass |
| **eight half-size reads, two per entry** | 8 | 4 | **+4** | pass | **fail** |
| two double-size reads, each announcing two entries | 2 | 4 | -2 | pass | pass |

A surplus over-grants the consumer. A matching count and a deficit both behave correctly, so this is
not simply "the counting is unreliable"; it goes wrong in one direction only. The failing row is
[RatioTwoReadsPerAnnouncedSlot](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L796-L814);
the deficit row is
[RatioOneReadPerTwoAnnouncedSlotsCompletes](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L844-L855).

### The consumer's kind is the second condition

Same producer, same surplus. The only change is putting a compute kernel in the draining position, with
a data movement kernel after it to write the result out to DRAM where the test can check it
([run_compute_consumer_ratio_case](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L910-L1021)).
The compute kernel copies each entry straight through unchanged, so nothing is being tested except
which hardware performs the wait
([dfb_tile_copy_compute.cpp:36-51](tests/tt_metal/tt_metal/test_kernels/compute/dfb_tile_copy_compute.cpp#L36-L51)).

| Transfers per entry | surplus | Consumer | Wormhole | Quasar |
|---|---|---|---|---|
| one | 0 | data movement | pass | pass |
| two | +4 | data movement | pass | **fail** |
| one | 0 | compute | pass | pass |
| two | +4 | compute | pass | pass |
| eight | +28 | compute | pass | pass |
| eight | +28 | data movement | pass | **deadlocks** |

This is what condition 2 predicts: the compute consumer never reads the credit counter, so a wrong
count has nothing to act on.

The last two rows are what make that conclusion safe. A compute consumer passing the two-transfer case
on its own would prove little, because the compute kernel's copy takes time, so it might simply be
reading late enough that the correct data has already arrived, hiding the defect rather than being
immune to it. A surplus of 28 against a two-entry ring cannot be hidden that way: it stops the data
movement consumer completely, and the compute consumer delivers bit-exact output.

### What a surplus of exactly one looks like

The clearest picture comes from adding a single extra transfer on top of four reads that fill four
entries
([MinimalExtraReadNoScratchpad](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L584-L684)).
Its per-grant record on Quasar, with "unwritten" meaning the entry read as zero:

```
grants:  [unwritten, unwritten, entry 0's value, entry 1's value]
```

The first two grants hand over entries nothing has written. After that the consumer stays exactly two
entries behind, and two is the ring depth: being one grant ahead in a two-entry ring means reading the
position that was last written two entries ago.

Larger surpluses do not produce a pattern this tidy. Do not read the minimal reproduction's output as
the counter-example, though: that test reports the delivered data, not the per-grant record, and the two
are not comparable. See [What is still open](#what-is-still-open).

## Ruling out what it is not

**It is not about where a transfer lands.** That is the natural suspect, because in the embedding
operation the extra transfer reads into a scratchpad. One test sweeps eight variations of what else the
producer does
([ScratchpadUsePatternsThatDisturbTheBuffer](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L1097-L1265)).
Six fail on Quasar and all eight pass on Wormhole. The six failures are an extra transfer into a
scratchpad through its handle, that same transfer repeated every second entry, a transfer to the
scratchpad's own address reached through a plain pointer instead of the handle, a transfer to the far
end of a large scratchpad, a transfer to an unrelated address with a scratchpad reserved but untouched,
and a transfer to an unrelated address with **no scratchpad anywhere in the program**.

Of the two that pass on Quasar, one is the baseline that does nothing extra, so it carries no
information. The informative one is the other: the producer writes to that same scratchpad memory with
an ordinary store instruction instead of over the NoC, and the ring is undisturbed. A store is not a
transfer, so it does not change the count, which is exactly what the surplus explanation predicts.
[MinimalExtraReadNoScratchpad](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L584-L684)
is the same result again with no scratchpad declared at all.

**It is not Quasar's implicit sync.** Quasar has a feature, called implicit sync in the code, that can
announce an entry from the NoC transfer that filled it rather than waiting for `push_back`. That is the
obvious culprit, and there is an existing issue, #50328, describing a problem with it whose documented
fix is to switch the feature off. Switching it off here changes nothing: the failing case produces
bit-identical results with the feature on and off, down to the recorded grant values
([RatioTwoReadsPerAnnouncedSlotNoImplicitSync](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L822-L838)).
So this is a different defect that resembles that one, and it should not be filed as a duplicate or
closed as one.

That null result is less surprising than it first looks, and it is worth being precise about why rather
than presenting it as a mystery. Implicit sync applies to a specific form of the read call, one that
takes no size argument and lets the ring supply the transfer parameters itself
([noc.h:796-812](tt_metal/hw/inc/api/dataflow/noc.h#L796-L812)). Every kernel here passes an explicit
byte count, so it uses the ordinary form and these transfers were never on the implicit-sync path to
begin with. Switching the feature off was still worth doing, because it also covers the announcement
calls and because it is the documented fix for #50328, but it was never likely to be the cause. What
does count these transfers remains open.

## Why most of the operator library is unaffected

ResNet already runs on Quasar, which raises a fair question: if this defect is real, why does a whole
working model not hit it?

We audited every operation written for Quasar, all 27 of them under
[experimental/quasar/](ttnn/cpp/ttnn/operations/experimental/quasar/) (28 directories, one of which is
a shared helper rather than an operation), covering every program factory (the host-side code that
builds a program) and every ring in them. The answer is structural rather than
lucky, and it comes in three parts.

**Most operations put a compute kernel between the reader and the writer.** Eight do, so whatever their
transfer counts, condition 2 is not met and they cannot be affected. Matrix multiply is the clearest
case: every ring there has a compute kernel on exactly one end, never data movement on both. Two of the
eight are also confirmed to carry a real surplus, which is what makes this the interesting group rather
than a technicality: convolution's activation reader, and tilize, whose reader issues 32 transfers
between a single reserve and a single announcement.

**Most rings that do run data movement to data movement never announce anything.** Several operations
declare a producer and a consumer only to satisfy a validation rule, then use the ring purely as a
convenient address and never call `push_back` or `wait_front` at all. The remainder are strictly one
transfer per entry, or announce more entries than they fill, which is the harmless direction.

**About ten rings do match both conditions, and those are latent rather than safe.** They include four
of the five `pad` factories that handle plain unstructured rows, `fold`'s equivalent factory at four
transfers per announcement for the common two-by-two case, and several `slice` and `transpose`
factories where the surplus appears only for certain ways of splitting a tensor across cores. Two
caveats keep this from being a list of ten live bugs: one of the four `pad` factories is never actually
selected, so it cannot be reached at all, and in three of them the surplus disappears when there is no
padding to add. Only one of the ten sits on ResNet's path, `interleaved_to_sharded`, and it needs an
input arrangement ResNet never produces.

So the model's immunity is real but narrow, and it is not evidence that the operator library as a whole
is clear. Plenty of existing code has a surplus and is safe only because a compute kernel happens to be
draining its ring. Any operation that hands data straight from one data movement kernel to another,
which is common for changing data arrangements, padding and slicing, is one loop rewrite away from this.

The two embedding factories that were ported sit at the intersection of both conditions: a reader
feeding a writer with no compute kernel between them, a ring with real waits, and an index fetch that
gives them a surplus. (A third factory does use a compute kernel, but it was not ported and is covered
in the companion report.)

## Why the Wormhole results are the point

`wait_front` is part of Metal 2.0's public API
([dataflow_buffer.h:182](tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L182)) and is meant to hold on
every architecture. The kernel and host code are identical in both columns of every table above and
call only documented functions, yet the result is correct on one architecture and wrong on the other.
Either the platform breaks the promise or the promise is under-specified, and both are platform
problems rather than problems in the operation.

## What is still open

**The exact accounting.** We measured three transfer-to-announcement ratios, not a formula. A surplus
over-grants, a matching count is correct, and a deficit is correct. Working out what the count is
actually derived from is for whoever owns the mechanism.

**Where corruption turns into deadlock.** Outcomes get worse as the surplus grows, but the ring depth is
not the threshold, and our four data points do not isolate the cause:

| Surplus | Entries | Transfers per entry | Ring depth | Outcome on Quasar |
|---|---|---|---|---|
| +1 | 4 | 1, plus one extra transfer in the run | 2 | corruption, in the clean two-entry pattern shown above |
| +4 | 4 | 2 | 2 | corruption, with no tidy pattern |
| +8 | 8 | 2 | 2 | deadlock |
| +28 | 4 | 8 | 2 | deadlock |

A surplus of 4 already exceeds the depth of 2 and still only corrupts, so depth is not the boundary.
But the +4 and +8 rows have the **same** surplus per entry and differ only in how many entries the run
pushes, so the difference between corrupting and deadlocking may be run length rather than surplus size.
Separating those two would need a run holding one fixed while varying the other, which we did not do.

**Which side gains the credit.** The failing pattern reads more like the producer running ahead and
overwriting entries than the consumer being released early. Both descriptions fit, because the
hardware tracks free space and occupancy as two readings of the same pair of counters, so a single
accounting error can present as either side gaining. We did not separate them, and this document picks
the consumer-side description for consistency rather than because it is established.

**Silicon.** Everything Quasar here ran on craq-sim. If the gap is in the simulator rather than in the
thing it simulates, Quasar hardware may be unaffected and the fix belongs to a different team, so that
is worth confirming before anyone changes hardware behaviour. This partly limits the Wormhole
comparison above: one side of it is a real chip and the other is a model of a chip. What is measured is
still a violation of the documented promise, so it is a defect on whichever layer owns it, and not in
the operation.

**Filing status.** Not filed at the time of writing. The minimal reproduction is the one to attach.

## What to do about it meanwhile


## Appendix: running the tests

Both files are in `tests/tt_metal/tt_metal/api/metal2_host_api/`. Build them into the `unit_tests_api`
binary with `./build_metal.sh --build-tests` from the repository root.

The tests are written with gtest, the C++ test framework, where each file's tests belong to a named
suite and `--gtest_filter` selects which ones to run.

- [test_dfb_gen2_split_read_repro_hw.cpp](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_split_read_repro_hw.cpp#L69-L168),
  suite `Gen2DFBSplitReadReproTest`. One test, the minimal case.
- [test_dfb_gen2_credits_hw.cpp](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L1-L94),
  suite `Gen2DFBCreditsTest`. Sixteen tests: three vary the ratio, three vary where the extra transfer
  lands, four compare compute against data movement consumers, and six are controls.

The six controls all pass on both generations. Their job is to leave the two conditions as the only
things that matter, by varying everything else in turn:

| Control | What it varies |
|---|---|
| [ConsumerDoesNotRunAheadOfProducer](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L309-L331) | ring depth and entry size |
| [...RawWritePtr](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L339-L364) | how the producer addresses the ring |
| [...TwoCores](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L371-L475) | one core against two |
| [...TensorAccessor](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L484-L576) | how DRAM addresses are computed |
| [...NoImplicitSync](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L1504-L1517) | implicit sync off on a working program |
| [ConsumerObservesProducerPushCountAtEachGrant](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L1381-L1492) | nothing; it confirms entries really are full at grant time when the counts match |

A Quasar run needs the craq-sim environment exported, which is documented separately. Leave it unset
and the same binary runs on Wormhole, which is how both columns in every table above were produced.
`TT_METAL_SLOW_DISPATCH_MODE=1` appears on every command line below because the shared test fixture
these suites derive from skips every test unless it is set
([device_fixture.hpp:79-95](tests/tt_metal/tt_metal/common/device_fixture.hpp#L79-L95)). Without it the
run reports success having executed nothing.

**The minimal case on its own.** This is the one to attach to a bug report.

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 ./build_Release/test/tt_metal/unit_tests_api \
  --gtest_filter="Gen2DFBSplitReadReproTest.*"
```

Quasar: fails, identically on repeated runs. Wormhole: passes in well under a second.

**The full suite, everything except the two tests that deadlock.** Fourteen tests, about 16 seconds on
Quasar and about 1 second on Wormhole.

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 ./build_Release/test/tt_metal/unit_tests_api \
  --gtest_filter="Gen2DFBCreditsTest.*:-Gen2DFBCreditsTest.ScratchpadReadEveryEntry*:Gen2DFBCreditsTest.DmConsumerManyReads*"
```

Quasar: 10 pass and 4 fail, those being `RatioTwoReadsPerAnnouncedSlot`,
`RatioTwoReadsPerAnnouncedSlotNoImplicitSync`, `MinimalExtraReadNoScratchpad` and
`ScratchpadUsePatternsThatDisturbTheBuffer`. The last prints a pass or fail line for each of its eight
rows. Wormhole: all 14 pass.

Note the **single** dash in that filter. gtest reads everything after the first dash as the exclusion
list, colon separated, so writing `-A*:-B*` excludes only `A*` and lets `B*` run. Written that way this
filter selects 15 tests instead of 14 and appears to hang, because one of the tests it was meant to
exclude is a deadlock. Check any filter with `--gtest_list_tests` before trusting it.

**The two deadlocking tests, separately.** gtest has no per-test timeout, so give each an external one.
They are excluded above so they cannot stop the others from reporting.

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 timeout 120 ./build_Release/test/tt_metal/unit_tests_api \
  --gtest_filter="Gen2DFBCreditsTest.ScratchpadReadEveryEntryCompletes"

TT_METAL_SLOW_DISPATCH_MODE=1 timeout 120 ./build_Release/test/tt_metal/unit_tests_api \
  --gtest_filter="Gen2DFBCreditsTest.DmConsumerManyReadsPerAnnouncedSlot"
```

[ScratchpadReadEveryEntryCompletes](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L1275-L1371)
adds one extra transfer per entry across eight entries, a surplus of 8.
[DmConsumerManyReadsPerAnnouncedSlot](tests/tt_metal/tt_metal/api/metal2_host_api/test_dfb_gen2_credits_hw.cpp#L1083-L1095)
performs eight transfers per entry across four entries, a surplus of 28, and is the data movement half
of the compute comparison above. Wormhole: both pass in well under a second. Quasar: neither finishes,
and `timeout` kills them.
