# ttnn.conv1d deadlocks when run after a traced model on the same device

## Summary

A sequence of `ttnn.conv1d` ops that executes correctly in isolation **deadlocks the
device permanently** once another model on the same device has executed its captured
traces. The hang is not a crash: the host spins at ~106% CPU on one thread forever,
`SIGINT` cannot interrupt it, and killing the process leaves stale ethernet dispatch
kernels so the board needs `tt-smi -glx_reset_auto` before it can be opened again.

ttnn's own watcher identifies the failure as a **circular-buffer deadlock on 2 of 72
cores** inside the block-sharded multicast activation reader.

Reproduced on Wormhole (`tt-galaxy`, 8x9 compute grid), tt-metal
`851e7a852e4` (branch `tvardhineni/sdawle/qwen3-tts-device-decode-wip`),
ttnn `0.75.0rc10.dev551`.

## What is affected

The qwen3-tts 1.7B speech-tokenizer decoder (a conv vocoder: codec tokens -> 24 kHz
waveform). It shares one device with the model's "talker" transformer, which runs as
a set of captured traces. Decode must happen after the talker produces codes, so the
two cannot be separated.

This blocks moving the vocoder off CPU. On device the decode is ~0.32 s where CPU
takes ~7.3 s at the server's real threading budget (1 torch thread), so the decoder
currently burns roughly half of every request on the host.

## Observed behaviour

Three steps, in this order, every run:

1. **Decoder warmup succeeds.** At startup all conv shapes are prepared and executed
   for bucket lengths 64/128/192/256/320/384/448. Every conv works.
2. **Talker runs correctly.** Captures 9 bucketed decode traces plus prefill at
   startup, then executes them per request. Completes normally (~950 ms).
3. **Decode then deadlocks** — on the *same* bucket that warmed successfully in step
   1, with the same cached conv objects, in the same process.

Trace *capture* is harmless: step 1 happens after all traces are already captured and
works fine. It is trace *execution* in step 2 that poisons the device for conv.

## Evidence

### Host stack (py-spy, hung worker)

```
Thread (active): "asyncio_0"
    __call__ (ttnn/decorators.py:650)
    __call__ (ttnn_conv_decoder.py:103)          <- ttnn.conv1d
    transpose_conv1d_nhwc (speech_tokenizer.py:191)
    conv_decoder_block (speech_tokenizer.py:396)
    _conv_decoder_forward (speech_tokenizer.py:1439)
    forward (speech_tokenizer.py:1581)
    decode_audio_device (server.py:2904)
```

With an explicit `ttnn.synchronize_device()` placed after the conv stack, the hang
moves to that synchronize instead -- the convs enqueue asynchronously and the
device-side deadlock only surfaces when something waits.

### Device state (TT_METAL_WATCHER=10)

Two cores stuck, 70 healthy:

```
Device 0 worker core(x=0,y=0) virtual(x=18,y=18): CWFD,CRBW,   W,   W,   W  rmsg:D1G|BNt h_id:73965
Device 0 worker core(x=0,y=1) virtual(x=18,y=19): CWFD,CRBW,   W,   W,   W  rmsg:D1G|BNt h_id:73965
Device 0 worker core(x=1,y=0) virtual(x=19,y=18):    R,   R,   R,   R,   R  rmsg:D1D|BNT h_id:73966
   ... 69 more cores all R / h_id:73966 ...
```

The stuck pair is on dispatch `h_id:73965` while all 70 others advanced to `73966`.
BRISC is in `CWFD` (waiting for circular-buffer data), NCRISC in `CRBW` (waiting for
circular-buffer space), all three TRISCs spinning in `W`.

Kernel named in the same watcher dump:

```
NCRISC: reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp
```

Full watcher log: `watcher_hang_evidence.log` (1.2 MB, 76 dumps).

### Why that kernel is the suspect

`ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`
documents this exact hazard in its own comments:

```
// or mcast when the sender core is not a receiver core (it is only present in the input grid,
// mcast loopback will hang if the core isn't one of receivers) or just local write when it is
// in both input and output grids but is the only receiver core (will hang if mcast loopback is used)
```

The hang condition is a sender core present in the input grid but not the output
grid. That is consistent with 2 cores (column x=0) stuck as senders while the rest of
the grid advances.

## Ruled out by direct experiment

| Hypothesis | Result |
|---|---|
| Un-warmed decode length / first-use weight prepare | ✗ hangs on a length warmed minutes earlier |
| Missing queue synchronization after the traced model | ✗ added `synchronize_device`, no change |
| Conv shape or L1 pressure | ✗ rewrote the transposed conv as a sub-pixel decomposition: 8x fewer input positions (1046 -> 129) and 16 -> 2 taps, still hangs |
| Two command queues | ✗ hangs with `num_command_queues=1` and `use_2cq=False` |
| Sharded output readback | ✗ hangs before the readback; moving the output to interleaved DRAM first does not help |
| Non-power-of-two bucket lengths | ✗ was a *separate* bug, now fixed by the sub-pixel rewrite (all of 64..448 warm cleanly) |

## Why it cannot be worked around in model code

The convs must use `BLOCK_SHARDED`, which is the layout that uses the multicast
reader. They have no alternative: channel counts reach 1024 against 1.5 MB of L1 per
core, so `HEIGHT_SHARDED` (which does not split channels) has no valid config.

| Workaround attempted | Blocked by |
|---|---|
| `override_output_sharding_config` to force input grid == output grid | `conv2d.cpp:753` -- unsupported on the DRAM-sliced path these convs already require |
| Pin a square 8x8 core grid (`override_sharding_config`) | L1 overflow: circular buffers grow to 1,584,416 B vs 1,499,136 B limit |
| `act_block_h_override=32` to recover that L1 | `matmul_device_operation.cpp:411: bias_shape_padded[-1] == b_shape_padded[-1]` |
| `shard_layout=HEIGHT_SHARDED` | `op_slicing.cpp:266: found_valid_config` on the first bucket |

## Not fixed upstream

As of `origin/main` at `16327cdfae5` (**2059 commits** past our base `747b4f4a63ef`):

- `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`: **0 changes**
- `ttnn/cpp/ttnn/operations/conv/`: only 6 commits, none related (a validation fix,
  two cleanups, a Quasar Resnet kernel cleanup, fp32 depthwise conv1d, an LLK init
  cleanup)

## How to reproduce

Both repos must be checked out and on `PYTHONPATH`.

| Repo | Branch | HEAD |
|---|---|---|
| `tenstorrent/tt-metal` | `tvardhineni/sdawle/qwen3-tts-device-decode-wip` | `851e7a852e4` |
| `tenstorrent/tt-inference-server` | `tvardhineni/qwen3-tts-device-decode-wip` | `090ef472` |

### Case A -- conv-only sanity check (expected to PASS)

Confirms the decoder's conv sequence is fine with no traced model present.

```bash
cd tt-metal && source python_env/bin/activate
TT_REPRO_DEVICE_ID=0 TT_BUCKETS=64,128,192,256,320,384,448 \
PYTHONPATH=$PWD TT_METAL_HOME=$PWD \
python ttnn_conv_hang_repro/01_isolate_failing_conv.py
```

Expect `ALL BUCKETS WARMED -- bug did not reproduce`. Every conv1d call is printed
with its full parameters, so if it ever does hang the last `CALLING` line names the
exact conv.

### Case B -- the bug (expected to HANG)

Runs the full serving path: talker traces execute, then decode.

```bash
cd tt-inference-server/tt-media-server
source $TT_METAL_HOME/python_env/bin/activate
MODEL_RUNNER=tt-qwen3-tts MODEL_WEIGHTS_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
DEVICE=n150 DEVICE_IDS="(0)" IS_GALAXY=false ENVIRONMENT=development \
SERVICE_PORT=8000 TT_QWEN3_DEVICE_DECODE=1 \
TT_METAL_WATCHER=10 TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' \
PYTHONPATH=$TT_METAL_HOME:$PWD \
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --lifespan on
```

Startup logs, in order:

```
Device 0: On-device speech decoder ready; warmed and froze buckets [...]   <- all convs fine
Device 0: Warm-up synth text='テスト'
Inference time    941.7                                                   <- talker fine
Audio duration      0.80s
<hangs here, forever, ~106% CPU on one thread>
```

Confirm with `py-spy dump --pid <worker pid>` (the worker is the
`multiprocessing-fork` child with the largest RSS) and read
`generated/watcher/watcher.log` for the stuck cores.

**Recovery:** kill the worker, then `tt-smi -glx_reset_auto`. Plain `tt-smi -r` does
not work on Galaxy. Note that `tt-smi -ls` will still list all chips while the board
is unusable -- the real check is whether `ttnn.open_device` succeeds.

## What a fix needs to address

How the block-sharded conv derives its multicast sender/receiver grids when another
model's traces have already executed on the device and changed L1 occupancy. The
input and output parallel configs appear to diverge, leaving senders that are not
receivers -- the condition the kernel's own comments say will hang.

Ideally `conv1d` should also **fail loudly rather than deadlock** when it cannot
construct a valid multicast configuration. Explicitly setting `HEIGHT_SHARDED` raises
`found_valid_config` cleanly; auto-selected `BLOCK_SHARDED` hangs the chip instead,
which costs a board reset each time.
