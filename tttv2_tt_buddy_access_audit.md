# tt-buddy Access and CCL Knowledge Audit

## Scope and constraints

- Dedicated goal: locate any available `tenstorrent/tt-buddy` content and extract CCL debugging guidance relevant to the Attention2D axis-1 stall.
- No hardware commands or tests are permitted.
- This audit edits only this file.

## Checkpoint 1: Local filesystem, Git metadata, and caches

### Commands

```bash
git status --short -- tttv2_tt_buddy_access_audit.md
git remote -v
git config --show-origin --get-regexp 'url\..*\.insteadof|credential|http\..*extraheader'
find /home/gwang /tmp /opt /usr/local -xdev \( -iname '*tt-buddy*' -o -iname '*tt_buddy*' \) -print
rg -n -i --hidden --glob '!.git/objects/**' --glob '!build/**' --glob '!generated/**' \
  'tt[-_ ]buddy|tenstorrent/tt-buddy' /home/gwang/tt-metal /home/gwang/.config /home/gwang/.cache
find /home/gwang -xdev -type f \
  \( -path '*/.git/config' -o -path '*/.gitmodules' -o -path '*/.git/FETCH_HEAD' -o -path '*/.git/packed-refs' \) \
  -print0 | xargs -0 rg -n -i 'tt-buddy|tenstorrent/tt-buddy'
find /home/gwang/.cache /home/gwang/.local /root/.cache /root/.local /opt -xdev -type f | \
  rg -i 'tt[-_]?buddy|buddy.*(whl|tar|zip)|tenstorrent'
```

### Findings

- No path named `tt-buddy` or `tt_buddy` exists in the searched local roots.
- No local repository config, submodule file, fetch record, or packed ref mentions `tt-buddy`.
- No matching package/archive artifact was found in the inspected user, root, or `/opt` caches. The broad cache query returned Tenstorrent tooling files but no `buddy` match.
- The only repository text matches are lines 1578-1582 of `tttv2_2d_modules_work_log.md`, which document earlier unsuccessful access attempts; they do not contain repository content.
- The current repository has only `origin = git@github.com:tenstorrent/tt-metal.git`. No configured Git URL rewrite or usable credential/extra-header entry was reported.

### Conclusion

There is no confirmed local copy or cache of `tt-buddy`. Continue with remote endpoint and public-index discovery; do not treat a name-only filesystem search as proof that private content never existed on this host.

## Checkpoint 2: GitHub and mirror access paths

### Commands and endpoints

```bash
curl -L -sS -o /tmp/ttb_body -w 'status=%{http_code} final=%{url_effective} type=%{content_type}\n' URL
```

The command above was run for:

- `https://api.github.com/repos/tenstorrent/tt-buddy`
- `https://github.com/tenstorrent/tt-buddy`
- `https://raw.githubusercontent.com/tenstorrent/tt-buddy/main/README.md`
- `https://codeload.github.com/tenstorrent/tt-buddy/tar.gz/refs/heads/main`
- repository `forks`, `commits`, `contents`, and `git/refs/heads/main` API endpoints

Additional probes:

```bash
GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/tenstorrent/tt-buddy.git
ssh -o BatchMode=yes -o ConnectTimeout=10 -T git@github.com
GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=10' \
  git ls-remote git@github.com:tenstorrent/tt-buddy.git
curl -L -sS 'https://api.github.com/search/repositories?q=org:tenstorrent+tt-buddy'
curl -L -sS 'https://api.github.com/search/repositories?q=%22tenstorrent%2Ftt-buddy%22'
curl -L -sS -o /tmp/ttb_mirror -w 'status=%{http_code} final=%{url_effective}\n' \
  https://gitlab.com/tenstorrent/tt-buddy
```

### Findings

- Every unauthenticated GitHub page, REST repository endpoint, raw-content URL, and codeload URL returned HTTP 404. Public repository searches returned zero results.
- The GitLab URL redirected to sign-in and then returned a Cloudflare HTTP 403 page, so it neither confirms nor disproves a mirror.
- HTTPS Git access could not obtain credentials with prompts disabled.
- The host SSH key authenticated successfully to GitHub as `gwangTT`.
- SSH Git access is available: `git ls-remote git@github.com:tenstorrent/tt-buddy.git` returned `main` at commit `ba9021417442d59756aa8cdf154a25648c9a0de5` plus additional branches and pull-request refs.

### Conclusion

`tenstorrent/tt-buddy` is private or otherwise hidden from unauthenticated discovery, but it is accessible from this host through the configured GitHub SSH identity. Its contents can be inspected from a temporary clone without modifying the shared repository.

## Checkpoint 3: Private repository contents

### Commands

```bash
GIT_SSH_COMMAND='ssh -o BatchMode=yes' \
  git clone --filter=blob:none --depth 1 \
  git@github.com:tenstorrent/tt-buddy.git /tmp/tt-buddy-access-audit-20260819
git -C /tmp/tt-buddy-access-audit-20260819 rev-parse HEAD
find /tmp/tt-buddy-access-audit-20260819 -maxdepth 4 -type f -print | sort
sed -n '1,260p' knowledge/ccl.md
sed -n '1,260p' skills/debugger/{SKILL.md,triage.md,interpretation.md,scripts.md,watcher.md}
sed -n '1,220p' skills/run/recovery.md
sed -n '1,180p' knowledge/hardware/quirks.md
rg -n -i 'ccl|all.?gather|all.?reduce|reduce.?scatter|deadlock|hang|num_links|chunks_per_sync|num_workers_per_link|num_buffers_per_channel|tt-triage' .
```

### Findings

- The temporary clone succeeded at `ba9021417442d59756aa8cdf154a25648c9a0de5` (`main`).
- The repository is a knowledge/skills plugin. The directly relevant files are:
  - `knowledge/ccl.md`
  - `skills/debugger/SKILL.md`
  - `skills/debugger/triage.md`
  - `skills/debugger/interpretation.md`
  - `skills/debugger/scripts.md`
  - `skills/debugger/watcher.md`
  - `skills/run/recovery.md`
  - `knowledge/hardware/quirks.md`
- `knowledge/ccl.md` states that Galaxy uses four physical links and warns that an incorrect `num_links` can deadlock rather than return a soft error. It recommends deriving the value from a sibling model's CCL helper instead of hardcoding it.
- The same file says `chunks_per_sync`, `num_workers_per_link`, and `num_buffers_per_channel` have non-monotonic effects. Tuning should therefore be a small recorded matrix, one knob at a time, after correctness is established.
- The debugger skill requires diagnosis while the hung process remains alive. Its evidence funnel is: running op/mesh, per-core callstacks, mailbox/binary integrity, then fabric/hardware status.
- For a silent hang, it recommends `dump_aggregated_callstacks` plus `dump_running_operations`; the lowest active op ID is the first-hung op, and `[!]` entries in `dump_op_mesh` identify stragglers.
- Relevant stuck-call patterns are:
  - `noc_semaphore_wait`: writer missing the exact value or overshooting an equality wait.
  - `noc_semaphore_wait_min`: cross-core semaphore never reaching the minimum.
  - `noc_async_read_barrier`: read responses did not match issued reads.
  - `noc_async_write_barrier`: non-posted write acknowledgements did not match software counters.
  - `cb_wait_front` / `cb_reserve_back`: producer-consumer page-count mismatch or CB-capacity violation.
- If triage shows no stalled core, the guidance shifts to a host dispatcher/command-queue acknowledgement issue and recommends a watcher-enabled rerun. If the operation immediately before the stalled collective is a large matmul, `TT_MM_THROTTLE_PERF` is a candidate for Wormhole di/dt mitigation, not a verdict without triage evidence.
- Recovery order is diagnosis, process termination, reset, verification, then conditional cache clearing. Resetting while a process still owns a device handle is explicitly identified as unsafe.

## Checkpoint 4: Correlation with the Attention2D blocker

### Commands

```bash
nl -ba models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py | sed -n '330,390p'
nl -ba models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py | sed -n '455,530p'
nl -ba models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py | sed -n '90,180p'
nl -ba models/common/modules/tt_ccl.py | sed -n '145,205p'
nl -ba models/tt_transformers/tests/test_ccl_utils.py | sed -n '30,48p'
rg -n 'cluster_axis.?=.?1|reduce_scatter_minimal_async|all_gather_async|num_links' \
  tests/ttnn/unit_tests/operations/ccl models/demos/llama3_70b_galaxy models/common/modules/mlp
```

### Findings

1. **Primary mismatch: Attention axis-1 hardcodes one link.** The active non-persistent Attention path calls both `ttnn.reduce_scatter` and `ttnn.all_gather` with `cluster_axis=1`, `num_links=1`, and `Topology.Linear` at lines 347-368 of `test_attention_2d_wh_galaxy.py`. Its resource plans also encode one link at lines 459-521.
2. **Repository topology resolution says four links.** `models/common/modules/tt_ccl.py:get_num_links()` maps `TG` to `(4, 4)`, so either Galaxy axis resolves to four links. `models/tt_transformers/tests/test_ccl_utils.py` explicitly asserts `cluster_axis=1` returns 4 for a 32-device host-local Wormhole mesh.
3. **The known-good MLP2D axis-1 configuration uses Ring and four links.** Its decode reduce-scatter plan sets `Topology.Ring` and `num_links=4`; the paired all-gather inherits Ring and four links. This is stronger local evidence than a generic topology assumption because MLP2D has already passed on the same host and mesh.
4. **There is a shape-near precedent.** `tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py` includes an axis-1 QKV all-reduce case with shape `[1, 1, 32, 1280]`. It uses three links in that legacy specialized test, showing that payload geometry itself is supported, while also reinforcing that a one-link hardcode should not be assumed correct. The shared current helper remains the preferred source of truth.
5. The current stall follows a matmul, so the debugger skill's Wormhole di/dt branch is relevant only if live triage shows the collective waiting after the projection and the link/topology correction does not resolve it.

### Conclusion

The highest-confidence next experiment is not another buffer-placement variation. It is to remove the one-link axis-1 hardcode and use the shared topology-derived value, with the already-qualified MLP2D Ring/four-link setup as the initial configuration. This is the only newly identified configuration error with an explicit `tt-buddy` deadlock warning and matching local topology tests.

## Checkpoint 5: Public mirrors and documentation references

### Sources

- GitHub repository searches for `"tenstorrent/tt-buddy"`, `org:tenstorrent tt-buddy`, and `user:tenstorrent tt-buddy`: zero public results.
- General web searches for `"tenstorrent/tt-buddy"`, `"tt-buddy" Tenstorrent CCL`, and `site:github.com/tenstorrent "tt-buddy"`: no relevant indexed result.
- GitLab repository URL: access was blocked after sign-in redirect, so no mirror was confirmed.
- Official public corroboration: `https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/triage.html` documents `tt-triage`, Inspector dependency, multi-device selection, and remote-exalens fallback.
- Official public lab: `https://docs.tenstorrent.com/tt-metal/v0.70.1/tt-metalium/tt_metal/labs/matmul/lab1/lab1.html` recommends leaving a hang alive and using per-core callstacks to locate the stuck RISC-V code.

### Conclusion

No public mirror of `tt-buddy` was found. The content is available through this host's private GitHub SSH access, while its core triage workflow is independently supported by public TT-Metalium documentation.

## Actionable recommendations

1. **First correction:** resolve axis-1 `num_links` with `models.common.modules.tt_ccl.get_num_links(mesh_device, cluster_axis=1)`; on this WH Galaxy it must return 4. Mirror the hardware-qualified MLP2D axis-1 `Topology.Ring`, four-link reduce-scatter/all-gather configuration before changing memory placement again.
2. **Minimal isolation:** rerun only the shape `[1, 1, 32, 1280]` axis-1 collective reproducer with the corrected link/topology pair. Keep all other parameters fixed so the result attributes cleanly to the configuration change.
3. **Capture a live stall before reset:** while the process is still alive, run the complete triage with machine-readable output and aggregated callstacks:

   ```bash
   TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS=1 \
     tools/tt-triage.py --llm-output \
     --triage-summary-path=/tmp/attention-axis1-triage-summary.txt \
     2>&1 | tee /tmp/attention-axis1-triage-output.txt
   ```

4. **Interpret in order:** identify `[!]` mesh stragglers and the lowest active op ID; then classify the stuck frame as semaphore equality/minimum, NoC read/write barrier, or CB page-count wait. Check `check_core_magic` and `check_binary_integrity` before trusting mailbox-derived callstacks.
5. **Watcher fallback:** if live triage reports no stalled core, rerun the minimal reproducer with `TT_METAL_WATCHER=5`; inspect the final `TRIPPED`/`STOPPED` block rather than benign waypoint snapshots.
6. **Secondary matmul hypothesis:** only if the corrected Ring/four-link case still stalls immediately after the large QKV matmul, compare a run with the repository's Wormhole `TT_MM_THROTTLE_PERF` convention. Label this a candidate until callstacks and operation ordering support it.
7. **Tune last:** after a correct non-stalling baseline, sweep `chunks_per_sync`, `num_workers_per_link`, and `num_buffers_per_channel` independently over a small matrix. Do not infer directionality from one result.
8. **Recovery discipline:** preserve triage artifacts first, terminate the owning process, then reset and verify all 32 devices. Clear TT-Metal caches only when kernel C++/headers changed.

## Audit result

- `tt-buddy` availability: **confirmed through private SSH Git access**.
- Public availability: **not found**.
- Most actionable blocker finding: **Attention2D axis-1 uses one link despite the WH Galaxy topology helper, tests, known-good MLP2D path, and `tt-buddy` guidance indicating four links**.
- Hardware activity during this audit: **none**.
