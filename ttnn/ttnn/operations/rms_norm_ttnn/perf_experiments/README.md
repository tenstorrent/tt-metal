# `perf_experiments/` — DO NOT MAKE THESE DIRECTORIES PACKAGES

Every dir here is a self-contained perf-lab artifact: a forked descriptor, a copy of the
kernels, a bench driver.  They are run directly, with `sys.path.insert(0, HERE)`:

    scripts/tt-probe.sh rms_norm_ttnn < <dir>/<bench>.py

**None of them may contain an `__init__.py`.**  `ttnn/ttnn/operations/__init__.py` calls
`pkgutil.walk_packages(__path__)` and `exec_module`s every module it finds, so an
`__init__.py` anywhere under here turns the whole subtree into a package that
**executes on `import ttnn`**.

Perf 2 measured what that costs.  Two of these dirs set `os.environ["RMS_STAGE_ZONES"] = "1"`
at module scope (a profiling switch).  The shipped descriptor module is imported EARLIER in
the same walk and reads it as `False`; anything `_load`ed afterwards read `True` — so a
candidate build got the zone-instrumented kernels and its baseline got the clean ones.  That
is the entire "unattributed 4-5% same-build overhead" that blocked Perf 1's per-channel
multicast idea from graduating.  It was never the idea; it was this file.  A dir that also
opens a device at module scope hard-fails `import ttnn` outright.

If a bench needs to import a sibling's module, use `sys.path.insert(0, os.path.dirname(...))`
and a plain module import — never a package path through `ttnn.operations`.
