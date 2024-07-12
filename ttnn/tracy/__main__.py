# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tracy import *


def main():
    from optparse import OptionParser

    usage = "python -m tracy [-m module | scriptfile] [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option("-m", dest="module", action="store_true", help="Profile a library module.", default=False)
    parser.add_option("-p", dest="partial", action="store_true", help="Only profile enabled zones", default=False)
    parser.add_option("-l", dest="lines", action="store_true", help="Profile every line of python code", default=False)
    parser.add_option("-r", dest="report", action="store_true", help="Generate ops report", default=False)
    parser.add_option("-v", dest="verbose", action="store_true", help="More info is printed to stdout", default=False)
    parser.add_option(
        "--no-device", dest="device", action="store_false", help="Do not include device data", default=True
    )
    parser.add_option(
        "-o", "--output-folder", action="store", help="Artifact output folder", type="string", dest="output_folder"
    )
    parser.add_option(
        "-n",
        "--name-append",
        action="store",
        help="Custom name to be added to report name",
        type="string",
        dest="name_append",
    )
    parser.add_option(
        "-t", "--port", action="store", help="Internal port used by the script", type="string", dest="port"
    )
    parser.add_option(
        "--no-op-info-cache",
        dest="opInfoCache",
        action="store_false",
        help="Show full op info for cached ops as well",
        default=True,
    )
    parser.add_option(
        "--child-functions",
        type="string",
        help="Comma separated list of child function to have their duration included for parent OPs",
        action="callback",
        callback=split_comma_list,
    )

    if not sys.argv[1:]:
        parser.print_usage()
        sys.exit(2)

    originalArgs = sys.argv.copy()

    (options, args) = parser.parse_args()
    sys.argv[:] = args

    if options.port:
        port = options.port
    else:
        port = get_available_port()

    opInfoCacheStr = "TT_METAL_PROFILER_NO_CACHE_OP_INFO"
    if options.opInfoCache:
        if opInfoCacheStr in os.environ.keys():
            del os.environ[opInfoCacheStr]
    else:
        os.environ[opInfoCacheStr] = "1"

    if len(args) > 0:
        doReport = False
        if options.report:
            if not port:
                logger.error("No available port found")
                sys.exit(1)
            logger.info(f"Using port {port}")
            doReport, captureProcess = run_report_setup(options.verbose, port)

        if not doReport:
            if options.module:
                import runpy

                code = "run_module(modname, run_name='__main__')"
                globs = {
                    "run_module": runpy.run_module,
                    "modname": args[0],
                }
            else:
                progname = args[0]
                sys.path.insert(0, os.path.dirname(progname))
                with io.open_code(progname) as fp:
                    code = compile(fp.read(), progname, "exec")
                spec = importlib.machinery.ModuleSpec(name="__main__", loader=None, origin=progname)
                globs = {
                    "__spec__": spec,
                    "__file__": spec.origin,
                    "__name__": spec.name,
                    "__package__": None,
                    "__cached__": None,
                }

            if options.partial:
                tracy_state.doPartial = True

            if options.lines:
                tracy_state.doLine = True

            try:
                runctx(code, globs, None, options.partial)
            except BrokenPipeError as exc:
                # Prevent "Exception ignored" during interpreter shutdown.
                sys.stdout = None
                sys.exit(exc.errno)
        else:
            originalArgs.remove("-r")
            osCmd = " ".join(originalArgs[1:])

            testCommand = f"python -m tracy {osCmd}"

            envVars = dict(os.environ)
            # No Dispatch cores for op_report
            envVars["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"
            if options.device:
                envVars["TT_METAL_DEVICE_PROFILER"] = "1"
            else:
                if "TT_METAL_DEVICE_PROFILER" in envVars.keys():
                    del envVars["TT_METAL_DEVICE_PROFILER"]

            if port:
                envVars["TRACY_PORT"] = port

            testProcess = subprocess.Popen([testCommand], shell=True, env=envVars, preexec_fn=os.setsid)
            logger.info(f"Test process started")

            def signal_handler(sig, frame):
                os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
                captureProcess.terminate()
                captureProcess.communicate()
                sys.exit(3)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            testProcess.communicate()

            try:
                captureProcess.communicate(timeout=15)
                generate_report(options.output_folder, options.name_append, options.child_functions)
            except subprocess.TimeoutExpired as e:
                captureProcess.terminate()
                captureProcess.communicate()
                logger.error(
                    f"No profiling data could be captured. Please make sure you are on the correct build. Use scripts/build_scripts/build_with_profiler_opt.sh to build if you are not sure."
                )
                sys.exit(1)

    else:
        parser.print_usage()
    return parser


# When invoked as main program, invoke the profiler on a script
if __name__ == "__main__":
    main()
