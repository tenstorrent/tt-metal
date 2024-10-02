# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import importlib.machinery
import sys
import signal
import os
import io
import subprocess
import time
import socket

from loguru import logger

from tt_metal.tools.profiler.process_ops_logs import process_ops
from tt_metal.tools.profiler.common import (
    TT_METAL_HOME,
    PROFILER_BIN_DIR,
    PROFILER_ARTIFACTS_DIR,
    PROFILER_SCRIPTS_ROOT,
    TRACY_MODULE_PATH,
    TRACY_FILE_NAME,
    TRACY_OPS_TIMES_FILE_NAME,
    TRACY_OPS_DATA_FILE_NAME,
    TRACY_CAPTURE_TOOL,
    TRACY_CSVEXPROT_TOOL,
    generate_logs_folder,
)

import tracy.tracy_state

DEFAULT_CHILD_CALLS = ["CompileProgram", "HWCommandQueue_write_buffer"]


def signpost(header, message=None):
    import ttnn

    if message:
        ttnn.tracy_message(f"`TT_SIGNPOST: {header}\n{message}`")
        logger.info(f"{header} : {message} ")
    else:
        ttnn.tracy_message(f"`TT_SIGNPOST: {header}`")
        logger.info(f"{header}")


class Profiler:
    def __init__(self):
        from tracy.tracy_ttnn import tracy_marker_func, tracy_marker_line, finish_all_zones

        self.doProfile = tracy_state.doPartial and sys.gettrace() is None and sys.getprofile() is None
        self.doLine = tracy_state.doLine

        self.lineMarker = tracy_marker_line
        self.funcMarker = tracy_marker_func
        self.finishZones = finish_all_zones

    def enable(self):
        if self.doProfile:
            if self.doLine:
                sys.settrace(self.lineMarker)
            else:
                sys.setprofile(self.funcMarker)

    def disable(self):
        if self.doProfile:
            sys.settrace(None)
            sys.setprofile(None)
            self.finishZones()


def runctx(cmd, globals, locals, partialProfile):
    from tracy.tracy_ttnn import tracy_marker_func, finish_all_zones

    if not partialProfile:
        sys.setprofile(tracy_marker_func)

    try:
        exec(cmd, globals, locals)
    finally:
        sys.setprofile(None)
        finish_all_zones()


def run_report_setup(verbose, outputFolder, port):
    toolsReady = True

    logger.info("Verifying tracy profiling tools")
    toolsReady &= os.path.exists(PROFILER_BIN_DIR / TRACY_CAPTURE_TOOL)
    toolsReady &= os.path.exists(PROFILER_BIN_DIR / TRACY_CSVEXPROT_TOOL)

    logsFolder = generate_logs_folder(outputFolder)
    captureProcess = None
    if toolsReady:
        subprocess.run(
            f"rm -rf {logsFolder}; mkdir -p {logsFolder}",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        options = ""
        if port:
            options += f"-p {port}"

        captureCommand = (f"{PROFILER_BIN_DIR / TRACY_CAPTURE_TOOL} -o {logsFolder / TRACY_FILE_NAME} -f {options}",)
        if verbose:
            logger.info(f"Capture command: {captureCommand}")
            captureProcess = subprocess.Popen(captureCommand, shell=True)
        else:
            captureProcess = subprocess.Popen(
                captureCommand, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    else:
        logger.error(
            f"Tracy tools were not found. Please make sure you are on the correct build. Use scripts/build_scripts/build_with_profiler_opt.sh to build if you are not sure."
        )
        sys.exit(1)

    return toolsReady, captureProcess


def generate_report(outputFolder, nameAppend, childCalls):
    logsFolder = generate_logs_folder(outputFolder)
    tracyOutFile = logsFolder / TRACY_FILE_NAME
    timeOut = 15
    timeCount = 0
    while not os.path.exists(tracyOutFile):
        logger.warning(
            f"tracy capture out not found, will try again in 1 second. Run in verbose (-v) mode to see tracy capture info"
        )
        if timeCount > timeOut:
            logger.error(
                f"tracy capture output file {tracyOutFile} was not generated. Run in verbose (-v) mode to see tracy capture info"
            )
            sys.exit(1)
        timeCount += 1
        time.sleep(1)
    with open(logsFolder / TRACY_OPS_TIMES_FILE_NAME, "w") as csvFile:
        childCallStr = ""
        childCallsList = DEFAULT_CHILD_CALLS
        if childCalls:
            childCallsList = list(set(childCalls + DEFAULT_CHILD_CALLS))
        if childCallsList:
            childCallStr = f"-x {','.join(childCallsList)}"
        subprocess.run(
            f"{PROFILER_BIN_DIR / TRACY_CSVEXPROT_TOOL} -u -p TT_DNN {childCallStr} {logsFolder / TRACY_FILE_NAME}",
            shell=True,
            check=True,
            stdout=csvFile,
            stderr=subprocess.DEVNULL,
        )

    logger.info(f"Host side ops time report generated at {logsFolder / TRACY_OPS_TIMES_FILE_NAME}")

    with open(logsFolder / TRACY_OPS_DATA_FILE_NAME, "w") as csvFile:
        subprocess.run(
            f'{PROFILER_BIN_DIR / TRACY_CSVEXPROT_TOOL} -m -s ";" {logsFolder / TRACY_FILE_NAME}',
            shell=True,
            check=True,
            stdout=csvFile,
            stderr=subprocess.DEVNULL,
        )

    logger.info(f"Host side ops data report generated at {logsFolder / TRACY_OPS_DATA_FILE_NAME}")

    process_ops(outputFolder, nameAppend, True)


def get_available_port():
    ip = socket.gethostbyname(socket.gethostname())

    for port in range(8086, 8500):
        try:
            serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serv.bind((ip, port))
            return str(port)
        except PermissionError as e:
            pass
        except OSError as e:
            pass
    return None


def split_comma_list(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(","))
