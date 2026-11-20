#!/bin/bash
# usage: run_wait.sh <budget_s> <label> <cases> <variants> [--dev] [env...]
# Queues for the device like any run, but variants that would start more than <budget_s> seconds from now are skipped
# (RMB_DEADLINE), so the job never outlives the caller's 10-minute cap while it holds the device.
budget=$1; shift
label=$1; cases=$2; variants=$3; shift 3
dev=()
if [ "$1" == "--dev" ]; then dev=(--dev); shift; fi
exec "$(dirname "$0")/run.sh" "$label" "$cases" "$variants" "${dev[@]}" RMB_DEADLINE=$(( $(date +%s) + budget )) "$@"
