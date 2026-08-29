"""Running one measurement draw in a fresh process, and getting its result back.

Every compile-sensitive measurement here runs one configuration per process. That
is not tidiness: ``torch.compile``'s failure mode on this model is **per-process**
(`f-2026-08-28-compile-miscompilation`), so two configurations sharing a process
share a draw and the comparison between them measures nothing. Tearing a CUDA
model down and building another in one process also leaves the second run emitting
``cuBLAS ... there was no current CUDA context`` from inside the backward --
recoverable, but the process is then no longer in the state the measurement
assumes.

This module holds the parent side of that pattern, extracted from
``experiments/dim_batch_diagnosis.py`` when a second driver needed it. Per
``experiments/README.md`` a driver is a script and never an import target, so
logic two drivers share is library code by definition.

The protocol is one line on stdout: the child prints ``RESULT_PREFIX`` followed by
a JSON payload, the parent parses it and passes everything else through.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

#: Marker for the child's result line. Anything else the child prints is treated
#: as progress output and forwarded.
RESULT_PREFIX = "@@DIAG_RESULT@@ "

#: Per-child ceiling. Measured children take 42-95s, so this is generous by ~4x.
#: It was 900s, which is worse than useless: nothing legitimate runs that long, and
#: the longer the ceiling the longer a hang looks like work.
DEFAULT_TIMEOUT_S = 300


def emit_result(payload: dict[str, Any]) -> None:
    """Child side: hand one result back to the parent."""
    print(RESULT_PREFIX + json.dumps(payload, default=str), flush=True)


def run_child(script: Path | str, label: str, extra: list[str], *,
              timeout_s: int = DEFAULT_TIMEOUT_S,
              quiet: bool = False) -> dict[str, Any] | None:
    """Run one child to completion, announcing it first.

    ``script`` is the driver to re-invoke -- normally ``Path(__file__)``, since the
    pattern is a driver spawning its own ``--child`` mode.

    Output is captured, so nothing the child prints is visible until it exits and
    without the announcement below a two-minute compile looks like a hang. The
    announcement is flushed explicitly because this output is usually piped to a
    log, where Python block-buffers and an unflushed line would not appear either.

    Returns the child's payload, or ``None`` if it timed out or produced no result.
    A missing draw is reported and skipped rather than aborting the run: the other
    configurations are still worth having.
    """
    cmd = [sys.executable, str(Path(script).resolve()), *extra]
    # A COMPLETE line, not a trailing "label ... " stub. A pager or a log reads by
    # line and holds a newline-less fragment until a newline arrives, so the
    # in-progress announcement was invisible through `| more` -- which made a run
    # that was merely slow look frozen. Two lines per child is a small price for
    # progress that survives being piped.
    print(f"  -> {label} starting", flush=True)
    started = time.time()

    # start_new_session puts the child in its own process GROUP so a timeout can kill
    # its descendants too. torch inductor spawns compile-worker subprocesses, and
    # killing only the direct child orphans them -- they keep a GPU context and burn
    # memory for the rest of the run.
    #
    # Note this is orphan hygiene, NOT a hang fix. subprocess.run(timeout=...) was
    # suspected of deadlocking in its post-kill drain; a reproduction with a
    # grandchild deliberately holding the pipe showed it returns promptly, because on
    # POSIX it calls process.wait() rather than communicate(). The theory was wrong
    # and the test is what said so.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"TIMED OUT after {timeout_s}s -- killing the process group", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError) as exc:
            print(f"     could not kill the group: {exc}", flush=True)
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = "", ""
            print("     drain still blocked after the group kill; giving up on it",
                  flush=True)
        print("     the other configurations still run; this one is recorded as missing.",
              flush=True)
        return None
    elapsed = time.time() - started

    payload = None
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX):])
        elif line.strip() and not quiet:
            print(line, flush=True)

    if payload is None:
        print(f"     {label} NO RESULT after {elapsed:.0f}s (exit {proc.returncode})",
              flush=True)
        for t in (stderr or "<no stderr>").strip().splitlines()[-4:]:
            print(f"     {t}", flush=True)
    else:
        note = ""
        if "recompile_limit" in (stderr or ""):
            note = "  !! dynamo hit its recompile limit -- output suspect"
            payload["dynamo_limit_hit"] = True
        ident = payload.get("identity_distance")
        detail = f"identity_distance={ident:.6f}  " if ident is not None else ""
        print(f"     {label} done: {detail}({elapsed:.0f}s){note}", flush=True)
        payload.setdefault("elapsed_s", round(elapsed, 1))
    return payload
