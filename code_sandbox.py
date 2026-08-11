"""
Code Sandbox

Executes a short, untrusted Python snippet (the agent's execute_python
tool) with real OS-level sandboxing via macOS's Seatbelt (sandbox-exec),
not just resource limits. Verified by hand (see commit history / PR
description for the test battery) to actually block:

  - all outbound network access -- sockets, DNS resolution, AND anything
    a spawned child process tries (e.g. shelling out to curl to work
    around no urllib access; Seatbelt restrictions are inherited by child
    processes, confirmed empirically)
  - filesystem writes anywhere except a dedicated scratch directory
  - unbounded execution time (hard wall-clock timeout via subprocess) and
    unbounded CPU time (RLIMIT_CPU, as a second layer independent of the
    wall-clock timeout) and unbounded process count (RLIMIT_NPROC, against
    fork bombs)

Two deliberate, honestly-documented gaps, not silently swept under the rug:

1. Reads are NOT restricted. A first attempt tried `(deny default)` plus a
   hand-built allowlist of exactly which paths Python's own startup needs
   to read -- it crashed Python's own interpreter startup on this machine
   (the allowlist was necessarily incomplete/fragile: exact paths needed
   vary by Python build and would break on a version bump). The profile
   here instead starts from `(allow default)` and denies only network and
   non-scratch writes, which is reliable but means a snippet could read a
   local file and print its contents to stdout, which the agent (and the
   trace log) would then see. Not protection against local secret
   exfiltration via stdout -- only against writes, network, and runaway
   execution, which is what was actually asked for.

2. No hard memory cap. Setting RLIMIT_AS in `preexec_fn` reliably crashed
   the forked-but-not-yet-exec'd Python process before it could even call
   execve (a known CPython footgun: the fork is still running full Python,
   which needs a bit more address space to finish the exec call than the
   limit allows). The wall-clock timeout and CPU-time limit bound runaway
   *loops*, but not a single large one-shot allocation.
"""

import os
import resource
import shutil
import subprocess

# uses the real system/homebrew Python (not this project's venv), so the
# sandbox doesn't need read access to arbitrary installed third-party
# packages -- stdlib only, which is enough for short lookup/compute
# snippets. resolved to the real (non-symlink) path: sandbox-exec's exec
# resolution fails on the venv's symlinked launcher (realpath is denied
# under the sandbox profile before the target is even known), so this
# must be the actual interpreter binary, not a symlink to it.
_PYTHON_BIN = os.path.realpath(shutil.which("python3") or "/usr/bin/python3")

SCRATCH_DIR = "/private/tmp/foresite_code_sandbox"
TIMEOUT_SECONDS = 5
CPU_SECONDS_LIMIT = 5
MAX_PROCESSES = 32
_OUTPUT_CHAR_LIMIT = 4000


def _sandbox_profile() -> str:
    return f"""
(version 1)
(allow default)
(deny network*)
(deny file-write* (require-not (subpath "{SCRATCH_DIR}")))
"""


def _set_resource_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS_LIMIT, CPU_SECONDS_LIMIT))
    resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))


def execute_python(code: str) -> dict:
    """
    runs `code` in the sandbox, returns {stdout, stderr, returncode, timed_out}
    stdout/stderr are truncated to _OUTPUT_CHAR_LIMIT chars each
    """
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    try:
        result = subprocess.run(
            ["sandbox-exec", "-p", _sandbox_profile(), _PYTHON_BIN, "-c", code],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            preexec_fn=_set_resource_limits,
        )
        return {
            "stdout": result.stdout[-_OUTPUT_CHAR_LIMIT:],
            "stderr": result.stderr[-_OUTPUT_CHAR_LIMIT:],
            "returncode": result.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "", "returncode": None, "timed_out": True}


if __name__ == "__main__":
    print(execute_python("print(sum(range(100)))"))
    print(execute_python("import socket; socket.create_connection(('8.8.8.8', 53), timeout=2)"))
    print(execute_python("open('/tmp/escape.txt', 'w').write('x')"))
    print(execute_python("while True: pass"))
