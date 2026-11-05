"""
Provenance record for ``svensemble run`` (written as io.RUN_PROVENANCE).

The record is written when a run starts (``status: "running"``) and rewritten
when it ends (``"succeeded"`` or ``"failed"``), so a record still at
``"running"`` means the process was killed before it could finish.
"""

import hashlib
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from . import __version__
from . import io as sv_io

_PACKAGES = ("pandas", "numpy", "scikit-learn", "xgboost", "pysam", "cyvcf2")


def _plain(obj):
    """Paths to str, sets/tuples to lists, recursively -- for a human-readable record."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_plain(v) for v in obj]
    return obj


def now() -> str:
    """Current local time, ISO 8601 with timezone."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_state() -> dict | None:
    """Commit and dirty flag when the package is imported from a git checkout.

    Only reported when the repository root is the directory containing the
    package, so an installed package inside some unrelated repository is not
    attributed that repository's commit.
    """
    pkg_root = Path(__file__).resolve().parent.parent

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(pkg_root), *args],
            capture_output=True, text=True, timeout=10,
        )

    try:
        top = git("rev-parse", "--show-toplevel")
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != pkg_root:
            return None
        commit = git("rev-parse", "HEAD").stdout.strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": dirty}


def _package_versions() -> dict:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def file_info(path: Path | None) -> dict | None:
    """Path, size and mtime of an input file (not checksummed: inputs can be large)."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def output_info(path: Path) -> dict:
    """Path, size and sha256 of a published output."""
    path = Path(path)
    return {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}


def start(
    *,
    command: str | None,
    parameters: dict,
    inputs: dict,
    model: dict,
) -> dict:
    """The record for a run that is starting.

    *parameters* and *inputs* are stored with Paths as plain strings. In
    *model*, ``config`` is stored tag-encoded (io.dict_json_encode) so it
    survives JSON losslessly (tuples, inf).
    """
    model = {**model, "config": sv_io.dict_json_encode(model.get("config"))}
    return {
        "status": "running",
        "start_time": now(),
        "end_time": None,
        "duration_s": None,
        "error": None,
        "command": command,
        "argv": list(sys.argv) if command is not None else None,
        "cwd": os.getcwd(),
        "parameters": _plain(parameters),
        "model": model,
        "software": {
            "svensemble": __version__,
            "git": _git_state(),
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "host": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "tmpdir": os.environ.get("TMPDIR"),
        },
        "inputs": _plain(inputs),
        "outputs": None,
    }


def finish(
    record: dict,
    *,
    error: BaseException | None,
    outputs: dict[str, Path] | None,
    start_monotonic: float,
    end_monotonic: float,
) -> dict:
    """Mark *record* finished: succeeded with output hashes, or failed with the error."""
    record["status"] = "failed" if error is not None else "succeeded"
    record["end_time"] = now()
    record["duration_s"] = round(end_monotonic - start_monotonic, 3)
    if error is not None:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    if outputs is not None:
        record["outputs"] = {name: _plain(output_info(p)) for name, p in outputs.items()}
    return record


def write(record: dict, dirs: list[Path]) -> None:
    """Write *record* as io.RUN_PROVENANCE into each of *dirs* (atomically)."""
    for d in dirs:
        sv_io.write_config(record, Path(d) / sv_io.RUN_PROVENANCE)
