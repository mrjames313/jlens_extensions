"""Verify this machine is configured before a slow fit finds out it isn't.

Run it with::

    python -m jlens_extensions.setup_check

It checks that the three variables are set, that both roots exist and are genuinely
writable, and it *reports which filesystem each root resolves to*. That last part is
the point: the failure this guards against is scratch silently sitting on a network
mount, which a fit would discover only by running slowly for hours.

Exit status is 0 when everything passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from .config import ConfigError, Config, load

#: Filesystems that should never hold the scratch root. The checkpoint loop rewrites
#: the whole running sum every prompt, so a networked scratch turns a fit into a file
#: transfer benchmark.
NETWORK_FS = frozenset(
    {
        "nfs", "nfs4", "cifs", "smbfs", "smb3", "afs", "afpfs", "ncpfs",
        "glusterfs", "ceph", "lustre", "beegfs", "9p", "davfs",
        "fuse.sshfs", "fuse.s3fs", "fuse.rclone", "fuse.gcsfuse",
    }
)


def _mount_table() -> list[tuple[Path, str, str]]:
    """``(mount_point, device, fstype)`` from /proc/mounts, longest mount point first.

    Returns an empty list off Linux; callers degrade to reporting the mount point
    alone, found by walking up device boundaries.
    """
    try:
        raw = Path("/proc/mounts").read_text()
    except OSError:
        return []
    rows = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            device, mount, fstype = parts[0], parts[1].replace("\\040", " "), parts[2]
            rows.append((Path(mount), device, fstype))
    rows.sort(key=lambda row: len(str(row[0])), reverse=True)
    return rows


def _walk_up_to_mount(path: Path) -> Path:
    """Find the mount point by climbing until the device id changes."""
    path = path if path.exists() else path.parent
    try:
        dev = path.stat().st_dev
    except OSError:
        return path
    current = path
    while current != current.parent:
        parent = current.parent
        try:
            if parent.stat().st_dev != dev:
                return current
        except OSError:
            return current
        current = parent
    return current


def describe_filesystem(path: Path) -> tuple[Path, str, str]:
    """``(mount_point, device, fstype)`` for whichever filesystem holds *path*."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    for mount, device, fstype in _mount_table():
        try:
            probe.relative_to(mount)
        except ValueError:
            continue
        return mount, device, fstype
    return _walk_up_to_mount(probe), "unknown", "unknown"


def is_writable(path: Path) -> tuple[bool, str]:
    """Actually write a file rather than trusting ``os.access``.

    ``os.access`` consults permission bits, which can disagree with reality on
    read-only mounts, full filesystems, and ACL-governed directories. Since the
    cost of being wrong here is a fit that dies partway through, do the real thing.
    """
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".jlens-writecheck-"):
            pass
        return True, ""
    except OSError as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def _gb(n: int) -> str:
    return f"{n / 1e9:,.1f} GB"


def _report_root(label: str, path: Path, note: str, problems: list[str]) -> None:
    print(f"  {label:<20} {path}{note}")

    if not path.exists():
        print(f"  {'':<20}   MISSING -- create it, or point the variable elsewhere")
        problems.append(f"{label} does not exist: {path}")
        return
    if not path.is_dir():
        print(f"  {'':<20}   NOT A DIRECTORY")
        problems.append(f"{label} is not a directory: {path}")
        return

    mount, device, fstype = describe_filesystem(path)
    if fstype == "unknown":
        # No /proc/mounts, i.e. not Linux. Say so plainly rather than printing
        # "unknown (unknown)", which reads as a failure -- and note the loss,
        # since the network-filesystem guard has no type to test against.
        print(
            f"  {'':<20}   filesystem  mounted at {mount} "
            f"(type needs /proc/mounts; network-fs guard inactive)"
        )
    else:
        print(f"  {'':<20}   filesystem  {device} ({fstype}) mounted at {mount}")

    usage = shutil.disk_usage(path)
    print(f"  {'':<20}   free        {_gb(usage.free)} of {_gb(usage.total)}")

    ok, why = is_writable(path)
    print(f"  {'':<20}   writable    {'yes' if ok else 'NO -- ' + why}")
    if not ok:
        problems.append(f"{label} is not writable: {path}")

    return None


def check(cfg: Config, *, create: bool = False) -> list[str]:
    """Print the report. Returns a list of problems; empty means everything passed."""
    problems: list[str] = []

    print("jlens_extensions setup check\n")
    print(f"  {'JLENS_MACHINE':<20} {cfg.machine}")

    _report_root("JLENS_ARTIFACT_ROOT", cfg.artifact_root, "", problems)

    note = "   [derived -- JLENS_SCRATCH_ROOT unset]" if cfg.scratch_is_derived else ""
    _report_root("JLENS_SCRATCH_ROOT", cfg.scratch_root, note, problems)

    # The plan's specific hazard, called out rather than left to be read off the table.
    if cfg.scratch_root.exists():
        _, _, fstype = describe_filesystem(cfg.scratch_root)
        if fstype in NETWORK_FS:
            problems.append(
                f"scratch root is on a network filesystem ({fstype}). The checkpoint "
                f"loop rewrites the entire running sum every prompt; put it on local NVMe."
            )

    if cfg.scratch_is_derived:
        print(
            "\n  Scratch follows the artifact root because JLENS_SCRATCH_ROOT is unset.\n"
            "  That is the documented single-disk configuration, not an oversight."
        )

    print("\n  derived paths")
    if create:
        cfg.ensure_dirs()
    for path in cfg.all_dirs:
        state = "exists" if path.exists() else "missing (run with --create)"
        try:
            shown = path.relative_to(cfg.artifact_root.parent)
        except ValueError:
            shown = path
        print(f"  {'':<20}   {str(shown):<44} {state}")

    print(f"\n  profile for this machine: {cfg.profile_path}")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m jlens_extensions.setup_check",
        description="Verify the JLENS_* environment before running a fit.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="create the derived subdirectories (never the roots themselves)",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load()
    except ConfigError as exc:
        # The whole point of the task: name the variable, never fall back.
        print(f"jlens_extensions setup check: FAILED\n\n{exc}", file=sys.stderr)
        return 1

    problems = check(cfg, create=args.create)

    if problems:
        print("\nFAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
