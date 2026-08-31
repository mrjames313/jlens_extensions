"""Shared test helpers.

Chiefly: importing an ``experiments/`` driver on a machine that is not the box.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_driver(name: str):
    """Import ``experiments/<name>.py`` with a stubbed config, leaving no trace.

    Drivers are scripts, not import targets (see ``experiments/README.md``), so this
    goes through ``importlib`` rather than a normal import. Two things have to be
    faked or undone, and both were found the hard way:

    **The config.** A driver calls ``jx_config.load()`` at module scope, which raises
    off-box because ``JLENS_*`` is unset. Replacing ``sys.modules`` is *not* enough:
    drivers say ``from jlens_extensions import config``, which is an attribute lookup
    on the already-imported package, so the real module wins as soon as any earlier
    test has imported it. The package attribute has to be patched too -- which is why
    a test that passes alone can fail in the suite, purely on file ordering.

    **sys.path.** A driver puts ``harness/`` on it at import, exactly as
    ``fit_lens.py`` needs. Left behind, that makes ``jlens`` importable and trips
    ``test_scaffold``'s assertion that ``harness/`` is never installed.
    """
    package = importlib.import_module("jlens_extensions")
    stub = types.ModuleType("jlens_extensions.config")

    class _Cfg:
        machine = "test-box"
        artifact_root = Path("/tmp/does-not-exist")

        def hf_env(self):
            return {}

    stub.load = lambda: _Cfg()
    stub.ConfigError = RuntimeError

    real_module = sys.modules.get("jlens_extensions.config")
    real_attr = getattr(package, "config", None)
    saved_path = list(sys.path)

    sys.modules["jlens_extensions.config"] = stub
    package.config = stub
    driver_name = f"_driver_{name}"
    try:
        spec = importlib.util.spec_from_file_location(
            driver_name, REPO / "experiments" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        # Register before exec: @dataclass resolves its own module through
        # sys.modules[cls.__module__] to build __eq__/__repr__, and an unregistered
        # name makes that None. A driver defining a dataclass fails at import
        # otherwise, with an error that points at dataclasses.py rather than here.
        sys.modules[driver_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        if real_module is not None:
            sys.modules["jlens_extensions.config"] = real_module
        else:
            sys.modules.pop("jlens_extensions.config", None)
        if real_attr is not None:
            package.config = real_attr
        else:
            delattr(package, "config")
