"""Compatibility entry point; protobuf regeneration is part of build_native_core.py."""
import argparse
from pathlib import Path
import os
from build_native_core import regenerate_protobuf
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source-dir", type=Path, required=True)
p.add_argument("--build-dir", type=Path, required=True)
p.add_argument("--protobuf-root", type=Path, required=True)
p.add_argument("--library-dir", type=Path, action="append", default=[])
a = p.parse_args()
a.build_dir.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = ":".join(map(str, a.library_dir)) + ":" + env.get("LD_LIBRARY_PATH", "")
regenerate_protobuf(a.source_dir.resolve(), a.build_dir.resolve(), a.protobuf_root.resolve(), env)
