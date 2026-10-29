"""Fetch or verify the pinned CUDA 12.8 Git LFS sparse-convolution library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from common import sha256
from build_native_core import SPCONV_SHA256

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source-dir", type=Path, required=True)
a = p.parse_args()
target = a.source_dir / "libraries/3DSparseConvolution/libspconv/lib/x86_64_cuda12.8/libspconv.so"
if sha256(target) == SPCONV_SHA256:
    print("verified existing libspconv:", SPCONV_SHA256)
    raise SystemExit(0)
pointer = target.read_text()
if not pointer.startswith("version https://git-lfs.github.com/spec/v1"):
    raise ValueError("Refusing to replace an unexpected sparse-convolution library")
fields = dict(row.split(" ", 1) for row in pointer.splitlines() if " " in row)
oid, size = fields["oid"].split(":", 1)[1], int(fields["size"])
if oid != SPCONV_SHA256:
    raise ValueError("Source checkout has a different Git LFS object")
body = json.dumps({"operation": "download", "transfers": ["basic"], "objects": [{"oid": oid, "size": size}]}).encode()
request = urllib.request.Request("https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution.git/info/lfs/objects/batch",
    data=body, headers={"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
with urllib.request.urlopen(request, timeout=30) as response:
    action = json.load(response)["objects"][0]["actions"]["download"]
partial = target.with_name("libspconv.so.download")
digest, count = hashlib.sha256(), 0
request = urllib.request.Request(action["href"], headers=action.get("header", {}))
with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as output:
    for chunk in iter(lambda: response.read(1024 * 1024), b""):
        output.write(chunk)
        digest.update(chunk)
        count += len(chunk)
if count != size or digest.hexdigest() != oid:
    raise ValueError("Downloaded libspconv failed SHA-256/size verification")
os.replace(partial, target)
print("verified downloaded libspconv:", oid)
