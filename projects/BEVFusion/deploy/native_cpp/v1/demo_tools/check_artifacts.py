"""Verify published demo assets and ensure the ZIP contains only the offline HTML."""
import argparse,hashlib,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--docs-dir",type=Path,required=True)
a=p.parse_args()
manifest=json.loads((a.docs_dir/"demo/demo_manifest.json").read_text())
for name,metadata in manifest["files"].items():
    path=a.docs_dir/name
    assert path.stat().st_size==metadata["bytes"],name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata["sha256"],name
with zipfile.ZipFile(a.docs_dir/"demo/bevfusion_v1_interactive.zip") as archive:
    assert archive.namelist()==["bevfusion_v1_interactive.html"]
    html=archive.read("bevfusion_v1_interactive.html")
    assert html==(a.docs_dir/"demo/bevfusion_v1_interactive.html").read_bytes()
assert b"window.BEVFUSION_DEMO_DATA=" in html
assert b"ftyp" in (a.docs_dir/"assets/bevfusion_v1_demo.mp4").read_bytes()[:32]
assert (a.docs_dir/"assets/bevfusion_v1_demo.gif").read_bytes().startswith((b"GIF87a",b"GIF89a"))
print("published_demo_artifacts_valid")
