"""Package selected actual deployment frames into one offline interactive HTML."""
import argparse,base64,copy,io,json
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--viewer-dir",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--frames-per-condition",type=int,default=12)
p.add_argument("--step",type=int,default=3)
p.add_argument("--max-points",type=int,default=4000)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
source=json.loads((a.viewer_dir/"manifest.json").read_text())
manifest=copy.deepcopy(source);frames={}
baseline=len(json.loads((a.viewer_dir/source["conditions"][0]["frames"][0]).read_text())["points"])//3
for condition,original in zip(manifest["conditions"],source["conditions"]):
    selected=original["frames"][::a.step][:a.frames_per_condition]
    condition["frames"]=selected
    for name in selected:
        data=json.loads((a.viewer_dir/name).read_text())
        points=np.asarray(data["points"]).reshape(-1,3)
        limit=min(len(points),max(100,int(round(a.max_points*len(points)/max(baseline,1)))))
        keep=np.linspace(0,len(points)-1,limit,dtype=np.int64)
        data["points"]=[round(float(value),2) for value in points[keep].reshape(-1)]
        for camera in data["cameras"]:
            if camera["image"]:
                im=Image.open(a.viewer_dir/camera["image"]).convert("RGB").resize((352,128))
                buffer=io.BytesIO();im.save(buffer,format="JPEG",quality=64,optimize=True)
                camera["image"]="data:image/jpeg;base64,"+base64.b64encode(buffer.getvalue()).decode()
        frames[name]=data
assets=Path(__file__).resolve().parent/"viewer"
html=(assets/"index.html").read_text()
html=html.replace('<link rel="stylesheet" href="style.css">',"<style>"+(assets/"style.css").read_text()+"</style>")
html=html.replace("六路相机、点云与检测框，同一帧一起看。",f"真实部署精选回放 · 共 {len(frames)} 帧 · 完整评测每种条件 81 帧")
html=html.replace("nuScenes mini val · 两个场景","精选帧快放 · 评测使用完整 81 帧")
payload=json.dumps({"manifest":manifest,"frames":frames},separators=(",",":"),ensure_ascii=True).replace("<","\\u003c")
html=html.replace('<script src="app.js"></script>',
                  "<script>window.BEVFUSION_DEMO_DATA="+payload+";</script><script>"+(assets/"app.js").read_text()+"</script>")
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(html,encoding="utf-8")
print("offline_demo_ready",a.output,"bytes",a.output.stat().st_size,"frames",len(frames),flush=True)
