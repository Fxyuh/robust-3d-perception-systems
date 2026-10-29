"""Create a local interactive viewer from actual frozen-v1 native predictions."""
import argparse,json,shutil,pickle
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--input-dir",type=Path,required=True)
p.add_argument("--validation-dir",type=Path,required=True)
p.add_argument("--ann-file",type=Path,required=True)
p.add_argument("--output-dir",type=Path,required=True)
p.add_argument("--max-points",type=int,default=8000)
a=p.parse_args()
if a.max_points<100:raise ValueError("max-points must be at least 100")
a.output_dir.mkdir(parents=True,exist_ok=False)
assets=Path(__file__).resolve().parent/"viewer"
for name in ("index.html","app.js","style.css"):shutil.copy2(assets/name,a.output_dir/name)
with a.ann_file.open("rb") as file:infos=pickle.load(file)
names=list(infos["data_list"][0]["images"])
comparison=json.loads((a.validation_dir/"comparison.json").read_text())
manifest={"method":comparison["method"],"camera_names":names,"weights":comparison["weights"],
          "visualization_note":"Display point subsampling only; evaluation uses complete inputs.","conditions":[]}
labels={"normal":"正常输入","camera_drop_3_lidar_drop_75":"缺 3 路相机 + 丢 75% 点","camera_drop_6_lidar_drop_75":"相机全失效 + 丢 75% 点"}
mean=np.asarray([123.675,116.28,103.53],np.float32).reshape(1,3,1,1)
std=np.asarray([58.395,57.12,57.375],np.float32).reshape(1,3,1,1)
for row in comparison["results"]:
    condition=row["condition"];spec={"name":condition,"label":labels[condition],"metrics":row["native_v1"],"frames":[]}
    inputs=a.input_dir/condition;source=json.loads((inputs/"manifest.json").read_text())
    for ordinal,entry in enumerate(source["samples"],1):
        index=entry["dataset_index"];d=inputs/entry.get("directory",f"{index:03d}")
        prediction_path=a.validation_dir/"inference"/condition/"predictions"/f"{index:03d}.json"
        prediction=json.loads(prediction_path.read_text())
        health=json.loads((d/"health.json").read_text())
        target=a.output_dir/"frames"/condition/f"{index:03d}";target.mkdir(parents=True)
        image=np.fromfile(d/"img.f16",np.float16).reshape(6,3,256,704).astype(np.float32)
        image=np.clip(image*std+mean,0,255).astype(np.uint8)
        points=np.fromfile(d/"points.f32",np.float32).reshape(-1,5)[:,:3]
        display_limit=max(100,int(round(a.max_points*len(points)/max(health["points_before"],1))))
        keep=np.linspace(0,len(points)-1,min(len(points),display_limit),dtype=np.int64)
        points=np.round(points[keep],2)
        l2i=np.fromfile(d/"lidar2img.f32",np.float32).reshape(6,4,4)
        aug=np.fromfile(d/"img_aug_matrix.f32",np.float32).reshape(6,4,4)
        cameras=[]
        for camera,name in enumerate(names):
            available=bool(health["available_mask"]&(1<<camera));url=None
            if available:
                im=Image.fromarray(image[camera].transpose(1,2,0)[:,:,::-1])
                path=target/f"camera_{camera}.jpg";im.save(path,quality=76,optimize=True)
                url=path.relative_to(a.output_dir).as_posix()
            cameras.append({"name":name,"available":available,"image":url,
                            "lidar2img":l2i[camera].reshape(-1).astype(float).tolist(),
                            "augmentation":aug[camera].reshape(-1).astype(float).tolist()})
        boxes=[]
        for box,score,label in zip(prediction["boxes_3d"],prediction["scores"],prediction["labels"]):
            boxes.append([int(label),round(float(score),5),*[round(float(v),4) for v in box[:7]]])
        frame={"sample_idx":prediction["sample_idx"],"route":prediction["route"],"model_ms":prediction["model_ms"],
               "point_count":prediction["point_count"],"points":points.reshape(-1).astype(float).tolist(),
               "boxes":boxes,"cameras":cameras}
        path=target/"frame.json";path.write_text(json.dumps(frame,separators=(",",":"),allow_nan=False))
        spec["frames"].append(path.relative_to(a.output_dir).as_posix())
        if ordinal%20==0 or ordinal==len(source["samples"]):print("viewer",condition,ordinal,flush=True)
    manifest["conditions"].append(spec)
(a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
print("viewer_ready",a.output_dir,flush=True)
