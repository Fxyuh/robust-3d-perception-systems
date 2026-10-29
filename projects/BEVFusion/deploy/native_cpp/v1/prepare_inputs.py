"""Prepare the frozen protocol with the original transform and seeded workers."""
import sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import parser,setup,load_config,write_json
p=parser(__doc__)
p.add_argument("--data-root",required=True)
p.add_argument("--output-dir",type=Path,required=True)
p.add_argument("--max-samples",type=int)
a=setup(p.parse_args())
import numpy as np
import projects.BEVFusion.bevfusion
import reference_degradation
from mmengine.runner import Runner
cfg=load_config(a,a.data_root)
a.output_dir.mkdir(parents=True,exist_ok=False)
mean=np.asarray([123.675,116.28,103.53],np.float32).reshape(1,3,1,1)
std=np.asarray([58.395,57.12,57.375],np.float32).reshape(1,3,1,1)
conditions=[("normal",0,0.0),("camera_drop_3_lidar_drop_75",3,0.75),("camera_drop_6_lidar_drop_75",6,0.75)]
for name,drop,ratio in conditions:
    dcfg=copy.deepcopy(cfg.test_dataloader);dcfg["batch_size"]=1;dcfg["num_workers"]=2
    pipeline=dcfg["dataset"]["pipeline"]
    if drop:
        i=next(i for i,x in enumerate(pipeline) if x["type"]=="Pack3DDetInputs")
        pipeline.insert(i,dict(type="V1FrozenSensorInputDegradation",camera_drop_count=drop,lidar_drop_ratio=ratio,seed=a.seed))
        pipeline[i+1]["meta_keys"]=list(pipeline[i+1].get("meta_keys",[]))+["sensor_degradation"]
    loader=Runner.build_dataloader(dcfg,seed=a.seed)
    limit=len(loader.dataset) if a.max_samples is None else a.max_samples
    if not 1<=limit<=len(loader.dataset):raise ValueError("Invalid sample count")
    out=a.output_dir/name;out.mkdir()
    manifest={"dataset":"nuScenes v1.0-mini val","dataset_size":len(loader.dataset),"count":limit,"seed":a.seed,
              "num_workers":2,"condition":name,"camera_drop_count":drop,"lidar_drop_ratio":ratio,"samples":[],
              "degradation_statistics":{"points_before":0,"points_after":0,"dropped_camera_counts":{}}}
    iterator=iter(loader)
    try:
        for index,batch in enumerate(iterator):
            if index>=limit:break
            meta=batch["data_samples"][0].metainfo;sid=int(meta["sample_idx"])
            image=batch["inputs"]["img"][0].numpy().astype(np.float32)
            points=np.ascontiguousarray(batch["inputs"]["points"][0].numpy(),dtype=np.float32)
            if not 0<len(points)<=300000:raise ValueError("Point count outside native contract")
            degradation=meta.get("sensor_degradation",{})
            dropped=degradation.get("dropped_camera_indices",[])
            target=out/f"{index:03d}";target.mkdir()
            np.ascontiguousarray((image-mean)/std,dtype=np.float16).tofile(target/"img.f16")
            points.tofile(target/"points.f32")
            for key in ("cam2lidar","cam2img","lidar2img","img_aug_matrix"):
                value=np.ascontiguousarray(meta[key],dtype=np.float32);assert value.shape==(6,4,4)
                value.tofile(target/(key+".f32"))
            health={"sample_idx":sid,"available_mask":sum(1<<i for i in range(6) if i not in dropped),
                    "dropped_camera_indices":dropped,"points_before":degradation.get("points_before",len(points)),
                    "points_after":len(points)}
            write_json(target/"health.json",health)
            stats=manifest["degradation_statistics"]
            stats["points_before"]+=degradation.get("points_before",0)
            stats["points_after"]+=degradation.get("points_after",0)
            for camera in degradation.get("dropped_camera_names",[]):
                stats["dropped_camera_counts"][camera]=stats["dropped_camera_counts"].get(camera,0)+1
            manifest["samples"].append({"dataset_index":index,"sample_idx":sid,"directory":f"{index:03d}"})
            if (index+1)%20==0 or index+1==limit:print("prepared",name,index+1,"/",limit,flush=True)
    finally:
        if hasattr(iterator,"_shutdown_workers"):iterator._shutdown_workers()
    write_json(out/"manifest.json",manifest)
