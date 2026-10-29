"""Verify prepared v1 corruption against the original evaluation transform."""
import sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import parser,setup,load_config,write_json
p=parser(__doc__)
p.add_argument("--data-root",required=True)
p.add_argument("--input-dir",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=setup(p.parse_args())
import numpy as np
import projects.BEVFusion.bevfusion
import reference_degradation
from mmengine.runner import Runner
report={}
for name,count,ratio in (("normal",0,0.0),("camera_drop_3_lidar_drop_75",3,0.75),("camera_drop_6_lidar_drop_75",6,0.75)):
    cfg=load_config(a,a.data_root);dcfg=copy.deepcopy(cfg.test_dataloader.dataset)
    if count:
        pipeline=dcfg["pipeline"]
        index=next(i for i,x in enumerate(pipeline) if x["type"]=="Pack3DDetInputs")
        pipeline.insert(index,dict(type="V1FrozenSensorInputDegradation",camera_drop_count=count,lidar_drop_ratio=ratio,seed=a.seed))
        pipeline[index+1]["meta_keys"]=list(pipeline[index+1].get("meta_keys",[]))+["sensor_degradation"]
    loader_cfg=copy.deepcopy(cfg.test_dataloader)
    loader_cfg["dataset"]=dcfg;loader_cfg["batch_size"]=1;loader_cfg["num_workers"]=2
    loader=Runner.build_dataloader(loader_cfg,seed=a.seed);checked=[]
    iterator=iter(loader)
    for index,batch in enumerate(iterator):
        if index not in (0,len(loader.dataset)-1):continue
        sample={"inputs":{k:v[0] for k,v in batch["inputs"].items()},"data_samples":batch["data_samples"][0]}
        d=a.input_dir/name/f"{index:03d}"
        points=np.fromfile(d/"points.f32",np.float32).reshape(-1,5)
        np.testing.assert_array_equal(points,sample["inputs"]["points"].numpy())
        image=sample["inputs"]["img"].numpy().astype(np.float32)
        expected=((image-np.asarray([123.675,116.28,103.53],np.float32).reshape(1,3,1,1))/np.asarray([58.395,57.12,57.375],np.float32).reshape(1,3,1,1)).astype(np.float16)
        actual=np.fromfile(d/"img.f16",np.float16).reshape(6,3,256,704)
        np.testing.assert_array_equal(actual,expected)
        health=json.loads((d/"health.json").read_text())
        dropped=sample["data_samples"].metainfo.get("sensor_degradation",{}).get("dropped_camera_indices",[])
        assert health["dropped_camera_indices"]==dropped
        checked.append(index)
    if hasattr(iterator,"_shutdown_workers"):iterator._shutdown_workers()
    report[name]={"checked_indices":checked,"points_exact":True,"normalized_image_fp16_exact":True,"camera_masks_exact":True}
write_json(a.output,report);print(json.dumps(report,indent=2),flush=True)
