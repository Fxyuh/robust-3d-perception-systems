from common import get_args, load_config, seed_everything
args = get_args("camera_fp16_io.py", export=False, stage=False)
from pathlib import Path
import sys
import onnx
from onnx import TensorProto,helper
source=(args.work_dir / "camera_native_01/camera_with_depth.onnx")
directory=(args.work_dir / "camera_native_02")
directory.mkdir(parents=True,exist_ok=False)
model=onnx.load(str(source))
graph=model.graph
for item in graph.input:
    name=item.name
    assert item.type.tensor_type.elem_type==TensorProto.FLOAT
    converted=name+"_float32_internal"
    for node in graph.node:
        for index,value in enumerate(node.input):
            if value==name: node.input[index]=converted
    graph.node.insert(0,helper.make_node("Cast",[name],[converted],name=name+"_to_float32",to=TensorProto.FLOAT))
    item.type.tensor_type.elem_type=TensorProto.FLOAT16
for item in graph.output:
    name=item.name
    assert item.type.tensor_type.elem_type==TensorProto.FLOAT
    converted=name+"_float32_internal"
    found=0
    for node in graph.node:
        for index,value in enumerate(node.output):
            if value==name:
                node.output[index]=converted
                found+=1
    assert found==1,(name,found)
    for node in graph.node:
        for index,value in enumerate(node.input):
            if value==name: node.input[index]=converted
    graph.node.append(helper.make_node("Cast",[converted],[name],name=name+"_to_float16",to=TensorProto.FLOAT16))
    item.type.tensor_type.elem_type=TensorProto.FLOAT16
out=directory/"camera_native_fp16_io.onnx"
onnx.checker.check_model(model)
onnx.save(model,str(out))
print("io",[(x.name,x.type.tensor_type.elem_type) for x in graph.input],[(x.name,x.type.tensor_type.elem_type) for x in graph.output],flush=True)
print("onnx_ok",out,out.stat().st_size,flush=True)
