from common import get_args, load_config, seed_everything
args = get_args("convert_native_stages_fp16.py", export=False, stage=False)
from pathlib import Path
import sys
import onnx
from onnx import TensorProto,helper
src=(args.work_dir / "stages_native_01")
dst=(args.work_dir / "stages_native_02")
dst.mkdir(parents=True,exist_ok=False)
for filename in ("camera_vtransform.onnx","fuser.onnx"):
    model=onnx.load(str(src/filename))
    graph=model.graph
    for item in graph.input:
        name=item.name
        assert item.type.tensor_type.elem_type==TensorProto.FLOAT
        inside=name+"_float32_internal"
        for node in graph.node:
            for i,value in enumerate(node.input):
                if value==name: node.input[i]=inside
        graph.node.insert(0,helper.make_node("Cast",[name],[inside],name=name+"_to_float32",to=TensorProto.FLOAT))
        item.type.tensor_type.elem_type=TensorProto.FLOAT16
    for item in graph.output:
        name=item.name
        assert item.type.tensor_type.elem_type==TensorProto.FLOAT
        inside=name+"_float32_internal"
        found=0
        for node in graph.node:
            for i,value in enumerate(node.output):
                if value==name: node.output[i]=inside;found+=1
        assert found==1,(filename,name,found)
        for node in graph.node:
            for i,value in enumerate(node.input):
                if value==name: node.input[i]=inside
        graph.node.append(helper.make_node("Cast",[inside],[name],name=name+"_to_float16",to=TensorProto.FLOAT16))
        item.type.tensor_type.elem_type=TensorProto.FLOAT16
    onnx.checker.check_model(model)
    out=dst/filename
    onnx.save(model,str(out))
    print("converted",out,[(x.name,x.type.tensor_type.elem_type) for x in graph.input],[(x.name,x.type.tensor_type.elem_type) for x in graph.output],flush=True)
