"""Adapt selected ONNX boundary tensors to the native half-precision interface."""
import argparse
from pathlib import Path
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--source",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--keep-fp32",action="append",default=[])
p.add_argument("--python-path",action="append",default=[])
a=p.parse_args()
sys.path.extend(a.python_path)
import onnx
from onnx import helper,TensorProto
if a.output.exists():raise FileExistsError(a.output)
model=onnx.load(str(a.source));g=model.graph
for item in g.input:
    if item.name in a.keep_fp32:continue
    name=item.name;internal=name+"_fp32_internal"
    for node in g.node:
        for i,value in enumerate(node.input):
            if value==name:node.input[i]=internal
    g.node.insert(0,helper.make_node("Cast",[name],[internal],name=name+"_to_fp32",to=TensorProto.FLOAT))
    item.type.tensor_type.elem_type=TensorProto.FLOAT16
for item in g.output:
    name=item.name;internal=name+"_fp32_internal"
    for node in g.node:
        for i,value in enumerate(node.output):
            if value==name:node.output[i]=internal
    for node in g.node:
        for i,value in enumerate(node.input):
            if value==name:node.input[i]=internal
    g.node.append(helper.make_node("Cast",[internal],[name],name=name+"_to_fp16",to=TensorProto.FLOAT16))
    item.type.tensor_type.elem_type=TensorProto.FLOAT16
onnx.checker.check_model(model)
onnx.save(model,str(a.output))
