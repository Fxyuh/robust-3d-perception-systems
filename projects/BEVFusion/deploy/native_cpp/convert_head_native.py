from common import get_args, load_config, seed_everything
args = get_args("convert_head_native.py", export=False, stage=False)
from pathlib import Path
import sys
import onnx
from onnx import TensorProto,helper
source=(args.work_dir / "head_export_03/head_top200.onnx")
directory=(args.work_dir / "head_native_01")
directory.mkdir(parents=True,exist_ok=False)
model=onnx.load(str(source))
graph=model.graph
assert len(graph.input)==1 and graph.input[0].name=="bev_features"
graph.input[0].name="middle"
graph.input[0].type.tensor_type.elem_type=TensorProto.FLOAT16
graph.node.insert(0,helper.make_node("Cast",["middle"],["bev_features"],name="middle_to_float32",to=TensorProto.FLOAT))
shapes={x.name:[d.dim_value for d in x.type.tensor_type.shape.dim] for x in graph.output}
del graph.output[:]
def add_cast_output(original,target,shape):
    inside=original+"_float32_internal"
    found=0
    for node in graph.node:
        for i,value in enumerate(node.output):
            if value==original:
                node.output[i]=inside
                found+=1
    assert found==1,(original,found)
    for node in graph.node:
        for i,value in enumerate(node.input):
            if value==original: node.input[i]=inside
    graph.node.append(helper.make_node("Cast",[inside],[target],name=target+"_to_float16",to=TensorProto.FLOAT16))
    graph.output.append(helper.make_tensor_value_info(target,TensorProto.FLOAT16,shape))
for original,target in (("center","reg"),("height","height"),("dim","dim"),("rot","rot"),("vel","vel")):
    add_cast_output(original,target,shapes[original])
graph.node.append(helper.make_node("Sigmoid",["heatmap"],["head_score_sigmoid"],name="HeadScoreSigmoid"))
graph.node.append(helper.make_node("Mul",["head_score_sigmoid","query_heatmap_score"],["head_score_weighted"],name="HeadScoreWeighted"))
assert any("/Cast_5_output_0" in n.output for n in graph.node), "Unexpected exported class-gate graph"
graph.node.append(helper.make_node("Mul",["head_score_weighted","/Cast_5_output_0"],["head_score_float32"],name="HeadScoreClassGate"))
graph.node.append(helper.make_node("Cast",["head_score_float32"],["score"],name="HeadScoreToFloat16",to=TensorProto.FLOAT16))
graph.output.append(helper.make_tensor_value_info("score",TensorProto.FLOAT16,[1,10,200]))
onnx.checker.check_model(model)
out=directory/"head_bbox.onnx"
onnx.save(model,str(out))
print("io",[(x.name,x.type.tensor_type.elem_type) for x in graph.input],[(x.name,x.type.tensor_type.elem_type) for x in graph.output],flush=True)
print("onnx_ok",out,out.stat().st_size,flush=True)
