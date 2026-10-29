from common import get_args, load_config, seed_everything
args = get_args("rewrite_head_topk.py", export=False, stage=False)
from pathlib import Path
import sys
import onnx
from onnx import TensorProto, helper
source=(args.work_dir / "head_export_02/head.onnx")
out_dir=(args.work_dir / "head_export_03")
out_dir.mkdir(parents=True,exist_ok=False)
model=onnx.load(str(source))
nodes=model.graph.node
matches=[(i,n) for i,n in enumerate(nodes) if n.op_type=="TopK"]
assert len(matches)==1
idx,topk=matches[0]
const=helper.make_node("Constant",[],["topk_200"],name="TopK_200_Constant",value=helper.make_tensor("topk_200_tensor",TensorProto.INT64,[1],[200]))
nodes.insert(idx,const)
topk.input[1]="topk_200"
out=out_dir/"head_top200.onnx"
onnx.checker.check_model(model)
onnx.save(model,str(out))
print("rewritten",out,out.stat().st_size,flush=True)
