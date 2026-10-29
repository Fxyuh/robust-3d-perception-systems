from __future__ import annotations
from common import get_args, load_config, seed_everything
args = get_args("export_sparse_native_v2.py", export=True, stage=False)
seed_everything(args.seed)
import sys
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
import projects.BEVFusion.bevfusion
from mmdet3d.registry import MODELS
import onnx
from onnx import helper, TensorProto

out=(args.work_dir / "sparse_native_02")
out.mkdir(parents=True,exist_ok=False)
cfg=load_config(args)
init_default_scope(cfg.get("default_scope","mmdet3d"))
model=MODELS.build(cfg.model)
load_checkpoint(model,str(args.checkpoint),map_location="cpu")
enc=model.pts_middle_encoder.eval()
nodes=[]
weights=[]
index=0

def add_tensor(name,value):
    a=value.detach().cpu().numpy().astype(np.float16)
    weights.append(helper.make_tensor(name,TensorProto.FLOAT16,list(a.shape),a.tobytes(),raw=True))

def conv(x,m,bn,stage,activation="None"):
    global index
    index+=1
    name=f"conv{index}"
    w=m.weight.detach().float()
    if m.bias is None:
        b=torch.zeros(w.shape[0],dtype=torch.float32)
    else:
        b=m.bias.detach().float()
    scale=bn.weight.detach().float()/torch.sqrt(bn.running_var.detach().float()+bn.eps)
    w=w*scale.reshape(-1,1,1,1,1)
    b=(b-bn.running_mean.detach().float())*scale+bn.bias.detach().float()
    add_tensor(name+".weight",w)
    add_tensor(name+".bias",b)
    y=f"t{index}"
    is_subm=m.__class__.__name__=="SubMConv3d"
    nodes.append(helper.make_node("SparseConvolution",[x,name+".weight",name+".bias"],[y],name,
        kernel_size=list(m.kernel_size),stride=list(m.stride),padding=list(m.padding),
        dilation=list(m.dilation),subm=int(is_subm),inverse=0,
        rulebook=("subm"+str(stage)) if is_subm else (m.indice_key or ""),
        activation=activation,precision="fp16",output_precision="fp16",output_bound=160000))
    return y

def relu(x):
    global index
    index+=1
    y=f"t{index}"
    nodes.append(helper.make_node("Relu",[x],[y],f"relu{index}"))
    return y

def block(x,m,stage):
    identity=x
    y=conv(x,m.conv1,m.norm1,stage,"ReLU")
    y=conv(y,m.conv2,m.norm2,stage)
    global index
    index+=1
    z=f"t{index}"
    nodes.append(helper.make_node("Add",[y,identity],[z],f"add{index}",precision="fp16",output_precision="fp16"))
    return relu(z)

def seq(x,m,stage):
    return conv(x,m[0],m[1],stage,"ReLU")

x=seq("features",enc.conv_input,1)
for i,stage in enumerate(enc.encoder_layers,1):
    for child in stage:
        if child.__class__.__name__=="SparseBasicBlock":
            x=block(x,child,i)
        else:
            x=seq(x,child,i)
x=seq(x,enc.conv_out,5)
nodes.append(helper.make_node("ScatterDense",[x],["dense"],"scatter",
    input_spatial_shape=[180,180,2],output_shape=[1,128,180,180,2],
    format="xyz",output_layout="NCHW"))
nodes.append(helper.make_node("Transpose",["dense"],["permuted"],"transpose",dims=[0,1,4,2,3]))
nodes.append(helper.make_node("Reshape",["permuted"],["lidar_bev"],"reshape",dims=[1,256,180,180]))
graph=helper.make_graph(nodes,"robust_epoch4_sparse",[
    helper.make_tensor_value_info("features",TensorProto.FLOAT16,[None,5])],
    [helper.make_tensor_value_info("lidar_bev",TensorProto.FLOAT16,[1,256,180,180])],initializer=weights)
onnx_model=helper.make_model(graph,opset_imports=[helper.make_operatorsetid("",11)])
p=out/"lidar_sparse.onnx"
onnx.save(onnx_model,str(p))
print("exported",p,"nodes",len(nodes),"convs",sum(n.op_type=="SparseConvolution" for n in nodes),"bytes",p.stat().st_size,flush=True)
