from common import get_args, load_config, seed_everything
args = get_args("build_native_camera_fp16_io.py", export=False, stage=False)
from pathlib import Path
import sys
import tensorrt as trt
p=(args.work_dir / "camera_native_02/camera_native_fp16_io.onnx")
out=p.with_name("camera_native_fp16.engine")
if out.exists(): raise FileExistsError(out)
logger=trt.Logger(trt.Logger.WARNING)
builder=trt.Builder(logger)
network=builder.create_network(0)
parser=trt.OnnxParser(network,logger)
if not parser.parse_from_file(str(p)):
    raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
for i in range(network.num_inputs):
    assert network.get_input(i).dtype==trt.float16
for i in range(network.num_outputs):
    assert network.get_output(i).dtype==trt.float16
config=builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,args.workspace_gib*(1024**3))
serialized=builder.build_serialized_network(network,config)
if serialized is None: raise RuntimeError("builder returned None")
out.write_bytes(serialized)
runtime=trt.Runtime(logger)
engine=runtime.deserialize_cuda_engine(serialized)
print("engine_io",[(engine.get_tensor_name(i),str(engine.get_tensor_dtype(engine.get_tensor_name(i))),tuple(engine.get_tensor_shape(engine.get_tensor_name(i)))) for i in range(engine.num_io_tensors)],flush=True)
print("engine_ok",out,out.stat().st_size,flush=True)
