"""Build a static TensorRT engine using the existing runtime."""
import argparse,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--onnx",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--python-path",action="append",default=[])
a=p.parse_args()
sys.path.extend(a.python_path)
import tensorrt as trt
if a.output.exists():raise FileExistsError(a.output)
logger=trt.Logger(trt.Logger.WARNING)
b=trt.Builder(logger);n=b.create_network(0);parser=trt.OnnxParser(n,logger)
if not parser.parse_from_file(str(a.onnx)):
    raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
cfg=b.create_builder_config();cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,3*1024**3)
engine=b.build_serialized_network(n,cfg)
if engine is None:raise RuntimeError("TensorRT build failed")
a.output.write_bytes(engine)
print("engine_ready",a.output,flush=True)
