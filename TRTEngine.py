import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from collections import namedtuple
import sys

logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, namespace="")

sys.path.insert(0, r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.12.0.36\python")

Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

class TRTEngine:
    def __init__(self, engine_path):
        print(f"Loading TensorRT engine from: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize the engine!")

        self.context = self.engine.create_execution_context()
        self.bindings = []
        self.stream = cuda.Stream()

        device = cuda.Device(0)
        print(f"Using GPU: {device.name()}")

        nb_bindings = self.engine.num_io_tensors

        for index in range(nb_bindings):
            name = self.engine.get_tensor_name(index)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            binding = Binding(name, dtype, shape, host_mem, device_mem)
            self.bindings.append(binding)

        # 判断哪些是输入输出
       # self.inputs = [b for b in self.bindings if self.engine.is_execution_binding_input(self.engine.get_tensor_index(b.name))]
        #num_inputs = self.engine.num_io_tensors
        
        self.inputs = []
        self.outputs = []
        
        for b in self.bindings:
            model = self.engine.get_tensor_mode(b.name)
            if model == trt.TensorIOMode.INPUT:
                self.inputs.append(b)
            else:
                self.outputs.append(b)
        #self.outputs = [b for b in self.bindings if not self.engine.is_execution_binding_input(self.engine.get_tensor_index(b.name))]

    def infer(self, input_array):
        if not input_array.flags['C_CONTIGUOUS']:
            input_array = np.ascontiguousarray(input_array)

        # 只处理第一个输入绑定
        np.copyto(self.inputs[0].data, input_array.ravel())
        cuda.memcpy_htod_async(self.inputs[0].ptr, self.inputs[0].data, self.stream)

        for binding in self.bindings:
            self.context.set_tensor_address(binding.name, int(binding.ptr))

            self.context.execute_async_v3(stream_handle=self.stream.handle)


       # self.context.execute_async_v2(
        #    bindings=[b.ptr for b in self.bindings],
         #   stream_handle=self.stream.handle
        #)

        for output in self.outputs:
            cuda.memcpy_dtoh_async(output.data, output.ptr, self.stream)
        self.stream.synchronize()

        return [output.data.reshape(output.shape) for output in self.outputs]
