# -*- coding: utf-8 -*-
"""YOLO .pt -> ONNX and ONNX -> TensorRT engine (optional TRT install)."""


def export_yolo_pt_to_onnx(weight_path, imgsz=1280, **kwargs):
    """Export a YOLO .pt checkpoint to ONNX next to weights (Ultralytics)."""
    from ultralytics import YOLO

    model = YOLO(weight_path)
    return model.export(imgsz=imgsz, format="onnx", **kwargs)


def build_tensorrt_engine(
    max_batch_size=1,
    onnx_file_path="",
    engine_file_path="",
    fp16_mode=False,
    save_engine=False,
    input_dynamic=False,
):
    """
    Build or load a TensorRT engine from ONNX (legacy onnx2engine API).
    Requires: pip install tensorrt (and matching CUDA).
    """
    import tensorrt as trt
    import os

    TRT_LOGGER = trt.Logger()

    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    def get_engine(
        max_batch_size=1,
        onnx_file_path="",
        engine_file_path="",
        fp16_mode=False,
        save_engine=False,
        input_dynamic=False,
    ):
        if os.path.exists(engine_file_path):
            print("Reading engine from file: {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser:
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            builder.max_batch_size = max_batch_size
            if fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
            if not os.path.exists(onnx_file_path):
                raise FileNotFoundError("ONNX file {} not found!".format(onnx_file_path))
            print("loading onnx file from path {} ...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Begining onnx file parsing")
                parser.parse(model.read())
            print("Completed parsing of onnx file")
            print("Building an engine from file{}' this may take a while...".format(onnx_file_path))
            if input_dynamic:
                profile = builder.create_optimization_profile()
                profile.set_shape("input", (1, 3, 32, 32), (1, 3, 32, 320), (1, 3, 32, 640))
                config.add_optimization_profile(profile)
            print(network.get_layer(network.num_layers - 1).get_output(0).shape)
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")
            if save_engine and engine is not None:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    return get_engine(
        max_batch_size,
        onnx_file_path,
        engine_file_path,
        fp16_mode,
        save_engine,
        input_dynamic,
    )
