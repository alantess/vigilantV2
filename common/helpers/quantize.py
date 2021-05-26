import torch


def quantize(model, config="qnnpack"):
    example = torch.randn(1, 3, 512, 512)
    model.load()
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig(config)
    # model_fp32_fused = torch.quantization.fuse_modules(model, [['base']])
    model_prepared = torch.quantization.prepare(model)
    model_prepared(example)
    model_int_8 = torch.quantization.convert(model_prepared)
    traced_script_module = torch.jit.trace(model, example)
    cpp_model_dir = "saved_models/quantize_lanes.pt"
    traced_script_module.save(cpp_model_dir)
    print("QUANTIZED MODEL SAVED.")

    return model_int_8
