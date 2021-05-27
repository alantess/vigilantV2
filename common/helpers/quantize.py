import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import convert_fx, prepare_fx
from torch.quantization import QuantStub, DeQuantStub


def quantize(model, data_loader, config="fbgemm"):
    example = torch.randn(1, 3, 512, 512)
    # Configuration
    prep_config_dict = {"non_traceable_module_name": ["base", "deconv"]}
    qconfig = get_default_qconfig(config)
    qconfig_dict = {"": qconfig}
    model.load()
    model.eval()
    # Prepare Model
    model_prepared = prepare_fx(model,
                                qconfig_dict,
                                prepare_custom_config_dict=prep_config_dict)

    calibrate(model_prepared, data_loader)
    model_int_8 = convert_fx(model_prepared)
    # Model Description
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print("ORIGINAL")
    print("Number of Parameters: {:.1f}M".format(params / 1e6))
    print(f"Number of Parameters: {params}M")
    params = sum([np.prod(p.size()) for p in model_int_8.parameters()])
    print("QUANTIZED")
    print("Number of Parameters: {:.6f}M".format(params / 1e6))
    print(f"Number of Parameters: {params}M")

    print_size_of_model(model_int_8)
    model_int_8(example)
    torch.jit.save(torch.jit.script(model_int_8),
                   "../app/desktop/models/traced_quantize_lanesNet.pt")

    return model_int_8


def restructure_model(model):
    model = nn.Sequential(QuantStub(), model, DeQuantStub())
    return model


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def calibrate(model, data_loader):
    print("Calibrating model...")
    loop = tqdm(data_loader)
    model.eval()
    with torch.no_grad():
        for (X, _) in (loop):
            model(X)
