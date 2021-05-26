import torch


def save_torchscript(model):
    model.load()
    example = torch.randn(1, 3, 512, 512)
    traced_script_module = torch.jit.trace(model, example)
    cpp_model_dir = "../app/desktop/models/traced_lanesNet.pt"
    traced_script_module.save(cpp_model_dir)
    print("MODEL SAVED.")
