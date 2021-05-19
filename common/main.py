import torch
from torch import optim
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from train import *
from lanes.dataset import LanesDataset
from lanes.model import LanesSegNet

if __name__ == '__main__':
    # Default Params
    SEED = 98
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    BATCH_SIZE = 4
    PIN_MEM = True
    NUM_WORKERS = 4
    IMG_SIZE = 512
    EPOCHS = 10
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    # Preprocess Data
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ])

    # Image Path
    img_path = "/media/alan/seagate/vigilant_datasets/images/images/100k/train/"
    img_val_path = "/media/alan/seagate/vigilant_datasets/images/images/100k/val/"
    img_test_path = "/media/alan/seagate/vigilant_datasets/images/images/100k/test/"
    # Lanes Path
    lanes_colormask_trainpath = "/media/alan/seagate/vigilant_datasets/lanes/labels/lane/colormaps/train/"
    lanes_mask_trainpath = "/media/alan/seagate/vigilant_datasets/lanes/labels/lane/masks/train/"
    lanes_colormask_valpath = "/media/alan/seagate/vigilant_datasets/lanes/labels/lane/colormaps/val/"
    lanes_mask_valpath = "/media/alan/seagate/vigilant_datasets/lanes/labels/lane/masks/val/"

    # Lane Segmentation
    loss_fn = torch.nn.MSELoss()
    model = LanesSegNet(4)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    trainset = LanesDataset(img_path, lanes_mask_trainpath, preprocess, 6)
    valset = LanesDataset(img_val_path, lanes_mask_valpath, preprocess, 10)
    train_loader = DataLoader(trainset,
                              BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEM,
                              shuffle=True)
    val_loader = DataLoader(valset,
                            BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEM,
                            shuffle=False)

    print("Training size: ", len(trainset))
    print("Validation size: ", len(valset))
    fit_model(model, optimizer, train_loader, loss_fn, device, EPOCHS,
              val_loader)

    test_model(model, val_loader, device, False)
