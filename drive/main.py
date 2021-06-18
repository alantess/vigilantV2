import sys

sys.path.insert(0, '..')
import json
import os
import torch
from torch import optim
import argparse
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from common.helpers.video import Display
from common.helpers.train import *
from common.helpers.support import *
from common.helpers.transfer_model import *
from lanes.dataset import LanesDataset
from lanes.model import LanesSegNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AI over the self driving space.')
    parser.add_argument('--test',
                        type=bool,
                        default=False,
                        help='Trains Models')
    parser.add_argument('--batch',
                        type=int,
                        default=16,
                        help='Batch size of input')

    parser.add_argument(
        '--Model',
        type=int,
        default=0,
        help='Model #0: Lanes Net \n Model #1: Quantized Lanes Net')

    args = parser.parse_args()

    SEED = 98
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    BATCH_SIZE = args.batch
    PIN_MEM = True
    NUM_WORKERS = 4
    IMG_SIZE = 512
    CLASSES = 4
    if not args.test:
        print("Train Mode.")
        EPOCHS = 100
    else:
        print("Test Mode.")
        EPOCHS = 1

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    # Preprocess Data
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ])

    # Video Paths
    video_url = "../etc/videos/driving.mp4"
    display = Display(video_url)

    # Loads paths
    json_file = open('paths.json')
    json_str = json_file.read()
    paths = json.loads(json_str)

    # Lane Segmentation
    loss_fn = torch.nn.MSELoss()
    model = LanesSegNet(CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    trainset = LanesDataset(paths["img_path"], paths["lanes_mask_trainpath"],
                            preprocess, 6)
    calibrated_set = LanesDataset(paths["img_path"],
                                  paths["lanes_mask_trainpath"], preprocess,
                                  36)
    valset = LanesDataset(paths["img_val_path"], paths["lanes_mask_valpath"],
                          preprocess, 10)
    calibrated_loader = DataLoader(calibrated_set,
                                   BATCH_SIZE,
                                   num_workers=NUM_WORKERS,
                                   pin_memory=PIN_MEM,
                                   shuffle=True)

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

    if not args.test:
        fit_model(model, optimizer, train_loader, loss_fn, device, EPOCHS,
                  val_loader)
    else:
        display.show(model, device)
