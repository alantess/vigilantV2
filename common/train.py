import torch
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import torchvision as vision
import matplotlib.pyplot as plt
import numpy as np


def fit_model(model,
              optimizer,
              data_loader,
              loss_fn,
              device,
              epochs,
              val_data_loader=None,
              load_model=False):
    scaler = torch.cuda.amp.GradScaler()
    best_score = np.inf
    model.to(device)

    if load_model:
        model.load()
        print('MODEL LOADED')

    print('Starting...')
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(data_loader)
        # Training
        for i, (img, mask) in enumerate(loop):
            # visualize(img,mask)
            img = img.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.float32)

            for p in model.parameters():
                p.grad = None

            with torch.cuda.amp.autocast():
                pred = model(img)
                loss = loss_fn(pred, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"EPOCH {epoch}\t Loss: {total_loss:.5f}")

        # Validation
        if val_data_loader:
            print("Validation:")
            val_loop = tqdm(val_data_loader)
            val_total_loss = 0
            with torch.no_grad():
                for j, (val_img, val_mask) in enumerate(val_loop):
                    val_img = val_img.to(device, dtype=torch.float32)
                    val_mask = val_mask.to(device, dtype=torch.float32)

                    with torch.cuda.amp.autocast():
                        val_pred = model(val_img)
                        val_loss = loss_fn(val_pred, val_mask)

                    val_total_loss += val_loss.item()
            # visualize(val_img, val_pred, True, epoch)
            print(f"Validation Loss: {val_total_loss:.5f}")

        # Save Model
        if val_total_loss < best_score:
            best_score = val_total_loss
            model.save()
            print("MODEL SAVED")

    print("DONE.")


def test_model(model, data_loader, device, default=False):
    img, mask = next(iter(data_loader))
    mask = mask.to(dtype=torch.float32)

    if default:
        img = img.to(device, dtype=torch.float32)
        model.load()
        model.to(device)
        with torch.no_grad():
            pred = model(img)

        visualize(img, pred, True)
    else:
        visualize(img, mask)


# Visualization
def visualize(img, mask, train_mode=False, epoch=0, iteration=0):
    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    img = invTrans(img)
    img = img.cpu()
    if train_mode:
        mask = mask.detach().cpu()
    # print(mask.size())
    # print(torch.unique(mask))
    img = vision.utils.make_grid(img)
    mask = vision.utils.make_grid(mask)
    grayscale = vision.transforms.Grayscale()
    mask = grayscale(mask)

    # plt.imshow(img.permute(1, 2, 0))
    plt.imshow(mask.permute(1, 2, 0), cmap='rainbow', alpha=0.3)
    save_name = 'img/' + str(epoch) + '_val.jpg'
    # plt.savefig(save_name)
    # plt.show()
    # plt.close()
