import torch
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
              load_model=False,
              segmentation=True):
    scaler = torch.cuda.amp.GradScaler()
    best_score = np.inf
    model = model.to(device)

    if load_model:
        model.load()
        print('MODEL LOADED')

    print('Starting...')
    for epoch in range(epochs):
        total_loss = 0
        # Training
        for i, (img, mask) in enumerate(data_loader, 0):
            # visualize(img,mask)
            img = img.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)
            if segmentation:
                mask = mask.squeeze(1)

            for p in model.parameters():
                p.grad = None

            with torch.cuda.amp.autocast():
                pred = model(img)
                loss = loss_fn(pred, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 200 == 0:
                print(f"[{i}/{epoch}]: {loss.item():.5f}")

            total_loss += loss.item()

        print(f"EPOCH {epoch}\t Loss: {total_loss:.5f}")

        # Validation
        if val_data_loader:
            print("Validation:")
            val_total_loss = 0
            with torch.no_grad():
                for j, (val_img, val_mask) in enumerate(val_data_loader, 0):
                    val_img = val_img.to(device, dtype=torch.float32)
                    val_mask = val_mask.to(device, dtype=torch.long)
                    if segmentation:
                        val_mask = val_mask.squeeze(1)

                    with torch.cuda.amp.autocast():
                        val_pred = model(val_img)
                        val_loss = loss_fn(val_pred, val_mask)

                    val_total_loss += val_loss.item()
            print(f"Validation Loss: {val_total_loss:.5f}")

        # Save Model
        if val_total_loss < best_score:
            best_score = total_loss
            model.save()
            print("MODEL SAVED")

    print("DONE.")


def test_model(model, data_loader, device):
    model.load()
    model.to(device)
    img, mask = next(iter(data_loader))
    mask = mask.to(dtype=torch.long)

    img = img.to(device, dtype=torch.float32)

    pred = model(img)
    visualize(img, mask)


# Visualization
def visualize(img, mask, train_mode=False):
    print(torch.unique(mask))
    img = img / 2 + 0.5
    plt.figure(figsize=(3, 4))
    img = img.cpu()
    if train_mode:
        print(mask.size())
        mask = mask.argmax(1).detach().cpu()

    img = vision.utils.make_grid(img)
    mask = vision.utils.make_grid(mask)
    grayscale = vision.transforms.Grayscale()
    mask = grayscale(mask)

    # plt.imshow(img.permute(1, 2, 0))
    plt.imshow(mask.permute(1, 2, 0), cmap='jet', alpha=0.4)
    plt.show()
