import cv2 as cv
import torch
from torchvision import transforms
import numpy as np


def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img


def show_video(url, model=None, device=None):
    FRAME_SIZE = (1280, 720)
    tensor_resize = transforms.Resize(FRAME_SIZE[::-1])
    model.load()
    model.to(device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    alpha = 0.25

    cap = cv.VideoCapture(url)
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # # Frame to Tensor
        frame_tensor = preprocess(frame)
        frame_tensor = frame_tensor.to(device, dtype=torch.float32)

        # # Forward Pass
        with torch.no_grad():
            mask = model(frame_tensor.unsqueeze(0))
            mask = tensor_resize(mask)
            mask = mask[0].permute(1, 2, 0)
            mask = mask.mul(255).clamp(0, 255)
            mask = mask.detach().cpu().numpy().astype(np.uint8)

        # Display Output

        frame = cv.resize(frame, FRAME_SIZE, interpolation=cv.INTER_AREA)
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        mask = cv.applyColorMap(mask, cv.COLORMAP_TWILIGHT_SHIFTED)

        merged = frame.copy()
        cv.addWeighted(mask, alpha, merged, 1 - alpha, 0, merged)
        merged = increase_brightness(merged, value=20)

        cv.imshow('DEMO', merged)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
