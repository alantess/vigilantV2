import cv2 as cv
from .support import *
import torch
from torch.cuda.amp import autocast
from torchvision import transforms
import numpy as np


class Display(object):
    def __init__(self, url):
        self.url = url

    def increase_brightness(self, img, value=30):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv.merge((h, s, v))
        img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return img

    def show(self,
             model=None,
             device=None,
             load_model=True,
             quantized_model=False):
        if load_model and not quantized_model:
            model.load()
            model.to(device)

        FRAME_SIZE = (1280, 720)
        tensor_resize = transforms.Resize(FRAME_SIZE[::-1])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        alpha = 0.55

        frame_tensor = torch.zeros((1, 3, 512, 512),
                                   device=device,
                                   dtype=torch.float32)
        cap = cv.VideoCapture(self.url)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv.resize(frame, FRAME_SIZE, interpolation=cv.INTER_AREA)

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            if model:
                #  Frame to Tensor
                frame_input = preprocess(frame)
                frame_tensor[0] = frame_input

                #  Forward Pass
                with torch.no_grad():
                    with autocast():
                        mask = model(frame_tensor)
                        mask = tensor_resize(mask)
                        mask = mask[0].permute(1, 2, 0)
                        mask = mask.mul(255).clamp(0, 255)
                    mask = mask.detach().cpu().numpy().astype(np.float32)
                    mask = apply_sharpen_filter(mask,
                                                alpha=50).astype(np.uint8)
                # Display Output
                mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
                mask = cv.applyColorMap(mask, cv.COLORMAP_TWILIGHT_SHIFTED)

                merged = frame.copy()
                cv.addWeighted(mask, alpha, merged, 1 - alpha, 0, merged)
                merged = self.increase_brightness(merged, value=20)

                cv.imshow('DEMO', merged)
            else:
                cv.imshow('DEMO', frame)

            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
