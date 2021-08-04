import os
import cv2
import torch
from models.pipepline import Pixel_Anchor

NUM_CLASS = 2

def main():
    model = Pixel_Anchor(2)
    x = torch.randn(1, 3, 224, 224)
    model_outputs = model(x)
    
    for i in model_outputs:
        print(i.shape)

if __name__ == '__main__':
    main()