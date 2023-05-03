import torch
import torchvision

if __name__ == "__main__":
    vgg = torchvision.models.vgg11(
        torchvision.models.VGG11_Weights.IMAGENET1K_V1)
