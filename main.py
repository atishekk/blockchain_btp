from models.vgg import VGG11

if __name__ == "__main__":
    model = VGG11.new()
    print(model.blocks())
