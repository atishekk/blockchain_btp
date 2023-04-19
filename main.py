from models.vgg import VGG11

if __name__ == "__main__":
    model = VGG11.new()
    print(model.blocks())
    VGG11.encode(model, "db.sqlite")
    dec_model = VGG11.decode("db.sqlite")
    print(dec_model.blocks())
