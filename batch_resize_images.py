from PIL import Image
import os, sys

path = "/home/dennis/Downloads/knight_sprites/png"


def resize(directory):
    dirs = os.listdir(directory)
    for item in dirs:
        if os.path.isfile(os.path.join(path, item)):
            im = Image.open(os.path.join(path, item))
            f, e = os.path.splitext(os.path.join(path, item))
            imResize = im.resize((500, 500), Image.ANTIALIAS)
            imResize.save(f + " resized.png", "PNG", quality=90)


if __name__ == "__main__":
    resize(path)
