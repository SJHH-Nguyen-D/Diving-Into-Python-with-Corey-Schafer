from PIL import Image
import os, sys

path = "/home/dennis/Desktop/Link to datascience_job_portfolio/images_to_resize_for_blog"


def resize(directory):
    dirs = os.listdir(directory)
    for item in dirs:
        if os.path.isfile(os.path.join(path, item)):
            im = Image.open(os.path.join(path, item))
            f, e = os.path.splitext(os.path.join(path, item))
            imResize = im.resize((1038, 437), Image.ANTIALIAS)
            imResize.save(f + " resized.jpeg", "JPEG", quality=100)
    print("Finished Resizing!!!")


if __name__ == "__main__":
    resize(path)
