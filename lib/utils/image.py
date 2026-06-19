from PIL import Image

def get_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size