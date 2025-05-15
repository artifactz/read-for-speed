import uuid, contextlib
from pathlib import Path
from PIL import Image


class _TileImageWriter:
    def __init__(self, folder: str | Path, prefix: str = None, max_width: int = 50000, image_mode="L"):
        """
        :param folder: Folder to save the images.
        :param max_width: Maximum width of the tiled image. If the total width exceeds this, it will save the current
                          image and start a new one.
        :param image_mode: Image mode for the new image. Default is "L" (grayscale).
        """
        self.folder = Path(folder)
        self.prefix = prefix
        self.max_width = max_width
        self.image_mode = image_mode
        self.images = []
        self.image_size = None

    def write(self, img: Image.Image):
        if self.image_size is None:
            self.image_size = img.size
        elif self.image_size != img.size:
            raise ValueError("All images must have the same size")

        self.images.append(img)

        if len(self.images) * self.image_size[0] > self.max_width:
            self.save()

    def save(self):
        if not self.images:
            return

        total_width = len(self.images) * self.image_size[0]
        total_height = self.image_size[1]

        new_image = Image.new(self.image_mode, (total_width, total_height))

        for i, img in enumerate(self.images):
            new_image.paste(img, (i * self.image_size[0], 0))
            img.close()

        filename = f"{self.prefix or ''}{uuid.uuid4()}.png"
        new_image.save(self.folder / filename)
        new_image.close()

        self.images = []


@contextlib.contextmanager
def tile_image_writer(folder: str | Path, prefix: str = None, max_width=50000):
    """
    Context manager for writing tiled images to a folder.
    :param folder: Folder to save the images.
    :param max_width: Maximum width of the tiled image. If the total width exceeds this, it will save the current image
                      and start a new one.
    """
    writer = _TileImageWriter(folder, prefix, max_width)
    try:
        yield writer
    finally:
        writer.save()


def read_tile_image(path: str | Path, tile_width: int = None) -> list[Image.Image]:
    """
    Reads an image and splits it into tiles of the specified width.
    :param path: Path to the image file.
    :param tile_width: Width of each tile. If None, the height of the image is used as the tile width, assuming square
                       tiles.
    :return: List of tile images.
    """
    img = Image.open(path)

    tile_width = tile_width or img.size[1]
    if img.size[0] % tile_width != 0:
        raise ValueError("Image width must be divisible by tile width")

    num_tiles = img.size[0] // tile_width
    tile_height = img.size[1]

    tiles = []
    for i in range(num_tiles):
        left = i * tile_width
        right = left + tile_width
        tile = img.crop((left, 0, right, tile_height))
        tiles.append(tile)

    return tiles
