import base64
from pathlib import Path
import fitz
from PIL import Image, ImageFile
import io
from vllm.assets.image import ImageAsset


def encode_image(image: Image.Image, image_format="PNG") -> str:
    im_file = io.BytesIO()
    image.save(im_file, format=image_format)
    im_bytes = im_file.getvalue()
    im_64 = base64.b64encode(im_bytes).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{im_64}"


def convert_pdf_to_image(pdf_path: Path) -> ImageFile:
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # Load the first page
    pix = page.get_pixmap()

    img = Image.open(io.BytesIO(pix.tobytes("jpg")))

    return encode_image(img, image_format="JPEG")


def get_asset(path: Path) -> ImageFile:
    if path.suffix == ".pdf":
        img = convert_pdf_to_image(path)
    else:
        img = ImageAsset(path)

    return img
