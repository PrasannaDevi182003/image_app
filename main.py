# main.py
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageFilter, ImageEnhance
import shutil
import cv2
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/download")
def download_image():
    return FileResponse("static/edited_image.jpg", media_type="image/jpeg", filename="edited_image.jpg")

@app.post("/edit", response_class=HTMLResponse)
async def edit_image(
    request: Request,
    file: UploadFile = File(None),
    brightness: str = Form(None),
    noise: str = Form(None),
    width: str = Form(None),
    height: str = Form(None),
    rotate: str = Form(None),
    flip: str = Form(None),
    filter: str = Form(None),
    face_blur: str = Form(None),
    crop_x: str = Form(None),
    crop_y: str = Form(None),
    crop_w: str = Form(None),
    crop_h: str = Form(None),
    reset: str = Form(None)
):
    # Load new file if uploaded
    if file:
        with open("static/original.jpg", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        shutil.copy("static/original.jpg", "static/edited_image.jpg")

    # Reset to original
    if reset:
        shutil.copy("static/original.jpg", "static/edited_image.jpg")

    img = Image.open("static/edited_image.jpg")

    # Resize
    if width and height:
        try:
            img = img.resize((int(width), int(height)))
        except:
            pass

    # Brightness
    if brightness == "Increase":
        img = img.point(lambda p: min(255, int(p * 1.2)))
    elif brightness == "Decrease":
        img = img.point(lambda p: max(0, int(p * 0.8)))

    # Noise removal
    if noise == "Remove Noise":
        img = img.filter(ImageFilter.MedianFilter(size=3))

    # Rotate
    if rotate:
        try:
            img = img.rotate(-int(rotate), expand=True)
        except:
            pass

    # Flip
    if flip == "horizontal":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == "vertical":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Filters
    if filter == "bw":
        img = img.convert("L").convert("RGB")
    elif filter == "sepia":
        sepia = img.convert("RGB")
        w, h = sepia.size
        pixels = sepia.load()
        for py in range(h):
            for px in range(w):
                r, g, b = pixels[px, py]
                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                pixels[px, py] = (min(255, tr), min(255, tg), min(255, tb))
        img = sepia
    elif filter == "invert":
        img = Image.fromarray(255 - np.array(img))
    elif filter == "sharpen":
        img = img.filter(ImageFilter.SHARPEN)
    elif filter == "contrast":
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)

    # Face blur using OpenCV
    if face_blur == "on":
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = cv_img[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(face, (51, 51), 30)
            cv_img[y:y+h, x:x+w] = blurred
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    # Crop
    if crop_x and crop_y and crop_w and crop_h:
        try:
            x, y, w, h = int(crop_x), int(crop_y), int(crop_w), int(crop_h)
            img = img.crop((x, y, x + w, y + h))
        except:
            pass

    # Save edited image
    img.save("static/edited_image.jpg")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "edited": True,
        "edited_image": "/static/edited_image.jpg"
    })
