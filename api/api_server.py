from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import os
from models.main_pipeline import ComicTranslationPipeline

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("OPENAI_API_KEY")
FONT_PATH = "./static/NanumGothic-Bold.ttf"
pipeline = ComicTranslationPipeline(api_key=API_KEY, font_path=FONT_PATH)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png"
}

@app.post("/translate")
async def translate_comic(file: UploadFile = File(...)):
    filename = file.filename
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="지원하지 않는 이미지 확장자입니다.")

    # 파일을 메모리에서 바로 읽어 PIL 이미지로 변환
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    # 파이프라인이 PIL.Image 또는 BytesIO를 지원해야 함
    result_image, _ = pipeline.translate_comic_with_ocr_pil(
        image=image,
        image_name=filename
    )

    # 결과 이미지를 메모리 버퍼에 저장
    output_buffer = BytesIO()
    result_image.save(output_buffer, format=image.format)
    output_buffer.seek(0)

    media_type = MEDIA_TYPES[ext]
    return StreamingResponse(output_buffer, media_type=media_type, headers={
        "Content-Disposition": f"attachment; filename=translated_{filename}"
    })
