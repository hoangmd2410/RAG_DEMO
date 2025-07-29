from openai import AsyncOpenAI
import base64
from rag.utils import encode_image, crop_all_pages
import asyncio
client = AsyncOpenAI(api_key="123", base_url="https://ydbna6rhr1f7pc-8000.proxy.runpod.net/v1")

model = "nanonets/Nanonets-OCR-s"

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

async def ocr_page_with_nanonets_s(img_base64):
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑️ for check boxes.",
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=15000
    )
    return response.choices[0].message.content

async def ocr_list_pages(list_img_base64):
    list_task = []
    for img_base64 in list_img_base64:
        list_task.append(ocr_page_with_nanonets_s(img_base64))
    list_ocr_text = await asyncio.gather(*list_task)
    ocr_text = "\n".join(list_ocr_text)
    return ocr_text