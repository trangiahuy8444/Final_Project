import gradio as gr
from PIL import Image
import json
from config import *
from db import *
from RelTR.inference import *
import os

# Hàm kiểm tra định dạng của ảnh
def is_valid_image(image):
    valid_extensions = ['.jpg', '.jpeg', '.png']  # Các định dạng hợp lệ
    if isinstance(image, Image.Image):  # Kiểm tra nếu đối tượng là hình ảnh PIL
        return True  # Luôn trả về True vì hình ảnh đã hợp lệ
    return False

# Hàm xử lý ảnh
def retrieval_image(image):
    if not is_valid_image(image):
        raise ValueError("Invalid image format. Please upload a .jpg, .jpeg, or .png image.")
    model = load_model('./RelTR/ckpt/checkpoint0149.pth')
    predictions = predict(image, model)

    dataFrame = query_images_for_multiple_subject_relation_object(predictions)

    unique_images = set(dataFrame)  # Loại bỏ các URL trùng

    # # Hiển thị các hình ảnh từ query
    # images = []
    # for img_url in unique_images:
    #     img = Image.open(img_url)
    #     images.append(img)
    # return images

    images = []
    for img_url in unique_images:
        # Tải ảnh từ URL
        try:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            images.append(img)
        except Exception as e:
            print(f"Error loading image from {img_url}: {e}")

    return images

# Tạo giao diện Gradio
interface = gr.Interface(
    fn=retrieval_image,  # Hàm xử lý ảnh tải lên
    inputs=gr.Image(type="pil", label="Upload an Image"),  # Input là ảnh
    outputs=gr.Gallery(label="Similar Images"),  # Hiển thị các ảnh tương tự
    title="Image Retrieval",  # Tiêu đề giao diện
    description="Upload an image to retrieve similar images based on subject, relation, and object."
)

# Chạy giao diện
interface.launch()