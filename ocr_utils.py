# ocr_utils.py
import easyocr
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import streamlit as st

@st.cache_resource
def load_model(language_code: str) -> easyocr.Reader:
    """Loads the OCR model with the specified language."""
    try:
        return easyocr.Reader([language_code], model_storage_directory=".")
    except Exception as e:
        st.error(f"Failed to load the OCR model: {e}")
        raise

def process_image(image, apply_preprocessing: bool,
                  grayscale: bool, contrast: bool, contrast_factor: float,
                  sharpen: bool, sharpen_factor: float,
                  denoise: bool, denoise_radius: int) -> Image.Image:
    """Handles image processing."""
    try:
        # Open the original image
        input_image = Image.open(image)

        if apply_preprocessing:
            # Preprocess the image based on user options
            processed_image = preprocess_image(
                input_image,
                grayscale,
                contrast,
                contrast_factor,
                sharpen,
                sharpen_factor,
                denoise,
                denoise_radius
            )
        else:
            processed_image = input_image

        return processed_image

    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        return None

def preprocess_image(image: Image.Image, grayscale: bool, contrast: bool,
                     contrast_factor: float, sharpen: bool, sharpen_factor: float,
                     denoise: bool, denoise_radius: int) -> Image.Image:
    """Applies preprocessing steps to the image to optimize OCR results."""
    if grayscale:
        image = ImageOps.grayscale(image)

    if contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

    if sharpen:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpen_factor)

    if denoise:
        image = image.filter(ImageFilter.MedianFilter(size=denoise_radius))

    return image

def perform_ocr(image: Image.Image, language_code: str) -> str:
    """Performs OCR and returns the extracted text."""
    reader = load_model(language_code)
    results = reader.readtext(np.array(image), detail=0, paragraph=True)
    # Combine the results into a single string
    ocr_text = '\n'.join(results)
    return ocr_text
