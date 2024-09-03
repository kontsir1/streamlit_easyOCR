import easyocr as ocr  # OCR
import streamlit as st  # Web App
from PIL import Image, ImageOps, ImageEnhance  # Image Processing
import numpy as np  # Image Processing
from easyocr import Reader


def main():
    # Set up the Streamlit app
    st.title("Easy OCR - Extract Text from Images")
    st.markdown("## Optical Character Recognition - Using `easyocr` and `streamlit`")

    # Language selection in the sidebar
    language = st.sidebar.selectbox(
        "Choose OCR Language",
        ("English (UK)", "French", "Italian")
    )

    # Map the selection to easyocr language codes
    language_code = map_language_to_code(language)

    # Image upload section
    image = st.file_uploader(label="Upload your image here", type=["png", "jpg", "jpeg"])

    if image is not None:
        process_image(image, language_code)
    else:
        st.write("Please upload an image to proceed.")


def map_language_to_code(language: str) -> str:
    """Maps language selection to easyocr language codes."""
    language_map = {
        "English (UK)": "en",
        "French": "fr",
        "Italian": "it"
    }
    return language_map.get(language, "en")  # Default to English


def process_image(image, language_code: str):
    """Handles image processing and OCR."""
    try:
        # Open and display the original image
        input_image = Image.open(image)
        st.image(input_image, caption="Original Image", use_column_width=True)

        # Preprocess the image for better OCR results
        processed_image = preprocess_image(input_image)
        st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Perform OCR on the processed image
        result_text = perform_ocr(processed_image, language_code)
        st.write("**Extracted Text:**")
        st.write(result_text)

    except Exception as e:
        st.error(f"An error occurred: {e}")


def preprocess_image(image: Image.Image) -> Image.Image:
    """Applies preprocessing steps to the image to optimize OCR results."""
    # Convert image to grayscale
    grayscale_image = ImageOps.grayscale(image)

    # Enhance the image contrast
    enhancer = ImageEnhance.Contrast(grayscale_image)
    enhanced_image = enhancer.enhance(2.0)

    # Further process the image if needed (e.g., thresholding)
    # You can add more preprocessing steps here if needed

    return enhanced_image


@st.cache_resource
def load_model(language_code: str) -> Reader:
    """Loads the OCR model with the specified language."""
    try:
        return ocr.Reader([language_code], model_storage_directory=".")
    except Exception as e:
        st.error(f"Failed to load the OCR model: {e}")
        raise


def perform_ocr(image: Image.Image, language_code: str) -> list:
    """Performs OCR on the given image using the specified language."""
    reader = load_model(language_code)
    result = reader.readtext(np.array(image))

    # Extract and return text from OCR result
    return [text[1] for text in result]


if __name__ == "__main__":
    main()
