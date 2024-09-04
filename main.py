import easyocr as ocr  # OCR
import streamlit as st  # Web App
from PIL import Image, ImageOps, ImageEnhance  # Image Processing
import numpy as np  # Image Processing
from easyocr import Reader
import difflib  # For comparing texts

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

    # Preprocessing toggle in the sidebar
    apply_preprocessing = st.sidebar.checkbox("Apply Image Preprocessing", value=True)

    # Image upload section
    image = st.file_uploader(label="Upload your image here", type=["png", "jpg", "jpeg"])

    # Input field for the original text
    original_text = st.text_area("Enter the original text here:", height=200)

    if image is not None:
        ocr_text = process_image(image, language_code, apply_preprocessing)
        st.write("**Extracted Text (before comparison):**")
        st.write(ocr_text)

        if original_text:
            # Compare OCR text with the original text
            highlighted_text = compare_texts(ocr_text, original_text)
            st.write("**Comparison Result:**")
            st.markdown(highlighted_text, unsafe_allow_html=True)
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

def process_image(image, language_code: str, apply_preprocessing: bool) -> str:
    """Handles image processing and OCR."""
    try:
        # Open the original image
        input_image = Image.open(image)

        if apply_preprocessing:
            # Preprocess the image for better OCR results
            processed_image = preprocess_image(input_image)
        else:
            processed_image = input_image

        # Display images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(input_image, caption="Original Image", use_column_width=True)

        with col2:
            st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Perform OCR on the processed image
        result_text = perform_ocr(processed_image, language_code)
        ocr_text = " ".join(result_text)
        return ocr_text

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

def preprocess_image(image: Image.Image) -> Image.Image:
    """Applies preprocessing steps to the image to optimize OCR results."""
    # Convert image to grayscale
    grayscale_image = ImageOps.grayscale(image)

    # Enhance the image contrast
    enhancer = ImageEnhance.Contrast(grayscale_image)
    enhanced_image = enhancer.enhance(2.0)

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

def compare_texts(ocr_text: str, original_text: str) -> str:
    """Compares OCR text with the original text and highlights differences."""
    # Use difflib to compare texts character by character
    diff = difflib.ndiff(list(original_text), list(ocr_text))
    highlighted_text = []

    for char in diff:
        if char.startswith(' '):  # no difference
            highlighted_text.append(f'<span style="color:green">{char[2:]}</span>')
        elif char.startswith('-'):  # missing in OCR (in original but not in OCR)
            highlighted_text.append(f'<span style="color:red">{char[2:]}</span>')
        elif char.startswith('+'):  # extra in OCR (in OCR but not in original)
            highlighted_text.append(f'<span style="color:blue">{char[2:]}</span>')

    # Join the list into a string and return the highlighted HTML
    return ''.join(highlighted_text)


if __name__ == "__main__":
    main()
