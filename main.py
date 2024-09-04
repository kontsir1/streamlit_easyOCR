import easyocr as ocr  # OCR
import streamlit as st  # Web App
from PIL import Image, ImageOps, ImageEnhance, ImageFilter  # Image Processing
import numpy as np  # Image Processing
from easyocr import Reader
import difflib  # For comparing texts

def main():
    # Set up the Streamlit app
    st.title("Easy OCR - Extract Text from Images")
    st.markdown("## Optical Character Recognition - Using easyocr and streamlit")

    # Language selection in the sidebar
    language = st.sidebar.selectbox(
        "Choose OCR Language",
        ("English (UK)", "French", "Italian", "Polish", "German")
    )

    # Map the selection to easyocr language codes
    language_code = map_language_to_code(language)

    # Preprocessing options in the sidebar
    apply_preprocessing = st.sidebar.checkbox("Apply Image Preprocessing", value=True)
    
    if apply_preprocessing:
        # Preprocessing options
        grayscale = st.sidebar.checkbox("Convert to Grayscale", value=True)
        contrast = st.sidebar.checkbox("Enhance Contrast", value=True)
        contrast_factor = st.sidebar.slider("Contrast Enhancement Factor", 1.0, 3.0, 2.0)
        sharpen = st.sidebar.checkbox("Sharpen Image", value=True)
        sharpen_factor = st.sidebar.slider("Sharpening Factor", 1.0, 3.0, 1.5)
        denoise = st.sidebar.checkbox("Reduce Noise", value=True)
        denoise_radius = st.sidebar.slider("Denoise Radius", 1, 9, 3, step=2)

    # Image upload section
    image = st.file_uploader(label="Upload your image here", type=["png", "jpg", "jpeg"])

    # Input field for the original text
    original_text = st.text_area("Enter the original text here:", height=200)

    if image is not None:
        ocr_text = process_image(image, language_code, apply_preprocessing, grayscale, contrast, contrast_factor, sharpen, sharpen_factor, denoise, denoise_radius)
        
        st.write("**Extracted Text (before comparison):**")
        # Display OCR text line by line to preserve paragraph structure
        for line in ocr_text:
            st.text(line)

        if original_text:
            # Compare OCR text with the original text
            highlighted_text = compare_texts(ocr_text, original_text)
            st.write("**Comparison Result:**")
            st.write(highlighted_text)
    else:
        st.write("Please upload an image to proceed.")

def map_language_to_code(language: str) -> str:
    """Maps language selection to easyocr language codes."""
    language_map = {
        "English (UK)": "en",
        "French": "fr",
        "Italian": "it",
        "Polish": "pl",
        "German": "de"
    }
    return language_map.get(language, "en")  # Default to English

def process_image(image, language_code: str, apply_preprocessing: bool, grayscale: bool, contrast: bool, contrast_factor: float, sharpen: bool, sharpen_factor: float, denoise: bool, denoise_radius: int) -> list:
    """Handles image processing and OCR."""
    try:
        # Open the original image
        input_image = Image.open(image)

        if apply_preprocessing:
            # Preprocess the image based on user options
            processed_image = preprocess_image(input_image, grayscale, contrast, contrast_factor, sharpen, sharpen_factor, denoise, denoise_radius)
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
        
        # Return text as a list of lines
        return result_text

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

def preprocess_image(image: Image.Image, grayscale: bool, contrast: bool, contrast_factor: float, sharpen: bool, sharpen_factor: float, denoise: bool, denoise_radius: int) -> Image.Image:
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

    # Extract and return text from OCR result as a list of lines
    return [text[1] for text in result]

def compare_texts(ocr_text: list, original_text: str) -> str:
    """Compares OCR text with the original text and highlights differences."""
    ocr_lines = "\n".join(ocr_text).splitlines()
    original_lines = original_text.splitlines()

    # Use difflib to compare lines and highlight differences
    diff = difflib.ndiff(original_lines, ocr_lines)
    highlighted_text = []

    for line in diff:
        if line.startswith(' '):  # no difference
            highlighted_text.append(line[2:])
        elif line.startswith('-'):  # missing in OCR
            highlighted_text.append(f'-- {line[2:]} --')
        elif line.startswith('+'):  # extra in OCR
            highlighted_text.append(f'++ {line[2:]} ++')

    return "\n".join(highlighted_text)

if __name__ == "__main__":
    main()
