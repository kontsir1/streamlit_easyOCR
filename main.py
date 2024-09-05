import easyocr as ocr  # OCR
import streamlit as st  # Web App
from PIL import Image, ImageOps, ImageEnhance, ImageFilter  # Image Processing
import numpy as np  # Image Processing
from easyocr import Reader
import difflib  # For comparing texts
from streamlit_cropper import st_cropper  # For image cropping

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

    # Initialize preprocessing variables
    grayscale = contrast = sharpen = denoise = False
    contrast_factor = sharpen_factor = 1.0
    denoise_radius = 1

    if apply_preprocessing:
        # Preprocessing options
        grayscale = st.sidebar.checkbox("Convert to Grayscale", value=True)
        contrast = st.sidebar.checkbox("Enhance Contrast", value=True)
        contrast_factor = st.sidebar.slider("Contrast Enhancement Factor", 1.0, 3.0, 2.0)
        sharpen = st.sidebar.checkbox("Sharpen Image", value=True)
        sharpen_factor = st.sidebar.slider("Sharpening Factor", 1.0, 3.0, 1.5)
        denoise = st.sidebar.checkbox("Reduce Noise", value=True)
        denoise_radius = st.sidebar.slider("Denoise Radius", 1, 7, 3, step=2)

    # Enable batch processing
    batch_process = st.sidebar.checkbox("Enable Batch Processing", value=False)

    if batch_process:
        images = st.file_uploader("Upload images (multiple selection allowed)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    else:
        images = st.file_uploader("Upload your image here", type=["png", "jpg", "jpeg"])

    # Input field for the original text
    original_text = st.text_area("Enter the original text here:", height=200)

    if images:
        if not batch_process:
            images = [images]

        for idx, image in enumerate(images):
            st.write(f"**Processing Image {idx + 1}:**")

            # Allow the user to crop the image before OCR
            cropped_image = st_cropper(Image.open(image), realtime_update=True)

            ocr_results = process_image(cropped_image, language_code, apply_preprocessing, grayscale, contrast, contrast_factor, sharpen, sharpen_factor, denoise, denoise_radius)
            
            st.write("**Extracted Text with Confidence Scores:**")
            # Display OCR text with confidence scores
            for line, confidence in ocr_results:
                st.write(f"Text: {line} (Confidence: {confidence * 100:.2f}%)")

            if original_text:
                # Extract text only for comparison
                ocr_text = [line for line, _ in ocr_results]
                highlighted_text = compare_texts(ocr_text, original_text)
                st.write("**Comparison Result:**")
                st.markdown(highlighted_text, unsafe_allow_html=True)


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

def process_image(image, language_code: str, apply_preprocessing: bool, grayscale: bool = False, contrast: bool = False, contrast_factor: float = 1.0, sharpen: bool = False, sharpen_factor: float = 1.0, denoise: bool = False, denoise_radius: int = 1) -> list:
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
    """Performs OCR on the given image using the specified language and returns text with confidence scores."""
    reader = load_model(language_code)
    results = reader.readtext(np.array(image))

    # Extract and return text with confidence scores as a list of tuples (text, confidence)
    return [(text[1], text[2]) for text in results]

def compare_texts(ocr_text: list, original_text: str) -> str:
    """Compares OCR text with the original text and highlights differences using HTML and CSS."""
    ocr_lines = "\n".join(ocr_text).splitlines()
    original_lines = original_text.splitlines()

    # Use difflib to compare lines and highlight differences
    diff = difflib.ndiff(original_lines, ocr_lines)
    highlighted_text = []

    for line in diff:
        if line.startswith(' '):  # no difference
            highlighted_text.append(f'<span>{line[2:]}</span>')
        elif line.startswith('-'):  # missing in OCR
            highlighted_text.append(f'<span style="color:red; background-color:#ffcccc;">{line[2:]}</span>')
        elif line.startswith('+'):  # extra in OCR
            highlighted_text.append(f'<span style="color:green; background-color:#ccffcc;">{line[2:]}</span>')

    return "<br>".join(highlighted_text)

if __name__ == "__main__":
    main()
