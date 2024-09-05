import easyocr as ocr  # OCR
import streamlit as st  # Web App
from PIL import Image, ImageOps, ImageEnhance, ImageFilter  # Image Processing
import numpy as np  # Image Processing
from easyocr import Reader
import difflib  # For comparing texts
import io  # For file download

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
    grayscale = contrast = sharpen = denoise = edge_detection = False
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
        edge_detection = st.sidebar.checkbox("Apply Edge Detection", value=False)

    # Image upload section for batch processing
    images = st.file_uploader("Upload your images here", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # Input field for the original text
    original_text = st.text_area("Enter the original text here:", height=200)

    if images:
        for image in images:
            with st.spinner(f'Processing {image.name}...'):
                ocr_results = process_image(image, language_code, apply_preprocessing, grayscale, contrast, contrast_factor, sharpen, sharpen_factor, denoise, denoise_radius, edge_detection)

            st.write(f"**Extracted Text with Confidence Scores from {image.name}:**")
            for line, confidence in ocr_results:
                st.write(f"Text: {line} (Confidence: {confidence * 100:.2f}%)")

            if original_text:
                ocr_text = [line for line, _ in ocr_results]
                highlighted_text = compare_texts(ocr_text, original_text)
                st.markdown("**Comparison Result:**")
                st.markdown(highlighted_text, unsafe_allow_html=True)

        # Option to download the extracted text as a file
        st.download_button(
            label="Download Extracted Text",
            data=download_text(ocr_results),
            file_name="extracted_text.txt",
            mime="text/plain"
        )


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


def process_image(image, language_code: str, apply_preprocessing: bool, grayscale: bool = False, contrast: bool = False,
                  contrast_factor: float = 1.0, sharpen: bool = False, sharpen_factor: float = 1.0,
                  denoise: bool = False, denoise_radius: int = 1, edge_detection: bool = False) -> list:
    """Handles image processing and OCR."""
    try:
        # Open the original image
        input_image = Image.open(image)

        if apply_preprocessing:
            # Preprocess the image based on user options
            processed_image = preprocess_image(input_image, grayscale, contrast, contrast_factor, sharpen, sharpen_factor, denoise, denoise_radius, edge_detection)
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


def preprocess_image(image: Image.Image, grayscale: bool, contrast: bool, contrast_factor: float, sharpen: bool,
                     sharpen_factor: float, denoise: bool, denoise_radius: int, edge_detection: bool) -> Image.Image:
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

    if edge_detection:
        image = image.filter(ImageFilter.FIND_EDGES)

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
            highlighted_text.append(f'<span style="color:red">-- {line[2:]} --</span>')
        elif line.startswith('+'):  # extra in OCR
            highlighted_text.append(f'<span style="color:green">++ {line[2:]} ++</span>')

    return "\n".join(highlighted_text)


def download_text(ocr_results: list) -> io.StringIO:
    """Returns the OCR results as a downloadable text file."""
    buffer = io.StringIO()
    for line, confidence in ocr_results:
        buffer.write(f"Text: {line} (Confidence: {confidence * 100:.2f}%)\n")
    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    main()
