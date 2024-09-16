import easyocr as ocr  # OCR
import streamlit as st  # Web App
from PIL import Image, ImageOps, ImageEnhance, ImageFilter  # Image Processing
import numpy as np  # Image Processing
from easyocr import Reader
from difflib import HtmlDiff
from bs4 import BeautifulSoup  # For HTML parsing and manipulation

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

    # Image upload section for single image processing
    image = st.file_uploader("Upload your image here", type=["png", "jpg", "jpeg"])

    # Input field for the original text
    original_text = st.text_area("Enter the original text here:", height=200)

    if image:
        with st.spinner(f'Processing {image.name}...'):
            ocr_results = process_image(
                image,
                language_code,
                apply_preprocessing,
                grayscale,
                contrast,
                contrast_factor,
                sharpen,
                sharpen_factor,
                denoise,
                denoise_radius
            )

        # Reconstruct text preserving spaces
        reconstructed_lines = reconstruct_text_with_spaces(ocr_results)

        # Display OCR results
        st.write(f"**Extracted Text from {image.name}:**")
        extracted_text = '\n'.join(reconstructed_lines)
        st.text_area("OCR Extracted Text:", extracted_text, height=200)

        # Comparison with original text
        if original_text:
            # Use the reconstructed lines for comparison
            diff_html = compare_texts(reconstructed_lines, original_text)
            st.markdown("**Comparison Result:**")
            # Render the HTML diff using Streamlit components
            st.components.v1.html(diff_html, height=600, scrolling=True)

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

def process_image(image, language_code: str, apply_preprocessing: bool,
                  grayscale: bool = False, contrast: bool = False,
                  contrast_factor: float = 1.0, sharpen: bool = False,
                  sharpen_factor: float = 1.0, denoise: bool = False,
                  denoise_radius: int = 1) -> list:
    """Handles image processing and OCR."""
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

        # Display images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(input_image, caption="Original Image", use_column_width=True)

        with col2:
            st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Perform OCR on the processed image
        ocr_results = perform_ocr(processed_image, language_code)

        # Return OCR results
        return ocr_results

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

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

@st.cache_resource
def load_model(language_code: str) -> Reader:
    """Loads the OCR model with the specified language."""
    try:
        return ocr.Reader([language_code], model_storage_directory=".")
    except Exception as e:
        st.error(f"Failed to load the OCR model: {e}")
        raise

def perform_ocr(image: Image.Image, language_code: str) -> list:
    """Performs OCR on the given image using the specified language and returns bounding boxes, text, and confidence scores."""
    reader = load_model(language_code)
    results = reader.readtext(np.array(image), detail=1, paragraph=False)
    # Each item in results: (bbox, text, confidence)
    return results

def reconstruct_text_with_spaces(ocr_results):
    """Reconstructs text from OCR results, preserving spaces based on bounding box positions."""
    reconstructed_lines = []
    line = ''
    previous_bbox = None
    previous_y_min = None

    # Calculate average character width to set thresholds
    char_widths = []
    for result in ocr_results:
        bbox, text, confidence = result
        x_coords = [point[0] for point in bbox]
        char_width = (max(x_coords) - min(x_coords)) / max(len(text), 1)
        char_widths.append(char_width)
    if char_widths:
        average_char_width = np.mean(char_widths)
    else:
        average_char_width = 10  # default value if no text detected

    # Define thresholds
    single_space_threshold = average_char_width * 0.5
    double_space_threshold = average_char_width * 1.5
    new_line_threshold = average_char_width * 2

    for result in ocr_results:
        bbox, text, confidence = result
        x_min = min(point[0] for point in bbox)
        y_min = min(point[1] for point in bbox)
        if previous_bbox:
            previous_x_max = max(point[0] for point in previous_bbox)
            previous_y_min = min(point[1] for point in previous_bbox)
            gap_x = x_min - previous_x_max
            gap_y = abs(y_min - previous_y_min)
            # Check if we're still on the same line
            if gap_y > new_line_threshold:
                # New line
                reconstructed_lines.append(line)
                line = text
            else:
                # Same line
                if gap_x > double_space_threshold:
                    spaces = '  '  # Double space
                elif gap_x > single_space_threshold:
                    spaces = ' '   # Single space
                else:
                    spaces = ''    # No space
                line += spaces + text
        else:
            line = text
        previous_bbox = bbox
        previous_y_min = y_min
    if line:
        reconstructed_lines.append(line)
    return reconstructed_lines

def compare_texts(ocr_text: list, original_text: str) -> str:
    """Compares OCR text with the original text and returns an HTML diff without links in the legend."""
    from difflib import HtmlDiff
    from bs4 import BeautifulSoup

    original_lines = original_text.splitlines()
    ocr_lines = ocr_text  # ocr_text is already a list of lines

    # Generate the diff HTML
    html_diff = HtmlDiff().make_file(
        original_lines,
        ocr_lines,
        fromdesc='Original Text',
        todesc='OCR Text',
        context=True,
        numlines=2
    )

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_diff, 'html.parser')

    # Find the legend table
    legend_table = soup.find('table', {'summary': 'Legends'})

    # Remove all <a> tags within the legend table
    if legend_table:
        for a_tag in legend_table.find_all('a'):
            a_tag.replace_with_children()

    # Return the modified HTML
    return str(soup)

if __name__ == "__main__":
    main()
