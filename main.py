import easyocr as ocr  # OCR
import streamlit as st  # Web App
from PIL import Image, ImageOps, ImageEnhance, ImageFilter  # Image Processing
import numpy as np  # Image Processing
from easyocr import Reader
from difflib import HtmlDiff
from bs4 import BeautifulSoup  # For HTML parsing and manipulation

def main():
    # Set up the Streamlit app
    st.title("OCR Text Comparison Tool")
    st.markdown("## Extract and Compare Text from Images")

    # Language selection in the sidebar
    available_languages = {
        "English": "en",
        "French": "fr",
        "Italian": "it",
        "Polish": "pl",
        "German": "de",
        "Spanish": "es",
        "Chinese (Simplified)": "ch_sim",
        "Chinese (Traditional)": "ch_tra",
        "Japanese": "ja",
        "Korean": "ko",
        "Russian": "ru",
        "Arabic": "ar",
    }
    language_options = st.sidebar.multiselect(
        "Choose OCR Language(s)",
        list(available_languages.keys()),
        default=["English"]
    )
    language_codes = [available_languages[lang] for lang in language_options]

    # Preprocessing options in the sidebar
    st.sidebar.markdown("### Image Preprocessing")
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

    # Adjustable space detection thresholds
    st.sidebar.markdown("### Space Detection Thresholds")
    line_threshold = st.sidebar.slider("Line Grouping Threshold", 5, 20, 10)
    double_space_threshold = st.sidebar.slider("Double Space Threshold", 15, 40, 20)
    single_space_threshold = st.sidebar.slider("Single Space Threshold", 5, 14, 10)

    # Image upload section for single image processing
    uploaded_file = st.file_uploader(
        "Upload your image here", type=["png", "jpg", "jpeg"]
    )

    # Input field for the original text
    original_text = st.text_area("Enter the original text here:", height=200)

    if uploaded_file:
        with st.spinner(f'Processing {uploaded_file.name}...'):
            ocr_results = process_image(
                uploaded_file,
                language_codes,
                apply_preprocessing,
                grayscale,
                contrast,
                contrast_factor,
                sharpen,
                sharpen_factor,
                denoise,
                denoise_radius
            )
        if ocr_results:
            # Reconstruct text preserving spaces
            reconstructed_lines = reconstruct_text_with_spaces(
                ocr_results,
                line_threshold,
                double_space_threshold,
                single_space_threshold
            )

            # Display OCR results
            st.markdown("### OCR Extracted Text")
            extracted_text = '\n'.join(reconstructed_lines)
            st.text_area("OCR Extracted Text", extracted_text, height=200)

            # Display original and processed images
            with st.expander("View Images"):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(Image.open(uploaded_file), caption="Original Image", use_column_width=True)
                with col2:
                    processed_image = get_processed_image(
                        uploaded_file,
                        apply_preprocessing,
                        grayscale,
                        contrast,
                        contrast_factor,
                        sharpen,
                        sharpen_factor,
                        denoise,
                        denoise_radius
                    )
                    st.image(processed_image, caption="Processed Image", use_column_width=True)

            # Comparison with original text
            if original_text:
                # Generate diff HTML with color highlights
                diff_html = compare_texts(reconstructed_lines, original_text)
                st.markdown("### Comparison Result")
                # Render the HTML diff using Streamlit components
                st.components.v1.html(diff_html, height=400, scrolling=True)
        else:
            st.error(f"No text found in {uploaded_file.name}. Please try with a different image.")

def process_image(image, language_codes: list, apply_preprocessing: bool,
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

        # Perform OCR on the processed image
        ocr_results = perform_ocr(processed_image, language_codes)

        # Return OCR results
        return ocr_results

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

def get_processed_image(image, apply_preprocessing: bool,
                        grayscale: bool, contrast: bool, contrast_factor: float,
                        sharpen: bool, sharpen_factor: float,
                        denoise: bool, denoise_radius: int) -> Image.Image:
    """Returns the processed image for display."""
    input_image = Image.open(image)
    if apply_preprocessing:
        return preprocess_image(
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
        return input_image

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
def load_model(language_codes: list) -> Reader:
    """Loads the OCR model with the specified languages."""
    try:
        return ocr.Reader(language_codes, model_storage_directory=".")
    except Exception as e:
        st.error(f"Failed to load the OCR model: {e}")
        raise

def perform_ocr(image: Image.Image, language_codes: list) -> list:
    """Performs OCR and returns bounding boxes, text, and confidence scores."""
    reader = load_model(language_codes)
    results = reader.readtext(np.array(image), detail=1, paragraph=False)
    # Each item in results: (bbox, text, confidence)
    return results

def reconstruct_text_with_spaces(ocr_results, line_threshold, double_space_threshold, single_space_threshold):
    """Reconstructs text from OCR results, preserving spaces based on bounding box positions."""
    lines = {}
    for result in ocr_results:
        bbox, text, confidence = result
        # Calculate the center of the bounding box to determine the line
        y_center = np.mean([point[1] for point in bbox])
        # Group words that are on the same line
        line_key = None
        for key in lines.keys():
            if abs(key - y_center) < line_threshold:
                line_key = key
                break
        if line_key is None:
            line_key = y_center
            lines[line_key] = []
        lines[line_key].append((bbox, text))

    # Sort lines by their y position
    sorted_lines = [lines[key] for key in sorted(lines.keys())]

    reconstructed_lines = []
    for line in sorted_lines:
        # Sort words in the line by their x position
        line.sort(key=lambda x: np.mean([point[0] for point in x[0]]))
        line_text = ''
        previous_bbox = None
        for bbox, text in line:
            x_min = min(point[0] for point in bbox)
            if previous_bbox:
                previous_x_max = max(point[0] for point in previous_bbox)
                gap = x_min - previous_x_max
                # Determine the number of spaces based on the gap
                if gap > double_space_threshold:
                    spaces = '  '  # Double space
                elif gap > single_space_threshold:
                    spaces = ' '   # Single space
                else:
                    spaces = ''
                line_text += spaces + text
            else:
                line_text = text
            previous_bbox = bbox
        reconstructed_lines.append(line_text)
    return reconstructed_lines

def compare_texts(ocr_text: list, original_text: str) -> str:
    """Compares OCR text with the original text and returns an HTML diff without any links."""
    from difflib import HtmlDiff
    from bs4 import BeautifulSoup

    original_lines = original_text.splitlines()
    ocr_lines = ocr_text  # ocr_text is already a list of lines

    # Generate the diff HTML
    html_diff = HtmlDiff(wrapcolumn=80).make_file(
        original_lines,
        ocr_lines,
        fromdesc='Original Text',
        todesc='OCR Text',
        context=False,
        numlines=0
    )

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_diff, 'html.parser')

    # Remove all <a> tags within the entire HTML
    for a_tag in soup.find_all('a'):
        a_tag.replace_with_children()  # Replace <a> tag with its children (text)

    # Return the modified HTML
    return str(soup)

if __name__ == "__main__":
    main()
