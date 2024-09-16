import streamlit as st  # Web App
from PIL import Image, ImageOps, ImageEnhance, ImageFilter  # Image Processing
import numpy as np  # Image Processing
from difflib import HtmlDiff
from bs4 import BeautifulSoup  # For HTML parsing and manipulation
import pytesseract
from pytesseract import Output

# Set the Tesseract executable path if necessary
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update with your path

def main():
    # Set up the Streamlit app
    st.title("Tesseract OCR - Extract Text from Images")
    st.markdown("## Optical Character Recognition - Using Tesseract and Streamlit")

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
    image_file = st.file_uploader("Upload your image here", type=["png", "jpg", "jpeg"])

    # Input field for the original text
    original_text = st.text_area("Enter the original text here:", height=200)

    if image_file:
        with st.spinner(f'Processing {image_file.name}...'):
            ocr_data = process_image(
                image_file,
                apply_preprocessing,
                grayscale,
                contrast,
                contrast_factor,
                sharpen,
                sharpen_factor,
                denoise,
                denoise_radius
            )

        # Check if ocr_data contains 'level' key
        if ocr_data and 'level' in ocr_data:
            # Compute average space width
            average_space_width = compute_average_space_width(ocr_data)

            # Reconstruct text preserving spaces
            reconstructed_text = reconstruct_text_with_spaces(ocr_data, average_space_width)

            # Display OCR results
            st.write(f"**Extracted Text from {image_file.name}:**")
            st.text_area("OCR Extracted Text:", reconstructed_text, height=200)

            # Comparison with original text
            if original_text:
                diff_html = compare_texts(reconstructed_text, original_text)
                st.markdown("**Comparison Result:**")
                # Render the HTML diff using Streamlit components
                st.components.v1.html(diff_html, height=600, scrolling=True)
        else:
            st.error("OCR data is empty or missing expected keys. Please check Tesseract installation and configurations.")
            st.write("OCR Data Keys:", ocr_data.keys() if ocr_data else "No data returned.")

def process_image(image_file, apply_preprocessing: bool,
                  grayscale: bool = False, contrast: bool = False,
                  contrast_factor: float = 1.0, sharpen: bool = False,
                  sharpen_factor: float = 1.0, denoise: bool = False,
                  denoise_radius: int = 1):
    """Handles image processing and OCR."""
    try:
        # Open the original image
        input_image = Image.open(image_file)

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
        ocr_data = perform_ocr(processed_image)

        return ocr_data

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return {}

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

def perform_ocr(image: Image.Image) -> dict:
    """Performs OCR using Tesseract and returns character-level data."""
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config=custom_config,
        lang='eng'
    )
    return data

def compute_average_space_width(ocr_data: dict) -> float:
    """Computes the average width of spaces between words."""
    n_boxes = len(ocr_data['level'])
    gaps = []
    previous_word = None
    for i in range(n_boxes):
        if ocr_data['text'][i].strip() == '':
            continue
        left = ocr_data['left'][i]
        width = ocr_data['width'][i]
        if previous_word:
            gap = left - (previous_word['left'] + previous_word['width'])
            gaps.append(gap)
        previous_word = {'left': left, 'width': width}
    if gaps:
        return np.mean(gaps)
    else:
        return 10  # Default value if no gaps found

def reconstruct_text_with_spaces(ocr_data: dict, average_space_width: float) -> str:
    """Reconstructs text from OCR data, preserving spaces."""
    if 'level' not in ocr_data:
        st.error("OCR data does not contain 'level' key. Please check Tesseract installation and configurations.")
        return ''
    n_boxes = len(ocr_data['level'])
    lines = {}
    for i in range(n_boxes):
        if ocr_data['text'][i].strip() == '':
            continue
        line_num = ocr_data['line_num'][i]
        word_num = ocr_data['word_num'][i]
        left = ocr_data['left'][i]
        width = ocr_data['width'][i]
        text = ocr_data['text'][i]

        if line_num not in lines:
            lines[line_num] = []
        lines[line_num].append({
            'word_num': word_num,
            'left': left,
            'width': width,
            'text': text
        })

    # Reconstruct text line by line
    reconstructed_text = ''
    for line_num in sorted(lines.keys()):
        words = lines[line_num]
        # Sort words by their position
        words.sort(key=lambda x: x['left'])
        line = ''
        previous_word = None
        for word in words:
            if previous_word:
                # Calculate the gap between words
                gap = word['left'] - (previous_word['left'] + previous_word['width'])
                # Determine the number of spaces based on the gap
                if gap > average_space_width * 1.5:
                    spaces = '  '  # Double space
                elif gap > average_space_width * 0.5:
                    spaces = ' '   # Single space
                else:
                    spaces = ''
                line += spaces + word['text']
            else:
                line = word['text']
            previous_word = word
        reconstructed_text += line + '\n'

    return reconstructed_text.strip()

def compare_texts(ocr_text: str, original_text: str) -> str:
    """Compares OCR text with the original text and returns an HTML diff without links in the legend."""
    from difflib import HtmlDiff
    from bs4 import BeautifulSoup

    original_lines = original_text.splitlines()
    ocr_lines = ocr_text.splitlines()

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
