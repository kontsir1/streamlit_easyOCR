# main.py
import streamlit as st
from ocr_utils import process_image, perform_ocr
from constants import AVAILABLE_LANGUAGES
from PIL import Image
from difflib import HtmlDiff
from bs4 import BeautifulSoup

def main():
    # Set up the Streamlit app
    st.title("OCR Text Comparison Tool")
    st.markdown("## Extract and Compare Text from Images")

    # Language selection
    language = st.sidebar.selectbox(
        "Choose OCR Language",
        list(AVAILABLE_LANGUAGES.keys())
    )
    language_code = AVAILABLE_LANGUAGES[language]

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

    # Image upload section
    uploaded_file = st.file_uploader(
        "Upload your image here", type=["png", "jpg", "jpeg"]
    )

    # Input field for the original text
    original_text = st.text_area("Enter the original text here:", height=200)

    if uploaded_file:
        with st.spinner(f'Processing {uploaded_file.name}...'):
            processed_image = process_image(
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
            if processed_image is not None:
                ocr_text = perform_ocr(processed_image, language_code)
            else:
                st.error("Failed to process the image.")
                return

        if ocr_text.strip():
            # Display OCR results
            st.markdown("### OCR Extracted Text")
            st.text_area("OCR Extracted Text", ocr_text, height=200)

            # Display original and processed images
            with st.expander("View Images"):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(Image.open(uploaded_file), caption="Original Image", use_column_width=True)
                with col2:
                    st.image(processed_image, caption="Processed Image", use_column_width=True)

            # Comparison with original text
            if original_text.strip():
                diff_html = compare_texts(ocr_text, original_text)
                st.markdown("### Comparison Result")
                # Render the HTML diff using Streamlit components
                st.components.v1.html(diff_html, height=400, scrolling=True)
        else:
            st.error(f"No text found in {uploaded_file.name}. Please try with a different image.")

def compare_texts(ocr_text: str, original_text: str) -> str:
    """Compares OCR text with the original text and returns an HTML diff without any links."""
    original_lines = original_text.splitlines()
    ocr_lines = ocr_text.splitlines()

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
