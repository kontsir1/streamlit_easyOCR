import easyocr as ocr
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import numpy as np


def main():
    st.title("Easy OCR - Extract Text from Images")
    st.markdown("## Optical Character Recognition - Using `easyocr` and `streamlit`")

    language = st.sidebar.selectbox("Choose OCR Language", ("English (UK)", "French", "Italian"))
    language_code = map_language_to_code(language)
    apply_preprocessing = st.sidebar.checkbox("Apply Image Preprocessing", value=True)

    image = st.file_uploader(label="Upload your image here", type=["png", "jpg", "jpeg"])

    if image is not None:
        process_image(image, language_code, apply_preprocessing)

        # Input field for comparison
        st.markdown("## Compare OCR Text with Input Text")
        input_text = st.text_area("Enter text to compare with OCR result:", height=200)

        if st.button("Compare Texts"):
            # Display OCR and input text side by side for comparison
            col1, col2 = st.columns(2)
            with col1:
                st.write("**OCR Text:**")
                ocr_text = st.session_state.get("ocr_text", "")
                st.write(ocr_text)

            with col2:
                st.write("**Input Text:**")
                st.write(input_text)

            if ocr_text == input_text:
                st.success("The OCR text matches the input text.")
            else:
                st.error("The OCR text does not match the input text.")

    else:
        st.write("Please upload an image to proceed.")


def map_language_to_code(language: str) -> str:
    language_map = {
        "English (UK)": "en",
        "French": "fr",
        "Italian": "it"
    }
    return language_map.get(language, "en")


def process_image(image, language_code: str, apply_preprocessing: bool):
    try:
        input_image = Image.open(image)

        if apply_preprocessing:
            processed_image = preprocess_image(input_image)
        else:
            processed_image = input_image

        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(processed_image, caption="Processed Image", use_column_width=True)

        result_text = perform_ocr(processed_image, language_code)

        # Save OCR result in session state for comparison
        st.session_state["ocr_text"] = result_text
        st.write("**Extracted Text:**")
        st.write(result_text)

    except Exception as e:
        st.error(f"An error occurred: {e}")


def preprocess_image(image: Image.Image) -> Image.Image:
    grayscale_image = ImageOps.grayscale(image)
    enhancer = ImageEnhance.Contrast(grayscale_image)
    enhanced_image = enhancer.enhance(2.0)
    return enhanced_image


@st.cache_resource
def load_model(language_code: str) -> ocr.Reader:
    try:
        return ocr.Reader([language_code], model_storage_directory=".")
    except Exception as e:
        st.error(f"Failed to load the OCR model: {e}")
        raise


def perform_ocr(image: Image.Image, language_code: str) -> str:
    reader = load_model(language_code)
    result = reader.readtext(np.array(image), detail=1)

    # Sort results based on position (y-axis first, then x-axis)
    result.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

    # Concatenate text maintaining structure
    structured_text = "\n".join([text[1] for text in result])

    return structured_text


if __name__ == "__main__":
    main()
