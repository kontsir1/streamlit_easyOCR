import easyocr as ocr  # OCR
import streamlit as st  # Web App
from PIL import Image  # Image Processing
import numpy as np  # Image Processing
from easyocr import Reader


def main():
    # title
    st.title("Easy OCR - Extract Text from Images")

    # subtitle
    st.markdown("## Optical Character Recognition - Using `easyocr`, `streamlit`")

    # image uploader
    image = st.file_uploader(label="Upload your image here", type=["png", "jpg", "jpeg"])

    if image is not None:

        input_image = Image.open(image)  # read image
        st.image(input_image)  # display image

        with st.spinner(" AI is at Work! "):
            reader = load_model()  # load model
            result = reader.readtext(np.array(input_image))

            result_text = []  # empty list for results

            for text in result:
                result_text.append(text[1])

            st.write(result_text)
    else:
        st.write("Upload an Image")


@st.cache
def load_model() -> Reader:
    return ocr.Reader(["en"], model_storage_directory=".")


if __name__ == "__main__":
    main()
