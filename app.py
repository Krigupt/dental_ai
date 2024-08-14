import streamlit as st
from tooth_detection import run_model
import os

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    placeholder = st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.read())

    # Create a button to segment the image
    b1 = st.empty()
    b1click = b1.button('Segment')
    if b1click:
        b1.empty()

        # Run the model on the uploaded image
        has_disease, fp = run_model(image_path)

        # Display processed image
        processed_image_path = 'static/output/output.jpg'
        if os.path.exists(processed_image_path):
            st.image(processed_image_path, caption="Processed Image", use_column_width=True)
        else:
            st.error("Processed image not found.")

        # Display result based on whether disease is detected
        if not has_disease:
            st.subheader(':green[NO DISEASES FOUND -- YOU HAVE VERY HEALTHY TEETH]')
        else:
            st.subheader(':red[DISEASE DETECTED]')
