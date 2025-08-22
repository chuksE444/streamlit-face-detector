import streamlit as st
import cv2
import numpy as np
import io

st.title("Face Detection App")

st.markdown("""
This app allows you to upload an image and detect faces in it.
You can adjust the parameters for face detection, choose rectangle color/thickness,
and download the processed image.
""")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image into OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the original image
    st.image(image, channels="BGR", caption="Original Image", use_column_width=True)

    st.sidebar.header("Adjust Detection Parameters")

    # Adjust scaleFactor
    scale_factor = st.sidebar.slider("Scale Factor", 1.01, 2.0, 1.1, 0.01)
    st.sidebar.write(f"Current Scale Factor: {scale_factor}")

    # Adjust minNeighbors
    min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5, 1)
    st.sidebar.write(f"Current Min Neighbors: {min_neighbors}")

    # Choose rectangle color
    color_picker = st.sidebar.color_picker("Choose Rectangle Color", "#FF0000")
    # Convert hex to BGR
    hex_color = color_picker.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    color_bgr = (b, g, r)

    # Rectangle thickness
    rect_thickness = st.sidebar.slider("Rectangle Thickness", 1, 10, 2)

    # Load the Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if face_cascade.empty():
        st.error("Error loading cascade classifier. Please check your OpenCV installation.")
    else:
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )

        # Draw rectangles
        output_img = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(output_img, (x, y), (x + w, y + h), color_bgr, rect_thickness)

        # Show processed image
        st.image(output_img, channels="BGR", caption="Image with Detected Faces", use_column_width=True)
        st.write(f"Found {len(faces)} faces in the image.")

        # Download button
        is_success, buffer = cv2.imencode(".png", output_img)
        if is_success:
            st.download_button(
                label="Download Image with Detected Faces",
                data=buffer.tobytes(),
                file_name="image_with_faces.png",
                mime="image/png"
            )
