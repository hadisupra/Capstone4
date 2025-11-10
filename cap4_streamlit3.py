import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter
import cv2
print("NumPy version:", np.__version__)
print("OpenCV version:", cv2.__version__)

# Load YOLOv8 model
model = YOLO("best.pt")  # Replace with "yolov8n.pt" if needed

# Page setup
st.set_page_config(page_title="Deep Learning YOLOv12 Vehicle Detection", layout="centered")
st.title("🚦 YOLOv12 Vehicle Detection")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Show uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to numpy and run inference
        image_np = np.array(image)
        results = model(image_np, conf=0.25)

        # Extract class names and detected class indices
        names = model.names
        detections = results[0].boxes.cls.cpu().numpy()

        # Count cars and buses
        car_count = sum(1 for cls in detections if names[int(cls)] == 'car')
        bus_count = sum(1 for cls in detections if names[int(cls)] == 'bus')

        # Display detection metrics
        st.subheader("📊 Detection Summary")
        st.write(f"🚗 **Cars Detected:** {car_count}")
        st.write(f"🚌 **Buses Detected:** {bus_count}")

        # Full class breakdown
        class_counts = Counter([names[int(cls)] for cls in detections])
        st.markdown("### 🔍 Full Class Breakdown")
        for label, count in class_counts.items():
            st.write(f"- **{label}**: {count}")

        # Show annotated image
        st.markdown("### 🖼️ Detection Result")
        results[0].save(filename="result.jpg")
        st.image("result.jpg", caption="Annotated Image", use_column_width=True)

    except Exception as e:
        st.error(f"⚠️ Detection failed: {e}")
else:
    st.info("👈 Please upload an image to begin.")
