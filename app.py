import streamlit as st
import pandas as pd
import os
from PIL import Image
import time
import io
from image_classifier import ImageClassifier
from utils import preprocess_image, get_image_display_size, display_educational_content

# Set page configuration
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state for classification history
if 'history' not in st.session_state:
    st.session_state.history = []

def main():
    # App title and description
    st.title("📷 Image Classification App")
    st.write("Upload an image and get predictions using a pre-trained deep learning model")
    
    # Sidebar for options and educational content
    with st.sidebar:
        st.header("Options")
        model_option = st.selectbox(
            "Select a pre-trained model",
            ["MobileNetV2", "EfficientNetB0"],
            index=0
        )
        
        top_k = st.slider(
            "Number of predictions to show",
            min_value=1,
            max_value=10,
            value=5
        )
        
        st.divider()
        display_educational_content()
        
    # Initialize the classifier
    classifier = ImageClassifier(model_name=model_option)
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    # Display the main content
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                width, height = get_image_display_size(image, max_width=400)
                st.image(image, width=width)
                
                # Display image info
                file_details = {
                    "Filename": uploaded_file.name,
                    "Format": image.format,
                    "Size": f"{uploaded_file.size / 1024:.1f} KB",
                    "Dimensions": f"{image.width} x {image.height} px"
                }
                st.write("Image details:")
                for key, value in file_details.items():
                    st.write(f"- **{key}:** {value}")
            
            # Process and classify the image
            with col2:
                st.subheader("Classification Results")
                
                with st.spinner("Classifying image..."):
                    # Preprocess image for the model
                    processed_img = preprocess_image(image, model_name=model_option)
                    
                    # Get predictions
                    start_time = time.time()
                    predictions = classifier.predict(processed_img, top_k=top_k)
                    elapsed_time = time.time() - start_time
                    
                    # Display results
                    st.write(f"Classified in {elapsed_time:.2f} seconds")
                    
                    # Create a DataFrame for cleaner display
                    results_df = pd.DataFrame(
                        predictions,
                        columns=["Class", "Confidence (%)"]
                    )
                    
                    # Display as a table with progress bars
                    for i, row in results_df.iterrows():
                        class_name = row["Class"]
                        confidence = row["Confidence (%)"]
                        
                        # Use columns to display class name and confidence bar side by side
                        cols = st.columns([3, 7])
                        with cols[0]:
                            st.write(f"**{class_name}**")
                        with cols[1]:
                            st.progress(confidence / 100)
                            st.write(f"{confidence:.2f}%")
                    
                    # Add to history
                    history_entry = {
                        "filename": uploaded_file.name,
                        "model": model_option,
                        "top_prediction": predictions[0][0],
                        "confidence": predictions[0][1],
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.history.append(history_entry)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please try uploading a different image.")
    
    # Display classification history
    if st.session_state.history:
        st.divider()
        st.subheader("Classification History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

if __name__ == "__main__":
    main()
