import numpy as np
from PIL import Image
import streamlit as st
import io

def preprocess_image(image, model_name="MobileNetV2"):
    """
    Preprocess an image for the specified model.
    
    Args:
        image (PIL.Image): The input image.
        model_name (str): Name of the model for which to preprocess the image.
        
    Returns:
        np.array: Preprocessed image array ready for the model.
    """
    # Use a standard size for all models
    target_size = (224, 224)
    
    # Resize the image
    image = image.resize(target_size)
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    
    # Simple normalization to [0,1]
    img_array = img_array / 255.0
    
    return img_array

def get_image_display_size(image, max_width=800, max_height=600):
    """
    Calculate an appropriate display size for an image while maintaining aspect ratio.
    
    Args:
        image (PIL.Image): The image to resize.
        max_width (int): Maximum display width.
        max_height (int): Maximum display height.
        
    Returns:
        tuple: (width, height) dimensions for display.
    """
    # Get original dimensions
    orig_width, orig_height = image.size
    
    # If image is already smaller than max dimensions, return original size
    if orig_width <= max_width and orig_height <= max_height:
        return orig_width, orig_height
    
    # Calculate aspect ratio
    aspect_ratio = orig_width / orig_height
    
    # Scale down to fit within max dimensions
    if aspect_ratio > 1:  # Width > Height
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
    else:  # Height >= Width
        new_height = max_height
        new_width = int(new_height * aspect_ratio)
    
    # Check if height still exceeds max_height
    if new_height > max_height:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)
    
    return new_width, new_height

def display_educational_content():
    """
    Display educational content about image classification in the sidebar.
    """
    st.header("How Image Classification Works")
    
    st.write("""
    ### What is Image Classification?
    
    Image classification is a computer vision task where the algorithm identifies 
    what's depicted in an image. The models we're using have been trained on 
    millions of images from the ImageNet dataset.
    
    ### How It Works:
    
    1. **Preprocessing**: Images are resized and normalized to match the model's requirements.
    
    2. **Feature Extraction**: The model extracts important features from the image using 
    convolutional neural networks (CNNs).
    
    3. **Classification**: The model analyzes the features and predicts the probability 
    of the image belonging to different classes.
    
    ### About the Models:
    
    **MobileNetV2** is a lightweight model designed for mobile devices.
    
    **EfficientNetB0** is optimized for accuracy and efficiency.
    
    Both models have been pre-trained on the ImageNet dataset, which contains 
    over 1 million images across 1,000 categories.
    """)
