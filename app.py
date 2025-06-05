import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import time
import seaborn as sns

# Set page title and configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# Title and description
st.title("MNIST Digit Recognition with Machine Learning")
st.markdown("""
This application allows you to build, train, and evaluate a machine learning model for recognizing handwritten digits
using the MNIST dataset. You can customize various parameters, visualize the results, and see the model's
performance on test data.
""")

# Function to plot sample images
def plot_sample_images(x_data, y_data, n_samples=10):
    # Select random samples
    indices = np.random.choice(len(x_data), n_samples, replace=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # Display the image
        img = x_data[idx].reshape(8, 8)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {y_data[idx]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax
    )
    
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    
    return fig

# Sidebar for model selection and parameters
st.sidebar.header("Model Parameters")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Support Vector Machine (SVM)", "Random Forest", "Neural Network (MLP)"],
    index=0
)

# Parameters based on model type
if model_type == "Support Vector Machine (SVM)":
    kernel = st.sidebar.selectbox(
        "Kernel Type", 
        ["linear", "poly", "rbf", "sigmoid"],
        index=2
    )
    C = st.sidebar.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0, 0.1)
    gamma = st.sidebar.selectbox(
        "Gamma", 
        ["scale", "auto", 0.01, 0.1, 1.0],
        index=0
    )
    
elif model_type == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, 10)
    max_depth = st.sidebar.slider("Maximum Depth", 5, 30, 10, 1)
    min_samples_split = st.sidebar.slider("Minimum Samples to Split", 2, 10, 2, 1)
    
else:  # Neural Network (MLP)
    hidden_layer_sizes = st.sidebar.slider("Hidden Layer Size", 10, 200, 100, 10)
    activation = st.sidebar.selectbox(
        "Activation Function", 
        ["relu", "tanh", "logistic"],
        index=0
    )
    alpha = st.sidebar.select_slider(
        "Alpha (Regularization)", 
        options=[0.0001, 0.001, 0.01, 0.1, 1.0],
        value=0.0001,
        format_func=lambda x: f"{x:.4f}"
    )
    max_iter = st.sidebar.slider("Maximum Iterations", 100, 1000, 500, 100)

# Dataset sampling option
st.sidebar.header("Dataset Options")
sample_size = st.sidebar.slider("Sample Size (% of full dataset)", 10, 100, 20, 10)
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, 5)

# Function to load and preprocess data
@st.cache_data
def load_digits_data(sample_percentage=100, test_percentage=20):
    st.info("Loading digits dataset...")
    # Load the digits dataset
    digits = datasets.load_digits()
    
    # Calculate how many samples to use based on percentage
    n_samples = len(digits.images)
    sample_count = int(n_samples * sample_percentage / 100)
    
    # Flatten the images
    data = digits.images[:sample_count].reshape((sample_count, -1))
    
    # Create train/test split
    test_size = test_percentage / 100
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target[:sample_count], test_size=test_size, random_state=42, stratify=digits.target[:sample_count]
    )
    
    return X_train, X_test, y_train, y_test, digits.images[:sample_count], digits.target[:sample_count]

# Main content
st.header("Digits Dataset")

# Load data
X_train, X_test, y_train, y_test, images, target = load_digits_data(sample_size, test_size)

st.write(f"Training set size: {len(X_train)} samples")
st.write(f"Test set size: {len(X_test)} samples")

# Display sample images
st.subheader("Sample Images from Digits Dataset")
fig = plot_sample_images(images, target)
st.pyplot(fig)

# Build and train model section
st.header("Build and Train Model")

col1, col2 = st.columns(2)

with col1:
    # Build model button
    if st.button("Build Model"):
        # Clear previous model if exists
        if 'model' in st.session_state:
            del st.session_state['model']
            
        with st.spinner("Building the model..."):
            # Create model based on selection
            if model_type == "Support Vector Machine (SVM)":
                model = svm.SVC(kernel=kernel, C=C, gamma=gamma)
                st.session_state['model_info'] = f"SVM with {kernel} kernel, C={C}, gamma={gamma}"
            
            elif model_type == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    min_samples_split=min_samples_split, 
                    random_state=42
                )
                st.session_state['model_info'] = f"Random Forest with {n_estimators} trees, max_depth={max_depth}"
            
            else:  # Neural Network (MLP)
                model = MLPClassifier(
                    hidden_layer_sizes=(hidden_layer_sizes,), 
                    activation=activation, 
                    alpha=alpha, 
                    max_iter=max_iter,
                    random_state=42
                )
                st.session_state['model_info'] = f"Neural Network with {hidden_layer_sizes} hidden units, activation={activation}"
            
            st.session_state['model'] = model
            st.success(f"Model built: {st.session_state['model_info']}")

with col2:
    # Train model button
    if st.button("Train Model") and 'model' in st.session_state:
        with st.spinner("Training model... This may take a while."):
            start_time = time.time()
            
            model = st.session_state['model']
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            st.session_state['is_trained'] = True
            st.success(f"Training completed in {training_time:.2f} seconds!")

# Model evaluation if trained
if 'is_trained' in st.session_state and st.session_state['is_trained']:
    st.header("Model Evaluation")
    
    with st.spinner("Evaluating model on test data..."):
        model = st.session_state['model']
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = metrics.accuracy_score(y_test, y_pred)
        st.write(f"Test Accuracy: **{accuracy:.4f}**")
        
        # Classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Display metrics in a more structured way
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precision (avg)", f"{report['weighted avg']['precision']:.4f}")
        with col2:
            st.metric("Recall (avg)", f"{report['weighted avg']['recall']:.4f}")
        with col3:
            st.metric("F1-Score (avg)", f"{report['weighted avg']['f1-score']:.4f}")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = plot_confusion_matrix(cm)
        st.pyplot(fig)
        
        # Display some predictions
        st.subheader("Sample Predictions")
        
        # Select random samples from test set
        num_samples = 10
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            # Get the image and prediction
            img = X_test[idx].reshape(8, 8)
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            
            # Display the image
            axes[i].imshow(img, cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)

# About section
st.header("About")
st.markdown("""
This application demonstrates how to build and train machine learning models for digit recognition 
using the scikit-learn digits dataset, which is a simplified version of the MNIST dataset with 8x8 images.

**Technologies used:**
- Python with Streamlit
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn
""")
