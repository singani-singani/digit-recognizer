import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

def plot_training_history(history):
    """
    Plot the training and validation metrics from model training history.
    
    Parameters:
    -----------
    history : History object
        History object returned from model.fit()
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with training history plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm):
    """
    Plot a confusion matrix.
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix to plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with confusion matrix plot
    """
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

def plot_sample_images(x_train, y_train, n_samples=10):
    """
    Plot sample images from the MNIST dataset.
    
    Parameters:
    -----------
    x_train : numpy.ndarray
        Training images
    y_train : numpy.ndarray
        Training labels
    n_samples : int
        Number of samples to display
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with sample images
    """
    # Select random samples
    indices = np.random.choice(len(x_train), n_samples, replace=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # Display the image
        axes[i].imshow(x_train[idx], cmap='gray')
        axes[i].set_title(f"Label: {y_train[idx]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig
