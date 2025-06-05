import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, models

def build_model(input_shape, hidden_layers=1, neurons=[128], activation='relu', learning_rate=0.001):
    """
    Build a neural network model for MNIST digit recognition.
    
    Parameters:
    -----------
    input_shape : int
        The number of input features (784 for flattened MNIST images)
    hidden_layers : int
        Number of hidden layers
    neurons : list
        List of neurons in each hidden layer
    activation : str
        Activation function to use ('relu', 'tanh', or 'sigmoid')
    learning_rate : float
        Learning rate for the optimizer
    
    Returns:
    --------
    model : keras.Model
        The compiled neural network model
    """
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(input_shape,)))
    
    # Hidden layers
    for i in range(hidden_layers):
        model.add(layers.Dense(neurons[i], activation=activation))
    
    # Output layer (10 classes for digits 0-9)
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, x_train, y_train, batch_size=64, epochs=5, validation_split=0.2):
    """
    Train the neural network model.
    
    Parameters:
    -----------
    model : keras.Model
        The compiled model to train
    x_train : numpy.ndarray
        Training data features
    y_train : numpy.ndarray
        Training data labels (one-hot encoded)
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs to train for
    validation_split : float
        Fraction of training data to use for validation
    
    Returns:
    --------
    history : History object
        Training history
    """
    history = model.fit(
        x_train, 
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1
    )
    
    return history
