import numpy as np

def convert_to_binary_labels(y_train, current_class):
    """
    Convert the multi-class labels to binary labels for the current class.
    
    Parameters:
        y_train: The original multi-class labels (numpy array).
        current_class: The class label for which binary labels are generated.
        
    Returns:
        binary_y_train: Binary labels for the current class (1 for current class, 0 otherwise).
    """
    return (y_train == current_class).astype(int)

def train_model_for_class(model_class, X_train, binary_y_train):
    """
    Train a binary classifier for the current class.
    
    Parameters:
        model_class: The classifier class to be used for training.
        X_train: The training features (numpy array).
        binary_y_train: Binary labels for the current class (numpy array).
        
    Returns:
        model: The trained model.
    """
    model = model_class()  # Initialize a new model instance
    model.fit(X_train, binary_y_train)  # Train the model
    return model

def get_class_probabilities(model, X_test):
    """
    Get predicted probabilities for the current class from the trained model.
    
    Parameters:
        model: The trained classifier for the current class.
        X_test: The test features for prediction (numpy array).
        
    Returns:
        probabilities: Predicted probabilities for the current class (numpy array).
    """
    return model.predict_proba(X_test)  # Get probabilities for the test data

def one_vs_all_custom(model_class, X_train, y_train, X_test):
    """
    Custom implementation of One-vs-All for multi-class classification.
    
    Parameters:
        model_class: A class that implements fit and predict_proba methods.
                     A new instance will be created for each class.
        X_train: Training features (numpy array or similar structure).
        y_train: Training labels (numpy array, one-dimensional).
        X_test: Test features for prediction (numpy array or similar structure).
    
    Returns:
        predictions: Array of predicted class labels for X_test.
        models: List of trained binary classifiers, one for each class.
    """
    classes = np.unique(y_train)  # Get all unique class labels
    models = []  # To store trained models
    scores = np.zeros((X_test.shape[0], len(classes)))  # To store scores for each class

    for i, c in enumerate(classes):
        binary_y_train = convert_to_binary_labels(y_train, c)  # Convert labels to binary for the current class
        
        model = train_model_for_class(model_class, X_train, binary_y_train)  # Train the model for the current class
        models.append(model)  # Append trained model to the list
        
        scores[:, i] = get_class_probabilities(model, X_test)  # Get probabilities for the current class
    
    # Final predictions are the classes with the highest score
    predictions = np.argmax(scores, axis=1)
    
    return predictions, models
