from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def prepare_data_with_smote(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets and applies SMOTE to balance the training set.

    Parameters:
        X (numpy.ndarray or pandas.DataFrame): Features.
        y (numpy.ndarray or pandas.Series): Target labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generator to ensure reproducibility.

    Returns:
        X_train_resampled (numpy.ndarray or pandas.DataFrame): Resampled training features.
        X_test (numpy.ndarray or pandas.DataFrame): Testing features.
        y_train_resampled (numpy.ndarray or pandas.Series): Resampled training target.
        y_test (numpy.ndarray or pandas.Series): Testing target.
    """
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Apply SMOTE to the training set
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test
