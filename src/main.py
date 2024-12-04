import argparse
import pandas as pd
from data_split import *
from one_vs_all_method import *
from svm_hard_margin import run_model_svm_hard_margin
from svm_smo import run_model_svm_smo
from logistic_regression import run_model_log_reg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main(args):
    data = pd.read_csv("data/preprocessed_data.csv")

    # Step 1: Split the data into features (X) and target (y)
    X = data.drop('readmitted', axis=1)  # Drop the 'readmitted' column for features
    y = data['readmitted']  # Target variable

    X_train, X_test, y_train, y_test = prepare_data_with_smote(X, y)

    # Step 2: Apply One-vs-All (OvA) strategy if needed (based on multi-class or binary classification)
    print("Applying One-vs-All strategy...")
    X_train, y_train, X_test, y_test = one_vs_all_custom(X_train, y_train, X_test, y_test)

    # Step 3: Train and evaluate models based on the input arguments
    if args.model == 'hard_margin':
        print("Training Hard Margin SVM...")
        # # Step 2: Apply One-vs-All (OvA) strategy if needed (based on multi-class or binary classification)
        # print("Applying One-vs-All strategy...")
        # X_train, y_train, X_test, y_test = one_vs_all_custom(HardMarginSVM, X_train, y_train, X_test, y_test)
        # svm_model = HardMarginSVM(learning_rate=args.learning_rate, n_iter=args.n_iter)
        accuracy, precision, recall, f1, conf_matrix = run_model_svm_hard_margin(X_train, X_test, y_train, y_test)

    elif args.model == 'svm_smo':
        # Step 2: Apply One-vs-All (OvA) strategy if needed (based on multi-class or binary classification)
        print("Applying One-vs-All strategy...")
        X_train, y_train, X_test, y_test = one_vs_all_custom(svm_smo, X_train, y_train, X_test, y_test)
        print("Training SVM with SMO...")
        accuracy, precision, recall, f1, conf_matrix = run_model_svm_smo(X_train, X_test, y_train, y_test)
    
    elif args.model == 'logistic_log_reg':
        # Step 2: Apply One-vs-All (OvA) strategy if needed (based on multi-class or binary classification)
        print("Applying One-vs-All strategy...")
        X_train, y_train, X_test, y_test = one_vs_all_custom(logistic_log_reg, X_train, y_train, X_test, y_test)
        print("Training Logistic Regression...")
        logistic_model = run_model_log_reg(X_train, X_test, y_train, y_test)
        logistic_model.fit(X_train, y_train)
        
        # y_pred = logistic_model.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred, average='binary' if len(set(y)) == 2 else 'micro')
        # recall = recall_score(y_test, y_pred, average='binary' if len(set(y)) == 2 else 'micro')
        # f1 = f1_score(y_test, y_pred, average='binary' if len(set(y)) == 2 else 'micro')
        # conf_matrix = confusion_matrix(y_test, y_pred)
    
    else:
        print(f"Model {args.model} is not recognized.")
        return

    # # Print final evaluation metrics
    # print("\nModel evaluation metrics:")
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")
    # print(f"Confusion Matrix:\n{conf_matrix}")

# if __name__ == '__main__':
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description='Model training and evaluation.')

#     # Required arguments
#     parser.add_argument('--input_file', type=str, required=True, help='Path to the input dataset file.')
#     parser.add_argument('--model', type=str, choices=['hard_margin', 'smo', 'logistic'], required=True, help='Model to train: "hard_margin", "smo", or "logistic"')

#     # Data split parameters
#     parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use as test set.')

#     # Parse arguments
#     args = parser.parse_args()

#     # Step 1: Preprocess and save data
#     print("Preprocessing data...")
#     preprocess_data(args.input_file)

#     # Call main function
#     main(args)
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Model training and evaluation.')

    # Required arguments
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input dataset file.')
    parser.add_argument('--model', type=str, choices=['hard_margin', 'smo', 'logistic'], required=True, help='Model to train: "hard_margin", "smo", or "logistic"')

    # # Data split parameters
    # parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use as test set.')

    # Parse arguments
    args = parser.parse_args()

    # Call main function from data_preprocess.py
    main(args)
