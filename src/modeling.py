import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

def train_model(X_train, y_train, max_depth=5):
    """
    Train a Decision Tree Classifier.
    
    Parameters:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        max_depth (int): Max depth of tree

    Returns:
        DecisionTreeClassifier: Trained model
    """
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test, save_dir="../outputs/figures/"):
    """
    Evaluate the classifier and save visualizations.

    Parameters:
        clf (DecisionTreeClassifier): Trained classifier
        X_test (DataFrame): Test features
        y_test (Series): True test labels
        save_dir (str): Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Predict and print classification report
    y_pred = clf.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Plot and save confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.show()

    # Plot and save decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X_test.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
    plt.title("Decision Tree")
    plt.savefig(f"{save_dir}/decision_tree.png")
    plt.show()
