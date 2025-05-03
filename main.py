from src.preprocessing import load_data, preprocess_data, split_data
from src.modeling import train_model, evaluate_model

def main():
    print("Loading data...")
    df = load_data("data/bank-additional-full.csv")

    print("Preprocessing data...")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training model...")
    clf = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(clf, X_test, y_test)

if __name__ == "__main__":
    main()
