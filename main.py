import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def main():
    # Task 1: Load data
    df = pd.read_csv(DATA_URL)

    print("First 5 rows:")
    print(df.head())

    # Task 2: Separate X and y
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Task 3: Split data into 75% training and 25% test data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Task 4: Create model
    model = LogisticRegression(max_iter=1000)

    # Task 5: Train model
    model.fit(X_train, y_train)

    # Task 6: Predict
    y_pred = model.predict(X_test)

    # Task 7: Accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    _, _, false_negatives, _ = cm.ravel()

    print("\nAccuracy Score:")
    print(accuracy)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nFalse Negatives:")
    print(false_negatives)


if __name__ == "__main__":
    main()
