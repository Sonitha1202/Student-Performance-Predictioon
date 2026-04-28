import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("=== Student Performance Prediction System ===\n")

# Sample dataset
# study_hours, attendance, previous_marks, result
# result: 1 = Pass, 0 = Fail

data = {
    "study_hours": [2, 3, 5, 6, 1, 4, 7, 8, 2, 5, 6, 3, 7, 1, 4],
    "attendance": [60, 65, 75, 80, 50, 70, 90, 95, 55, 78, 85, 68, 88, 45, 72],
    "previous_marks": [40, 45, 60, 70, 35, 55, 80, 85, 38, 62, 75, 50, 82, 30, 58],
    "result": [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1]
}

# Create DataFrame
_df = pd.DataFrame(data)

# Features and target
X = _df[["study_hours", "attendance", "previous_marks"]]
y = _df["result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Enter student details for prediction:\n")

study_hours = float(input("Study Hours per Day: "))
attendance = float(input("Attendance Percentage: "))
previous_marks = float(input("Previous Marks: "))

new_student = [[study_hours, attendance, previous_marks]]
result = model.predict(new_student)

print("\nPrediction Result:")

if result[0] == 1:
    print("The student is likely to PASS")
else:
    print("The student is likely to FAIL")
