import pandas as pd
from sklearn.impute import SimpleImputer

# Load raw dataset
data = pd.read_csv("../Data/Autism-Child-Data.csv")

print("Original shape:", data.shape)

# Target column
target_column = "Class/ASD"

# Separate features and target
X = data.drop(columns=[target_column])
y = data[target_column].map({"YES": 1, "NO": 0})

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Handle missing values (categorical → most frequent)
imputer = SimpleImputer(strategy="most_frequent")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_encoded),
    columns=X_encoded.columns
)

# Combine features and target back
processed_data = X_imputed.copy()
processed_data["ASD_Risk"] = y

print("Processed shape:", processed_data.shape)
print(processed_data.head())

# Save preprocessed dataset
processed_data.to_csv("../Data/Autism_Questionnaire_Preprocessed.csv", index=False)

print("Preprocessing completed and file saved!")
