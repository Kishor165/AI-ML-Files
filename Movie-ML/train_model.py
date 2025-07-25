import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load your CSV
df = pd.read_csv("movie_data.csv")

# Select features and target
X = df[["Age", "Gender", "WatchFrequency", "GenrePreference"]]
y = df["Interested"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save to movie_model.pkl
with open("movie_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as movie_model.pkl")
