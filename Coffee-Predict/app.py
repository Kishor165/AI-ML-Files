import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Coffee Prediction App", layout="centered")

# -------- Data Preparation --------
data = {
    'Weather': ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Rainy'],
    'TimeOfDay': ['Morning', 'Morning', 'Afternoon', 'Afternoon', 'Evening', 'Morning', 'Morning', 'Afternoon', 'Evening', 'Morning'],
    'SleepQuality': ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor'],
    'Mood': ['Tired', 'Fresh', 'Tired', 'Energetic', 'Tired', 'Fresh', 'Tired', 'Tired', 'Energetic', 'Tired'],
    'BuyCoffee': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Label Encoding
df_encoded = df.copy()
label_encoders = {}

for col in df.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df_encoded.drop('BuyCoffee', axis=1)
y = df_encoded['BuyCoffee']

# Train model
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# -------- Streamlit UI --------
st.title("☕ Coffee Purchase Predictor")
st.write("Predict whether a customer will buy coffee based on their mood, sleep, weather, and time of day.")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        weather = st.selectbox("Weather", df['Weather'].unique())
        sleep_quality = st.selectbox("Sleep Quality", df['SleepQuality'].unique())
    with col2:
        time_of_day = st.selectbox("Time of Day", df['TimeOfDay'].unique())
        mood = st.selectbox("Mood", df['Mood'].unique())

    submitted = st.form_submit_button("Predict")

# -------- Prediction --------
if submitted:
    input_data = pd.DataFrame([{
        'Weather': label_encoders['Weather'].transform([weather])[0],
        'TimeOfDay': label_encoders['TimeOfDay'].transform([time_of_day])[0],
        'SleepQuality': label_encoders['SleepQuality'].transform([sleep_quality])[0],
        'Mood': label_encoders['Mood'].transform([mood])[0],
    }])

    prediction = model.predict(input_data)
    predicted_label = label_encoders['BuyCoffee'].inverse_transform(prediction)[0]

    st.subheader("🎯 Prediction Result:")
    st.success(f"Customer will **{predicted_label}** buy coffee.")

# -------- Tree Visualization --------
st.subheader("🌳 Decision Tree (ID3 - Entropy)")
fig = plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=label_encoders['BuyCoffee'].classes_, filled=True)
st.pyplot(fig)
