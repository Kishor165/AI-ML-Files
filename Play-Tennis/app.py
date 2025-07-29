import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier ,plot_tree

data = {
 'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain',
'Overcast', 'Sunny',
 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast',
'Rain'],
 'Temperature':['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
'Mild',
 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
 'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal',
'Normal', 'High',
 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
 'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong',
'Strong', 'Weak',
 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
 'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}


df = pd.DataFrame(data)

df_encoded = df.copy()
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df[column])
        label_encoders[column] = le

x = df_encoded.drop('PlayTennis', axis=1)
y = df_encoded['PlayTennis']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x, y)

st.title("Play Tennis Prediction with 103 Decision Tree")

# User input
st.sidebar.header("Input Weather Conditions")
def user_input():
 outlook = st.sidebar.selectbox("Outlook", df['Outlook'].unique())
 temp = st.sidebar.selectbox("Temperature", df['Temperature'].unique())
 humidity = st.sidebar.selectbox("Humidity", df['Humidity'].unique())
 wind = st.sidebar.selectbox("Wind", df['Wind'].unique())
 return pd.DataFrame([[outlook, temp, humidity, wind]], columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
input_df = user_input()

# Encode input
input_encoded = input_df.copy()
for col in input_encoded.columns:
 input_encoded[col] = label_encoders[col].transform(input_encoded[col])

 # Prediction
prediction = model.predict(input_encoded)[0]
prediction_label = label_encoders['PlayTennis'].inverse_transform([prediction])[0]
st.subheader("Prediction:")
st.success(f"The model predicts: {prediction_label}")
st.subheader("Input Values:")
st.write(input_df)
st.subheader("Training Data:")
st.dataframe(df)