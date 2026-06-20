import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation="relu", input_shape=(4,)),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")
])

model.load_weights("iris_weights.weights.h5")

# Load Scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load Label Encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸"
)

st.title("🌸 Iris Flower Prediction")

st.write("Enter flower measurements below")

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

if st.button("Predict"):

    sample = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    # Apply scaler
    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    pred_class = np.argmax(prediction)

    flower = label_encoder.inverse_transform(
        [pred_class]
    )[0]

    confidence = np.max(prediction) * 100

    st.success(
        f"Predicted Flower: {flower}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )
