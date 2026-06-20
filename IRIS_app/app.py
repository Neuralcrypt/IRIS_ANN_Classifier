import streamlit as st
import numpy as np
import pickle
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# File Paths
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "iris_weights.weights.h5"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

# -------------------------------
# Load Model
# -------------------------------

model = Sequential([
    Dense(16, activation="relu", input_shape=(4,)),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")
])

model.load_weights(WEIGHTS_PATH)

# -------------------------------
# Load Scaler
# -------------------------------

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# Load Label Encoder
# -------------------------------

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classification")
st.markdown(
    "Predict the species of an Iris flower using an Artificial Neural Network."
)

st.metric(
    "Model Accuracy",
    "96%"
)

st.divider()

# -------------------------------
# Inputs
# -------------------------------

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Species"):

    sample = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(
        sample_scaled,
        verbose=0
    )

    pred_class = np.argmax(prediction)

    flower_name = label_encoder.inverse_transform(
        [pred_class]
    )[0]

    confidence = np.max(prediction) * 100

    st.success(
        f"🌼 Predicted Species: {flower_name}"
    )

    st.info(
        f"🎯 Confidence: {confidence:.2f}%"
    )

    st.subheader("Prediction Probabilities")

    st.write(
        {
            label_encoder.inverse_transform([0])[0]:
                round(float(prediction[0][0]) * 100, 2),

            label_encoder.inverse_transform([1])[0]:
                round(float(prediction[0][1]) * 100, 2),

            label_encoder.inverse_transform([2])[0]:
                round(float(prediction[0][2]) * 100, 2),
        }
    )

st.divider()

st.caption(
    "Built using TensorFlow, Scikit-Learn and Streamlit"
)
