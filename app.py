import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort

# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    """Loads a pre-trained ONNX model and returns the inference session."""
    return ort.InferenceSession(model_path)

# Load your two models
try:
    health_model = load_model('crop_health_model.onnx')
    yield_model = load_model('yield_model.onnx')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- STREAMLIT APP INTERFACE ---
st.title('🌾 Crop Health and Yield Prediction')

st.write("""
Upload your agriculture dataset CSV file to get a combined prediction for crop health and potential yield.
""")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the uploaded data
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data (First 5 Rows)")
        st.dataframe(df.head())

        # Make a copy for processing
        df_processed = df.copy()

        # --- FEATURE SELECTION ---
        # Updated according to your dataset column names
        health_feature_columns = ['Temperature', 'Humidity', 'Rainfall']
        yield_feature_columns = ['Soil_pH', 'Soil_Moisture']  # Adjusted for your dataset

        # Ensure all required columns exist
        required_cols = set(health_feature_columns + yield_feature_columns)
        if not required_cols.issubset(df_processed.columns):
            st.error(f"CSV is missing required columns. Please ensure it contains: {list(required_cols)}")
        else:
            # Convert to NumPy float32 arrays
            health_features = df_processed[health_feature_columns].to_numpy(dtype=np.float32)
            yield_features = df_processed[yield_feature_columns].to_numpy(dtype=np.float32)

            # Run inference on Health Model
            health_input_name = health_model.get_inputs()[0].name
            health_output_name = health_model.get_outputs()[0].name
            health_predictions = health_model.run([health_output_name], {health_input_name: health_features})[0]

            # Run inference on Yield Model
            yield_input_name = yield_model.get_inputs()[0].name
            yield_output_name = yield_model.get_outputs()[0].name
            yield_predictions = yield_model.run([yield_output_name], {yield_input_name: yield_features})[0]

            # Add predictions to dataframe
            df['Health_Prediction_Label'] = health_predictions.flatten()
            df['Yield_Prediction_Label'] = yield_predictions.flatten()

            # Display results
            st.write("### Combined Prediction Results")
            st.dataframe(df)

            # Download button
            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8')

            csv_output = convert_df_to_csv(df)
            st.download_button(
                label="Download results as CSV",
                data=csv_output,
                file_name='agriculture_predictions.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
