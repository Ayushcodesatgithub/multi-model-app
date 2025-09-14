import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort

# --- MODEL LOADING ---

# Use caching to load models only once and improve performance
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
    st.stop() # Stop the app if models can't be loaded

# --- STREAMLIT APP INTERFACE ---

st.title('🌾 Crop Health and Yield Prediction')

st.write("""
Upload a CSV file with crop data to get a combined prediction for crop health and potential yield.
""")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the uploaded data
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(df.head())

        # Make a copy for processing to keep the original data unchanged
        df_processed = df.copy()
        
        # --- PREDICTION LOGIC ---
        
        # IMPORTANT: Replace these with the actual column names your models expect.
        # The order of columns must match the order the models were trained on.
        health_feature_columns = ['Temperature', 'Humidity', 'Rainfall'] # Example columns
        yield_feature_columns = ['N', 'P', 'K', 'Ph'] # Example columns

        # Ensure all required columns are in the uploaded file
        required_cols = set(health_feature_columns + yield_feature_columns)
        if not required_cols.issubset(df_processed.columns):
            st.error(f"CSV is missing required columns. Please ensure it contains: {list(required_cols)}")
        else:
            # 2. Prepare data for each model and run inference
            # Convert selected features to a NumPy array of type float32, which is standard for ONNX models.
            
            # Crop Health Prediction
            health_features = df_processed[health_feature_columns].to_numpy(dtype=np.float32)
            health_input_name = health_model.get_inputs()[0].name
            health_output_name = health_model.get_outputs()[0].name
            health_predictions = health_model.run([health_output_name], {health_input_name: health_features})[0]
            
            # Yield Prediction
            yield_features = df_processed[yield_feature_columns].to_numpy(dtype=np.float32)
            yield_input_name = yield_model.get_inputs()[0].name
            yield_output_name = yield_model.get_outputs()[0].name
            yield_predictions = yield_model.run([yield_output_name], {yield_input_name: yield_features})[0]

            # Add predictions as new columns to the DataFrame
            # The .flatten() is used to convert the output array into a 1D format suitable for a DataFrame column.
            df['Health Prediction'] = health_predictions.flatten()
            df['Yield Prediction (tons/hectare)'] = yield_predictions.flatten()

            # --- COMBINED PREDICTION OUTPUT ---
            
            st.write("### Combined Prediction Results")

            # This is a simple example of a "combined output". 
            # You can create more complex rules based on your specific needs.
            def get_recommendation(row):
                # Example: Let's assume health prediction is categorical (0=Unhealthy, 1=Healthy)
                # and yield is numerical.
                is_healthy = row['Health Prediction'] == 1 # Change this condition based on your model's output
                good_yield = row['Yield Prediction (tons/hectare)'] > 4 # Example threshold
                
                if is_healthy and good_yield:
                    return "✅ Excellent Prospect"
                elif is_healthy and not good_yield:
                    return "⚠️ Healthy, but Low Yield Expected"
                else:
                    return "❌ Poor Prospect (Unhealthy)"

            df['Recommendation'] = df.apply(get_recommendation, axis=1)
            
            # Display the final DataFrame with all predictions and recommendations
            st.dataframe(df)

            # Optionally, add a download button for the results
            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8')

            csv_output = convert_df_to_csv(df)
            st.download_button(
                label="Download results as CSV",
                data=csv_output,
                file_name='predictions.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")