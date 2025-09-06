import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Sales Predictor Pro",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load Assets
# ----------------------------
@st.cache_resource
def load_assets():
    """
    Loads all necessary assets from a single joblib file.
    This includes all trained models, the feature scaler, and the list of model column names.
    """
    try:
        assets = joblib.load("sales_classifier.joblib")
        return assets['models'], assets['scaler'], assets['columns']
    except FileNotFoundError:
        st.error("Asset file 'sales_classifier.joblib' not found. Please ensure it's in the same directory as the app.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading the asset file: {e}")
        return None, None, None

# Load the assets into global variables
models, scaler, model_columns = load_assets()

# ----------------------------
# Prediction Function
# ----------------------------
def predict(data: pd.DataFrame, model, scaler, model_cols):
    """
    Preprocesses the input data and returns model predictions.
    """
    df = data.copy()
    
    # Standardize 'Item_Fat_Content' values
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'Non-Edible': 'Low Fat'})
    
    # Perform one-hot encoding
    df_encoded = pd.get_dummies(df)
    
    # Align columns with the model's training columns
    final_df = df_encoded.reindex(columns=model_cols, fill_value=0)
    
    # Scale the features
    scaled_data = scaler.transform(final_df)
    
    # Make predictions
    predictions = model.predict(scaled_data)
    
    return predictions

# ----------------------------
# Home Page Content
# ----------------------------
def show_home_page():
    st.title("🛒 Welcome to Sales Predictor Pro")
    st.markdown("""
    ## About This Application
    
    Sales Predictor Pro is a machine learning-powered tool that predicts grocery item sales performance 
    into three categories: **Low**, **Medium**, and **High**. This helps retailers optimize their inventory 
    and marketing strategies.
    
    ### Features
    - **Multiple Algorithm Support**: Choose from various machine learning models
    - **Single Prediction**: Predict sales for individual products
    - **Batch Prediction**: Upload a CSV file to predict sales for multiple products at once
    - **Model Comparison**: Select different algorithms to see how they perform
    
    ### Available Algorithms
    - **Default Models**: SVM_rbf, KNN_k5, ANN_100
    - **Tuned Models**: SVM_linear, SVM_poly, KNN_k3, KNN_k7, ANN_2layers, ANN_3layers
    
    ### How to Use
    1. Navigate to the Single Prediction tab to test individual products
    2. Use the Batch Prediction tab to process multiple products from a CSV file
    3. Select your preferred algorithm from the sidebar
    4. View and download your predictions
    
    *Note: Based on our testing, SVM_linear generally provides the most accurate predictions.*
    """)
    
    # Add some visual elements
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Single Prediction**\n\nTest individual products with detailed inputs")
    with col2:
        st.success("**Batch Processing**\n\nUpload CSV files for bulk predictions")
    with col3:
        st.warning("**Multiple Models**\n\nCompare different ML algorithms")

# ----------------------------
# Main Application
# ----------------------------
def main():
    # Sidebar for model selection
    st.sidebar.header("⚙️ Model Selection")
    
    if models is not None:
        model_names = list(models.keys())
        # Sort models with SVM_linear first (most accurate)
        if "SVM_linear" in model_names:
            model_names.remove("SVM_linear")
            model_names = ["SVM_linear"] + sorted(model_names)
        
        selected_model_name = st.sidebar.selectbox(
            "Choose Prediction Algorithm",
            model_names,
            help="SVM_linear typically provides the most accurate predictions based on our testing"
        )
        selected_model = models[selected_model_name]
    else:
        selected_model = None
    
    # Navigation
    st.sidebar.header("📱 Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Single Prediction", "Batch Prediction"])
    
    # Display selected page
    if page == "Home":
        show_home_page()
    elif page == "Single Prediction":
        show_single_prediction(selected_model)
    elif page == "Batch Prediction":
        show_batch_prediction(selected_model)

# ----------------------------
# Single Prediction Page
# ----------------------------
def show_single_prediction(model):
    st.header("📊 Single Prediction")
    st.write("Use the input fields below to enter the details for a product.")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Product Details")
        item_weight = st.number_input("Item Weight (kg)", min_value=0.1, value=12.5, step=0.1)
        item_fat_content = st.selectbox("Item Fat Content", ['Low Fat', 'Regular', 'Non-Edible'])
        item_visibility = st.slider("Item Visibility", min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.3f")
        item_mrp = st.number_input("Item MRP (Price)", min_value=10.0, value=140.0, step=0.5)
        item_type = st.selectbox("Item Type", [
            'Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 
            'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 
            'Meat', 'Breads', 'Hard Drinks', 'Others', 'Starchy Foods', 'Breakfast', 'Seafood'
        ])
    
    with col2:
        st.subheader("Outlet Details")
        outlet_establishment_year = st.slider("Outlet Establishment Year", 1985, 2025, 2002)
        outlet_size = st.selectbox("Outlet Size", ['Medium', 'Small', 'High'])
        outlet_location_type = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
        outlet_type = st.selectbox("Outlet Type", [
            'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'
        ])
    
    # Prediction button
    if st.button("🚀 Predict Sales Class", type="primary", use_container_width=True):
        if model is None:
            st.error("No model selected. Please choose a model from the sidebar.")
            return
            
        input_data = {
            'Item_Weight': item_weight, 
            'Item_Fat_Content': item_fat_content, 
            'Item_Visibility': item_visibility,
            'Item_Type': item_type, 
            'Item_MRP': item_mrp, 
            'Outlet_Establishment_Year': outlet_establishment_year,
            'Outlet_Size': outlet_size, 
            'Outlet_Location_Type': outlet_location_type, 
            'Outlet_Type': outlet_type
        }
        
        input_df = pd.DataFrame([input_data])
        prediction = predict(input_df, model, scaler, model_columns)
        
        st.subheader("Prediction Result")
        sales_class = prediction[0]
        
        # Display the result
        st.metric(label="Predicted Sales Class", value=sales_class)
        
        if sales_class == "High":
            st.success("This product is predicted to be a **top seller**.")
        elif sales_class == "Medium":
            st.info("This product is predicted to have **average sales performance**.")
        else:
            st.warning("This product is predicted to be a **slow-moving item**.")

# ----------------------------
# Batch Prediction Page
# ----------------------------
def show_batch_prediction(model):
    st.header("📂 Batch Prediction")
    st.write("Upload a CSV file containing product data to generate predictions for multiple items.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head())
            
            if st.button("🔮 Run Batch Prediction", type="primary", use_container_width=True):
                if model is None:
                    st.error("No model selected. Please choose a model from the sidebar.")
                    return
                    
                # Check required columns
                required_cols = [
                    'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 
                    'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 
                    'Outlet_Location_Type', 'Outlet_Type'
                ]
                
                if not all(col in batch_df.columns for col in required_cols):
                    st.error(f"Error: The uploaded CSV must contain these columns: {', '.join(required_cols)}")
                else:
                    with st.spinner('Processing your file... This may take a moment.'):
                        predictions = predict(batch_df[required_cols], model, scaler, model_columns)
                        result_df = batch_df.copy()
                        result_df['Predicted_Sales_Class'] = predictions
                        
                        st.subheader("Batch Prediction Results")
                        st.dataframe(result_df)
                        
                        # Download button
                        csv_results = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_results,
                            file_name='prediction_results.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# Run the app
if __name__ == "__main__":
    main()
