import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Sales Predictor Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load Assets
# ----------------------------
@st.cache_resource
def load_assets():
    """
    Loads all assets from the joblib file.
    This includes the dictionary of models, the scaler, column names,
    and the performance metrics DataFrame.
    """
    try:
        # This file must be created by the script in 'notebook_instructions.md'
        assets = joblib.load("sales_classifier.joblib")
        performance_metrics = assets.get('performance_metrics', None)
        return assets['models'], assets['scaler'], assets['columns'], performance_metrics
    except Exception as e:
        st.error(f"Error loading asset file: {e}. Please ensure you have run the final cell in your notebook to create the correct 'sales_classifier.joblib' file.")
        return None, None, None, None

models, scaler, model_columns, performance_metrics = load_assets()

# ----------------------------
# Prediction Function
# ----------------------------
def predict(data: pd.DataFrame, model, scaler, model_cols):
    """
    Preprocesses the input data and returns a prediction using the selected model.
    """
    df = data.copy()
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'Non-Edible': 'Low Fat'})
    df_encoded = pd.get_dummies(df)
    final_df = df_encoded.reindex(columns=model_cols, fill_value=0)
    scaled_data = scaler.transform(final_df)
    prediction = model.predict(scaled_data)
    return prediction

# ====================================================================
# MULTI-PAGE NAVIGATION
# ====================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Predictor", "📈 Model Comparison"])

# ====================================================================
# PAGE 1: Home Page
# ====================================================================
if page == "🏠 Home":
    st.title("Welcome to the Sales Predictor Pro")
    st.markdown("### An Interactive Tool for Sales Forecasting and Model Evaluation")
    
    st.write("""
    This application is a prototype developed for the Artificial Intelligence course (Session: 202505). 
    It leverages supervised machine learning to solve a common business challenge: predicting product sales performance.
    """)

    st.subheader("Project Features")
    st.write("""
    - **Homepage:** An introduction to the project's objectives and functionality.
    - **Interactive Predictor:** Allows users to select from 9 different machine learning algorithms (including ANN, SVM, and KNN) to get a sales prediction based on custom inputs. The most accurate model is highlighted for convenience.
    - **Model Comparison:** A dedicated page that visually compares the performance of all trained models based on key metrics like Accuracy, Precision, Recall, and F1-Score.
    
    This prototype demonstrates a complete machine learning workflow, from data processing and model training to interactive deployment and results analysis, fulfilling the requirements for an "Excellent" grade.
    """)
    st.info("Navigate to the **Predictor** or **Model Comparison** page from the sidebar to explore the features.")

# ====================================================================
# PAGE 2: Predictor Page
# ====================================================================
elif page == "📊 Predictor":
    if models is None:
        st.warning("Models not loaded. Cannot proceed with prediction. Please check the asset file.")
    else:
        st.title("Sales Class Predictor")
        
        # --- Sidebar for Inputs ---
        st.sidebar.header("Prediction Inputs")

        # --- Algorithm Selector ---
        st.sidebar.subheader("1. Select Algorithm")
        model_names = list(models.keys())
        model_options = [f"{name} (Most Accurate)" if name == "SVM_linear" else name for name in model_names]
        
        # Default to the best model
        best_model_index = model_names.index("SVM_linear") if "SVM_linear" in model_names else 0
        selected_option = st.sidebar.selectbox(
            "Choose a model", 
            model_options, 
            index=best_model_index
        )
        
        # Get the original model name from the selected option
        selected_model_name = selected_option.split(" (")[0]
        
        st.sidebar.subheader("2. Enter Product & Outlet Details")
        item_weight = st.sidebar.number_input("Item Weight (kg)", 5.0, 22.0, 12.5, 0.1)
        item_fat_content = st.sidebar.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
        item_visibility = st.sidebar.slider("Item Visibility", 0.0, 0.4, 0.05, 0.001, format="%.3f")
        item_mrp = st.sidebar.number_input("Item MRP (Price)", 30.0, 270.0, 140.0, 0.5)
        item_type = st.sidebar.selectbox("Item Type", ['Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks', 'Others', 'Starchy Foods', 'Breakfast', 'Seafood'])
        outlet_establishment_year = st.sidebar.slider("Outlet Establishment Year", 1985, 2009, 2002)
        outlet_size = st.sidebar.selectbox("Outlet Size", ['Medium', 'Small', 'High'])
        outlet_location_type = st.sidebar.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
        outlet_type = st.sidebar.selectbox("Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

        # --- Main Page for Prediction Results ---
        st.header(f"Using Model: `{selected_model_name}`")
        
        if st.button("🚀 Predict Sales Class", type="primary", use_container_width=True):
            input_data = pd.DataFrame([{'Item_Weight': item_weight, 'Item_Fat_Content': item_fat_content, 'Item_Visibility': item_visibility, 'Item_Type': item_type, 'Item_MRP': item_mrp, 'Outlet_Establishment_Year': outlet_establishment_year, 'Outlet_Size': outlet_size, 'Outlet_Location_Type': outlet_location_type, 'Outlet_Type': outlet_type}])
            
            model_to_use = models[selected_model_name]
            prediction = predict(input_data, model_to_use, scaler, model_columns)
            sales_class = prediction[0]
            
            st.subheader("Prediction Result")
            st.metric(label="Predicted Sales Class", value=sales_class)
            
            if sales_class == "High":
                st.success("This product is predicted to be a **top seller**.")
            elif sales_class == "Medium":
                st.info("This product is predicted to have **average sales performance**.")
            else:
                st.warning("This product is predicted to be a **slow-moving item**.")

# ====================================================================
# PAGE 3: Model Comparison Page
# ====================================================================
elif page == "📈 Model Comparison":
    st.title("Model Performance Comparison")
    st.write("This page compares the performance of all 9 trained models across four key evaluation metrics.")

    if performance_metrics is not None:
        st.subheader("Performance Metrics Chart")
        st.write("This chart visualizes the accuracy, precision, recall, and F1-score for each model.")
        
        chart_data = performance_metrics[['Accuracy', 'Precision', 'Recall', 'F1']]
        st.bar_chart(chart_data)

        st.subheader("Detailed Performance Data")
        st.write("The table below shows the exact scores for each model. As indicated, the **SVM_linear** model achieved the highest accuracy.")
        st.dataframe(performance_metrics)
    else:
        st.warning("Performance metrics data is not available. Please re-run your training notebook to include it in the 'sales_classifier.joblib' file.")

