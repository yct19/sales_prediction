import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Sales Predictor Pro",
    page_icon="💡",
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
    It expects the file to be a dictionary containing 'model', 'scaler', 'columns',
    and optionally 'feature_importances'.
    """
    try:
        assets = joblib.load("sales_classifier.joblib")
        model = assets['model']
        scaler = assets['scaler']
        model_columns = assets['columns']
        # --- NEW: Load feature importances if they exist ---
        feature_importances = assets.get('feature_importances', None)
        return model, scaler, model_columns, feature_importances
    except FileNotFoundError:
        st.error("Asset file 'sales_classifier.joblib' not found. Please ensure it's in the same directory as the app.")
        return None, None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the asset file: {e}")
        return None, None, None, None

model, scaler, model_columns, feature_importances = load_assets()

# ----------------------------
# Prediction Functions
# ----------------------------
def predict(data: pd.DataFrame, model, scaler, model_cols):
    """
    Preprocesses the input data and returns model predictions and probabilities.
    """
    df = data.copy()
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'Non-Edible': 'Low Fat'})
    df_encoded = pd.get_dummies(df)
    final_df = df_encoded.reindex(columns=model_cols, fill_value=0)
    scaled_data = scaler.transform(final_df)
    
    # --- NEW: Get both predictions and probabilities ---
    predictions = model.predict(scaled_data)
    try:
        probabilities = model.predict_proba(scaled_data)
    except AttributeError:
        # Some models like basic SVM don't have predict_proba without configuration
        probabilities = np.zeros((len(predictions), len(model.classes_)))

    return predictions, probabilities

# ----------------------------
# Main User Interface
# ----------------------------
st.title("💡 Sales Predictor Pro")
st.markdown("An advanced tool for predicting grocery sales performance with scenario analysis and model insights.")

if model is None:
    st.warning("Application cannot start because the model assets failed to load.")
else:
    # --- Sidebar for Inputs ---
    st.sidebar.header("📊 Prediction Inputs")
    st.sidebar.subheader("Product Details")
    item_weight = st.sidebar.number_input("Item Weight (kg)", 5.0, 22.0, 12.5, 0.1)
    item_fat_content = st.sidebar.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
    item_visibility = st.sidebar.slider("Item Visibility", 0.0, 0.4, 0.05, 0.001, format="%.3f")
    item_mrp = st.sidebar.number_input("Item MRP (Price)", 30.0, 270.0, 140.0, 0.5)
    item_type = st.sidebar.selectbox("Item Type", ['Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks', 'Others', 'Starchy Foods', 'Breakfast', 'Seafood'])

    st.sidebar.subheader("Outlet Details")
    outlet_establishment_year = st.sidebar.slider("Outlet Establishment Year", 1985, 2009, 2002)
    outlet_size = st.sidebar.selectbox("Outlet Size", ['Medium', 'Small', 'High'])
    outlet_location_type = st.sidebar.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
    outlet_type = st.sidebar.selectbox("Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

    # --- Main Page Tabs ---
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📈 Scenario Analysis", "🧠 Model Insights"])

    # --- TAB 1: Single Prediction ---
    with tab1:
        st.header("Single Item Sales Prediction")
        st.write("Use the inputs in the sidebar to get a prediction for a single product.")
        
        if st.button("🚀 Predict Sales Class", type="primary", use_container_width=True):
            input_data = pd.DataFrame([{'Item_Weight': item_weight, 'Item_Fat_Content': item_fat_content, 'Item_Visibility': item_visibility, 'Item_Type': item_type, 'Item_MRP': item_mrp, 'Outlet_Establishment_Year': outlet_establishment_year, 'Outlet_Size': outlet_size, 'Outlet_Location_Type': outlet_location_type, 'Outlet_Type': outlet_type}])
            prediction, probabilities = predict(input_data, model, scaler, model_columns)
            
            sales_class = prediction[0]
            confidence = probabilities.max() * 100
            
            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Sales Class", sales_class)
                st.metric("Model Confidence", f"{confidence:.2f}%")
            
            with col2:
                st.write("Prediction Probabilities:")
                prob_df = pd.DataFrame(probabilities, columns=model.classes_, index=["Probability"]).T
                st.bar_chart(prob_df)

    # --- TAB 2: Scenario Analysis ---
    with tab2:
        st.header("What-If Scenario Analysis")
        st.write("See how changing a key feature might impact the sales prediction.")
        
        scenario_feature = st.selectbox("Select Feature to Analyze", ["Item_MRP", "Item_Visibility"])
        
        if scenario_feature == "Item_MRP":
            min_val, max_val = 30.0, 270.0
            original_val = item_mrp
        else: # Item_Visibility
            min_val, max_val = 0.0, 0.4
            original_val = item_visibility
            
        st.write(f"Analyzing changes to **{scenario_feature}**. Original value is **{original_val}**.")
        
        # Create 20 data points for the analysis
        scenario_values = np.linspace(min_val, max_val, 20)
        scenario_results = []
        
        base_input = pd.DataFrame([{'Item_Weight': item_weight, 'Item_Fat_Content': item_fat_content, 'Item_Visibility': item_visibility, 'Item_Type': item_type, 'Item_MRP': item_mrp, 'Outlet_Establishment_Year': outlet_establishment_year, 'Outlet_Size': outlet_size, 'Outlet_Location_Type': outlet_location_type, 'Outlet_Type': outlet_type}])
        
        for val in scenario_values:
            temp_input = base_input.copy()
            temp_input[scenario_feature] = val
            pred, _ = predict(temp_input, model, scaler, model_columns)
            scenario_results.append(pred[0])
            
        results_df = pd.DataFrame({
            scenario_feature: scenario_values,
            'Predicted_Class': scenario_results
        })
        
        # Map classes to numbers for charting
        class_mapping = {"Low": 1, "Medium": 2, "High": 3}
        results_df['Predicted_Class_Num'] = results_df['Predicted_Class'].map(class_mapping)
        
        st.line_chart(results_df.rename(columns={scenario_feature:'x', 'Predicted_Class_Num':'y'}).set_index('x'))
        st.caption("Chart Key: 1=Low, 2=Medium, 3=High. The chart shows how the predicted class changes as the selected feature's value increases.")

    # --- TAB 3: Model Insights (XAI) ---
    with tab3:
        st.header("Model Insights - What Matters Most?")
        st.write("This chart shows the features that have the biggest impact on the model's predictions.")

        if feature_importances is not None:
            importance_df = pd.DataFrame(feature_importances.items(), columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values(by="Importance", ascending=False).head(15) # Show top 15
            
            st.bar_chart(importance_df.set_index('Feature'))
        else:
            st.info("Feature importance data is not available for this model type or was not included in the asset file.")
            st.write("To enable this chart, please use a model that supports feature importance (like SVM with a linear kernel, Logistic Regression, or XGBoost) and save the importances in the `sales_classifier.joblib` file.")

