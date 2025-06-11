import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import base64

# Set page configuration
st.set_page_config(
    page_title="Desktop Computer Price Prediction",
    page_icon="💻",
    layout="wide"
)

# Title and description
st.title("Desktop Computer Price Prediction Tool")
st.markdown("""
This app predicts the price of desktop computers based on hardware specifications.
Upload your dataset first, then either train a new model or use the trained model to predict prices.
""")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page:", ["Upload Dataset", "Price Prediction"])

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'cat_features' not in st.session_state:
    st.session_state.cat_features = None
if 'num_features' not in st.session_state:
    st.session_state.num_features = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'numeric_scalers' not in st.session_state:
    st.session_state.numeric_scalers = {}

# Function to preprocess data and train model
def preprocess_and_train(data):
    # Store raw data for later use in prediction
    st.session_state.raw_data = data.copy()
    
    # Identify numeric and categorical features
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    categorical_features = [col for col in categorical_features if col != 'Price' and col in data.columns]
    
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features = [col for col in numeric_features if col != 'Price' and col in data.columns]
    
    # Store feature types for later use
    st.session_state.cat_features = categorical_features
    st.session_state.num_features = numeric_features
    
    # Create a copy of data for processing
    processed_data = data.copy()
    
    # Label encode categorical features
    for cat_feature in categorical_features:
        le = LabelEncoder()
        processed_data[cat_feature] = le.fit_transform(processed_data[cat_feature].astype(str))
        # Store the encoder for later use
        st.session_state.label_encoders[cat_feature] = le
    
    # Standardize numeric features - optional but can improve model performance
    for num_feature in numeric_features:
        scaler = StandardScaler()
        processed_data[num_feature] = scaler.fit_transform(processed_data[[num_feature]])
        # Store the scaler for later use
        st.session_state.numeric_scalers[num_feature] = scaler
    
    # Separate features and target
    X = processed_data.drop('Price', axis=1)
    y = processed_data['Price']
    
    # Store feature names for later use
    st.session_state.feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForestRegressor with the best parameters from your analysis
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    st.session_state.model = model
    
    return model, X_test, y_test, processed_data

# Function to process user input before prediction
def process_user_input(user_input):
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_input])
    
    # Encode categorical features using the saved encoders
    for feature in st.session_state.cat_features:
        if feature in input_df.columns and feature in st.session_state.label_encoders:
            encoder = st.session_state.label_encoders[feature]
            input_df[feature] = encoder.transform(input_df[feature].astype(str))
    
    # Scale numeric features using the saved scalers
    for feature in st.session_state.num_features:
        if feature in input_df.columns and feature in st.session_state.numeric_scalers:
            scaler = st.session_state.numeric_scalers[feature]
            input_df[feature] = scaler.transform(input_df[[feature]])
    
    # Ensure all features used during training are present
    for feature in st.session_state.feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Default value for missing features
    
    # Keep only the features used during training
    input_df = input_df[st.session_state.feature_names]
    
    return input_df

# Function to download trained model
def get_download_link(model):
    """Generate a link to download the model"""
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/pkl;base64,{b64}" download="trained_model.pkl">Download Trained Model</a>'
    return href

# Upload Dataset Page
if page == "Upload Dataset":
    st.header("Upload Your Dataset")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load the data
        try:
            data = pd.read_csv(uploaded_file)
            
            st.session_state.data = data
            
            # Display dataset preview
            st.subheader("Dataset Preview")
            st.write(data.head())
            
            # Display basic statistics
            st.subheader("Dataset Statistics")
            st.write(data.describe())
            
            # Check if the dataset has the required 'Price' column
            if 'Price' not in data.columns:
                st.error("The dataset must contain a 'Price' column!")
            else:
                # Data visualization
                st.subheader("Data Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Price Distribution")
                    fig, ax = plt.subplots()
                    sns.histplot(data['Price'], kde=True, ax=ax)
                    st.pyplot(fig)
                
                with col2:
                    # Top correlations with Price
                    numeric_data = data.select_dtypes(include=['number'])
                    if 'Price' in numeric_data.columns and len(numeric_data.columns) > 1:
                        st.write("Top Feature Correlations with Price")
                        correlations = numeric_data.corr()['Price'].sort_values(ascending=False)
                        fig, ax = plt.subplots()
                        top_corr = correlations.drop('Price').head(10)
                        sns.barplot(x=top_corr.values, y=top_corr.index, ax=ax)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("Not enough numeric features to calculate correlations.")
                
                # Train model button
                if st.button("Train Model"):
                    # Create a progress indicator before starting the training
                    progress_placeholder = st.empty()
                    progress_bar = progress_placeholder.progress(0)
                    
                    # Update progress for different stages of training
                    progress_bar.progress(10)
                    progress_placeholder.text("Preprocessing data...")
                    
                    # Preprocess data
                    model, X_test, y_test, processed_data = preprocess_and_train(data)
                    
                    progress_bar.progress(50)
                    progress_placeholder.text("Training model...")
                    
                    # Model evaluation
                    progress_bar.progress(80)
                    progress_placeholder.text("Evaluating model performance...")
                    
                    y_pred = model.predict(X_test)
                    mse = np.mean((y_test - y_pred) ** 2)
                    rmse = np.sqrt(mse)
                    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
                    
                    # Complete the progress bar
                    progress_bar.progress(100)
                    progress_placeholder.text("Model training complete!")
                    
                    # After a short delay, replace with success message
                    import time
                    time.sleep(0.5)
                    progress_placeholder.success("Model trained successfully!")
                    
                    st.subheader("Model Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Squared Error", f"{mse:.2f}")
                    col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
                    col3.metric("R² Score", f"{r2:.4f}")
                    
                    # Feature importance plot
                    st.subheader("Feature Importance")
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_features = 10 if len(indices) > 10 else len(indices)
                    sns.barplot(x=importances[indices][:top_features], 
                              y=[st.session_state.feature_names[i] for i in indices][:top_features], 
                              ax=ax)
                    plt.title('Top Feature Importances')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Actual vs Predicted plot
                    st.subheader("Actual vs Predicted Prices")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_test, y_pred, alpha=0.7)
                    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
                    plt.title('Actual vs Predicted Prices')
                    plt.xlabel('Actual Price (TL)')
                    plt.ylabel('Predicted Price (TL)')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download model link
                    st.markdown(get_download_link(model), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error: {e}")

# Price Prediction Page
elif page == "Price Prediction":
    st.header("Predict Desktop Computer Price")
    
    # Check if model and raw data are available
    if st.session_state.model is None or st.session_state.raw_data is None:
        st.warning("Please upload a dataset and train the model first!")
    else:
        st.success("Model is ready for predictions")
        
        # Get the original column names from the raw dataset
        data = st.session_state.raw_data
        cat_features = st.session_state.cat_features
        num_features = st.session_state.num_features
        
        st.subheader("Enter Computer Specifications")
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        # Dictionary to store user inputs
        user_input = {}
        
        # Add input fields for categorical features
        with col1:
            for i, feature in enumerate(cat_features):
                if i % 2 == 0:  # Even index -> column 1
                    if feature in data.columns:
                        # Use original string values for dropdown
                        unique_values = data[feature].unique().tolist()
                        user_input[feature] = st.selectbox(f"{feature}", unique_values)
        
            # Add input fields for numeric features
            for i, feature in enumerate(num_features):
                if i % 2 == 0:  # Even index -> column 1
                    if feature in data.columns:
                        min_val = float(data[feature].min())
                        max_val = float(data[feature].max())
                        default_val = float(data[feature].mean())
                        
                        # Determine if it should be an integer or float
                        if data[feature].dtype == 'int64':
                            user_input[feature] = st.number_input(
                                f"{feature}",
                                min_value=int(min_val),
                                max_value=int(max_val),
                                value=int(default_val),
                                step=1
                            )
                        else:
                            user_input[feature] = st.number_input(
                                f"{feature}",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=float(default_val),
                                step=float((max_val - min_val) / 100)
                            )
        
        with col2:
            for i, feature in enumerate(cat_features):
                if i % 2 == 1:  # Odd index -> column 2
                    if feature in data.columns:
                        # Use original string values for dropdown
                        unique_values = data[feature].unique().tolist()
                        user_input[feature] = st.selectbox(f"{feature}", unique_values)
            
            # Add input fields for numeric features
            for i, feature in enumerate(num_features):
                if i % 2 == 1:  # Odd index -> column 2
                    if feature in data.columns:
                        min_val = float(data[feature].min())
                        max_val = float(data[feature].max())
                        default_val = float(data[feature].mean())
                        
                        # Determine if it should be an integer or float
                        if data[feature].dtype == 'int64':
                            user_input[feature] = st.number_input(
                                f"{feature}",
                                min_value=int(min_val),
                                max_value=int(max_val),
                                value=int(default_val),
                                step=1
                            )
                        else:
                            user_input[feature] = st.number_input(
                                f"{feature}",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=float(default_val),
                                step=float((max_val - min_val) / 100)
                            )
        
        # Create a button to make prediction
        if st.button("Predict Price"):
            try:
                # Process user input
                processed_input = process_user_input(user_input)
                
                # Make prediction
                predicted_price_tl = st.session_state.model.predict(processed_input)[0]
                
                # Convert TL to INR (1 TL = 2.27 INR)
                conversion_rate = 2.27
                predicted_price_inr = predicted_price_tl * conversion_rate
                
                # Display prediction
                st.subheader("Price Prediction")
                col1, col2 = st.columns(2)
                col1.success(f"Estimated Price: {predicted_price_tl:.2f} TL")
                col2.success(f"Estimated Price: {predicted_price_inr:.2f} INR")
                
                # Add explanation of currency conversion
                st.info(f"Conversion rate: 1 TL = {conversion_rate} INR")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.error(f"Details: {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2025 Desktop Computer Price Prediction Tool")
