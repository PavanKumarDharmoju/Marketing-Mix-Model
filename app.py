import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Function to remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Function to calculate ROI based on total marketing spend and Z_Revenue
def calculate_roi(df, spend_columns, revenue_column='Z_Revenue'):
    df['Total_Marketing_Spend'] = df[spend_columns].sum(axis=1)
    df['ROI_Total_Marketing'] = ((df[revenue_column] - df['Total_Marketing_Spend']) / df['Total_Marketing_Spend']) * 100
    return df

# Function to apply pre-trained models and get feature importance
def get_feature_importance(model_paths, feature_names):
    feature_importance = {}
    for model_name, model_path in model_paths.items():
        model = joblib.load(model_path)
        if hasattr(model, 'coef_') and len(model.coef_) == len(feature_names):
            feature_importance[model_name] = pd.Series(model.coef_, index=feature_names).sort_values(ascending=False)
        else:
            feature_importance[model_name] = "Feature importance length mismatch or not available."
    return feature_importance

# Title of the dashboard
st.title('Marketing Campaign Dashboard with Feature Importance, ROI Forecasting, and Insights')

# Sidebar for file upload
st.sidebar.header('Upload Marketing Data')
uploaded_file = st.sidebar.file_uploader('Upload CSV', type=['csv'])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file, delimiter=';')

    # Correct the feature list to match what the model expects (29 features)
    feature_names = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome',
                     'Recency', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
                     'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp1', 'AcceptedCmp2', 
                     'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 
                     'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
                     'MntGoldProds', 'Z_CostContact', 'Z_Revenue', 'Response', 'NumWebVisitsMonth', 'Recency']

    # Assuming the dataset uses Z_Revenue as a revenue proxy (adjust as needed)
    revenue_column = 'Z_Revenue'

    # Calculate total marketing spend and ROI
    marketing_spend_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    data = calculate_roi(data, marketing_spend_columns, revenue_column)

    # Remove outliers for ROI column
    data = remove_outliers(data, 'ROI_Total_Marketing')

    # Dataset Overview
    st.header('Dataset Overview')
    st.write(data.describe())

    # Visualizations
    st.header('Data Visualizations')

    # Correlation heatmap (only use numeric columns)
    st.subheader('Correlation Heatmap of Features')
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax1)
    ax1.set_title('Correlation Heatmap')
    st.pyplot(fig1)

    # Boxplot for income and marketing spend
    st.subheader('Income Distribution vs Total Marketing Spend')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Income', y='Total_Marketing_Spend', data=data, ax=ax2)
    ax2.set_title('Income vs Total Marketing Spend')
    ax2.set_xlabel('Income')
    ax2.set_ylabel('Total Marketing Spend')
    st.pyplot(fig2)

    # Response vs Recency
    st.subheader('Response vs Recency')
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Recency', y='Response', data=data, ax=ax3)
    ax3.set_title('Customer Response vs Recency')
    ax3.set_xlabel('Recency (Days since last purchase)')
    ax3.set_ylabel('Response (0 or 1)')
    st.pyplot(fig3)

    # Feature Importance for Response Models
    st.header('Feature Importance for Response Prediction')
    model_paths = {
        'BayesianRidge': 'BayesianRidge_best_model.pkl',
        'ElasticNet': 'ElasticNet_best_model.pkl',
        'Ridge': 'Ridge_best_model.pkl'
    }

    # Ensure X contains all relevant features (match with what the model expects)
    X = data[feature_names]

    # Display feature importance
    feature_importance = get_feature_importance(model_paths, feature_names)
    for model_name, importance in feature_importance.items():
        st.subheader(f'Feature Importance from {model_name}')
        if isinstance(importance, pd.Series):
            st.bar_chart(importance)
        else:
            st.write(importance)

    # Forecast ROI with ARIMA
    st.header('Quarterly Forecasting of ROI')

    # Prepare data for ARIMA forecasting
    data['Date'] = pd.date_range(start='2019-01-01', periods=len(data), freq='ME')  # Use 'ME' for month-end
    df_forecast = data[['Date', 'ROI_Total_Marketing']].rename(columns={'Date': 'ds', 'ROI_Total_Marketing': 'y'})
    df_forecast.set_index('ds', inplace=True)
    df_quarterly = df_forecast.resample('QE').mean()  # Use 'QE' for quarter-end

    arima_model = ARIMA(df_quarterly['y'], order=(5, 1, 0))
    arima_result = arima_model.fit()
    forecast = arima_result.forecast(steps=4)

    # Plot forecast
    st.subheader('Quarterly ROI Forecast for Total Marketing Spend')
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(df_quarterly.index, df_quarterly['y'], label='Historical ROI', color='blue')
    future_dates = pd.date_range(start=df_quarterly.index[-1], periods=5, freq='QE')[1:]
    ax4.plot(future_dates, forecast, label='Forecasted ROI', color='orange')
    ax4.set_title('Quarterly ROI Forecast for Total Marketing Spend')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('ROI (%)')
    ax4.legend()
    st.pyplot(fig4)

    # Provide Business Insights
    st.header('Business Insights')

    # Insights based on ROI forecast
    latest_actual = df_quarterly['y'].iloc[-1]
    first_forecast = forecast.iloc[0]
    percent_change = ((first_forecast - latest_actual) / latest_actual) * 100

    if percent_change > 0:
        insight_text = f"For the next quarter, the ROI is projected to increase by {percent_change:.2f}%. This suggests that the company may see improved returns, and allocating more resources could enhance overall profitability."
    else:
        insight_text = f"For the next quarter, the ROI is projected to decrease by {abs(percent_change):.2f}%. This might indicate diminishing returns, and the company may want to reassess the effectiveness of its marketing strategy."

    st.write(insight_text)

else:
    st.write('Please upload a CSV file to visualize and forecast.')
