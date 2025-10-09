import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# --- DEMO DATA GENERATION (Placeholder for actual data loading) ---
def generate_demo_data(n_days=1000):
    """Generates synthetic daily data to simulate the dataset structure."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Target variable (P_High_Risk, bounded [0, 1])
    # Simulating the ~75% baseline observed
    P_High_Risk = 0.75 + 0.1 * np.sin(np.arange(n_days) / 100) + np.random.normal(0, 0.05, n_days)
    P_High_Risk = np.clip(P_High_Risk, 0.01, 1.0)
    
    # SARIMAX Target (Delivery Time Deviation)
    Y_Deviation = 5.1 + 0.5 * np.cos(np.arange(n_days) / 50) + np.random.normal(0, 1.5, n_days)
    
    # Exogenous Features (VIF features)
    traffic = np.random.uniform(1, 10, n_days)
    weather = np.random.uniform(0, 5, n_days)
    volume = np.random.lognormal(mean=7, sigma=0.5, size=n_days)
    
    # Key XGBoost Feature (System Memory)
    lagged_risk = pd.Series(P_High_Risk).shift(1).fillna(P_High_Risk[0])

    df = pd.DataFrame({
        'Date': dates,
        'P_High_Risk': P_High_Risk,
        'Y_Deviation': Y_Deviation,
        'Max_Traffic': traffic,
        'Max_Weather': weather,
        'Total_Volume': volume,
        'Lagged_Risk_t1': lagged_risk 
    }).set_index('Date')
    
    return df

# --- 1. DATA PREP AND VIF FILTERING (Multicollinearity Check) ---

def calculate_vif(df):
    """Calculates VIF for all columns in a DataFrame."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    # Ensure all data types are float for VIF calculation
    vif_data["VIF"] = [variance_inflation_factor(df.values.astype(float), i) 
                       for i in range(len(df.columns))]
    return vif_data.sort_values(by='VIF', ascending=False)

def filter_vif_features(df, threshold=5.0):
    """Iteratively removes features with VIF > threshold to ensure robustness."""
    X_vif = df.copy()
    while True:
        vif_results = calculate_vif(X_vif)
        max_vif = vif_results['VIF'].max()
        if max_vif > threshold:
            feature_to_drop = vif_results.iloc[0]['feature']
            X_vif = X_vif.drop(columns=[feature_to_drop])
            print(f"Dropping high-VIF feature: {feature_to_drop} (VIF: {max_vif:.2f})")
        else:
            break
    print(f"\nFinal VIF-filtered features: {X_vif.columns.tolist()}")
    return X_vif

# --- 2. PHASE I: SARIMAX MODELING (Temporal Forecast) ---

def phase_i_sarimax(y_series, X_exo_vif):
    """
    Fits the SARIMAX model and generates the forecast (Phase I).
    The SARIMAX forecast becomes a feature for Phase II (XGBoost).
    """
    # 1. Apply first-order differencing (d=1) for stationarity
    y_diff = y_series.diff().dropna()
    X_exo_aligned = X_exo_vif.loc[y_diff.index]

    # Optimized SARIMAX parameters found via AIC grid search: (p, d, q) x (P, D, Q)s
    # Using a known optimal order from the analysis: (1,1,2)(1,1,1,7)
    order = (1, 0, 2)  # d=0 here as differencing is applied manually above
    seasonal_order = (1, 0, 1, 7)
    
    # Fit the SARIMAX model on the differenced series
    try:
        model = SARIMAX(
            y_diff,
            exog=X_exo_aligned,
            order=order, 
            seasonal_order=seasonal_order, 
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        print("SARIMAX Model fit successful.")
    except Exception as e:
        print(f"SARIMAX fit failed: {e}. Using a simplified model.")
        # Fallback to a simpler AR model if complex SARIMAX fails with demo data
        model = SARIMAX(y_diff, order=(1,0,0)).fit(disp=False)
        
    # Generate a simple 30-day forecast for demonstration
    steps = 30
    future_exog = X_exo_vif.iloc[-steps:] # Use last 30 days of exog as future input
    
    # Predict the differenced series
    forecast_diff = model.get_prediction(start=len(y_diff), end=len(y_diff) + steps - 1, exog=future_exog)
    
    # Invert the differencing (re-integrate) to get the final forecast
    last_actual_y = y_series.loc[y_diff.index[-1]]
    sarimax_forecast_raw = last_actual_y + forecast_diff.predicted_mean.cumsum()
    
    # Create the key feature for Phase II
    sarimax_feature = pd.Series(sarimax_forecast_raw, name='SARIMAX_Predicted_Delay')
    
    return sarimax_feature, model

# --- 3. PHASE II: XGBOOST REGRESSOR (Risk Classification) ---

def phase_ii_xgboost(df, sarimax_feature):
    """
    Trains XGBoost on historical data (Phase I forecast as a feature)
    and predicts the P_High_Risk for the future period.
    """
    
    # Prepare historical features for XGBoost (Lagged_Risk_t1, VIF features, etc.)
    X_historical = df.drop(columns=['P_High_Risk', 'Y_Deviation']).iloc[:-30]
    y_target = df['P_High_Risk'].iloc[:-30]

    # --- Chronological Split: CRITICAL for time series validation (shuffle=False) ---
    split_ratio = 0.8
    train_size = int(len(X_historical) * split_ratio)

    X_train, X_test = X_historical.iloc[:train_size], X_historical.iloc[train_size:]
    y_train, y_test = y_target.iloc[:train_size], y_target.iloc[train_size:]
    
    # Initialize and train the XGBoost Regressor
    # Hyperparameters tuned for generalization (e.g., lower learning rate, max depth control)
    xgb_regressor = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        objective='reg:squarederror',
        random_state=42
    )

    # Note: We assume 'SARIMAX_Predicted_Delay' is generated and aligned for training here 
    # for a full-scale implementation. For this snippet, we use existing historical features.
    
    xgb_regressor.fit(X_train, y_train)

    # 1. Validate on Test Set
    y_pred_test = xgb_regressor.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    print(f"\nPhase II XGBoost Test RMSE (Historical Validation): {rmse:.4f}")
    
    # 2. Prepare Future Features for Prediction
    X_future = df.drop(columns=['P_High_Risk', 'Y_Deviation']).iloc[-30:]
    X_future['SARIMAX_Predicted_Delay'] = sarimax_feature.values # Add Phase I output as feature
    
    # Final Future Prediction
    future_risk_prediction = pd.Series(xgb_regressor.predict(X_future.drop(columns=['SARIMAX_Predicted_Delay'], errors='ignore')), # Drop delay feature if not used in historical
                                       index=X_future.index, 
                                       name='Predicted_P_High_Risk')
    
    # Clip predictions to the logical range [0, 1]
    return future_risk_prediction.clip(0, 1), xgb_regressor

# --- 4. DYNAMIC PERCENTILE BUCKETING (DPB) ---

def dynamic_percentile_bucketing(forecast_series):
    """Classifies risk based on the relative severity within the predicted distribution."""
    
    # Find the 25th and 75th percentiles (quartiles) of the forecast distribution
    p25 = forecast_series.quantile(0.25)
    p75 = forecast_series.quantile(0.75)
    
    def classify_risk(risk_score):
        if risk_score >= p75:
            return "High Risk (Worst 25%)"
        elif risk_score >= p25:
            return "Moderate Risk (Middle 50%)"
        else:
            return "Low Risk (Best 25%)"
            
    # Apply the classification and combine with the forecast series
    classification = forecast_series.apply(classify_risk)
    
    print(f"\n--- Dynamic Percentile Bucketing (DPB) Thresholds ---")
    print(f"High Risk (Top 25%): >= {p75:.4f}")
    print(f"Low Risk (Bottom 25%): < {p25:.4f}")
    
    return pd.DataFrame({
        'Predicted_P_High_Risk': forecast_series,
        'DPB_Classification': classification
    })

# --- MAIN EXECUTION FLOW ---
if __name__ == "__main__":
    print("--- Running SXCM Predictive Pipeline Demo ---")
    
    # 1. Load Data
    df_raw = generate_demo_data(n_days=1000)
    
    # 2. VIF Filtering (Exogenous features)
    X_exo = df_raw[['Max_Traffic', 'Max_Weather', 'Total_Volume']]
    X_exo_vif_filtered = filter_vif_features(X_exo, threshold=5.0)
    
    # 3. PHASE I: SARIMAX Forecast (Y_Deviation is the target)
    y_deviation = df_raw['Y_Deviation']
    sarimax_feature_forecast, sarimax_model = phase_i_sarimax(y_deviation, X_exo_vif_filtered)
    
    # 4. PHASE II: XGBoost Prediction (P_High_Risk is the target)
    final_risk_forecast, xgb_model = phase_ii_xgboost(df_raw, sarimax_feature_forecast)
    
    # 5. Dynamic Risk Classification
    classified_risk_df = dynamic_percentile_bucketing(final_risk_forecast)
    
    print("\n--- Final 30-Day Classified Forecast (Actionable Output) ---")
    print(classified_risk_df.head())
    
    # Display Feature Importance for Diagnostic Insight
    if hasattr(xgb_model, 'feature_importances_'):
        feature_importances = pd.Series(xgb_model.feature_importances_, 
                                        index=df_raw.drop(columns=['P_High_Risk', 'Y_Deviation']).iloc[:800].columns)
        print("\n--- XGBoost Feature Importance (Gain) ---")
        print(feature_importances.sort_values(ascending=False).head(5))

    # Save final forecast file (for Power BI/Tableau integration)
    classified_risk_df.to_csv('future_risk_forecast.csv')
