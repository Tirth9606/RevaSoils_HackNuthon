import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to prevent tkinter issues
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import RFE, SelectFromModel
import shap
import time
from warnings import filterwarnings
filterwarnings('ignore')

import tempfile

import os
from pathlib import Path

# Use a dedicated output directory
OUTPUT_DIR = Path("soil_analysis_output")
PREDICTIONS_FILE = OUTPUT_DIR / "soil_predictions.csv"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------
# Data Preprocessing
# -------------------
def preprocess_data(df):
    if 'Records' in df.columns:
        df['Soil_ID'] = df['Records'].str.split('-').str[0]
        df['Moisture_Level'] = df['Soil_ID'].str.split('_').str[1]
        df = df.drop(columns=['Records'])
    elif 'Soil_ID' not in df.columns:
        # Create a default Soil_ID if neither Records nor Soil_ID exists
        df['Soil_ID'] = [f"Sample_{i}" for i in range(1, len(df)+1)]
    
    aggregated = df.groupby('Soil_ID').median(numeric_only=True).reset_index()
    
    sensor_cols = ['Moist', 'EC (u/10 gram)', 'Ph', 
                 'Nitro (mg/10 g)', 'Posh Nitro (mg/10 g)', 'Pota Nitro (mg/10 g)']
    
    existing_sensor_cols = [col for col in sensor_cols if col in aggregated.columns]
    aggregated[existing_sensor_cols] = aggregated[existing_sensor_cols].replace(0, np.nan)
    
    if existing_sensor_cols:
        imputer = KNNImputer(n_neighbors=3)
        aggregated[existing_sensor_cols] = imputer.fit_transform(aggregated[existing_sensor_cols])
    
    return aggregated

def process_spectral_data(df):
    wavelengths = ['410', '435', '460', '485', '510', '535', 
                 '560', '585', '610', '645', '680', '705', 
                 '730', '760', '810', '860', '900', '940']
    
    existing_wavelengths = [w for w in wavelengths if w in df.columns]
    
    if existing_wavelengths:
        df[existing_wavelengths] = savgol_filter(df[existing_wavelengths], 
                                               window_length=11, 
                                               polyorder=2,
                                               axis=1)
        
        scaler = StandardScaler()
        spectral_data = scaler.fit_transform(df[existing_wavelengths])
        
        deriv = np.gradient(spectral_data, axis=1)
        
        if '860' in df.columns and '645' in df.columns:
            df['NDI'] = (df['860'] - df['645']) / (df['860'] + df['645'])
        if '730' in df.columns and '680' in df.columns:
            df['SIR'] = df['730'] / df['680']
        
        deriv_cols = [f'd{w}' for w in existing_wavelengths]
        df = pd.concat([df, pd.DataFrame(deriv, columns=deriv_cols)], axis=1)
    
    return df

# -------------------
# Feature Engineering
# -------------------
def create_features(df):
    existing_cols = df.columns.tolist()
    
    vis_cols = [f'{w}' for w in range(400, 700, 10) if f'{w}' in existing_cols]
    nir_cols = [f'{w}' for w in range(700, 1000, 10) if f'{w}' in existing_cols]
    
    if vis_cols:
        df['VIS_avg'] = df[vis_cols].mean(axis=1)
    if nir_cols:
        df['NIR_avg'] = df[nir_cols].mean(axis=1)
    
    wavelengths = [str(w) for w in range(400, 1000, 10) if str(w) in existing_cols]
    if wavelengths:
        pca = PCA(n_components=min(5, len(wavelengths)))
        pca_features = pca.fit_transform(df[wavelengths])
        df_pca = pd.DataFrame(pca_features, columns=[f'PCA_{i+1}' for i in range(pca_features.shape[1])])
        df = pd.concat([df, df_pca], axis=1)
    
    return df

# -------------------
# Feature Selection
# -------------------
def select_features(X, y, method='importance', threshold=0.5):
    """Select features using specified method"""
    if method == 'correlation':
        corr = pd.concat([X, y], axis=1).corr()[y.name].abs()
        selected = corr[corr > threshold].dropna().index.tolist()
        selected.remove(y.name)
        return selected
    elif method == 'rfe':
        selector = RFE(RandomForestRegressor(n_estimators=50), n_features_to_select=10)
        selector.fit(X, y)
        return X.columns[selector.support_].tolist()
    elif method == 'importance':
        model = RandomForestRegressor(n_estimators=50)
        model.fit(X, y)
        selector = SelectFromModel(model, threshold=f'{threshold}*mean')
        selector.fit(X, y)
        return X.columns[selector.get_support()].tolist()
    else:
        return X.columns.tolist()

# -------------------
# Modeling (Optimized)
# -------------------
def train_models(X, y):
    start_time = time.time()
    timeout = 1200  # 20 minutes timeout
    
    models = {
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'n_jobs': [-1]
            },
            'pipeline': None
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'pipeline': None
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'svr__C': [0.1, 1, 10, 100],
                'svr__gamma': ['scale', 'auto', 0.01, 0.1],
                'svr__kernel': ['rbf', 'poly']
            },
            'pipeline': make_pipeline(StandardScaler(), SVR())
        }
    }
    
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, config in models.items():
        if time.time() - start_time > timeout:
            print(f"Timeout reached during {name} training")
            break
            
        print(f"\nTraining {name}...")
        try:
            scores = []
            best_params_list = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
                if time.time() - start_time > timeout:
                    print(f"Timeout during {name} fold {fold}")
                    break
                    
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                if config['pipeline']:
                    gs = GridSearchCV(
                        config['pipeline'],
                        config['params'],
                        cv=3,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=1
                    )
                else:
                    gs = GridSearchCV(
                        config['model'],
                        config['params'],
                        cv=3,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=1
                    )
                
                gs.fit(X_train, y_train)
                pred = gs.predict(X_test)
                
                fold_scores = {
                    'R2': r2_score(y_test, pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                    'MAE': mean_absolute_error(y_test, pred)
                }
                scores.append(fold_scores)
                best_params_list.append(gs.best_params_)
                
                print(f"Fold {fold} completed - R2: {fold_scores['R2']:.3f}")
                print(f"Best params: {gs.best_params_}")
            
            if scores:
                best_idx = np.argmax([s['R2'] for s in scores])
                results[name] = {
                    'R2': np.mean([s['R2'] for s in scores]),
                    'RMSE': np.mean([s['RMSE'] for s in scores]),
                    'MAE': np.mean([s['MAE'] for s in scores]),
                    'best_params': best_params_list[best_idx],
                    'is_pipeline': config['pipeline'] is not None,
                    'model': gs.best_estimator_
                }
                
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
            continue
    
    return pd.DataFrame.from_dict(results, orient='index')

# -------------------
# Visualization (Updated with proper figure handling)
# -------------------
def plot_spectral_profiles(df):
    try:
        wavelengths = [c for c in df.columns if c.isdigit()]
        if wavelengths:
            wavelengths = sorted(wavelengths, key=lambda x: int(x))
            fig, ax = plt.subplots(figsize=(12, 6))
            for idx in np.random.choice(len(df), min(5, len(df)), replace=False):
                ax.plot([int(w) for w in wavelengths], df.iloc[idx][wavelengths], alpha=0.5)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Normalized Reflectance')
            ax.set_title('Spectral Profiles')
            plt.tight_layout()
            plt.show()
            plt.close(fig)
    except Exception as e:
        print(f"Error plotting spectral profiles: {str(e)}")

def analyze_correlations(df, target='Moist'):
    try:
        if target in df.columns:
            numeric_df = df.select_dtypes(include=[np.number])
            if target in numeric_df.columns:
                correlations = numeric_df.corr()[target].sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                correlations[1:11].plot(kind='bar', ax=ax)
                ax.set_title(f'Top Features Correlated with {target}')
                plt.tight_layout()
                plt.show()
                plt.close(fig)
    except Exception as e:
        print(f"Error plotting correlations: {str(e)}")

# -------------------
# [Previous imports remain the same...]

# -------------------
# -------------------
# Main Workflow (Updated with CSV output containing only predictions)
# -------------------
if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_csv("soildata.csv")
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit()
    
    # Preprocess data
    processed_df = preprocess_data(df)
    spectral_df = process_spectral_data(processed_df)
    final_df = create_features(spectral_df)
    
    print("\nAvailable columns:", final_df.columns.tolist())
    
    # Prepare features and target
    cols_to_drop = ['Soil_ID', 'Moisture_Level']
    features = final_df.drop(columns=[col for col in cols_to_drop if col in final_df.columns])
    
    # Define all potential target columns
    sensor_targets = [
        'Capacitity Moist', 'Temp', 'Moist', 'EC (u/10 gram)', 
        'Ph', 'Nitro (mg/10 g)', 'Posh Nitro (mg/10 g)', 'Pota Nitro (mg/10 g)'
    ]
    
    # Filter to only include targets that exist in the data
    available_targets = [col for col in sensor_targets if col in final_df.columns]
    
    if not available_targets:
        print("No suitable target columns found")
        exit()
    
    print(f"\nPredicting targets: {available_targets}")
    
    # Prepare feature matrix
    numeric_features = features.select_dtypes(include=[np.number])
    
    # Initialize results storage
    all_results = {}
    best_models = {}
    
    # Initialize predictions dataframe EARLY (FIXED)
    predictions_df = final_df[['Soil_ID']].copy()
    
    for target_col in available_targets:
        print(f"\n=== Processing target: {target_col} ===")
        
        targets = final_df[target_col]
        
        # EDA for this target
        analyze_correlations(final_df, target_col)
        
        # Feature selection
        print("Selecting features...")
        selected_features = select_features(
            numeric_features, 
            targets, 
            method='importance', 
            threshold=0.5
        )
        print(f"Selected {len(selected_features)} features for {target_col}")
        
        if len(selected_features) == 0:
            print("No features selected, using all features")
            selected_features = numeric_features.columns.tolist()
        
        X = numeric_features[selected_features]
        
        # Model training
        print("Starting model training...")
        results = train_models(X, targets)
        
        if results.empty:
            print(f"Model training failed for {target_col}")
            continue
        
        print(f"\nModel Performance for {target_col}:")
        print(results[['R2', 'RMSE', 'MAE']])
        
        # Store results
        all_results[target_col] = results
        best_model_name = results['R2'].idxmax()
        best_models[target_col] = {
            'model': results.loc[best_model_name, 'model'],
            'features': selected_features,
            'performance': results.loc[best_model_name, ['R2', 'RMSE', 'MAE']]
        }
        
        # Generate and store predictions for this target
        try:
            predictions = best_models[target_col]['model'].predict(X)
            predictions_df[target_col] = predictions
        except Exception as e:
            print(f"Error generating predictions for {target_col}: {str(e)}")
            continue
        
        # Feature importance visualization
        current_model = best_models[target_col]['model']
        try:
            if hasattr(current_model, 'feature_importances_'):
                importances = current_model.feature_importances_
            elif isinstance(current_model, Pipeline) and hasattr(current_model.steps[-1][1], 'feature_importances_'):
                importances = current_model.steps[-1][1].feature_importances_
            else:
                importances = None
            
            if importances is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                pd.Series(importances, index=selected_features).sort_values().plot(kind='barh', ax=ax)
                ax.set_title(f'Feature Importances for {target_col}')
                plt.tight_layout()
                plt.show()
                plt.close(fig)
        except Exception as e:
            print(f"Error plotting feature importance: {str(e)}")
        
        # SHAP analysis
        print("\nRunning SHAP analysis...")
        try:
            sample = X.sample(min(100, len(X)), random_state=42)
            
            if isinstance(current_model, Pipeline):
                explainer = shap.KernelExplainer(
                    lambda x: current_model.predict(pd.DataFrame(x, columns=X.columns)),
                    shap.sample(pd.DataFrame(sample), 100))
            else:
                explainer = shap.TreeExplainer(current_model)
            
            shap_values = explainer.shap_values(sample)
            
            fig = plt.figure()
            shap.summary_plot(shap_values, sample, plot_type='bar', show=False)
            plt.title(f'SHAP Analysis for {target_col}')
            plt.tight_layout()
            plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"SHAP error for {target_col}: {str(e)}")
    
    # Summary of all predictions
    print("\n=== Final Summary ===")
    for target, info in best_models.items():
        print(f"\nTarget: {target}")
        print(f"Best model: {type(info['model']).__name__}")
        print(f"R2: {info['performance']['R2']:.3f}")
        print(f"RMSE: {info['performance']['RMSE']:.3f}")
        print(f"MAE: {info['performance']['MAE']:.3f}")
        print(f"Number of features used: {len(info['features'])}")
    
    # Generate predictions for all samples and all targets
    # Generate predictions for all samples and all targets
print("\nGenerating predictions for all samples...")
predictions_df = final_df[['Soil_ID']].copy()

successful_targets = 0
for target, info in best_models.items():
    try:
        X_all = numeric_features[info['features']]
        predictions = info['model'].predict(X_all)
        predictions_df[target] = predictions
        successful_targets += 1
    except Exception as e:
        print(f"Error generating predictions for {target}: {str(e)}")
        continue

# Verify we have predictions to save
if successful_targets == 0:
    print("\nError: No predictions were successfully generated!")
    print("Possible causes:")
    print("- No valid target columns found")
    print("- All model trainings failed")
    print("- Feature selection failed for all targets")
    exit()

# Save with robust error handling
try:
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"\nSuccess! Predictions saved to {PREDICTIONS_FILE}")
    
    # Verify the file was written
    if not PREDICTIONS_FILE.exists():
        raise Exception("File was not created despite no errors")
        
except Exception as e:
    print(f"\nFailed to save predictions: {str(e)}")
    print("\nAttempting alternative save methods...")
    
    # Try saving to different locations
    alt_locations = [
        Path.home() / "soil_predictions.csv",  # User home directory
        Path.cwd() / "soil_predictions_alt.csv",  # Current directory
        Path(tempfile.gettempdir()) / "soil_predictions.csv"  # Temp directory
    ]
    
    saved = False
    for location in alt_locations:
        try:
            predictions_df.to_csv(location, index=False)
            print(f"Successfully saved to alternative location: {location}")
            saved = True
            break
        except Exception as alt_e:
            print(f"Failed to save to {location}: {str(alt_e)}")
    
    if not saved:
        print("\nCritical: Could not save predictions to any location!")
        print("Final predictions data:")
        print(predictions_df.head())
        exit()
    # Example prediction for all targets (first sample)
    sample_idx = 0
    sample_data = numeric_features.iloc[[sample_idx]]
    print("\n=== Example Prediction ===")
    print(f"Sample data (first row):")
    
    for target, info in best_models.items():
        try:
            # Select only the features used for this target
            X_sample = sample_data[info['features']]
            prediction = info['model'].predict(X_sample)
            actual = final_df[target].iloc[sample_idx]
            
            print(f"\n{target}:")
            print(f"  Actual: {actual:.2f}")
            print(f"  Predicted: {prediction[0]:.2f}")
            print(f"  Difference: {abs(actual - prediction[0]):.2f}")
        except Exception as e:
            print(f"Error predicting {target}: {str(e)}")
    if not predictions_df.empty:
        predictions_df.to_csv(PREDICTIONS_FILE, index=False)
        print(f"\nAll predictions saved to {PREDICTIONS_FILE}")
    else:
        print("\nNo predictions were generated")
        exit()
    # Plot spectral profiles at the end
    plot_spectral_profiles(final_df)