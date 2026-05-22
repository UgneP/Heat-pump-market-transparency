import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import shap
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the heat pump data from CSV file"""
    csv_path = Path("data/processed/filtered_hplib_w_prices_and_decoding.csv")
    df = pd.read_csv(csv_path)
    return df

def clean_data(df):
    """Clean and prepare data for regression analysis"""
    # Convert numeric columns to proper types
    numeric_columns = ['Price', 'SCOP', 'Rated Power low T [kW]', 
                      'SPL outdoor high Power [dBA]', 'Max. water heating temperature [°C]']
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to string first, then numeric to handle mixed types
            df[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
    
    # Create configuration groups
    if 'Config' in df.columns and 'Storage' in df.columns:
        df['Configuration_Group'] = df['Config'].astype(str) + ' - ' + df['Storage'].astype(str)
        
        # Define the 5 specific groups we want to analyze
        target_groups = [
            'monoblock - none',
            'monoblock - hydraulic module', 
            'monoblock - warm-water tank',
            'split - hydraulic module',
            'split - warm-water tank'
        ]
        
        # Filter to only include the target groups
        df = df[df['Configuration_Group'].isin(target_groups)].copy()
    
    # Remove rows with missing essential data
    essential_columns = ['Price', 'SCOP', 'Rated Power low T [kW]', 
                        'SPL outdoor high Power [dBA]', 'Max. water heating temperature [°C]', 
                        'Manufacturer']
    df_clean = df.dropna(subset=essential_columns).copy()
    
    # Ensure all numeric columns are actually numeric
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove any rows that still have NaN values after conversion
    df_clean = df_clean.dropna(subset=essential_columns).copy()
    
    # Ensure manufacturer column is string type
    df_clean['Manufacturer'] = df_clean['Manufacturer'].astype(str)
    
    return df_clean

def prepare_regression_data(df):
    """Prepare data for regression analysis"""
    # Create dummy variables for categorical features
    df_reg = df.copy()
    
    # Encode manufacturer as dummy variables
    manufacturer_dummies = pd.get_dummies(df_reg['Manufacturer'], prefix='Brand', drop_first=True)
    df_reg = pd.concat([df_reg, manufacturer_dummies], axis=1)
    
    # Encode configuration if available
    if 'Configuration_Group' in df_reg.columns:
        config_dummies = pd.get_dummies(df_reg['Configuration_Group'], prefix='Config', drop_first=True)
        df_reg = pd.concat([df_reg, config_dummies], axis=1)
    
    # Encode refrigerant if available
    if 'Refrigerant' in df_reg.columns:
        refrigerant_dummies = pd.get_dummies(df_reg['Refrigerant'], prefix='Refrigerant', drop_first=True)
        df_reg = pd.concat([df_reg, refrigerant_dummies], axis=1)
    
    # Ensure all dummy variables are numeric (0/1)
    dummy_columns = [col for col in df_reg.columns if col.startswith(('Brand_', 'Config_', 'Refrigerant_'))]
    for col in dummy_columns:
        df_reg[col] = df_reg[col].astype(int)
    
    # Ensure numeric columns are float type
    numeric_columns = ['SCOP', 'Rated Power low T [kW]', 'SPL outdoor high Power [dBA]', 
                      'Max. water heating temperature [°C]', 'Price']
    for col in numeric_columns:
        if col in df_reg.columns:
            df_reg[col] = df_reg[col].astype(float)
    
    return df_reg

def print_regression_table(model, X, y, title):
    """Print regression results in Stata-like format"""
    print(f"\n{title.upper()}")
    print("=" * 80)
    
    # ANOVA Table
    n_obs = model.nobs
    k = model.df_model  # number of predictors
    df_model = k
    df_resid = model.df_resid
    df_total = df_model + df_resid
    
    ss_model = model.ess  # explained sum of squares
    ss_resid = model.ssr  # residual sum of squares  
    ss_total = model.centered_tss  # total sum of squares
    
    ms_model = ss_model / df_model if df_model > 0 else 0
    ms_resid = ss_resid / df_resid if df_resid > 0 else 0
    ms_total = ss_total / df_total if df_total > 0 else 0
    
    print(f"\nSource          {'SS':<15} {'df':<8} {'MS':<15}     Number of obs = {n_obs:>8.0f}")
    print("-" * 70 + f"     F({df_model:>3.0f},{df_resid:>6.0f}) = {model.fvalue:>8.2f}")
    print(f"Model     {ss_model:>15.8f} {df_model:>8.0f} {ms_model:>15.8f}     Prob > F      = {model.f_pvalue:>8.4f}")
    print(f"Residual  {ss_resid:>15.8f} {df_resid:>8.0f} {ms_resid:>15.8f}     R-squared     = {model.rsquared:>8.4f}")
    print("-" * 70 + f"     Adj R-squared = {model.rsquared_adj:>8.4f}")
    print(f"Total     {ss_total:>15.8f} {df_total:>8.0f} {ms_total:>15.8f}     Root MSE      = {np.sqrt(ms_resid):>8.5f}")
    
    # Coefficients Table
    print(f"\n{'-'*80}")
    print(f"{'Variable':<20} {'Coef.':<12} {'Std. Err.':<12} {'t':<8} {'P>|t|':<8} {'[95% Conf. Interval]'}")
    print(f"{'-'*80}")
    
    for var_name, coef, stderr, tval, pval, ci_low, ci_high in zip(
        X.columns, model.params, model.bse, model.tvalues, model.pvalues,
        model.conf_int()[0], model.conf_int()[1]):
        
        print(f"{var_name:<20} {coef:>11.7f} {stderr:>11.7f} {tval:>7.2f} {pval:>7.3f}   {ci_low:>10.7f}  {ci_high:>10.7f}")
    
    print(f"{'-'*80}")

def perform_linear_regression_full_dataset(df):
    """Perform linear regression on the full dataset"""
    print("\n" + "="*80)
    print("LINEAR REGRESSION ANALYSIS - FULL DATASET")
    print("="*80)
    
    df_reg = prepare_regression_data(df)
    
    # Define features for regression
    feature_columns = ['SCOP', 'Rated Power low T [kW]', 'SPL outdoor high Power [dBA]', 
                      'Max. water heating temperature [°C]']
    
    # Add brand dummy variables
    brand_columns = [col for col in df_reg.columns if col.startswith('Brand_')]
    feature_columns.extend(brand_columns)
    
    # Add configuration dummy variables
    config_columns = [col for col in df_reg.columns if col.startswith('Config_')]
    feature_columns.extend(config_columns)
    
    # Add refrigerant dummy variables
    refrigerant_columns = [col for col in df_reg.columns if col.startswith('Refrigerant_')]
    feature_columns.extend(refrigerant_columns)
    
    # Prepare feature matrix and target
    X = df_reg[feature_columns].copy()
    y = df_reg['Price'].copy()
    
    # Count total brands for reporting
    total_brands = df_reg['Manufacturer'].nunique()
    total_configs = df_reg['Configuration_Group'].nunique() if 'Configuration_Group' in df_reg.columns else 0
    total_refrigerants = df_reg['Refrigerant'].nunique() if 'Refrigerant' in df_reg.columns else 0
    
    print(f"Dataset size: {len(X)} observations")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Features included:")
    print(f"  - 4 numerical features: {feature_columns[:4]}")
    print(f"  - {len(brand_columns)} brand dummies (from {total_brands} total brands)")
    print(f"  - {len(config_columns)} configuration dummies (from {total_configs} total configurations)")
    if refrigerant_columns:
        print(f"  - {len(refrigerant_columns)} refrigerant dummies (from {total_refrigerants} total refrigerants)")
    
    # Fit the model using statsmodels for detailed statistics
    X_sm = sm.add_constant(X)  # Add intercept
    model = sm.OLS(y, X_sm).fit()
    
    print(f"\nREGRESSION RESULTS:")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.2f}")
    print(f"F-statistic p-value: {model.f_pvalue:.2e}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    
    # Print detailed regression output in Stata-like format
    print_regression_table(model, X_sm, y, "Linear Regression - Full Dataset")
    
    # Model diagnostics
    print(f"\nMODEL DIAGNOSTICS:")
    
    # Residuals analysis
    residuals = model.resid
    fitted = model.fittedvalues
    
    print(f"Mean of residuals: {residuals.mean():.6f}")
    print(f"Standard deviation of residuals: {residuals.std():.2f}")
    
    # Normality test (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    print(f"Jarque-Bera normality test: JB = {jb_stat:.4f}, p-value = {jb_pvalue:.4f}")
    if jb_pvalue > 0.05:
        print("  [OK] Residuals appear normally distributed")
    else:
        print("  [WARNING] Residuals may not be normally distributed")
    
    # Heteroscedasticity test (Breusch-Pagan)
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_sm)
    print(f"Breusch-Pagan heteroscedasticity test: LM = {bp_stat:.4f}, p-value = {bp_pvalue:.4f}")
    if bp_pvalue > 0.05:
        print("  [OK] Homoscedasticity assumption satisfied")
    else:
        print("  [WARNING] Heteroscedasticity detected")
    
    # Durbin-Watson test for autocorrelation
    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson test for autocorrelation: DW = {dw_stat:.4f}")
    if 1.5 < dw_stat < 2.5:
        print("  [OK] No significant autocorrelation")
    else:
        print("  [WARNING] Possible autocorrelation in residuals")
    
    return model, X, y, feature_columns

def perform_log_linear_regression_full_dataset(df):
    """Perform log-linear regression on the full dataset"""
    print("\n" + "="*80)
    print("LOG-LINEAR REGRESSION ANALYSIS - FULL DATASET")
    print("="*80)
    
    df_reg = prepare_regression_data(df)
    
    # Define features for regression
    feature_columns = ['SCOP', 'Rated Power low T [kW]', 'SPL outdoor high Power [dBA]', 
                      'Max. water heating temperature [°C]']
    
    # Add brand dummy variables
    brand_columns = [col for col in df_reg.columns if col.startswith('Brand_')]
    feature_columns.extend(brand_columns)
    
    # Add configuration dummy variables
    config_columns = [col for col in df_reg.columns if col.startswith('Config_')]
    feature_columns.extend(config_columns)
    
    # Add refrigerant dummy variables
    refrigerant_columns = [col for col in df_reg.columns if col.startswith('Refrigerant_')]
    feature_columns.extend(refrigerant_columns)
    
    # Prepare feature matrix and target
    X = df_reg[feature_columns].copy()
    y = df_reg['Price'].copy()
    
    # Count total brands for reporting
    total_brands = df_reg['Manufacturer'].nunique()
    total_configs = df_reg['Configuration_Group'].nunique() if 'Configuration_Group' in df_reg.columns else 0
    total_refrigerants = df_reg['Refrigerant'].nunique() if 'Refrigerant' in df_reg.columns else 0
    
    print(f"Dataset size: {len(X)} observations")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Features included:")
    print(f"  - 4 numerical features: {feature_columns[:4]}")
    print(f"  - {len(brand_columns)} brand dummies (from {total_brands} total brands)")
    print(f"  - {len(config_columns)} configuration dummies (from {total_configs} total configurations)")
    if refrigerant_columns:
        print(f"  - {len(refrigerant_columns)} refrigerant dummies (from {total_refrigerants} total refrigerants)")
    
    # Transform target variable to log scale (handling any zero values)
    y_log = np.log(y + 1)  # Adding 1 to handle potential zeros
    
    # Fit the model using statsmodels for detailed statistics
    X_sm = sm.add_constant(X)  # Add intercept
    model = sm.OLS(y_log, X_sm).fit()
    
    print(f"\nREGRESSION RESULTS (in log scale):")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.2f}")
    print(f"F-statistic p-value: {model.f_pvalue:.2e}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    
    # Print detailed regression output in Stata-like format
    print_regression_table(model, X_sm, y_log, "Log-Linear Regression - Full Dataset")
    
    # Model diagnostics
    print(f"\nMODEL DIAGNOSTICS:")
    
    # Residuals analysis
    residuals = model.resid
    fitted = model.fittedvalues
    
    print(f"Mean of residuals: {residuals.mean():.6f}")
    print(f"Standard deviation of residuals: {residuals.std():.2f}")
    
    # Normality test (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    print(f"Jarque-Bera normality test: JB = {jb_stat:.4f}, p-value = {jb_pvalue:.4f}")
    if jb_pvalue > 0.05:
        print("  [OK] Residuals appear normally distributed")
    else:
        print("  [WARNING] Residuals may not be normally distributed")
    
    # Heteroscedasticity test (Breusch-Pagan)
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_sm)
    print(f"Breusch-Pagan heteroscedasticity test: LM = {bp_stat:.4f}, p-value = {bp_pvalue:.4f}")
    if bp_pvalue > 0.05:
        print("  [OK] Homoscedasticity assumption satisfied")
    else:
        print("  [WARNING] Heteroscedasticity detected")
    
    # Durbin-Watson test for autocorrelation
    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson test for autocorrelation: DW = {dw_stat:.4f}")
    if 1.5 < dw_stat < 2.5:
        print("  [OK] No significant autocorrelation")
    else:
        print("  [WARNING] Possible autocorrelation in residuals")
    
    # Calculate predictions in original scale for comparison
    y_pred_log = model.predict(X_sm)
    y_pred_original = np.exp(y_pred_log) - 1
    
    # Calculate R² in original scale
    r2_original = r2_score(y, y_pred_original)
    print(f"\nPERFORMANCE IN ORIGINAL SCALE:")
    print(f"R-squared (original scale): {r2_original:.4f}")
    
    return model, X, y, y_log, feature_columns


def perform_advanced_modeling(df):
    """Perform advanced modeling with gradient boosting and SHAP analysis"""
    print("\n" + "="*80)
    print("ADVANCED MODELING WITH GRADIENT BOOSTING AND SHAP ANALYSIS")
    print("="*80)
    
    df_reg = prepare_regression_data(df)
    
    # Define all available features
    feature_columns = ['SCOP', 'Rated Power low T [kW]', 'SPL outdoor high Power [dBA]', 
                      'Max. water heating temperature [°C]']
    
    # Add categorical features
    brand_columns = [col for col in df_reg.columns if col.startswith('Brand_')]
    config_columns = [col for col in df_reg.columns if col.startswith('Config_')]
    refrigerant_columns = [col for col in df_reg.columns if col.startswith('Refrigerant_')]
    
    all_features = feature_columns + brand_columns + config_columns + refrigerant_columns
    
    # Filter to only include features that exist in the dataframe
    available_features = [col for col in all_features if col in df_reg.columns]
    
    X = df_reg[available_features].copy()
    y = df_reg['Price'].copy()
    
    print(f"Dataset size: {len(X)} observations")
    print(f"Total features: {len(available_features)}")
    print(f"- Numerical features: {len(feature_columns)}")
    print(f"- Brand dummies: {len(brand_columns)}")
    print(f"- Configuration dummies: {len(config_columns)}")
    print(f"- Refrigerant dummies: {len(refrigerant_columns)}")
    
    # Split data for advanced modeling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Gradient Boosting Model
    print(f"\nGRADIENT BOOSTING MODEL")
    print("-" * 40)
    
    # Fit gradient boosting model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_gb = gb_model.predict(X_test)
    
    # Calculate metrics
    r2_gb = r2_score(y_test, y_pred_gb)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    
    print(f"R² Score: {r2_gb:.4f}")
    print(f"RMSE: {np.sqrt(mse_gb):.2f} EUR")
    print(f"MAE: {mae_gb:.2f} EUR")
    
    # Feature importance from gradient boosting
    print(f"\nGRADIENT BOOSTING FEATURE IMPORTANCE")
    print("-" * 50)
    
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")
    
    # SHAP Analysis
    print(f"\nSHAP ANALYSIS")
    print("-" * 30)
    
    try:
        # Create SHAP explainer for gradient boosting model
        explainer = shap.TreeExplainer(gb_model)
        shap_values = explainer.shap_values(X_test)
        
        # --- SHAP summary beeswarm plot ---
        import matplotlib.pyplot as plt
        # --- Short brand mapping (from descriptive_analysis.py) ---
        brand_short_map = {
            'bosch thermotechnik gmbh': 'Bosch',
            'viessmann climate solutions se': 'Viessmann',
            'ait-deutschland gmbh': 'AIT',
            'daikin europe n.v.': 'Daikin',
            'bosch thermotechnik gmbh (buderus)': 'Buderus',
            'samsung electronics air conditioner europe b.v.': 'Samsung',
            'wolf gmbh': 'Wolf',
            'mitsubishi electric air conditioning systems europe ltd': 'Mitsubishi',
            'panasonic marketing europe gmbh': 'Panasonic',
            'lg electronics inc.': 'LG',
            'johnson controls-hitachi air-conditioning spain': 'Johnson / Hitachi',
            'johnson controls-hitachi airconditionning spain': 'Johnson / Hitachi',
            'johnson controls-hitachi air conditioning spain': 'Johnson / Hitachi',
            'toshiba air conditioning': 'Toshiba',
            'vaillant gmbh': 'Vaillant',
        }
        def short_brand_from_dummy(dummy):
            if dummy.startswith('Brand_'):
                # Remove 'Brand_' prefix and map
                long = dummy[6:].strip().lower()
                # Special case for Johnson / Hitachi
                if 'johnson' in long and 'hitachi' in long:
                    return 'Brand_Johnson / Hitachi'
                return f"Brand_{brand_short_map.get(long, long.title())}"
            return dummy
        # Map feature names for SHAP plots
        short_feature_names = [short_brand_from_dummy(f) for f in X_test.columns]
        # Calculate mean absolute SHAP values
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # SHAP summary plot with short names (only top 10 features)
        # Get indices of top 10 features by mean absolute SHAP value
        top10_idx = np.argsort(mean_shap_values)[-10:][::-1]
        shap_values_top10 = shap_values[:, top10_idx]
        X_test_top10 = X_test.iloc[:, top10_idx]
        short_feature_names_top10 = [short_feature_names[i] for i in top10_idx]
        shap.summary_plot(
            shap_values_top10, X_test_top10, feature_names=short_feature_names_top10, show=False
        )
        plt.xlabel('SHAP value (impact on model output)', labelpad=18, fontweight='bold')
        # Remove y-axis label
        plt.ylabel('')
        # Center the title over the whole figure, reduce vertical distance
        plt.suptitle('SHAP Value Distribution for Top Features', fontsize=14, fontweight='bold', y=0.98, ha='center')
        plt.tight_layout(pad=1.5)
        plt.savefig('reports/figures/linear_regression_analysis/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        # --- End SHAP summary plot ---
        
        # Calculate mean absolute SHAP values
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        shap_importance = pd.DataFrame({
            'feature': available_features,
            'mean_shap_value': mean_shap_values
        }).sort_values('mean_shap_value', ascending=False)
        # Add short feature names for plotting
        shap_importance['feature_short'] = [short_brand_from_dummy(f) for f in shap_importance['feature']]
        
        print("SHAP Feature Importance (Mean |SHAP value|):")
        for i, (_, row) in enumerate(shap_importance.head(10).iterrows()):
            feature_name = row['feature']
            shap_val = row['mean_shap_value']
            
            # Calculate EUR impact
            print(f"{i+1:2d}. {feature_name:<35} {shap_val:.2f} EUR average impact")
        
        # Calculate specific EUR impacts for numerical features
        print(f"\nPRICE IMPACT ANALYSIS")
        print("-" * 40)
        
        numerical_features = ['SCOP', 'Rated Power low T [kW]', 'SPL outdoor high Power [dBA]', 
                            'Max. water heating temperature [°C]']
        
        for feature in numerical_features:
            if feature in available_features:
                feature_idx = available_features.index(feature)
                
                # Calculate correlation between feature values and SHAP values
                feature_values = X_test.iloc[:, feature_idx]
                feature_shap = shap_values[:, feature_idx]
                
                # Linear regression to estimate EUR per unit
                if len(feature_values) > 1 and feature_values.std() > 0:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(feature_values, feature_shap)
                    
                    if feature == 'SCOP':
                        unit = 'SCOP point'
                    elif feature == 'Rated Power low T [kW]':
                        unit = 'kW'
                    elif feature == 'SPL outdoor high Power [dBA]':
                        unit = 'dBA'
                    elif feature == 'Max. water heating temperature [°C]':
                        unit = '°C'
                    else:
                        unit = 'unit'
                    
                    print(f"{feature}:")
                    print(f"  Impact: {slope:.2f} EUR per {unit}")
                    print(f"  Correlation: r = {r_value:.3f}, p = {p_value:.4f}")
        
        return {
            'gb_model': gb_model,
            'shap_explainer': explainer,
            'feature_names': available_features,
            'gb_r2': r2_gb,
            'shap_importance': shap_importance
        }
        
    except Exception as e:
        print(f"[WARNING] SHAP analysis failed: {e}")
        return {
            'gb_model': gb_model,
            'gb_r2': r2_gb
        }

def create_regression_visualizations(df, models_dict, figures_dir):
    """Create visualizations for regression analysis"""
    
    figures_dir.mkdir(exist_ok=True)
    
    # 1. Actual vs Predicted Price Plot (GBM)
    if 'gb_model' in models_dict:
        # Prepare data
        df_reg = prepare_regression_data(df)
        available_features = models_dict.get('feature_names', [])
        X = df_reg[available_features]
        y = df_reg['Price']
        # Make predictions
        y_pred = models_dict['gb_model'].predict(X)
        plt.figure(figsize=(10, 8))
        plt.scatter(y, y_pred, alpha=0.6, color='blue', s=30)
        min_price = min(y.min(), y_pred.min())
        max_price = max(y.max(), y_pred.max())
        plt.plot([min_price, max_price], [min_price, max_price], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Price (EUR)', fontsize=12)
        plt.ylabel('Predicted Price (EUR)', fontsize=12)
        plt.title('Gradient Boosting Model: Actual vs Predicted Price', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        r2 = models_dict.get('gb_r2', 0)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(figures_dir / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
    # 1b. Actual vs Predicted Price Plot (Linear Regression)
    if 'full_model' in models_dict:
        # Use statsmodels linear regression
        full_model = models_dict['full_model']
        X_full = full_model.model.exog
        y_full = full_model.model.endog
        y_pred_full = full_model.predict(X_full)
        plt.figure(figsize=(10, 8))
        plt.scatter(y_full, y_pred_full, alpha=0.6, color='green', s=30)
        min_price = min(y_full.min(), y_pred_full.min())
        max_price = max(y_full.max(), y_pred_full.max())
        plt.plot([min_price, max_price], [min_price, max_price], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Price (EUR)', fontsize=12)
        plt.ylabel('Predicted Price (EUR)', fontsize=12)
        plt.title('Linear Regression: Actual vs Predicted Price', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        r2 = full_model.rsquared
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(figures_dir / 'actual_vs_predicted_linear.png', dpi=300, bbox_inches='tight')
        plt.close()
    # 1c. Actual vs Predicted Price Plot (Log-Linear Regression)
    if 'log_model_sm' in models_dict and 'log_model_data' in models_dict:
        # Use statsmodels log-linear regression with full dataset
        log_model_sm = models_dict['log_model_sm']
        log_data = models_dict['log_model_data']
        X_log = sm.add_constant(log_data['X'])
        y_log = log_data['y_log']
        y_actual = log_data['y']
        y_pred_log = log_model_sm.predict(X_log)
        y_pred_log_linear = np.exp(y_pred_log) - 1
        plt.figure(figsize=(10, 8))
        plt.scatter(y_actual, y_pred_log_linear, alpha=0.6, color='orange', s=30)
        min_price = min(y_actual.min(), y_pred_log_linear.min())
        max_price = max(y_actual.max(), y_pred_log_linear.max())
        plt.plot([min_price, max_price], [min_price, max_price], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Price (EUR)', fontsize=12)
        plt.ylabel('Predicted Price (EUR)', fontsize=12)
        plt.title('Log-Linear Regression: Actual vs Predicted Price', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Calculate R² in original scale
        from sklearn.metrics import r2_score
        r2 = r2_score(y_actual, y_pred_log_linear)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(figures_dir / 'actual_vs_predicted_loglinear.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. SHAP Summary Plot (if available)
    if 'shap_importance' in models_dict:
        plt.figure(figsize=(12, 8))
        # Only show the top 10 features
        shap_data = models_dict['shap_importance'].head(10)
        # Reverse the order for plotting
        shap_data = shap_data.iloc[::-1]
        y_labels = shap_data['feature_short'] if 'feature_short' in shap_data else shap_data['feature']
        plt.barh(range(len(shap_data)), shap_data['mean_shap_value'])
        plt.yticks(range(len(shap_data)), y_labels, fontweight='normal', fontsize=13)
        plt.xlabel('Mean |SHAP Value| (EUR)', labelpad=18, fontweight='bold', fontsize=13)
        # Center the title over the whole figure - reduced y position and increased font size
        plt.suptitle('SHAP Feature Importance', fontsize=18, fontweight='bold', y=0.98, ha='center')
        plt.grid(True, alpha=0.3, axis='x')
        # Color bars by importance (reverse colorway to match reversed order)
        colors = plt.cm.viridis(np.linspace(1, 0, len(shap_data)))
        for i, (bar, color) in enumerate(zip(plt.gca().patches, colors)):
            bar.set_color(color)
        plt.tight_layout(pad=1.5)
        plt.savefig(figures_dir / 'shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Residuals Plot (GBM)
    if 'gb_model' in models_dict:
        plt.figure(figsize=(10, 6))
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='blue', s=30)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.xlabel('Predicted Price (EUR)', fontsize=12)
        plt.ylabel('Residuals (EUR)', fontsize=12)
        plt.title('Residuals Plot - Gradient Boosting Model', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'residuals_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    # 3b. Residuals Plot (Linear Regression)
    if 'full_model' in models_dict:
        full_model = models_dict['full_model']
        X_full = full_model.model.exog
        y_full = full_model.model.endog
        y_pred_full = full_model.predict(X_full)
        residuals_full = y_full - y_pred_full
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred_full, residuals_full, alpha=0.6, color='green', s=30)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.xlabel('Predicted Price (EUR)', fontsize=12)
        plt.ylabel('Residuals (EUR)', fontsize=12)
        plt.title('Residuals Plot - Linear Regression', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'residuals_plot_linear.png', dpi=300, bbox_inches='tight')
        plt.close()
    # 3c. Residuals Plot (Log-Linear Regression)
    if 'log_model_sm' in models_dict and 'log_model_data' in models_dict:
        log_model_sm = models_dict['log_model_sm']
        log_data = models_dict['log_model_data']
        X_log = sm.add_constant(log_data['X'])
        y_log = log_data['y_log']
        y_actual = log_data['y']
        y_pred_log = log_model_sm.predict(X_log)
        y_pred_log_linear = np.exp(y_pred_log) - 1
        residuals_log = y_actual - y_pred_log_linear
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred_log_linear, residuals_log, alpha=0.6, color='orange', s=30)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.xlabel('Predicted Price (EUR)', fontsize=12)
        plt.ylabel('Residuals (EUR)', fontsize=12)
        plt.title('Residuals Plot - Log-Linear Regression', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'residuals_plot_loglinear.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"[SUCCESS] Regression visualizations saved to: {figures_dir}")



def main():
    """Main function to run the complete regression analysis"""
    print("Starting Heat Pump Linear Regression Analysis...")
    
    # Load and clean data
    df = load_data()
    df = clean_data(df)
    
    print(f"Loaded {len(df)} records for regression analysis")
    
    # Print data overview
    print(f"\nDATA OVERVIEW:")
    print(f"Total manufacturers: {df['Manufacturer'].nunique()}")
    print(f"Manufacturers: {', '.join(sorted(df['Manufacturer'].unique()))}")
    if 'Configuration_Group' in df.columns:
        print(f"Total configurations: {df['Configuration_Group'].nunique()}")
        print(f"Configurations: {', '.join(sorted(df['Configuration_Group'].unique()))}")
        print(f"Configuration distribution:")
        config_counts = df['Configuration_Group'].value_counts()
        for config, count in config_counts.items():
            print(f"  {config}: {count} units ({count/len(df)*100:.1f}%)")
    if 'Refrigerant' in df.columns:
        print(f"Total refrigerants: {df['Refrigerant'].nunique()}")
        print(f"Refrigerants: {', '.join(sorted(df['Refrigerant'].unique()))}")
        print(f"Refrigerant distribution:")
        refrigerant_counts = df['Refrigerant'].value_counts()
        for ref, count in refrigerant_counts.items():
            print(f"  {ref}: {count} units ({count/len(df)*100:.1f}%)")
    
    # 1. Linear regression on full dataset
    full_model, X_full, y_full, features_full = perform_linear_regression_full_dataset(df)
    # 2. Log-linear regression on full dataset
    log_model, X_log_full, y_log_full, y_log_full_data, features_log_full = perform_log_linear_regression_full_dataset(df)
    # 3. Advanced modeling with SHAP
    advanced_models = perform_advanced_modeling(df)
    # --- Ensure all models are available for plotting ---
    models_dict = dict(advanced_models)
    models_dict['full_model'] = full_model
    models_dict['log_model_sm'] = log_model
    models_dict['log_model_data'] = {
        'X': X_log_full,
        'y': y_log_full,
        'y_log': y_log_full_data
    }
    # 4. Create overall visualizations
    analysis_name = Path(__file__).stem
    figures_dir = Path("reports/figures") / analysis_name
    create_regression_visualizations(df, models_dict, figures_dir)
    
    print(f"\n[SUCCESS] Linear regression analysis complete!")
    print(f"Model performance summary:")
    # Calculate R² for log-linear model in original scale
    y_pred_log = log_model.predict(sm.add_constant(X_log_full))
    y_pred_original = np.exp(y_pred_log) - 1
    log_r2_original = r2_score(y_log_full, y_pred_original)
    
    # Calculate RMSE and MAE for linear regression
    y_pred_linear = full_model.predict(sm.add_constant(X_full))
    linear_rmse = np.sqrt(mean_squared_error(y_full, y_pred_linear))
    linear_mae = mean_absolute_error(y_full, y_pred_linear)
    
    # Calculate RMSE and MAE for log-linear regression (in original scale)
    log_rmse = np.sqrt(mean_squared_error(y_log_full, y_pred_original))
    log_mae = mean_absolute_error(y_log_full, y_pred_original)
    
    print(f"   Linear regression R²: {full_model.rsquared:.3f}, RMSE: {linear_rmse:.2f} EUR, MAE: {linear_mae:.2f} EUR")
    print(f"   Log-linear model R²: {log_r2_original:.3f}, RMSE: {log_rmse:.2f} EUR, MAE: {log_mae:.2f} EUR")
    if advanced_models:
        gb_rmse = np.sqrt(mean_squared_error(y_full, advanced_models['gb_model'].predict(X_full)))
        gb_mae = mean_absolute_error(y_full, advanced_models['gb_model'].predict(X_full))
        print(f"   Gradient boosting R²: {advanced_models.get('gb_r2', 0):.3f}, RMSE: {gb_rmse:.2f} EUR, MAE: {gb_mae:.2f} EUR")
    
    return {
        'full_model': full_model,
        'log_model': log_model,
        'advanced_models': advanced_models
    }

if __name__ == "__main__":
    main() 