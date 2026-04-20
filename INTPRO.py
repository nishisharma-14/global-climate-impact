import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, norm
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score


sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# DATA MANIPULATION WITH NumPy & Pandas

print("=" * 65)
print("  SECTION 1: DATA LOADING & CLEANING")
print("=" * 65)

# ---------- Load Dataset ----------
df = pd.read_csv("global_warming_dataset.csv")
print(f"\n Dataset loaded successfully.")
print(f"       Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ----------Inspect Dataset ----------
print("--- First 5 Rows ---")
print(df.head())

print("\n--- Column Data Types ---")
print(df.dtypes)


# ---------- Handle Missing Values ----------
print("\n--- Missing Values Per Column ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "  No missing values found.")


# ---------- After missing values check — ADD THIS ----------
co2_arr  = np.array(df['CO2_Emissions'])
co2_norm = (co2_arr - co2_arr.min()) / (co2_arr.max() - co2_arr.min())
df['CO2_Normalized']    = co2_norm
df['Decade']            = (df['Year'] // 10) * 10
df['Era']               = np.where(df['Year'] < 1980, 'Pre-1980', 'Post-1980')
df['Emission_Category'] = pd.cut(df['CO2_Normalized'],
                                  bins=[-np.inf, 0.33, 0.66, np.inf],
                                  labels=['Low', 'Medium', 'High'])


print("\n Data cleaning and preparation complete.\n")

# ─────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE 2 : DATA VISUALIZATION — Matplotlib & Seaborn
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  SECTION 2: DATA VISUALIZATION")
print("=" * 65)
 
# ── Plot 1 : Temperature Anomaly over Years (Matplotlib line) ───────
yearly = df.groupby('Year')['Temperature_Anomaly'].mean().reset_index()
plt.figure(figsize=(12, 5))
plt.plot(yearly['Year'], yearly['Temperature_Anomaly'],
         color='crimson', linewidth=2, label='Avg Temp Anomaly')
plt.axhline(0, color='grey', linestyle='--', linewidth=1, label='Baseline 0 C')
plt.fill_between(yearly['Year'], yearly['Temperature_Anomaly'], 0,
                 alpha=0.18, color='crimson')
plt.title('Global Temperature Anomaly Trend  (1900 - 2023)',
          fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (C)')
plt.legend()
plt.tight_layout()
plt.show()

 
# ── Plot 2 : CO2 Emissions Distribution (Matplotlib histogram) ──────
plt.figure(figsize=(11, 5))
plt.hist(df['CO2_Emissions'], bins=60, color='steelblue',
         edgecolor='white', alpha=0.85)
plt.title('Distribution of CO2 Emissions', fontsize=14, fontweight='bold')
plt.xlabel('CO2 Emissions')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
 

# ── Plot 3 : Seaborn Heatmap — Correlation Matrix ───────────────────
key_cols = ['Temperature_Anomaly', 'CO2_Emissions', 'Methane_Emissions',
            'Sea_Level_Rise', 'Fossil_Fuel_Usage', 'Renewable_Energy_Usage',
            'Deforestation_Rate', 'Policy_Score', 'Average_Temperature',
            'Air_Pollution_Index']
plt.figure(figsize=(12, 8))
sns.heatmap(df[key_cols].corr(), annot=True, fmt=".2f",
            cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
plt.title('Correlation Matrix — Global Warming Features',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
 
# ── Plot 4 : Boxplot — Temp Anomaly by Emission Category (Seaborn) ──
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x='Emission_Category', y='Temperature_Anomaly',
            hue='Emission_Category',
            palette={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
            legend=False)

plt.title('Temperature Anomaly by CO2 Emission Category',
          fontsize=14, fontweight='bold')
plt.xlabel('Emission Category')
plt.ylabel('Temperature Anomaly (C)')
plt.tight_layout()
plt.show()
 
print("\n All visualizations displayed.\n")


 
print("=" * 65)
print("  SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 65)


eda_cols = ['Temperature_Anomaly', 'CO2_Emissions', 'Sea_Level_Rise',
            'Methane_Emissions', 'Fossil_Fuel_Usage', 'Policy_Score',
            'Average_Temperature']


# ─────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE 3 : EDA — Summary Stats, Correlation, Outliers
# ─────────────────────────────────────────────────────────────────────────────


# ── 3.1 Summary Statistics ──────────────────────────────────────────
print("\n── Summary Statistics ──")
print(df[eda_cols].describe().round(3))
 
# ── 3.2 Skewness ───────────────────────────────────────────────────
print("\n── Skewness ──")
print(df[eda_cols].skew().round(4))
 
# ── 3.3 Correlation Ranking with Temperature_Anomaly ───────────────
print("\n── Correlations with Temperature_Anomaly (ranked) ──")
corr_rank = df[eda_cols].corr()['Temperature_Anomaly'].drop('Temperature_Anomaly')
print(corr_rank.sort_values(ascending=False).round(4))

# ── 3.4 Covariance ────────────────────────────────────────────────
print("\n── Covariance Matrix (3 key variables) ──")
print(df[['Temperature_Anomaly', 'CO2_Emissions',
           'Sea_Level_Rise']].cov().round(4))
 
# ── 3.5 Outlier Detection — IQR ────────────────────────────────────
print("\n── Outlier Detection (IQR Method) ──")
outlier_cols = ['Temperature_Anomaly', 'CO2_Emissions',
                'Sea_Level_Rise', 'Methane_Emissions']
for col in outlier_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out  = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"  {col:<28}: {n_out:>5} outliers   fence [{lo:.2f}, {hi:.2f}]")
 
# ── Plot 7 : Outlier Boxplots ───────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, col in zip(axes, outlier_cols):
    ax.boxplot(df[col], patch_artist=True,
               boxprops=dict(facecolor='#3498db', color='navy'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(col.replace('_', '\n'), fontsize=9, fontweight='bold')
plt.suptitle('Outlier Boxplots — Key Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
 
print("\n[OBJ 3]  Complete\n")


# ─────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE 4 : STATISTICAL ANALYSIS
#  Tests chosen because they suit this dataset:
#  -> T-test        : did CO2 levels differ Pre vs Post-1980?
# ─────────────────────────────────────────────────────────────────────────────


print("=" * 65)
print("  OBJECTIVE 4 : STATISTICAL ANALYSIS")
print("=" * 65)


#── 4.2  Independent T-Test ─────────────────────────────────────────
print("\n── 4.2  T-Test : CO2 Emissions — Pre-1980 vs Post-1980 ──")
pre  = df[df['Era'] == 'Pre-1980' ]['CO2_Emissions'].sample(1000, random_state=1)
post = df[df['Era'] == 'Post-1980']['CO2_Emissions'].sample(1000, random_state=2)
t_stat, t_p = ttest_ind(pre, post)
print(f"  Pre-1980  mean  : {pre.mean():,.2f}")
print(f"  Post-1980 mean  : {post.mean():,.2f}")
print(f"  T-statistic     : {t_stat:.4f}")
print(f"  p-value         : {t_p:.4e}")
print(f"  Result : {'Significant difference (Reject H0)' if t_p < 0.05 else 'No significant difference (Fail to Reject H0)'}")
 
print("\n[OBJ 4]  Complete\n")




 
# ─────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE 5 : PROBABILITY DISTRIBUTIONS & A/B TESTING
#  Distributions chosen to match actual data columns:
#  -> Normal  : Temperature_Anomaly  (continuous, symmetric)
#  -> Poisson : Extreme_Weather_Events (count data, integer >= 0)
# ─────────────────────────────────────────────────────────────────────────────


print("=" * 65)
print("  OBJECTIVE 5 : PROBABILITY DISTRIBUTIONS & A/B TESTING")
print("=" * 65)
 
 
# 5.1 Normal Distribution — Temperature_Anomaly
print("\n── 5.1  Normal Distribution — Temperature_Anomaly ──")
mu, sigma = df['Temperature_Anomaly'].mean(), df['Temperature_Anomaly'].std()
x_n = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
plt.figure(figsize=(10, 5))
plt.hist(df['Temperature_Anomaly'].sample(5000, random_state=1),
         bins=50, density=True, color='steelblue', alpha=0.6, label='Empirical')
plt.plot(x_n, norm.pdf(x_n, mu, sigma), 'r-', linewidth=2,
         label=f'Normal PDF  mu={mu:.2f}  sigma={sigma:.2f}')
plt.title('Normal Distribution — Temperature Anomaly', fontsize=13, fontweight='bold')
plt.xlabel('Temperature Anomaly (C)');  plt.ylabel('Density')
plt.legend();  plt.tight_layout();  plt.show()
print(f"  mu={mu:.4f}   sigma={sigma:.4f}")
 
 
print("\n[OBJ 5]  Complete\n")



# ─────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE 6 : MACHINE LEARNING — CRISP-DM FRAMEWORK
#  -> Logistic Regression: classify CO2 as High or Low
# ─────────────────────────────────────────────────────────────────────────────


print("=" * 65)
print("  OBJECTIVE 6 : MACHINE LEARNING  (CRISP-DM)")
print("=" * 65)

FEATURES = ['CO2_Emissions', 'Methane_Emissions', 'Sea_Level_Rise',
            'Fossil_Fuel_Usage', 'Renewable_Energy_Usage', 'Policy_Score']
df_ml = df[FEATURES + ['Average_Temperature']].dropna()
X, y  = df_ml[FEATURES], df_ml['Average_Temperature']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)
print(f"\n  Train: {X_tr.shape[0]}  |  Test: {X_te.shape[0]}")
 
# Linear Regression
lr = LinearRegression()
lr.fit(X_tr_sc, y_tr)
y_pred = lr.predict(X_te_sc)
mse = mean_squared_error(y_te, y_pred)
print(f"\n-- Linear Regression --")
print(f"  MSE={mse:.4f}  RMSE={np.sqrt(mse):.4f}  R2={r2_score(y_te, y_pred):.4f}")
for f, c in zip(FEATURES, lr.coef_):
    print(f"  {f:<28}: {c:+.4f}")
 
# Actual vs Predicted plot
plt.figure(figsize=(7, 4))
plt.scatter(y_te, y_pred, alpha=0.3, color='steelblue', s=15)
plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--', linewidth=2)
plt.title('Linear Regression - Actual vs Predicted')
plt.xlabel('Actual'); plt.ylabel('Predicted')
plt.tight_layout(); plt.show()
 
 


 
# =============================================================================
# SECTION 7 : POLICY SCORE vs EMISSION CATEGORY (Inferential Statistics)
# Unique column: Policy_Score — not used visually anywhere else.
# Syllabus: Descriptive & Inferential Statistics
# =============================================================================
 
# Descriptive: Policy_Score stats per emission group
print("\n-- Policy Score by Emission Category --")
print(df.groupby('Emission_Category')['Policy_Score']
        .agg(['mean', 'median', 'std']).round(3))
 
# Inferential: t-test — does Policy_Score differ between Low and High emitters?
low_policy  = df[df['Emission_Category'] == 'Low' ]['Policy_Score']
high_policy = df[df['Emission_Category'] == 'High']['Policy_Score']
plt.figure(figsize=(9, 4))
plt.hist(low_policy,  bins=40, alpha=0.6, color='#2ecc71', label='Low Emission',  density=True)
plt.hist(high_policy, bins=40, alpha=0.6, color='#e74c3c', label='High Emission', density=True)
plt.axvline(low_policy.mean(),  color='green', linestyle='--', linewidth=1.5, label=f'Low mean={low_policy.mean():.2f}')
plt.axvline(high_policy.mean(), color='red',   linestyle='--', linewidth=1.5, label=f'High mean={high_policy.mean():.2f}')
plt.title('Policy Score Distribution — Low vs High Emitters')
plt.xlabel('Policy Score'); plt.ylabel('Density'); plt.legend()
plt.tight_layout(); plt.show()
 
 
 
 
# =============================================================================
#  OBJECTIVE 8 : FEATURE ENGINEERING & CORRELATION-BASED SELECTION
#  [UNIQUE] Creates 3 meaningful climate ratio features from existing columns,
#  ranks ALL features by correlation with Temperature_Anomaly, and identifies
#  the top predictors — directly improving the ML pipeline in OBJ 6.
# =============================================================================
print("=" * 65)
print("  OBJECTIVE 8 : FEATURE ENGINEERING & CORRELATION-BASED SELECTION")
print("=" * 65)

# 8.1 Engineer 2 ratio features
print("\n── 8.1  Engineered Features ──")
df['CO2_per_GDP']         = df['CO2_Emissions']     / (df['GDP'] + 1)
df['Fossil_to_Renewable'] = df['Fossil_Fuel_Usage'] / (df['Renewable_Energy_Usage'] + 1)
print("  CO2_per_GDP        : emission intensity of the economy")
print("  Fossil_to_Renewable: fossil vs clean energy ratio")
 
# 8.2 Top 8 features by correlation with Temperature_Anomaly
print("\n── 8.2  Top 8 Features Correlated with Temperature_Anomaly ──")
numeric_df = df.select_dtypes(include=[np.number])
top_corr = (numeric_df.corr()['Temperature_Anomaly']
                       .drop('Temperature_Anomaly')
                       .abs()
                       .sort_values(ascending=False)
                       .head(8))
print(top_corr.round(4).to_string())
 
# 8.3 Bar chart of top 8 correlations
plt.figure(figsize=(10, 5))
top_corr.sort_values().plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Top 8 Features — Correlation with Temperature Anomaly',
          fontsize=13, fontweight='bold')
plt.xlabel('Absolute Correlation')
plt.tight_layout();  plt.show()
 
print("\n[OBJ 8]  Complete\n")
