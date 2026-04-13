import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, norm
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import classification_report


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
 
# ── Plot 5 : Scatter — CO2 vs Sea Level Rise by Era (Seaborn) ───────
sample = df.sample(3000, random_state=42)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sample, x='CO2_Emissions', y='Sea_Level_Rise',
                hue='Era',
                palette={'Pre-1980': '#3498db', 'Post-1980': '#e74c3c'},
                alpha=0.55, s=25)
plt.title('CO2 Emissions vs Sea Level Rise  (Pre / Post 1980)',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
 
# ── Plot 6 : Bar — Avg CO2 Emissions by Decade (Seaborn) ────────────
decade_co2 = df.groupby('Decade')['CO2_Emissions'].mean().reset_index()
plt.figure(figsize=(12, 5))
sns.barplot(data=decade_co2, x='Decade', y='CO2_Emissions',
            hue='Decade', palette='YlOrRd',
            edgecolor='black', legend=False)
plt.title('Average CO2 Emissions by Decade',
          fontsize=14, fontweight='bold')
plt.xlabel('Decade')
plt.ylabel('Average CO2 Emissions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
 
print(" 6 plots displayed\n")
 

print("\n All visualizations displayed.\n")


 
print("=" * 65)
print("  SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 65)


eda_cols = ['Temperature_Anomaly', 'CO2_Emissions', 'Sea_Level_Rise',
            'Methane_Emissions', 'Fossil_Fuel_Usage', 'Policy_Score',
            'Average_Temperature']

