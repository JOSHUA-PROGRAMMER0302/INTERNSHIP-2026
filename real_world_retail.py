"""
Real-World E-Commerce Retail Analysis Pipeline
Complete workflow: Load → Clean → Analyze → Visualize → RFM Segmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 12)

# ============================================================================
# STEP 1: LOADING & DATA INSPECTION
# ============================================================================
print("=" * 90)
print("STEP 1: LOADING & DATA INSPECTION")
print("=" * 90)

# Load Online Retail dataset (e-commerce data)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

try:
    print("\nLoading e-commerce dataset...")
    df = pd.read_excel(url)
    print("✓ Dataset loaded successfully!")
except Exception as e:
    print(f"⚠ Could not load from URL: {e}")
    print("Using sample e-commerce data generation...")
    # Fallback: Create sample e-commerce data
    np.random.seed(42)
    n_rows = 5000
    df = pd.DataFrame({
        'InvoiceNo': ['INV' + str(i).zfill(6) for i in range(n_rows)],
        'StockCode': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], n_rows),
        'Description': np.random.choice(['Widget A', 'Widget B', 'Gadget X', 'Tool Y', 'Item Z'], n_rows),
        'Quantity': np.random.randint(-10, 100, n_rows),
        'InvoiceDate': [datetime(2023, 1, 1) + timedelta(days=x) for x in range(n_rows)],
        'UnitPrice': np.random.uniform(5, 100, n_rows),
        'CustomerID': np.random.choice(range(10000, 11000), n_rows),
        'Country': np.random.choice(['UK', 'USA', 'France', 'Germany', 'Spain', 'Netherlands'], n_rows)
    })

print(f"\n{'─' * 90}")
print("DATASET OVERVIEW")
print(f"{'─' * 90}")
print(f"Shape: {df.shape} (rows × columns)")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\n{'─' * 90}")
print("DATA TYPES & INSPECTION")
print(f"{'─' * 90}")
print("\nData Types:")
print(df.dtypes)

print(f"\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("✓ No missing values detected!")

print(f"\nUnique Values Count:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique values")

# Ensure dates are parsed as datetime
if 'InvoiceDate' in df.columns:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"\n✓ InvoiceDate converted to datetime format")

# ============================================================================
# STEP 2: ADVANCED DATA CLEANING & FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 90)
print("STEP 2: ADVANCED DATA CLEANING & FEATURE ENGINEERING")
print("=" * 90)

print(f"\nInitial dataset: {df.shape[0]} rows")

# Remove rows with negative quantities or prices (returns)
print("\nRemoving invalid transactions...")
print(f"  Rows with negative Quantity: {(df['Quantity'] < 0).sum()}")
print(f"  Rows with negative UnitPrice: {(df['UnitPrice'] < 0).sum()}")

df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
print(f"  After removal: {df.shape[0]} rows")

# Handle missing values in CustomerID
print(f"\nHandling missing values...")
print(f"  Missing CustomerIDs: {df['CustomerID'].isnull().sum()}")
df = df.dropna(subset=['CustomerID'])  # Drop rows with missing CustomerID
df['CustomerID'] = df['CustomerID'].astype(int)

# Handle missing descriptions
if df['Description'].isnull().sum() > 0:
    df['Description'].fillna('Unknown Product', inplace=True)
    print(f"  Missing Descriptions filled: {df['Description'].isnull().sum()}")

print(f"✓ Cleaned dataset: {df.shape[0]} rows")

# Feature Engineering
print(f"\nEngineering new features...")

# 1. TotalRevenue
df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']
print(f"  ✓ TotalRevenue = Quantity × UnitPrice")

# 2. Month
df['Month'] = df['InvoiceDate'].dt.to_period('M')
df['MonthDate'] = df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
print(f"  ✓ Month extracted from InvoiceDate")

# 3. Hour
df['Hour'] = df['InvoiceDate'].dt.hour
print(f"  ✓ Hour extracted from InvoiceDate")

# Additional features
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
df['Date'] = df['InvoiceDate'].dt.date

print(f"\n✓ Feature engineering complete!")
print(f"New features added: TotalRevenue, Month, Hour, DayOfWeek, Date")

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS & BUSINESS METRICS
# ============================================================================
print("\n" + "=" * 90)
print("STEP 3: EXPLORATORY DATA ANALYSIS & BUSINESS METRICS")
print("=" * 90)

# 1. Total Revenue Generated
total_revenue = df['TotalRevenue'].sum()
print(f"\n{'█' * 90}")
print(f"{'█'} 1. TOTAL REVENUE GENERATED")
print(f"{'█'} Total Revenue: £{total_revenue:,.2f}")
print(f"{'█' * 90}")

# 2. Top 5 Best-Selling Products by Quantity
print(f"\n{'█' * 90}")
print(f"{'█'} 2. TOP 5 BEST-SELLING PRODUCTS (BY QUANTITY)")
print(f"{'█' * 90}")

top_products = df.groupby('Description').agg({
    'Quantity': 'sum',
    'TotalRevenue': 'sum'
}).sort_values('Quantity', ascending=False).head(5)

for idx, (product, row) in enumerate(top_products.iterrows(), 1):
    print(f"\n{idx}. {product}")
    print(f"   Total Quantity Sold: {int(row['Quantity']):,} units")
    print(f"   Total Revenue: £{row['TotalRevenue']:,.2f}")

# 3. Monthly Revenue Trends
print(f"\n{'█' * 90}")
print(f"{'█'} 3. MONTHLY REVENUE TRENDS")
print(f"{'█' * 90}")

monthly_revenue = df.groupby('MonthDate')['TotalRevenue'].sum().sort_index()
print(f"\nMonthly Revenue Summary:")
for date, revenue in monthly_revenue.items():
    print(f"  {date.strftime('%Y-%m')}: £{revenue:,.2f}")

print(f"\n✓ EDA and Business Metrics Analysis Complete!")

# ============================================================================
# STEP 4: DATA VISUALIZATION (SALES & TRENDS)
# ============================================================================
print("\n" + "=" * 90)
print("STEP 4: DATA VISUALIZATION - DASHBOARD")
print("=" * 90)

# Create figure with 3 subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('E-Commerce Retail Analysis Dashboard', fontsize=18, fontweight='bold', y=1.00)

# -------- SUBPLOT 1: MONTHLY REVENUE TRENDS --------
print("\nGenerating Monthly Revenue Trend Chart...")
monthly_revenue.plot(ax=axes[0, 0], color='#2E86AB', linewidth=2.5, marker='o', markersize=8)
axes[0, 0].fill_between(range(len(monthly_revenue)), monthly_revenue.values, alpha=0.3, color='#2E86AB')
axes[0, 0].set_title('Monthly Revenue Trends', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Revenue (£)', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# -------- SUBPLOT 2: TOP 5 COUNTRIES BY REVENUE --------
print("Generating Top 5 Countries Revenue Chart...")
country_revenue = df.groupby('Country')['TotalRevenue'].sum().sort_values(ascending=False).head(5)
colors = plt.cm.Set3(np.linspace(0, 1, len(country_revenue)))
country_revenue.plot(kind='bar', ax=axes[0, 1], color=colors, edgecolor='black', linewidth=1.5)
axes[0, 1].set_title('Top 5 Countries by Total Revenue', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Country', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Revenue (£)', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(country_revenue.values):
    axes[0, 1].text(i, v, f'£{v:,.0f}', ha='center', va='bottom', fontweight='bold')

# -------- SUBPLOT 3: SALES DISTRIBUTION BY HOUR --------
print("Generating Hourly Sales Distribution Chart...")
hourly_sales = df.groupby('Hour')['TotalRevenue'].sum()
axes[1, 0].bar(hourly_sales.index, hourly_sales.values, color='#A23B72', edgecolor='black', linewidth=1.2)
axes[1, 0].set_title('Sales Distribution by Hour of Day', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Revenue (£)', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(range(0, 24, 2))
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Identify peak shopping hours
peak_hour = hourly_sales.idxmax()
peak_revenue = hourly_sales.max()
axes[1, 0].axvline(x=peak_hour, color='red', linestyle='--', linewidth=2, label=f'Peak Hour: {peak_hour}:00')
axes[1, 0].legend(fontsize=10, loc='upper left')

# -------- SUBPLOT 4: QUANTITY SOLD BY DAY OF WEEK --------
print("Generating Day of Week Sales Chart...")
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_quantity = df.groupby('DayOfWeek')['Quantity'].sum().reindex(dow_order)
colors_dow = plt.cm.Spectral(np.linspace(0, 1, len(dow_quantity)))
axes[1, 1].bar(range(len(dow_quantity)), dow_quantity.values, color=colors_dow, edgecolor='black', linewidth=1.2)
axes[1, 1].set_xticks(range(len(dow_quantity)))
axes[1, 1].set_xticklabels(dow_quantity.index, rotation=45)
axes[1, 1].set_title('Sales Quantity by Day of Week', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Total Quantity', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save dashboard
dashboard_path = 'retail_sales_dashboard.png'
plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Dashboard saved to: {dashboard_path}")
plt.close()

# ============================================================================
# STEP 5: RFM (Recency, Frequency, Monetary) CUSTOMER SEGMENTATION
# ============================================================================
print("\n" + "=" * 90)
print("STEP 5: RFM CUSTOMER SEGMENTATION ANALYSIS")
print("=" * 90)

# Reference date (most recent date in dataset + 1 day)
reference_date = df['InvoiceDate'].max() + timedelta(days=1)
print(f"\nReference Date: {reference_date.strftime('%Y-%m-%d')}")

# Calculate RFM metrics
print(f"\nCalculating RFM metrics...")
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',  # Frequency
    'TotalRevenue': 'sum'  # Monetary
}).round(2)

rfm.columns = ['Recency', 'Frequency', 'Monetary']
print(f"✓ RFM metrics calculated for {len(rfm)} unique customers")

# Assign RFM Scores (1-5, where 5 is best)
print(f"\nAssigning RFM scores (1-5 scale)...")

# Recency: Lower is better (recently purchased)
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')

# Frequency: Higher is better (more purchases)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Monetary: Higher is better (more spending)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Convert scores to numeric
rfm['R_Score'] = rfm['R_Score'].astype(int)
rfm['F_Score'] = rfm['F_Score'].astype(int)
rfm['M_Score'] = rfm['M_Score'].astype(int)

# Combine into RFM Score
rfm['RFM_Score'] = (rfm['R_Score'].astype(str) + 
                    rfm['F_Score'].astype(str) + 
                    rfm['M_Score'].astype(str))

# Calculate overall RFM value (average of R, F, M scores)
rfm['RFM_Value'] = (rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']) / 3

# Sort by RFM Value
rfm_sorted = rfm.sort_values('RFM_Value', ascending=False)

print(f"\n{'█' * 90}")
print(f"{'█'} TOP 10 HIGHEST-VALUE CUSTOMER PROFILES")
print(f"{'█' * 90}")

for idx, (customer_id, row) in enumerate(rfm_sorted.head(10).iterrows(), 1):
    print(f"\n{idx}. Customer ID: {int(customer_id)}")
    print(f"   RFM Score: {row['RFM_Score']}")
    print(f"   Recency (days since last purchase): {int(row['Recency'])} days (Score: {int(row['R_Score'])}/5)")
    print(f"   Frequency (number of purchases): {int(row['Frequency'])} transactions (Score: {int(row['F_Score'])}/5)")
    print(f"   Monetary (total spend): £{row['Monetary']:,.2f} (Score: {int(row['M_Score'])}/5)")
    print(f"   Overall RFM Value: {row['RFM_Value']:.2f}/5")

# RFM Segmentation Categories
print(f"\n{'█' * 90}")
print(f"{'█'} RFM SEGMENTATION SUMMARY")
print(f"{'█' * 90}")

# Define customer segments based on RFM scores
def segment_customer(rfm_score):
    r, f, m = int(rfm_score[0]), int(rfm_score[1]), int(rfm_score[2])
    
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 4 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and m >= 4:
        return "Big Spenders"
    elif r <= 2 and f <= 2:
        return "At Risk"
    elif r <= 2 and m <= 2:
        return "Lost Customers"
    else:
        return "Regular Customers"

rfm['Segment'] = rfm['RFM_Score'].apply(segment_customer)

segment_counts = rfm['Segment'].value_counts()
print(f"\nCustomer Segments Distribution:")
for segment, count in segment_counts.items():
    pct = (count / len(rfm)) * 100
    print(f"  {segment:.<30} {count:>5} customers ({pct:>5.1f}%)")

print(f"\n{'█' * 90}")
print(f"✓ RFM ANALYSIS COMPLETE!")
print(f"{'█' * 90}")

# ============================================================================
# FINAL SUMMARY & INSIGHTS
# ============================================================================
print("\n" + "=" * 90)
print("FINAL SUMMARY & KEY INSIGHTS")
print("=" * 90)

print(f"""
{'─' * 90}
RETAIL BUSINESS METRICS:
{'─' * 90}
  • Total Transactions: {df.shape[0]:,}
  • Total Revenue: £{total_revenue:,.2f}
  • Total Quantity Sold: {df['Quantity'].sum():,} units
  • Unique Customers: {df['CustomerID'].nunique():,}
  • Unique Products: {df['Description'].nunique():,}
  • Countries Served: {df['Country'].nunique()}

{'─' * 90}
CUSTOMER INSIGHTS:
{'─' * 90}
  • Average Customer Lifetime Value: £{rfm['Monetary'].mean():,.2f}
  • Top Customer Spent: £{rfm['Monetary'].max():,.2f}
  • Average Purchase Frequency: {rfm['Frequency'].mean():.1f} transactions
  • Most Recent Purchase: {df['InvoiceDate'].max().strftime('%Y-%m-%d')}

{'─' * 90}
TEMPORAL INSIGHTS:
{'─' * 90}
  • Peak Shopping Hour: {peak_hour}:00 (£{peak_revenue:,.2f})
  • Analysis Period: {df['InvoiceDate'].min().strftime('%Y-%m-%d')} to {df['InvoiceDate'].max().strftime('%Y-%m-%d')}
  • Total Days Analyzed: {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} days

{'─' * 90}
VISUALIZATIONS CREATED:
{'─' * 90}
  ✓ {dashboard_path}
    - Monthly revenue trends
    - Top 5 countries by revenue
    - Hourly sales distribution
    - Day-of-week sales patterns

{'─' * 90}
NEXT STEPS:
{'─' * 90}
  • Use RFM segments for targeted marketing campaigns
  • Focus retention efforts on "At Risk" and "Lost Customers"
  • Capitalize on peak shopping hours with targeted promotions
  • Investigate top countries for expansion opportunities
{'─' * 90}
""")

print("✅ E-COMMERCE RETAIL ANALYSIS PIPELINE COMPLETE!")
print("=" * 90)
