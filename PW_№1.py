import pandas as pd
import numpy as np
import time
from scipy import stats

print("Формування датасету...")
n = 250000
data = {
    'order_id': np.arange(n),
    'order_date': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, n), unit='D'),
    'city': np.random.choice([' Kyiv ', 'Lviv', ' odesa', 'Dnipro ', 'Kharkiv'], n),
    'product': np.random.choice([' Laptop ', 'Mouse', ' Keyboard', 'Monitor', 'Table'], n),
    'price': np.random.lognormal(mean=5, sigma=1, size=n),
    'quantity': np.random.randint(1, 10, n).astype(float),
    'discount': np.random.uniform(0, 0.3, n),
    'rating': np.random.uniform(0, 6, n)
}
df = pd.DataFrame(data)


df.loc[np.random.choice(df.index, 10000), ['quantity', 'discount']] = np.nan
df.iloc[:500] = df.iloc[500:1000].values
df['total'] = df['quantity'] * df['price'] * (1 - df['discount']) * np.random.choice([1, 1, 1, 0.5], n)


print("\n1. Звіт по пропускам:")
missing = df.isnull().agg(['sum', 'mean']).T
missing['mean'] *= 100
print(missing)

check_before = df['total'].mean()
df['quantity'] = df.groupby('product')['quantity'].transform(lambda x: x.fillna(x.median()))
df['discount'] = df.groupby('product')['discount'].transform(lambda x: x.fillna(x.median()))

# --- 2. ОПТИМІЗАЦІЯ (CITY, PRODUCT) ---
mem_before = df.memory_usage(deep=True).sum()
t1 = time.time()
df.groupby('city')['order_id'].count()
time_before = time.time() - t1

df['city'] = df['city'].str.strip().str.title().astype('category')
df['product'] = df['product'].str.strip().str.title().astype('category')

mem_after = df.memory_usage(deep=True).sum()
t2 = time.time()
df.groupby('city')['order_id'].count()
time_after = time.time() - t2
print(f"2. Економія пам'яті: {(mem_before - mem_after)/1024**2:.2f} MB. Прискорення groupby: {time_before/time_after:.2f}x")


df['order_date'] = pd.to_datetime(df['order_date'])
subset = df[df['city'].isin(['Kyiv', 'Lviv'])]
best_month = subset.groupby(subset['order_date'].dt.month)['total'].mean().idxmax()
print(f"3. Місяць з макс. середнім total: {best_month}")


rev_before = df.groupby('city', observed=True)['total'].sum()
df = df.drop_duplicates(subset=['order_id'])
rev_after = df.groupby('city', observed=True)['total'].sum()
print("4. Зміна виручки після видалення дублікатів (перші 3 міста):")
print((rev_after - rev_before).head(3))


expected_total = df['quantity'] * df['price'] * (1 - df['discount'])
diff = np.abs(df['total'] - expected_total)
mae = diff.mean()
df['total'] = expected_total
print(f"5. Середня помилка (MAE) в total до виправлення: {mae:.2f}")


Q1, Q3 = df['price'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_no_outliers = df[~((df['price'] < Q1 - 1.5 * IQR) | (df['price'] > Q3 + 1.5 * IQR))]
pivot = df_no_outliers.pivot_table(index='city', columns='product', values='total', aggfunc='sum')

df.loc[(df['rating'] < 1) | (df['rating'] > 5), 'rating'] = np.nan
df['rating'] = df.groupby('product')['rating'].transform(lambda x: x.fillna(x.median()))

df = df.sort_values('order_date')
df['cum_sum'] = df['total'].cumsum()
half_rev_date = df[df['cum_sum'] >= (df['total'].sum() * 0.5)]['order_date'].iloc[0]
print(f"8. Дата 50% виручки: {half_rev_date.date()}")

catalog = pd.DataFrame({
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'cost_price': [800, 10, 20, 150]
})
df = df.merge(catalog, on='product', how='left')
missing_catalog = df['cost_price'].isna().mean()
print(f"9. Частка рядків без собівартості: {missing_catalog:.2%}")

prod_stats = df.groupby('product', observed=True)['total'].agg(['count','sum','mean','median','std']).sort_values('sum', ascending=False)
top_5 = prod_stats.head(5)

p = df['price'].values
p_norm = (p - p.min()) / (p.max() - p.min())
df['price_norm'] = p_norm
print(f"11. Діапазон нормування: [{p_norm.min()}, {p_norm.max()}]. Std change: {df['price'].std():.2f} -> {p_norm.std():.4f}")

z = (p - p.mean()) / p.std()
outliers_z = np.sum(np.abs(z) > 3)
print(f"12. Викидів Z-score (>3 std): {outliers_z}. Викидів IQR (п.6): {len(df) - len(df_no_outliers)}")

price_64 = df['price'].mean()
df['price'] = df['price'].astype('float32')
print(f"14. Похибка float64 vs float32: {abs(price_64 - df['price'].mean()):.8f}")

cols = ['price', 'quantity', 'discount', 'rating']
corr_np = np.corrcoef(df[cols].dropna(), rowvar=False)
corr_pd = df[cols].corr().values
print(f"15. Узгодженість кореляцій (Max Diff): {np.abs(corr_np - corr_pd).max():.2e}")

X = df[['price', 'quantity', 'discount']].values
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
X_bias = np.c_[np.ones(len(X_std)), X_std]
y = df['total'].values
theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
print(f"17. Коефіцієнти регресії (W0, W_price, W_qty, W_disc): {np.round(theta, 2)}")

daily_rev = df.groupby('order_date')['total'].sum()
roll_pd = daily_rev.rolling(window=7).mean()

def numpy_rolling_mean(a, n=7):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

roll_np = numpy_rolling_mean(daily_rev.values)
print(f"18. Макс різниця Rolling (PD vs NP): {np.abs(roll_pd.values[6:] - roll_np).max():.2e}")

df['month'] = df['order_date'].dt.month
seasonal_pivot = df.pivot_table(index='month', columns='city', values='total', aggfunc='sum', observed=True)
seasonality = (seasonal_pivot.max() - seasonal_pivot.min())
print(f"19. Місто з найбільшою сезонністю: {seasonality.idxmax()}")

def get_dqi(d):
    metrics = [
        d.isnull().mean().mean(),
        d.duplicated().mean(),
        ((d['rating'] < 1) | (d['rating'] > 5)).mean() if 'rating' in d else 0
    ]
    return 1 - np.mean(metrics)

print(f"20. Індекс якості (DQI) в кінці пайплайну: {get_dqi(df):.4f}")