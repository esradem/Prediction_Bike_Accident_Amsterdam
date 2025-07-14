# Amsterdam Bike Accident Hotspot Analysis (2021–2023)
# All analysis in this script uses only the gdf_ams_21_23.csv dataset, which contains accident, weather, and injury severity data for Amsterdam (2021–2023).
# No other datasets are loaded or referenced.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler, LabelEncoder
import hdbscan
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# 1. Load the dataset (ONLY gdf_ams_21_23.csv)
df = pd.read_csv('../data/cleaned/gdf_ams_21_23.csv')
print('Data loaded. Shape:', df.shape)
print(df.head())

# 2. Data Exploration
print('\nData Info:')
df.info()
print('\nMissing values:')
print(df.isnull().sum())
print('\nBasic statistics:')
print(df.describe())

# 3. HDBSCAN Clustering
coords = df[['longitude', 'latitude']].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

clusterer = hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean')
df['cluster'] = clusterer.fit_predict(coords_scaled)
print('\nCluster value counts:')
print(df['cluster'].value_counts())

# 4. Visualize Clusters (Matplotlib/Seaborn)
plt.figure(figsize=(10,8))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='cluster', palette='tab20', s=10, legend=None)
plt.title('Bike Accident Clusters in Amsterdam (2021–2023)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('accident_clusters_21_23.png', dpi=200)
plt.close()
print('Static cluster image saved as accident_clusters_21_23.png')

# 5. Visualize Clusters (Folium Interactive Map)
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)
for _, row in df.iterrows():
    folium.CircleMarker([row['latitude'], row['longitude']],
                        radius=3,
                        color='red' if row['cluster']!=-1 else 'gray',
                        fill=True,
                        fill_opacity=0.5,
                        popup=f"Cluster: {row['cluster']}").add_to(marker_cluster)
m.save('accident_clusters_21_23.html')
print('Interactive cluster map saved as accident_clusters_21_23.html')

# 6. Predict Future Hotspots (Spatial Only)
hotspot_counts = df[df['cluster']!=-1]['cluster'].value_counts().head(10)
hotspot_clusters = hotspot_counts.index.tolist()
df['future_hotspot_spatial'] = df['cluster'].apply(lambda x: 1 if x in hotspot_clusters else 0)
print('\nFuture hotspot (spatial only) label counts:')
print(df['future_hotspot_spatial'].value_counts())

# 7. Predict Future Hotspots (Spatial + Weather + Injury)
# Encode categorical columns
df_model = df.copy()
for col in ['weather_conditions', 'road_conditions', 'outcome']:
    if col in df_model.columns:
        le = LabelEncoder()
        df_model[f'{col}_encoded'] = le.fit_transform(df_model[col].astype(str))

# Select features
feature_cols = ['longitude', 'latitude']
for col in ['weather_conditions_encoded', 'road_conditions_encoded', 'outcome_encoded']:
    if col in df_model.columns:
        feature_cols.append(col)

X = df_model[feature_cols]
y = df_model['future_hotspot_spatial']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('\nRandomForest classification report (spatial+weather+injury):')
print(classification_report(y_test, y_pred))
df_model['future_hotspot_pred'] = clf.predict(X)

# 8. Visualize Predicted Future Hotspots (Folium)
m2 = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m2)

def popup_text(row):
    pred = "Future Hotspot" if row['future_hotspot_pred'] == 1 else "Not Hotspot"
    return (
        f"<b>Prediction:</b> {pred}<br>"
        f"<b>Longitude:</b> {row['longitude']}<br>"
        f"<b>Latitude:</b> {row['latitude']}<br>"
        f"<b>Weather:</b> {row.get('weather_conditions', 'N/A')}<br>"
        f"<b>Road:</b> {row.get('road_conditions', 'N/A')}<br>"
        f"<b>Outcome:</b> {row.get('outcome', 'N/A')}"
    )

for _, row in df_model.iterrows():
    color = 'red' if row['future_hotspot_pred'] == 1 else 'blue'
    folium.CircleMarker(
        [row['latitude'], row['longitude']],
        radius=5,  # slightly larger for visibility
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(popup_text(row), max_width=300)
    ).add_to(marker_cluster)

m2.save('future_hotspots_21_23.html')
print('Predicted future hotspot map saved as future_hotspots_21_23.html')

# 9. Visualize Predicted Future Hotspots (Matplotlib)
plt.figure(figsize=(10,8))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='future_hotspot_pred', palette={1:'red',0:'blue'}, s=10, legend='full')
plt.title('Predicted Future Hotspots (2021–2023)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('future_hotspots_21_23.png', dpi=200)
plt.close()
print('Predicted future hotspot image saved as future_hotspots_21_23.png') 