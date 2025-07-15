

##  Project Overview

This project aims to identify and predict future bike accident hotspots in Amsterdam using advanced spatial clustering and machine learning techniques. Leveraging real accident data from 2021–2023, the project provides actionable insights for urban planners, policymakers, and the public to improve cycling safety in the city.

---

##  Objectives

- **Detect spatial clusters (hotspots) of bike accidents** using HDBSCAN.
- **Predict future accident hotspots** by combining spatial and contextual features (weather, road, light, outcome).
- **Visualize accident clusters and predictions** with both static and interactive maps.
- **Aggregate and analyze cluster-level information** to inform safety interventions.
- **Summarize key findings and suggest future improvements** for data-driven urban safety.

---

## 📂 Dataset

- **Source:** `data/cleaned/gdf_ams_21_23.csv`
- **Description:** Contains all bike accidents in Amsterdam (2021–2023) with the following columns:
  - `longitude`, `latitude`
  - `weather_conditions`
  - `road_conditions`
  - `light_conditions`
  - `outcome` (injury severity)
  - `street_name`
  - (and other relevant features)

**Note:** All analyses and visualizations in this project are based solely on this dataset.

---

##  Methodology

1. **Data Preparation & Cleaning**
   - Notebook: [`notebook/01_gdf_ams_cleaning.ipynb`](notebook/01_gdf_ams_cleaning.ipynb)
   - Load and clean the dataset, handle missing values, and ensure correct data types.

2. **Exploratory Data Analysis (EDA)**
   - Notebook: [`notebook/02_EDA.ipynb`](notebook/02_EDA.ipynb)
   - Explore spatial and contextual patterns in accident data.
   - Visualize distributions, correlations, and trends.

3. **Spatial Clustering (Hotspot Detection)**
   - Apply HDBSCAN to identify accident clusters/hotspots.
   - Analyze cluster characteristics (center, spread, accident count, common conditions).

4. **Feature Engineering & Prediction**
   - Notebook: [`notebook/03_future_hotspot_prediction.ipynb`](notebook/03_future_hotspot_prediction.ipynb)
   - Encode categorical features (weather, road, light, outcome).
   - Train machine learning models to predict future hotspot risk.

5. **Visualization**
   - **Static Maps:** Matplotlib/Seaborn plots for publication-ready visuals.
   - **Interactive Maps:** Folium maps with:
     - All accident points
     - Cluster centers
     - Interactive popups (risk level, prediction score, accident count, top streets, etc.)
     - Color-coded risk levels (red/orange/yellow)
     - Legends and detailed tooltips
   - Example outputs:
     - [`amsterdam_bike_accidents_cluster.html`](amsterdam_bike_accidents_cluster.html)
     - [`hotspot_clusters_map.html`](hotspot_clusters_map.html)
     - [`future_hotspots_21_23.png`](future_hotspots_21_23.png)

6. **Cluster Analysis & Reporting**
   - Aggregate and summarize cluster-level statistics.
   - Markdown summaries of key findings and recommendations.

---

##  Outputs

- **Cleaned Data:** `data/cleaned/gdf_ams_21_23.csv`
- **Notebooks:**
  - Data cleaning: `notebook/01_gdf_ams_cleaning.ipynb`
  - EDA: `notebook/02_EDA.ipynb`
  - Prediction: `notebook/03_future_hotspot_prediction.ipynb`
- **Python Scripts:** See `python/` directory for reusable functions and analysis scripts.
- **Visualizations:**
  - Static: `future_hotspots_21_23.png`
  - Interactive: `amsterdam_bike_accidents_cluster.html`, `hotspot_clusters_map.html`
- **Cluster Analysis:** Aggregated tables and summaries in notebooks and HTML outputs.

---

##  Key Findings

- Several persistent accident hotspots were identified, often correlated with specific weather, road, and light conditions.
- Machine learning models using spatial and contextual features improved hotspot prediction accuracy.
- Interactive maps enable detailed exploration of risk factors and cluster characteristics.

---

##  Future Improvements

- Incorporate additional features (e.g., traffic volume, time of day, infrastructure data).
- Develop temporal models for real-time or seasonal hotspot prediction.
- Integrate live data feeds for dynamic dashboards.
- Expand to other cities or regions for comparative analysis.

---

## How to Use

1. Clone the repository and install dependencies (see `requirements.txt`).
2. Run the notebooks in order for data cleaning, EDA, and prediction.
3. Explore the generated visualizations and HTML maps for insights.

---

##  License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Data: Gemeente Amsterdam, KNMI, and other open data sources.
- Libraries: HDBSCAN, scikit-learn, Folium, Matplotlib, Seaborn, Pandas, Geopandas.


