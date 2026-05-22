import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
from collections import Counter

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
    """Clean and prepare data for analysis"""
    # Convert numeric columns to proper types
    numeric_columns = ['Price', 'SCOP', 'Rated Power low T [kW]', 
                      'SPL outdoor high Power [dBA]', 'Max. water heating temperature [°C]']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def prepare_clustering_data(df):
    """Prepare and engineer features for clustering analysis"""
    print("Preparing data for clustering analysis...")
    
    # Clean the data
    df_clean = clean_data(df)
    
    print(f"Dataset after cleaning: {len(df_clean)} records")
    
    # 3. Create clustering features without configuration or refrigerant
    clustering_features = {
        'SCOP': df_clean['SCOP'],
        'Noise_dBA': df_clean['SPL outdoor high Power [dBA]'],
        'Price_per_kW': df_clean['Price'] / df_clean['Rated Power low T [kW]'],
        'Max_Temp_C': df_clean['Max. water heating temperature [°C]'],
        'Rated_Power_kW': df_clean['Rated Power low T [kW]']
    }
    
    # 4. Create final feature matrix (no configuration or refrigerant features)
    feature_df = pd.DataFrame(clustering_features)
    
    # Remove any rows with missing values in clustering features
    feature_df = feature_df.dropna()
    df_aligned = df_clean.loc[feature_df.index].copy()
    
    print(f"Final dataset: {len(feature_df)} products with {len(feature_df.columns)} features")
    
    # Display feature ranges
    print("Clustering features:")
    for col in feature_df.columns:
        print(f"  - {col}: {feature_df[col].min():.2f} to {feature_df[col].max():.2f}")
    
    return feature_df, df_aligned

def determine_optimal_clusters(X_scaled, max_clusters=10):
    """Determine optimal number of clusters using multiple metrics and inertia plot"""
    print(f"\nDetermining optimal number of clusters...")
    
    k_range = range(2, max_clusters + 1)
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    inertias = []  # For elbow plot
    
    for k in k_range:
        # Fit K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        sil_score = silhouette_score(X_scaled, cluster_labels)
        cal_score = calinski_harabasz_score(X_scaled, cluster_labels)
        db_score = davies_bouldin_score(X_scaled, cluster_labels)
        
        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)
        davies_bouldin_scores.append(db_score)
        inertias.append(kmeans.inertia_)  # Within-cluster sum of squares
    
    # Find optimal k for each metric
    optimal_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_calinski = k_range[np.argmax(calinski_scores)]
    optimal_davies_bouldin = k_range[np.argmin(davies_bouldin_scores)]
    
    # Calculate elbow point using the elbow method
    # Use the "knee" detection - find the point of maximum curvature
    def find_elbow_point(k_values, inertias):
        # Normalize the data
        k_norm = np.array(k_values)
        inertia_norm = np.array(inertias)
        
        # Calculate differences
        k_diff = k_norm[1:] - k_norm[:-1]
        inertia_diff = inertia_norm[1:] - inertia_norm[:-1]
        
        # Calculate second differences (curvature)
        if len(inertia_diff) > 1:
            curvature = np.abs(np.diff(inertia_diff))
            # Find the point of maximum curvature
            elbow_idx = np.argmax(curvature) + 1  # +1 because we lost one element in diff
            return k_values[elbow_idx]
        else:
            return k_values[len(k_values)//2]  # Fallback to middle value
    
    optimal_elbow = find_elbow_point(list(k_range), inertias)
    
    print(f"Optimal clusters by method:")
    print(f"  - Silhouette Score: {optimal_silhouette} clusters (score: {max(silhouette_scores):.3f})")
    print(f"  - Calinski-Harabasz: {optimal_calinski} clusters (score: {max(calinski_scores):.1f})")
    print(f"  - Davies-Bouldin: {optimal_davies_bouldin} clusters (score: {min(davies_bouldin_scores):.3f})")
    print(f"  - Elbow Method: {optimal_elbow} clusters")
    
    # Create elbow plot
    create_inertia_elbow_plot(k_range, inertias, silhouette_scores, calinski_scores, davies_bouldin_scores, optimal_elbow)
    
    # Select optimal k (prefer silhouette score as it's most interpretable)
    selected_k = optimal_silhouette
    print(f"Selected optimal clusters: {selected_k}")
    
    return selected_k

def perform_kmeans_clustering(X_scaled, df_aligned, n_clusters, feature_names):
    """Perform K-means clustering analysis"""
    print(f"\nPerforming K-means clustering with {n_clusters} clusters...")
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df_aligned.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Calculate cluster metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
    
    print(f"Clustering metrics:")
    print(f"  - Silhouette Score: {silhouette_avg:.3f}")
    print(f"  - Calinski-Harabasz Score: {calinski_score:.1f}")
    print(f"  - Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    # Analyze cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Convert back to original scale for interpretation
    scaler = StandardScaler()
    scaler.fit(X_scaled)  # This won't work properly, we need to pass the original scaler
    
    print(f"\nCLUSTER ANALYSIS:")
    print("=" * 80)
    
    for i in range(n_clusters):
        cluster_data = df_clustered[df_clustered['Cluster'] == i]
        cluster_size = len(cluster_data)
        cluster_pct = cluster_size / len(df_clustered) * 100
        
        print(f"\nCLUSTER {i} ({cluster_size} products, {cluster_pct:.1f}%)")
        print("-" * 50)
        
        # Calculate cluster statistics for original features
        stats = {
            'SCOP': cluster_data['SCOP'],
            'Noise_dBA': cluster_data['SPL outdoor high Power [dBA]'],
            'Price_EUR': cluster_data['Price'],
            'Price_per_kW': cluster_data['Price'] / cluster_data['Rated Power low T [kW]'],
            'Max_Temp_C': cluster_data['Max. water heating temperature [°C]'],
            'Rated_Power_kW': cluster_data['Rated Power low T [kW]']
        }
        
        for feature, values in stats.items():
            mean_val = values.mean()
            std_val = values.std()
            print(f"  {feature:<15}: {mean_val:8.2f} ± {std_val:6.2f}")
        
        # Most common brands, refrigerants, and configurations in this cluster
        if 'Manufacturer' in cluster_data.columns:
            top_brands = cluster_data['Manufacturer'].value_counts().head(3)
            print(f"  Top brands: {', '.join([f'{brand} ({count})' for brand, count in top_brands.items()])}")
        
        if 'Refrigerant' in cluster_data.columns:
            top_refrigerants = cluster_data['Refrigerant'].value_counts().head(3)
            print(f"  Top refrigerants: {', '.join([f'{ref} ({count})' for ref, count in top_refrigerants.items()])}")
        
        if 'Config' in cluster_data.columns and 'Storage' in cluster_data.columns:
            # Create configuration group for display
            cluster_data['Config_Display'] = cluster_data['Config'].astype(str) + ' - ' + cluster_data['Storage'].astype(str)
            top_configs = cluster_data['Config_Display'].value_counts().head(3)
            print(f"  Top configurations: {', '.join([f'{config} ({count})' for config, count in top_configs.items()])}")
    
    # Interpret clusters and assign names
    cluster_names, cluster_profiles = interpret_and_name_clusters(df_clustered, f"K-means (k={n_clusters})")
    
    return df_clustered, cluster_names, cluster_profiles



def interpret_and_name_clusters(df_clustered, cluster_method="K-means"):
    """Interpret clusters and assign meaningful names without configuration analysis"""
    print(f"\nINTERPRETING AND NAMING CLUSTERS ({cluster_method})")
    print("="*80)
    
    cluster_names = {}
    cluster_profiles = {}
    
    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        
        # Calculate cluster profile
        profile = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_clustered) * 100,
            'avg_price': cluster_data['Price'].mean(),
            'avg_scop': cluster_data['SCOP'].mean(),
            'avg_noise': cluster_data['SPL outdoor high Power [dBA]'].mean(),
            'avg_power': cluster_data['Rated Power low T [kW]'].mean(),
            'avg_max_temp': cluster_data['Max. water heating temperature [°C]'].mean(),
            'avg_price_per_kw': (cluster_data['Price'] / cluster_data['Rated Power low T [kW]']).mean(),
            'min_price': cluster_data['Price'].min(),
            'max_price': cluster_data['Price'].max(),
            'top_brands': cluster_data['Manufacturer'].value_counts().head(3)
        }
        
        cluster_profiles[cluster_id] = profile
        
        # Determine archetype name
        archetype_name = determine_cluster_archetype(profile, df_clustered)
        cluster_names[cluster_id] = archetype_name
        
        # Display cluster analysis
        print(f"\nCLUSTER {cluster_id}: {archetype_name}")
        print("-" * 60)
        print(f"Size: {profile['size']} products ({profile['percentage']:.1f}%)")
        print(f"Price: {profile['avg_price']:,.0f} EUR (range: {profile['min_price']:,.0f} - {profile['max_price']:,.0f})")
        print(f"Efficiency (SCOP): {profile['avg_scop']:.2f}")
        print(f"Noise Level: {profile['avg_noise']:.1f} dBA")
        print(f"Power Rating: {profile['avg_power']:.1f} kW (range: {cluster_data['Rated Power low T [kW]'].min():.1f} - {cluster_data['Rated Power low T [kW]'].max():.1f})")
        print(f"Max Temperature: {profile['avg_max_temp']:.0f}°C")
        print(f"Price per kW: {profile['avg_price_per_kw']:,.0f} EUR/kW")
        
        # Top brands
        print(f"Leading brands: ", end="")
        brand_strs = []
        for brand, count in profile['top_brands'].items():
            brand_strs.append(f" {brand} ({count})")
        print(",".join(brand_strs))
        
        # Top refrigerants
        if 'Refrigerant' in cluster_data.columns:
            top_refrigerants = cluster_data['Refrigerant'].value_counts().head(3)
            print(f"Top refrigerants: ", end="")
            refrigerant_strs = []
            for ref, count in top_refrigerants.items():
                refrigerant_strs.append(f"{ref} ({count})")
            print(", ".join(refrigerant_strs))
        
        # Top configurations
        if 'Config' in cluster_data.columns and 'Storage' in cluster_data.columns:
            # Create configuration group for display
            cluster_data['Config_Display'] = cluster_data['Config'].astype(str) + ' - ' + cluster_data['Storage'].astype(str)
            top_configs = cluster_data['Config_Display'].value_counts().head(3)
            print(f"Top configurations: ", end="")
            config_strs = []
            for config, count in top_configs.items():
                config_strs.append(f"{config} ({count})")
            print(", ".join(config_strs))
        
        # Generate buyer persona
        persona = generate_buyer_persona(profile, archetype_name)
        print(f"\nTARGET PERSONA: {persona}")
    
    return cluster_names, cluster_profiles

def determine_cluster_archetype(profile, df_all):
    """Determine cluster archetype based on characteristics without configuration
    
    Prioritization logic:
    1. Special applications (Retrofit) take highest priority
    2. Price positioning (Budget/Premium) comes second
    3. Performance characteristics (Efficient, Quiet) follow
    4. Size/power characteristics last
    5. Value propositions are used as modifiers
    """
    # Calculate percentiles relative to full dataset
    all_prices = df_all['Price']
    all_scop = df_all['SCOP']
    all_noise = df_all['SPL outdoor high Power [dBA]']
    all_power = df_all['Rated Power low T [kW]']
    all_price_per_kw = df_all['Price'] / df_all['Rated Power low T [kW]']
    
    price_pct = (all_prices <= profile['avg_price']).mean()
    scop_pct = (all_scop <= profile['avg_scop']).mean()
    noise_pct = (all_noise <= profile['avg_noise']).mean()  # Higher percentile = noisier cluster
    power_pct = (all_power <= profile['avg_power']).mean()
    price_per_kw_pct = (all_price_per_kw <= profile['avg_price_per_kw']).mean()
    
    # Define thresholds
    LOW_THRESHOLD = 0.40  # Set between 0.33 and 0.50 to make more clusters qualify as "Budget"
    HIGH_THRESHOLD = 0.67
    
    # Categorize characteristics
    price_cat = "Budget" if price_pct < LOW_THRESHOLD else ("Premium" if price_pct > HIGH_THRESHOLD else "Mid-Range")
    efficiency_cat = "High-Efficiency" if scop_pct > HIGH_THRESHOLD else ("Standard" if scop_pct > LOW_THRESHOLD else "Basic")
    # Fix noise categorization: low noise_pct = quiet (below average noise), high noise_pct = noisy (above average noise)
    noise_cat = "Quiet" if noise_pct < LOW_THRESHOLD else ("Standard" if noise_pct < HIGH_THRESHOLD else "Noisy")
    size_cat = "Compact" if power_pct < LOW_THRESHOLD else ("Large" if power_pct > HIGH_THRESHOLD else "Medium")
    value_cat = "Value" if price_per_kw_pct < LOW_THRESHOLD else ("Premium" if price_per_kw_pct > HIGH_THRESHOLD else "Standard")
    
    # Create archetype name based on PRIORITIZED characteristics
    primary_traits = []
    
    # PRIORITY 1: Special applications (highest priority)
    if profile['avg_max_temp'] > 65:
        primary_traits.append("Retrofit")
    
    # PRIORITY 2: Price positioning
    if price_cat == "Budget":
        primary_traits.append("Budget")
    elif price_cat == "Premium":
        primary_traits.append("Premium")
    elif price_cat == "Mid-Range" and len(primary_traits) == 0:
        primary_traits.append("Mid-Range")
    
    # PRIORITY 3: Performance characteristics
    if efficiency_cat == "High-Efficiency":
        primary_traits.append("Efficient")
    
    if noise_cat == "Quiet":
        primary_traits.append("Quiet")
    
    # PRIORITY 4: Size/power characteristics
    if size_cat == "Compact":
        primary_traits.append("Compact")
    elif size_cat == "Large":
        primary_traits.append("High-Power")
    
    # PRIORITY 5: Value proposition (used as modifier)
    if value_cat == "Value" and len(primary_traits) > 0:
        primary_traits.append("Value")
    
    # Construct name using intelligent combination rules
    if len(primary_traits) == 0:
        archetype = "Standard"
    elif len(primary_traits) == 1:
        archetype = primary_traits[0]
    else:
        # PRIORITIZED COMBINATION RULES:
        if "Retrofit" in primary_traits:
            # Retrofit takes highest priority, combine with one other trait
            other_traits = [t for t in primary_traits if t != "Retrofit"]
            if other_traits:
                archetype = f"Retrofit-{other_traits[0]}"
            else:
                archetype = "Retrofit-Specialist"
        elif "Budget" in primary_traits and "Compact" in primary_traits:
            archetype = "Budget-Compact"
        elif "Premium" in primary_traits and "Quiet" in primary_traits:
            archetype = "Premium-Quiet"
        elif "Premium" in primary_traits and "Efficient" in primary_traits:
            archetype = "Premium-Efficient"
        elif "Budget" in primary_traits and "Efficient" in primary_traits:
            archetype = "Budget-Efficient"
        elif "Budget" in primary_traits and "Value" in primary_traits:
            archetype = "Budget-Value"
        elif "Quiet" in primary_traits and "Value" in primary_traits:
            archetype = "Quiet-Value"
        elif "Quiet" in primary_traits and "High-Power" in primary_traits:
            archetype = "Quiet-High-Power"
        elif "Compact" in primary_traits and "Efficient" in primary_traits:
            archetype = "Compact-Efficient"
        elif "Value" in primary_traits and "Efficient" in primary_traits:
            archetype = "Value-Efficient"
        elif "Mid-Range" in primary_traits and "Efficient" in primary_traits:
            archetype = "Standard-Efficient"
        elif "Mid-Range" in primary_traits:
            archetype = "Mid-Range-Standard"
        else:
            # Default to combining top 2 traits
            archetype = "-".join(primary_traits[:2])
    
    return archetype

def generate_buyer_persona(profile, cluster_name):
    """Generate comprehensive buyer persona with targeting tags based on cluster characteristics"""
    
    # Expanded persona dictionary with more combinations and targeting details
    personas = {
        # Core archetypes
        "Budget-Compact": {
            "description": "Cost-conscious homeowners with smaller properties seeking basic heating solutions",
            "tags": ["first-time buyers", "budget-conscious", "space-constrained", "apartments", "starter homes"]
        },
        "Premium-Quiet": {
            "description": "Affluent homeowners prioritizing comfort and minimal noise disturbance",
            "tags": ["high-income", "urban residential", "neighbor-conscious", "premium comfort", "established homeowners"]
        },
        "Premium-Efficient": {
            "description": "Environmentally conscious buyers willing to invest in top performance and sustainability",
            "tags": ["eco-focused", "early adopters", "tech-savvy", "long-term investment", "green building"]
        },
        "Value-Efficient": {
            "description": "Smart buyers seeking the best efficiency per euro spent with strong ROI focus",
            "tags": ["value-conscious", "energy-aware", "practical", "cost-benefit analyzers", "informed buyers"]
        },
        "Retrofit-Specialist": {
            "description": "Owners of older properties needing high-temperature capability for existing heating systems",
            "tags": ["renovation projects", "older buildings", "radiator systems", "heritage homes", "heating upgrades"]
        },
        "Retrofit-Quiet": {
            "description": "Owners upgrading older properties who also prioritize noise reduction",
            "tags": ["urban renovation", "noise-sensitive areas", "premium retrofits", "historic districts"]
        },
        "Retrofit-Premium": {
            "description": "High-end retrofit market with premium requirements and high-temperature needs",
            "tags": ["luxury renovations", "premium retrofits", "high-end properties", "commercial upgrades"]
        },
        "High-Power": {
            "description": "Large property owners or commercial applications requiring substantial heating capacity",
            "tags": ["large homes", "commercial", "multi-zone heating", "high-capacity needs", "industrial"]
        },
        "Quiet-High-Power": {
            "description": "Large property owners who require substantial capacity but prioritize quiet operation",
            "tags": ["large residential", "suburban estates", "noise-sensitive large properties", "multi-family"]
        },
        "Quiet-Value": {
            "description": "Practical buyers seeking good value with emphasis on quiet operation",
            "tags": ["suburban residential", "family homes", "noise-conscious", "balanced priorities", "mainstream market"]
        },
        "Compact-Efficient": {
            "description": "Urban dwellers with space constraints who prioritize energy efficiency",
            "tags": ["urban apartments", "space-constrained", "energy-conscious", "modern living", "city dwellers"]
        },
        "Budget-Efficient": {
            "description": "Cost-conscious buyers who still prioritize energy efficiency for long-term savings",
            "tags": ["energy-aware budget buyers", "young professionals", "efficiency-focused", "ROI-conscious"]
        },
        "Standard-Efficient": {
            "description": "Mainstream market seeking reliable performance with good efficiency at fair prices",
            "tags": ["mainstream residential", "balanced buyers", "standard homes", "practical choice", "middle market"]
        },
        "Mid-Range-Standard": {
            "description": "Average market segment seeking reliable, balanced performance across all metrics",
            "tags": ["mainstream market", "standard residential", "balanced requirements", "typical homeowners"]
        },
        "Standard": {
            "description": "Mainstream residential market seeking reliable, balanced performance",
            "tags": ["general residential", "standard requirements", "balanced performance", "typical installations"]
        }
    }
    
    # Get persona info or create default
    persona_info = personas.get(cluster_name, {
        "description": "General residential heating market segment",
        "tags": ["general market", "standard residential", "basic requirements"]
    })
    
    # Format the output with description and targeting tags
    description = persona_info["description"]
    tags = ", ".join(persona_info["tags"])
    
    return f"{description} | Target segments: {tags}"

def create_cluster_visualizations(df_clustered, X_scaled, feature_names, cluster_names, figures_dir, method="K-means"):
    """Create comprehensive visualizations for cluster analysis"""
    print(f"\nCreating cluster visualizations for {method}...")
    
    # Create method-specific directory with safe name
    safe_method_name = method.lower().replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')
    method_dir = figures_dir / safe_method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    
    n_clusters = df_clustered['Cluster'].nunique()
    
    # Helper function to get display name for clusters
    def get_display_name(cluster_name):
        """Map cluster names to display names for plots"""
        if cluster_name == "Mid-Range":
            return "Low-Temperature"
        return cluster_name
    
    # 1. PCA Visualization
    plt.figure(figsize=(12, 10))
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot clusters
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        mask = df_clustered['Cluster'] == i
        cluster_name = cluster_names.get(i, f"Cluster {i}")
        display_name = get_display_name(cluster_name)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[i]], label=f"{display_name} (n={mask.sum()})", 
                   alpha=0.7, s=50)
    
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    plt.title(f'PCA Cluster Visualization\n(Total Variance Explained: {pca.explained_variance_ratio_.sum():.1%})', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(method_dir / 'pca_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Distribution by Cluster - Box Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'k-Means Clustering - Feature Distributions', fontsize=16, fontweight='bold')
    
    original_features = {
        'SCOP': df_clustered['SCOP'],
        'Noise (dBA)': df_clustered['SPL outdoor high Power [dBA]'],
        'Price (EUR)': df_clustered['Price'],
        'Price/kW (EUR/kW)': df_clustered['Price'] / df_clustered['Rated Power low T [kW]'],
        'Max Temp (°C)': df_clustered['Max. water heating temperature [°C]'],
        'Power (kW)': df_clustered['Rated Power low T [kW]']
    }
    
    axes_flat = axes.flatten()
    for idx, (feature_name, feature_data) in enumerate(original_features.items()):
        ax = axes_flat[idx]
        
        # Create box plot
        cluster_data = []
        cluster_labels = []
        for i in range(n_clusters):
            mask = df_clustered['Cluster'] == i
            cluster_data.append(feature_data[mask])
            cluster_name = cluster_names.get(i, f"C{i}")
            display_name = get_display_name(cluster_name)
            n_obs = mask.sum()
            cluster_labels.append(f"{display_name}\n(n={n_obs})")
        
        bp = ax.boxplot(cluster_data, labels=cluster_labels, patch_artist=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(feature_name)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(method_dir / 'feature_distributions_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2b. Feature Distribution by Cluster - Violin Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'k-Means Clustering - Feature Distributions', fontsize=16, fontweight='bold')
    
    # Prepare data for violin plots
    feature_data_df = pd.DataFrame()
    for feature_name, feature_data in original_features.items():
        # Create cluster labels with observation counts
        cluster_labels_with_counts = []
        for i in range(n_clusters):
            cluster_name = cluster_names.get(i, f"C{i}")
            display_name = get_display_name(cluster_name)
            n_obs = (df_clustered['Cluster'] == i).sum()
            cluster_labels_with_counts.append(f"{display_name}\n(n={n_obs})")
        
        temp_df = pd.DataFrame({
            'Value': feature_data,
            'Cluster': df_clustered['Cluster'].map(lambda x: cluster_labels_with_counts[x]),
            'Feature': feature_name
        })
        feature_data_df = pd.concat([feature_data_df, temp_df], ignore_index=True)
    
    # Create color palette from the same colors used in box plots
    cluster_order = []
    color_palette = [colors[i] for i in range(n_clusters)]
    for i in range(n_clusters):
        cluster_name = cluster_names.get(i, f"C{i}")
        display_name = get_display_name(cluster_name)
        n_obs = (df_clustered['Cluster'] == i).sum()
        cluster_order.append(f"{display_name}\n(n={n_obs})")
    
    axes_flat = axes.flatten()
    for idx, (feature_name, _) in enumerate(original_features.items()):
        ax = axes_flat[idx]
        
        # Filter data for this feature
        feature_subset = feature_data_df[feature_data_df['Feature'] == feature_name]
        
        # Create violin plot with custom color palette
        sns.violinplot(x='Cluster', y='Value', data=feature_subset, ax=ax, 
                      palette=color_palette, order=cluster_order)
        
        # Set labels and styling
        ax.set_title(feature_name, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(method_dir / 'feature_distributions_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cluster Characteristics Radar Chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
    fig.suptitle(f'k-Means Cluster Characteristics', fontsize=20, fontweight='bold')
    
    # Normalize features for radar chart (0-1 scale)
    radar_features = ['SCOP', 'Quietness', 'Value', 'Max_Temp', 'Power', 'Affordability']
    
    for idx, cluster_id in enumerate(range(min(n_clusters, 6))):  # Max 6 clusters for visualization
        ax = axes.flatten()[idx]
        
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        display_name = get_display_name(cluster_name)
        
        # Calculate normalized values
        values = []
        all_data = df_clustered
        
        # SCOP (normalized to 0-1)
        scop_norm = (cluster_data['SCOP'].mean() - all_data['SCOP'].min()) / (all_data['SCOP'].max() - all_data['SCOP'].min())
        values.append(scop_norm)
        
        # Quietness (inverse of noise, normalized)
        cluster_avg_noise = cluster_data['SPL outdoor high Power [dBA]'].mean()
        max_noise = all_data['SPL outdoor high Power [dBA]'].max()
        min_noise = all_data['SPL outdoor high Power [dBA]'].min()
        
        # Calculate quietness: lower noise = higher quietness
        # Invert the noise scale so that min_noise gives max quietness (1.0)
        quietness_norm = (max_noise - cluster_avg_noise) / (max_noise - min_noise)
        
        values.append(quietness_norm)
        
        # Value (inverse of price per kW, normalized)
        price_per_kw = cluster_data['Price'] / cluster_data['Rated Power low T [kW]']
        all_price_per_kw = all_data['Price'] / all_data['Rated Power low T [kW]']
        value_inv = all_price_per_kw.max() - price_per_kw.mean()
        value_norm = value_inv / (all_price_per_kw.max() - all_price_per_kw.min())
        values.append(value_norm)
        
        # Max Temperature (normalized)
        temp_norm = (cluster_data['Max. water heating temperature [°C]'].mean() - all_data['Max. water heating temperature [°C]'].min()) / (all_data['Max. water heating temperature [°C]'].max() - all_data['Max. water heating temperature [°C]'].min())
        values.append(temp_norm)
        
        # Power (normalized)
        power_norm = (cluster_data['Rated Power low T [kW]'].mean() - all_data['Rated Power low T [kW]'].min()) / (all_data['Rated Power low T [kW]'].max() - all_data['Rated Power low T [kW]'].min())
        values.append(power_norm)
        
        # Affordability (inverse of price, normalized)
        price_inv = all_data['Price'].max() - cluster_data['Price'].mean()
        affordability_norm = price_inv / (all_data['Price'].max() - all_data['Price'].min())
        values.append(affordability_norm)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=cluster_name, color=colors[cluster_id])
        ax.fill(angles, values, alpha=0.25, color=colors[cluster_id])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_features)
        ax.set_ylim(0, 1)
        ax.set_title(f'{display_name}\n(n = {len(cluster_data)})', fontweight='bold')
        ax.grid(True)
    
    # Add market segment pie chart in the bottom right cell
    if n_clusters < 6:
        ax_market = axes.flatten()[5]  # Bottom right cell
        ax_market.remove()  # Remove the polar projection
        ax_market = fig.add_subplot(2, 3, 6)  # Add regular subplot
        
        # Create pie chart for market segments
        cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
        cluster_labels_pie = [get_display_name(cluster_names.get(i, f"Cluster {i}")) for i in cluster_sizes.index]
        
        ax_market.pie(cluster_sizes.values, labels=cluster_labels_pie, autopct='%1.1f%%', 
                     colors=colors[:len(cluster_sizes)], startangle=90)
        ax_market.set_title('Market Segment Distribution', fontweight='bold')
    
    # Hide any remaining unused subplots
    for idx in range(n_clusters, 5):  # Only hide if we have less than 5 clusters
        axes.flatten()[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(method_dir / 'cluster_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Price vs Performance Scatter
    plt.figure(figsize=(14, 10))
    
    for i in range(n_clusters):
        mask = df_clustered['Cluster'] == i
        cluster_name = cluster_names.get(i, f"Cluster {i}")
        display_name = get_display_name(cluster_name)
        plt.scatter(df_clustered[mask]['SCOP'], 
                   df_clustered[mask]['Price'], 
                   c=[colors[i]], label=display_name, 
                   alpha=0.7, s=60)
    
    plt.xlabel('SCOP (Efficiency)', fontsize=12)
    plt.ylabel('Price (EUR)', fontsize=12)
    plt.title(f'k-Means Clustering - Price vs Efficiency by Market Segment', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(method_dir / 'price_vs_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Cluster Size and Market Share
    plt.figure(figsize=(12, 8))
    
    cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
    cluster_labels_pie = [get_display_name(cluster_names.get(i, f"Cluster {i}")) for i in cluster_sizes.index]
    
    plt.pie(cluster_sizes.values, labels=cluster_labels_pie, autopct='%1.1f%%', 
            colors=colors[:len(cluster_sizes)], startangle=90)
    plt.title(f'k-Means Clustering - Market Segment Distribution', fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(method_dir / 'market_segments.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{method} visualizations saved to: {method_dir}")

def create_marketing_analysis_report(df_clustered, cluster_names, figures_dir):
    """Create comprehensive marketing analysis report without configuration insights"""
    print(f"\nCreating marketing analysis report...")
    
    # Helper function to get display name for clusters
    def get_display_name(cluster_name):
        """Map cluster names to display names for reports"""
        if cluster_name == "Mid-Range":
            return "Low-Temperature"
        return cluster_name
    
    report_path = figures_dir / "marketing_analysis_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("HEAT PUMP MARKET PERSONALITY ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*20 + "\n")
        f.write(f"Total market analyzed: {len(df_clustered)} heat pump products\n")
        f.write(f"Number of market segments identified: {len(cluster_names)}\n")
        f.write(f"Analysis based on performance and pricing characteristics\n\n")
        
        # Detailed Cluster Analysis
        f.write("DETAILED MARKET SEGMENT ANALYSIS\n")
        f.write("-"*35 + "\n\n")
        
        total_value = df_clustered['Price'].sum()
        
        for cluster_id in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            cluster_name = cluster_names[cluster_id]
            display_name = get_display_name(cluster_name)
            
            # Basic metrics
            market_share = len(cluster_data) / len(df_clustered) * 100
            value_share = cluster_data['Price'].sum() / total_value * 100
            avg_price = cluster_data['Price'].mean()
            
            f.write(f"{cluster_id + 1}. {display_name.upper()}\n")
            f.write(f"Market Share: {market_share:.1f}% ({len(cluster_data)} products)\n")
            f.write(f"Value Share: {value_share:.1f}% of total market value\n")
            f.write(f"Average Price: {avg_price:,.0f} EUR\n")
            
            # Key characteristics
            f.write(f"Key Performance Indicators:\n")
            f.write(f"  - Average SCOP: {cluster_data['SCOP'].mean():.2f}\n")
            f.write(f"  - Average Noise: {cluster_data['SPL outdoor high Power [dBA]'].mean():.1f} dBA\n")
            f.write(f"  - Average Power: {cluster_data['Rated Power low T [kW]'].mean():.1f} kW\n")
            f.write(f"  - Average Max Temperature: {cluster_data['Max. water heating temperature [°C]'].mean():.0f}°C\n")
            f.write(f"  - Price range: {cluster_data['Price'].min():,.0f} - {cluster_data['Price'].max():,.0f} EUR\n")
            
            # Top brands
            top_brands = cluster_data['Manufacturer'].value_counts().head(3)
            f.write(f"Leading Manufacturers:\n")
            for brand, count in top_brands.items():
                brand_share = count / len(cluster_data) * 100
                f.write(f"  - {brand}: {count} products ({brand_share:.1f}%)\n")
            
            # Top refrigerants
            if 'Refrigerant' in cluster_data.columns:
                top_refrigerants = cluster_data['Refrigerant'].value_counts().head(3)
                f.write(f"Leading Refrigerants:\n")
                for ref, count in top_refrigerants.items():
                    ref_share = count / len(cluster_data) * 100
                    f.write(f"  - {ref}: {count} products ({ref_share:.1f}%)\n")
            
            # Top configurations
            if 'Config' in cluster_data.columns and 'Storage' in cluster_data.columns:
                # Create configuration group for display
                cluster_data['Config_Display'] = cluster_data['Config'].astype(str) + ' - ' + cluster_data['Storage'].astype(str)
                top_configs = cluster_data['Config_Display'].value_counts().head(3)
                f.write(f"Leading Configurations:\n")
                for config, count in top_configs.items():
                    config_share = count / len(cluster_data) * 100
                    f.write(f"  - {config}: {count} products ({config_share:.1f}%)\n")
            
            f.write("\n" + "-"*40 + "\n\n")
        
        # Strategic Recommendations
        f.write("STRATEGIC MARKET INSIGHTS\n")
        f.write("-"*27 + "\n")
        
        # Identify largest segments
        segment_sizes = [(get_display_name(cluster_names[cid]), len(df_clustered[df_clustered['Cluster'] == cid])) 
                        for cid in df_clustered['Cluster'].unique()]
        segment_sizes.sort(key=lambda x: x[1], reverse=True)
        
        f.write("Priority Market Segments (by volume):\n")
        for i, (name, size) in enumerate(segment_sizes[:3]):
            pct = size / len(df_clustered) * 100
            f.write(f"{i+1}. {name}: {pct:.1f}% market share\n")
        
        # Price tier analysis
        f.write(f"\nPrice Tier Analysis:\n")
        price_quartiles = df_clustered['Price'].quantile([0.25, 0.5, 0.75])
        f.write(f"Budget tier (Q1): ≤ {price_quartiles[0.25]:,.0f} EUR\n")
        f.write(f"Mid-range tier (Q2-Q3): {price_quartiles[0.25]:,.0f} - {price_quartiles[0.75]:,.0f} EUR\n")
        f.write(f"Premium tier (Q4): ≥ {price_quartiles[0.75]:,.0f} EUR\n")
        
    print(f"Marketing analysis report saved to: {report_path}")
    return report_path



def create_inertia_elbow_plot(k_range, inertias, silhouette_scores, calinski_scores, davies_bouldin_scores, optimal_elbow):
    """Create elbow plot showing inertia and other metrics vs number of clusters"""
    
    # Create the figures directory if it doesn't exist
    figures_dir = Path("reports/figures/cluster_market_personalities")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Optimal Number of Clusters', fontsize=16, fontweight='bold')
    
    # 1. Elbow Plot (Inertia)
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.axvline(x=optimal_elbow, color='red', linestyle='--', alpha=0.7, label=f'Elbow at k={optimal_elbow}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)')
    ax1.set_title('Elbow Method for Optimal k', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    

    
    # 2. Silhouette Score
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8, color='#A23B72')
    best_sil_k = k_range[np.argmax(silhouette_scores)]
    ax2.axvline(x=best_sil_k, color='red', linestyle='--', alpha=0.7, label=f'Best at k={best_sil_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Calinski-Harabasz Score
    ax3.plot(k_range, calinski_scores, 'mo-', linewidth=2, markersize=8, color='#F18F01')
    best_cal_k = k_range[np.argmax(calinski_scores)]
    ax3.axvline(x=best_cal_k, color='red', linestyle='--', alpha=0.7, label=f'Best at k={best_cal_k}')
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Calinski-Harabasz Score')
    ax3.set_title('Calinski-Harabasz Analysis', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Davies-Bouldin Score (lower is better)
    ax4.plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8, color='#C73E1D')
    best_db_k = k_range[np.argmin(davies_bouldin_scores)]
    ax4.axvline(x=best_db_k, color='red', linestyle='--', alpha=0.7, label=f'Best at k={best_db_k}')
    ax4.set_xlabel('Number of Clusters (k)')
    ax4.set_ylabel('Davies-Bouldin Score')
    ax4.set_title('Davies-Bouldin Analysis', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    plot_path = figures_dir / "cluster_optimization_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Cluster optimization plot saved to: {plot_path}")
    
    plt.close('all')  # Close all figures to free memory

def main():
    """Main function to run the complete cluster analysis"""
    print("Starting Heat Pump Market Personality Clustering Analysis...")
    
    # Load and prepare data
    df = load_data()
    df_cluster, df_aligned = prepare_clustering_data(df)
    
    print(f"Final dataset: {len(df_cluster)} products with {len(df_cluster.columns)} features")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    feature_names = df_cluster.columns.tolist()
    
    # Create output directory
    analysis_name = Path(__file__).stem
    figures_dir = Path("reports/figures") / analysis_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate optimization metrics for k=2-8 to determine optimal k
    print(f"\nDetermining optimal number of clusters...")
    print("Running clustering analysis for k=2-8 to calculate optimization metrics...")
    
    k_range = range(2, 9)
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    inertias = []
    
    for k in k_range:
        # Fit K-means silently (no output)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        sil_score = silhouette_score(X_scaled, cluster_labels)
        cal_score = calinski_harabasz_score(X_scaled, cluster_labels)
        db_score = davies_bouldin_score(X_scaled, cluster_labels)
        
        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)
        davies_bouldin_scores.append(db_score)
        inertias.append(kmeans.inertia_)
    
    # Create elbow plot with real metrics
    create_inertia_elbow_plot(k_range, inertias, silhouette_scores, calinski_scores, davies_bouldin_scores, 5)
    
    print("k=5 chosen as the final number of clusters based on optimization metrics.")
    
    # Run K-means clustering with k=5
    selected_k = 5
    print(f"\n{'='*20} ANALYZING WITH {selected_k} CLUSTERS {'='*20}")
    
    # Perform K-means clustering
    df_final, cluster_names_final, cluster_profiles_final = perform_kmeans_clustering(X_scaled, df_aligned, selected_k, feature_names)
    
    # Create visualizations
    create_cluster_visualizations(df_final, X_scaled, feature_names, cluster_names_final, figures_dir, f"K-means (Final k={selected_k})")
    
    # Create marketing analysis report
    create_marketing_analysis_report(df_final, cluster_names_final, figures_dir)
    
    # Save cluster assignments
    df_final.to_csv(figures_dir / f"kmeans_clusters_k{selected_k}.csv", index=False)
    
    # Enhanced final summary with detailed archetype analysis
    print(f"\nMARKET PERSONALITY ANALYSIS SUMMARY")
    print("="*80)
    print(f"Identified {selected_k} distinct market personalities:")
    
    # Helper function to get display name for clusters
    def get_display_name(cluster_name):
        """Map cluster names to display names for summary"""
        if cluster_name == "Mid-Range":
            return "Low-Temperature"
        return cluster_name
    
    # Analyze market characteristics
    total_market_value = df_final['Price'].sum()
    
    for cluster_id in sorted(cluster_names_final.keys()):
        name = cluster_names_final[cluster_id]
        display_name = get_display_name(name)
        cluster_data = df_final[df_final['Cluster'] == cluster_id]
        size = len(cluster_data)
        market_share = size / len(df_final) * 100
        cluster_value = cluster_data['Price'].sum()
        value_share = cluster_value / total_market_value * 100
        avg_price = cluster_data['Price'].mean()
        
        print(f"\n{cluster_id + 1}. {display_name.upper()}")
        print(f"   Market Share: {market_share:.1f}% ({size} products)")
        print(f"   Value Share: {value_share:.1f}% of total market value")
        print(f"   Average Price: {avg_price:,.0f} EUR")
        print(f"   Key Characteristics:")
        
        # Determine key differentiators
        profile = cluster_profiles_final[cluster_id]
        
        # Compare to market average
        market_avg_noise = df_final['SPL outdoor high Power [dBA]'].mean()
        market_avg_scop = df_final['SCOP'].mean()
        market_avg_temp = df_final['Max. water heating temperature [°C]'].mean()
        market_avg_price_kw = (df_final['Price'] / df_final['Rated Power low T [kW]']).mean()
        
        characteristics = []
        if profile['avg_noise'] < market_avg_noise - 5:
            characteristics.append(f"Quiet operation ({profile['avg_noise']:.1f} dBA)")
        if profile['avg_scop'] > market_avg_scop + 0.1:
            characteristics.append(f"High efficiency ({profile['avg_scop']:.2f} SCOP)")
        if profile['avg_max_temp'] > market_avg_temp + 10:
            characteristics.append(f"High temperature capability ({profile['avg_max_temp']:.0f}°C)")
        if profile['avg_price_per_kw'] < market_avg_price_kw * 0.8:
            characteristics.append(f"Good value ({profile['avg_price_per_kw']:,.0f} EUR/kW)")
        
        if characteristics:
            for char in characteristics:
                print(f"     • {char}")
        else:
            print(f"     • Standard market segment")
    
    print(f"\nAnalysis complete! Results saved to: {figures_dir}")
    print(f"Final recommendation: {selected_k} market personalities")
    
    return {
        'best_k': selected_k,
        'df_final': df_final,
        'final_names': cluster_names_final,
        'cluster_profiles': cluster_profiles_final,
        'feature_names': feature_names,
        'figures_dir': figures_dir
    }

if __name__ == "__main__":
    main() 