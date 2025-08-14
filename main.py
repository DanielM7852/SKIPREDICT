# main.py - Advanced Skip Prediction with Contextual Features
#AVIV CHANGE
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import xgboost as xgb
import shap
from dotenv import load_dotenv
import logging

# --------------------------
# 1. Enhanced Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('spotify_skip_advanced.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

# --------------------------
# 2. Data Collection with New Features
# --------------------------
def get_enhanced_track_data(playlist_id: str, user_type: str = 'premium') -> pd.DataFrame:
    """Fetch tracks with extended features including contextual data"""
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv('SPOTIFY_CLIENT_ID'),
        client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
    ))
    
    try:
        playlist = sp.playlist_tracks(playlist_id)
        tracks = []
        
        for item in playlist['items']:
            track = item['track']
            if not track:
                continue
                
            # Audio features
            features = sp.audio_features(track['id'])[0]
            
            # NEW: Artist popularity and genre
            artist = sp.artist(track['artists'][0]['id'])
            
            tracks.append({
                # Core features
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'duration_ms': track['duration_ms'],
                
                # Audio features
                'danceability': features['danceability'],
                'energy': features['energy'],
                'key': features['key'],
                'loudness': features['loudness'],
                'tempo': features['tempo'],
                'time_signature': features['time_signature'],
                
                # NEW: Contextual features
                'artist_popularity': artist['popularity'],
                'artist_genres': ','.join(artist['genres'][:3]),
                'release_year': int(track['album']['release_date'][:4]),
                'explicit': int(track['explicit']),
                'user_type': user_type,  # 'premium' or 'free'
                
                # NEW: Playlist context
                'position_in_playlist': item['position'],
                'album_track_number': track['track_number']
            })
            
        return pd.DataFrame(tracks)
    
    except Exception as e:
        logger.error(f"Data collection error: {e}")
        raise

# --------------------------
# 3. Advanced Feature Engineering
# --------------------------
def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features distinguishing skip types with temporal context"""
    np.random.seed(42)
    
    # NEW: Simulate different skip types
    df['skip_time'] = np.random.randint(5000, df['duration_ms'], size=len(df))
    df['skip_type'] = np.where(
        df['skip_time'] > 0.9 * df['duration_ms'],
        'outro_skip',
        np.where(df['skip_time'] < 0.1 * df['duration_ms'], 'intro_skip', 'mid_skip')
    )
    
    # NEW: Temporal features
    df['duration_sec'] = df['duration_ms'] / 1000
    df['intro_length'] = np.random.uniform(5, 15, size=len(df))  # Simulated intro length
    df['outro_length'] = np.random.uniform(5, 30, size=len(df))  # Simulated outro length
    
    # NEW: Density metrics
    df['sections_per_min'] = np.random.randint(3, 10, size=len(df))  # Simulated sections
    df['density_score'] = df['energy'] * df['loudness'] / df['duration_sec']
    
    # NEW: User alignment
    df['genre_match'] = np.random.random(size=len(df))  # Simulated user preference
    df['tempo_deviation'] = abs(df['tempo'] - 120)  # Deviation from common tempo
    
    # NEW: Skip flags
    df['skipped'] = 1  # All are skips in this simulation
    df['early_skip'] = (df['skip_time'] < 30000).astype(int)
    
    return df

# --------------------------
# 4. Scientific Visualization Suite
# --------------------------
def create_advanced_visualizations(df: pd.DataFrame):
    """Generate publication-quality visualizations with slicing"""
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Skip Type Distribution by Feature (Violin Plot)
    plt.subplot(2, 2, 1)
    sns.violinplot(
        x='skip_type', 
        y='tempo', 
        hue='user_type',
        data=df,
        split=True,
        palette={'premium': '#1DB954', 'free': '#FF4D4D'},
        inner="quartile"
    )
    plt.title("Tempo Distribution by Skip Type and User Tier")
    plt.xlabel("Skip Type")
    plt.ylabel("Tempo (BPM)")
    
    # Plot 2: Skip Timing Heatmap
    plt.subplot(2, 2, 2)
    skip_bins = pd.cut(df['skip_time'], bins=np.linspace(0, df['duration_ms'].max(), 20))
    heatmap_data = df.groupby(['skip_type', skip_bins]).size().unstack().fillna(0)
    sns.heatmap(
        heatmap_data.T, 
        cmap='YlOrRd',
        annot=True, 
        fmt='g',
        cbar_kws={'label': 'Skip Count'}
    )
    plt.title("Skip Timing Distribution by Skip Type")
    plt.xlabel("Skip Type")
    plt.ylabel("Time into Song (binned)")
    
    # Plot 3: Conditional Probability Analysis
    plt.subplot(2, 2, 3)
    condition = (df['duration_sec'].between(120, 180)) & (df['genre_match'] > 0.7)
    sns.boxplot(
        x='skip_type',
        y='danceability',
        data=df[condition],
        palette=['#1DB954', '#FF4D4D', '#191414'],
        showfliers=False
    )
    plt.title("Danceability Impact on Skips\n(120-180s Songs, High Genre Match)")
    plt.xlabel("")
    
    # Plot 4: SHAP Summary (Placeholder - Added during modeling)
    plt.subplot(2, 2, 4)
    plt.text(0.3, 0.5, "SHAP Summary Plot Will Appear Here", ha='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/advanced_analysis.png')
    plt.close()

# --------------------------
# 5. Modeling with Skip Type Differentiation
# --------------------------
def train_advanced_models(df: pd.DataFrame):
    """Train models distinguishing skip types"""
    # Feature selection
    numeric_features = [
        'danceability', 'energy', 'loudness', 'tempo', 
        'duration_sec', 'density_score', 'tempo_deviation'
    ]
    categorical_features = ['key', 'time_signature', 'artist_genres']
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Prepare data
    X = df[numeric_features + categorical_features]
    y = df['skip_type']  # Multi-class target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=3,
            random_state=42
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    logger.info("\n=== Skip Type Classification Report ===")
    logger.info(classification_report(y_test, pipeline.predict(X_test)))
    
    # SHAP Analysis
    explainer = shap.Explainer(pipeline.named_steps['classifier'])
    X_processed = preprocessor.transform(X_test)
    shap_values = explainer(X_processed)
    
    plt.figure()
    shap.summary_plot(
        shap_values, 
        X_processed, 
        feature_names=numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out()),
        class_names=pipeline.classes_,
        show=False
    )
    plt.tight_layout()
    plt.savefig('plots/shap_summary_advanced.png')
    plt.close()
    
    return pipeline

# --------------------------
# 6. Main Execution
# --------------------------
def main():
    try:
        logger.info("Starting Advanced Skip Analysis...")
        
        # 1. Get enhanced data
        logger.info("Fetching premium user data...")
        df_premium = get_enhanced_track_data('37i9dQZF1DXcBWIGoYBM5M', 'premium')
        logger.info("Fetching free user data...")
        df_free = get_enhanced_track_data('37i9dQZF1DXcBWIGoYBM5M', 'free')
        df = pd.concat([df_premium, df_free])
        
        # 2. Engineer features
        logger.info("Engineering advanced features...")
        df = engineer_advanced_features(df)
        
        # 3. Visualize
        logger.info("Creating scientific visualizations...")
        create_advanced_visualizations(df)
        
        # 4. Model skip types
        logger.info("Training skip-type classifier...")
        model = train_advanced_models(df)
        
        logger.info("Analysis complete! Check the 'plots' directory.")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

if __name__ == "__main__":
    main()