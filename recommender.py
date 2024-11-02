#K-means
import numpy as np
import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict

class SignLingoKMeansRecommender:
    def __init__(self, data, n_clusters=3):
        """
        Initialize the K-means clustering based sign language video recommender
        
        Parameters:
        data (pd.DataFrame): DataFrame containing 'video_link', 'word', and optional metadata
        n_clusters (int): Number of clusters to create
        """
        self.nlp = spacy.load('en_core_web_md')
        self.data = data
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)  # For visualization
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Initialize and fit the model
        self._prepare_data()
        self._train_model()
        
    def _prepare_data(self):
        """Prepare data by creating feature vectors"""
        # Get word embeddings
        print("Generating word embeddings...")
        self.data['vector'] = self.data['word'].apply(lambda x: self.nlp(x).vector)
        
        # Create feature matrix
        X = np.vstack(self.data['vector'].values)
        
        # Add additional features if available
        additional_features = self._create_additional_features()
        if additional_features is not None:
            X = np.hstack([X, additional_features])
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(X)
        
        # Create PCA transformation for visualization
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
    def _create_additional_features(self):
        """Create additional features if available in the dataset"""
        features = []
        
        # Add difficulty if available
        if 'difficulty' in self.data.columns:
            features.append(self.data['difficulty'].values.reshape(-1, 1))
            
        # Add normalized views if available
        if 'views' in self.data.columns:
            views = np.log1p(self.data['views'].values).reshape(-1, 1)  # Log transform for better scaling
            features.append(views)
            
        # Add ratings if available
        if 'rating' in self.data.columns:
            features.append(self.data['rating'].values.reshape(-1, 1))
            
        # Add word length as a feature
        word_lengths = self.data['word'].str.len().values.reshape(-1, 1)
        features.append(word_lengths)
        
        if features:
            return np.hstack(features)
        return None
    
    def _train_model(self):
        """Train the K-means clustering model"""
        print("Training K-means model...")
        self.cluster_labels = self.kmeans.fit_predict(self.X_scaled)
        self.data['cluster'] = self.cluster_labels
        
        # Create cluster centers dictionary for quick lookup
        self.cluster_centers = {}
        for i in range(self.n_clusters):
            cluster_points = self.X_scaled[self.cluster_labels == i]
            self.cluster_centers[i] = np.mean(cluster_points, axis=0)
    
    def _get_word_features(self, word):
        """Get features for a single word"""
        # Get word embedding
        word_vector = self.nlp(word).vector
        
        # Create additional features if necessary
        additional = []
        if 'difficulty' in self.data.columns:
            additional.append(self.data['difficulty'].mean())
        if 'views' in self.data.columns:
            additional.append(np.log1p(self.data['views'].mean()))
        if 'rating' in self.data.columns:
            additional.append(self.data['rating'].mean())
        additional.append(len(word))
        
        # Combine features
        if additional:
            features = np.hstack([word_vector, additional])
        else:
            features = word_vector
            
        # Scale features
        return self.scaler.transform(features.reshape(1, -1))
    
    def recommend(self, current_word, top_n=5, include_similarity=True):
        """
        Recommend similar sign language videos based on the current word
        
        Parameters:
        current_word (str): The word to base recommendations on
        top_n (int): Number of recommendations to return
        include_similarity (bool): Whether to include similarity scores
        
        Returns:
        pd.DataFrame: Recommended videos with similarity scores
        """
        # Get features for current word
        current_features = self._get_word_features(current_word)
        
        # Find the closest cluster
        cluster_distances = [
            np.linalg.norm(current_features - center.reshape(1, -1))
            for center in self.cluster_centers.values()
        ]
        closest_cluster = np.argmin(cluster_distances)
        
        # Get words from the same cluster
        cluster_words = self.data[self.data['cluster'] == closest_cluster]
        
        # Calculate similarities within the cluster
        similarities = cosine_similarity(
            current_features,
            self.X_scaled[self.data['cluster'] == closest_cluster]
        )[0]
        
        # Create results DataFrame
        results = cluster_words.copy()
        results['similarity_score'] = similarities
        
        # Sort and get top recommendations
        recommendations = results.sort_values(
            by='similarity_score',
            ascending=False
        ).head(top_n)
        
        if include_similarity:
            return recommendations[['video_link', 'word', 'cluster', 'similarity_score']]
        return recommendations[['video_link', 'word', 'cluster']]
    
    # def visualize_clusters(self):
    #     """Visualize the clusters using PCA"""
    #     plt.figure(figsize=(10, 8))
        
    #     # Plot points
    #     scatter = plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
    #                         c=self.cluster_labels, cmap='viridis')
        
    #     # Add word labels
    #     for i, word in enumerate(self.data['word']):
    #         plt.annotate(word, (self.X_pca[i, 0], self.X_pca[i, 1]))
        
    #     # Add cluster centers
    #     centers_pca = self.pca.transform(self.kmeans.cluster_centers_)
    #     plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
    #                c='red', marker='x', s=200, linewidths=3)
        
    #     plt.title('Sign Language Words Clusters')
    #     plt.colorbar(scatter)
    #     plt.show()
        
    def analyze_clusters(self):
        """Analyze the composition of each cluster"""
        cluster_analysis = defaultdict(list)
        
        for cluster in range(self.n_clusters):
            cluster_words = self.data[self.data['cluster'] == cluster]['word'].tolist()
            cluster_analysis[f'Cluster {cluster}'] = {
                'words': cluster_words,
                'size': len(cluster_words)
            }
            
        return cluster_analysis

# Example usage with sample data
sample_data = pd.DataFrame({
    'video_link': [
        'https://www.youtube.com/watch?v=nuYcIMq8e5U', 
        'https://www.youtube.com/watch?v=JC80hJOObmg', 
        'https://www.youtube.com/watch?v=GYVZ3VpzJRI', 
        'https://www.youtube.com/watch?v=fVnCA91Bvwo', 
        'https://www.youtube.com/watch?v=wubfL2VbBLY',
        'https://www.youtube.com/watch?v=mKxgmzevEio',
        'https://www.youtube.com/watch?v=SUGgjeP54CQ',
        'https://www.youtube.com/watch?v=HufgJPpb1kQ',
        'https://www.youtube.com/watch?v=wDy5hBdrwoc'
    ],
    'word': ['apple', 'mango', 'banana', 'dog', 'cat', 'bird', 'car', 'bus', 'train'],
    'difficulty': [1, 2, 2, 3, 2, 2, 3, 3, 4],
    'views': [1000, 800, 900, 1200, 1100, 950, 1500, 1300, 1400],
    'rating': [4.5, 4.2, 4.3, 4.7, 4.6, 4.4, 4.8, 4.5, 4.6]
})

# Initialize recommender
recommender = SignLingoKMeansRecommender(sample_data, n_clusters=3)

# Get recommendations
print("\nRecommendations for 'apple':")
print(recommender.recommend('apple', top_n=3))

# Visualize clusters
#recommender.visualize_clusters()

# Analyze clusters
print("\nCluster Analysis:")
cluster_analysis = recommender.analyze_clusters()
for cluster, info in cluster_analysis.items():
    print(f"\n{cluster}:")
    print(f"Size: {info['size']}")
    print(f"Words: {', '.join(info['words'])}")