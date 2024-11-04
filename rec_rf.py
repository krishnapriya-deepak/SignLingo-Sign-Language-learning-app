import numpy as np
import pandas as pd
import spacy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


#FINAL

class SignLingoRandomForestRecommender:
    def __init__(self, data, n_estimators=100):
        """
        Initialize the Random Forest based sign language video recommender
        
        Parameters:
        data (pd.DataFrame): DataFrame containing 'video_link', 'word', and optional metadata
        n_estimators (int): Number of trees in the random forest
        """
        self.nlp = spacy.load('en_core_web_md')
        self.data = data
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        self.rf_models = {}  # Will store one RF model per feature
        
        # Initialize and fit the model
        self._prepare_data()
        self._train_model()
        
    def _prepare_data(self):
        """Prepare data by creating feature vectors"""
        print("Generating word embeddings...")
        self.data['vector'] = self.data['word'].apply(lambda x: self.nlp(x).vector)
        
        # Create feature matrix
        self.word_embeddings = np.vstack(self.data['vector'].values)
        
        # Add additional features if available
        self.additional_features = self._create_additional_features()
        
        # Combine all features
        if self.additional_features is not None:
            self.X = np.hstack([self.word_embeddings, self.additional_features])
        else:
            self.X = self.word_embeddings
            
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
    def _create_additional_features(self):
        """Create additional features if available in the dataset"""
        features = []
        
        # Add difficulty if available
        if 'difficulty' in self.data.columns:
            features.append(self.data['difficulty'].values.reshape(-1, 1))
            
        # Add normalized views if available
        if 'views' in self.data.columns:
            views = np.log1p(self.data['views'].values).reshape(-1, 1)
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
        """Train a Random Forest model for each feature dimension"""
        print("Training Random Forest models...")
        
        # Train a model for each feature dimension
        for i in range(self.X_scaled.shape[1]):
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=42
            )
            # Use all features to predict each individual feature
            y = self.X_scaled[:, i]
            X = np.delete(self.X_scaled, i, axis=1)
            rf.fit(X, y)
            self.rf_models[i] = rf
    
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
    
    def _calculate_anomaly_scores(self, features):
        """Calculate anomaly scores using Random Forest prediction errors"""
        anomaly_scores = []
        
        for i, rf_model in self.rf_models.items():
            # Remove the target feature
            X = np.delete(features, i, axis=1)
            # Predict the target feature
            y_pred = rf_model.predict(X)
            # Calculate prediction error
            y_true = features[:, i]
            error = np.abs(y_pred - y_true)
            anomaly_scores.append(error)
            
        return np.mean(anomaly_scores, axis=0)
    
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
        
        # Calculate anomaly scores for all words
        anomaly_scores = self._calculate_anomaly_scores(self.X_scaled)
        
        # Calculate similarities
        similarities = cosine_similarity(current_features, self.X_scaled)[0]
        
        # Combine similarity and anomaly scores
        # Higher similarity and lower anomaly score is better
        combined_scores = similarities - anomaly_scores
        
        # Create results DataFrame
        results = self.data.copy()
        results['similarity_score'] = combined_scores
        
        # Sort and get top recommendations
        recommendations = results.sort_values(
            by='similarity_score',
            ascending=False
        ).head(top_n)
        
        if include_similarity:
            return recommendations[['video_link', 'word', 'similarity_score']]
        return recommendations[['video_link', 'word']]
    
    def analyze_importance(self):
        """Analyze feature importance across all Random Forest models"""
        feature_importance = np.zeros(self.X_scaled.shape[1])
        
        for i, rf_model in self.rf_models.items():
            # Get feature importance for the current model
            # Note: we need to adjust the indices since each model excludes one feature
            importance = rf_model.feature_importances_
            # Insert 0 at position i (the target feature for this model)
            importance = np.insert(importance, i, 0)
            feature_importance += importance
            
        # Average the importance scores
        feature_importance /= len(self.rf_models)
        
        # Create feature names
        feature_names = [f'embedding_{i}' for i in range(self.word_embeddings.shape[1])]
        if self.additional_features is not None:
            additional_names = []
            if 'difficulty' in self.data.columns:
                additional_names.append('difficulty')
            if 'views' in self.data.columns:
                additional_names.append('views')
            if 'rating' in self.data.columns:
                additional_names.append('rating')
            additional_names.append('word_length')
            feature_names.extend(additional_names)
            
        return pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)


sample_data=pd.read_csv("enriched_filtered_categories.csv")


# Initialize recommender
recommender = SignLingoRandomForestRecommender(sample_data)

# Get recommendations
print("\nRecommendations for 'apple':")
print(recommender.recommend('apple', top_n=4))

# Analyze feature importance
print("\nFeature Importance Analysis:")
print(recommender.analyze_importance().head())