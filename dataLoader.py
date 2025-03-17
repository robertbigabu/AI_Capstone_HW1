import os
import numpy as np
import librosa
import librosa.display
from sklearn.decomposition import PCA


class DataLoader:
    def __init__(self, dataset_path, genres):
        self.dataset_path = dataset_path
        self.genres = genres
        self.features = []
        self.labels = []    
        
    def extract_features(self, file_path, dim=40):
        try:
            audio, sample_rate = librosa.load(file_path, sr=22050) # Load an audio file as a floating point time series.
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=dim) # Load an audio file as a floating point time series.
            mfccs_mean = np.mean(mfccs, axis=1) 
        
        except Exception as e:
            print("Error encountered while parsing file: ", file_path)
            return None

        return mfccs_mean
    
    def loading(self, dim):
        for genre in self.genres:
            genre_path = os.path.join(self.dataset_path, genre)
            for file in os.listdir(genre_path):
                file_path = os.path.join(genre_path, file)
                print(file_path)
                feature_vector = self.extract_features(file_path)
                self.features.append(feature_vector)
                self.labels.append(genre)
                
        # Convert features and labels to numpy arrays
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        # Apply PCA to reduce the dimensionality of the features
        pca = PCA(dim)
        self.features = pca.fit_transform(self.features)

        return self.features, self.labels