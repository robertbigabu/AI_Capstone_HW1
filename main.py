import dataLoader
from SVM import SVM
from randomForest import RandomForest
from kMeans import kMeans


DATASET_PATH = "Dataset"
GENRES = ["Hard Rock", "Metal", "Pop Rock", "Post Rock", "Punk Rock"]

features = []
labels = []

data = dataLoader.DataLoader(DATASET_PATH, GENRES) 
training_amount = 1.0 # Percentage of the dataset will be used for training
data_dim = 5 

# Load the dataset
print("Loading the dataset...")
features, labels = data.loading(dim=data_dim)
        
# Train the SVM model
print("SVM Model:")
SVM(features, labels, GENRES, training_amount)

# Train the Random Forest model
print("\nRandom Forest Model:") 
RandomForest(features, labels, GENRES, training_amount)

# Train the KMeans model
print("\nKMeans Model:")
kMeans(features, labels, GENRES, training_amount)
