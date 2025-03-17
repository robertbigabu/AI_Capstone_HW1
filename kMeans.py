import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

def kMeans(features, labels, GENRES, training_amount=1.0):
    # Get the training sets according to the training amount
    if training_amount < 1.0:
        X_train, _, y_train, _ = train_test_split(features, labels, train_size=training_amount)
    else:
        X_train = features
        y_train = labels
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Train the KMeans model
    model = KMeans(n_clusters=len(GENRES), random_state=42)
    model.fit(X_scaled)
    kmeans = model.predict(X_scaled)
    
    # Compute the Adjusted Rand Index
    ari = adjusted_rand_score(y_train, kmeans)
    print("Adjusted Rand Index: {:.2f}".format(ari))

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering')  
    plt.show()
    