from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder


def RandomForest(features, labels, GENRES, training_amount=1.0):
    # Get the training sets according to the training amount
    if training_amount < 1.0:
        X_train, _, y_train, _ = train_test_split(features, labels, train_size=training_amount)
    else:
        X_train = features
        y_train = labels

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Predict the labels
    y_pred_cv = cross_val_predict(model, X_scaled, y_encoded, cv=5)
    
    # Evaluate the model
    report = classification_report(y_encoded, y_pred_cv, target_names=GENRES)

    print("Classification Report:\n", report)