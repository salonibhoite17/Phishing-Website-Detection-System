import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

def train_and_save_model():
    print("Training new phishing detection model...")
    
    try:
        # Check if phishing.csv exists
        if not os.path.exists('phishing.csv'):
            print("WARNING: phishing.csv not found. Please download the dataset first.")
            print("You can get it from: https://www.kaggle.com/eswarchandt/phishing-website-detector")
            return False
            
        # Load the dataset
        data = pd.read_csv('phishing.csv')
        
        # Remove index column if it exists
        if 'Index' in data.columns:
            data = data.drop(['Index'], axis=1)
            
        # Split features and target
        X = data.drop(["class"], axis=1)
        y = data["class"]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        print("Training Gradient Boosting model...")
        gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)
        gbc.fit(X_train, y_train)
        
        # Create pickle directory if it doesn't exist
        os.makedirs('pickle', exist_ok=True)
        
        # Save the model to pickle file
        model_path = os.path.join('pickle', 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(gbc, f)
            
        print(f"Model successfully trained and saved to {model_path}")
        
        # Evaluate the model
        train_accuracy = gbc.score(X_train, y_train)
        test_accuracy = gbc.score(X_test, y_test)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        
        return True
    
    except Exception as e:
        print(f"Error training model: {e}")
        return False

if __name__ == "__main__":
    train_and_save_model()