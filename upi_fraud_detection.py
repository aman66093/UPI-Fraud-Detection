# UPI Fraud Detection System - Beginner Level
# This is a simplified version for learning purposes

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UPIFraudDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def generate_sample_data(self, n_samples=10000):
        """
        Generate synthetic UPI transaction data for training
        In real project, you'll get this from your database
        """
        print("📊 Generating sample UPI transaction data...")
        
        np.random.seed(42)
        
        # Create realistic transaction data
        data = []
        
        for i in range(n_samples):
            # Basic transaction info
            transaction_id = f"UPI{str(i).zfill(8)}"
            
            # Amount patterns (fraudulent tend to be higher or very small)
            if np.random.random() < 0.05:  # 5% fraud
                # Fraudulent transaction patterns
                amount = np.random.choice([
                    np.random.uniform(50000, 200000),  # High value fraud
                    np.random.uniform(1, 10),          # Small amount fraud
                    np.random.uniform(5000, 50000)     # Medium fraud
                ])
                is_fraud = 1
                
                # Fraudulent patterns
                hour = np.random.choice([1, 2, 3, 23, 22, 21])  # Unusual hours
                is_weekend = np.random.choice([0, 1], p=[0.3, 0.7])  # More on weekends
                failed_attempts = np.random.randint(3, 10)  # More failed attempts
                account_age = np.random.randint(1, 30)  # Newer accounts
                transaction_frequency = np.random.randint(20, 100)  # High frequency
                
            else:
                # Normal transaction patterns
                amount = np.random.lognormal(8, 1.5)  # Realistic amount distribution
                is_fraud = 0
                
                # Normal patterns
                hour = np.random.choice(range(6, 23))  # Business hours
                is_weekend = np.random.choice([0, 1], p=[0.7, 0.3])  # Less on weekends
                failed_attempts = np.random.randint(0, 3)  # Fewer failed attempts
                account_age = np.random.randint(30, 1000)  # Older accounts
                transaction_frequency = np.random.randint(1, 20)  # Normal frequency
            
            # Other features
            merchant_category = np.random.choice(['Food', 'Shopping', 'Bills', 'Transfer', 'Entertainment'])
            device_type = np.random.choice(['Android', 'iOS', 'Web'])
            location_risk = np.random.choice(['Low', 'Medium', 'High'], p=[0.7, 0.2, 0.1])
            
            # Add some correlation for fraud
            if is_fraud:
                location_risk = np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.3, 0.5])
                merchant_category = np.random.choice(['Transfer', 'Shopping'], p=[0.7, 0.3])
            
            data.append({
                'transaction_id': transaction_id,
                'amount': amount,
                'hour': hour,
                'is_weekend': is_weekend,
                'failed_attempts': failed_attempts,
                'account_age_days': account_age,
                'transaction_frequency_last_24h': transaction_frequency,
                'merchant_category': merchant_category,
                'device_type': device_type,
                'location_risk': location_risk,
                'is_fraud': is_fraud
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} transactions ({df['is_fraud'].sum()} fraudulent)")
        return df
    
    def preprocess_data(self, df):
        """
        Prepare data for machine learning
        """
        print("🔧 Preprocessing data...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Feature engineering
        processed_df['amount_log'] = np.log1p(processed_df['amount'])
        processed_df['is_high_amount'] = (processed_df['amount'] > processed_df['amount'].quantile(0.95)).astype(int)
        processed_df['is_night_transaction'] = ((processed_df['hour'] < 6) | (processed_df['hour'] > 22)).astype(int)
        processed_df['high_frequency'] = (processed_df['transaction_frequency_last_24h'] > 10).astype(int)
        processed_df['new_account'] = (processed_df['account_age_days'] < 30).astype(int)
        
        # Encode categorical variables
        categorical_columns = ['merchant_category', 'device_type', 'location_risk']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                processed_df[col + '_encoded'] = self.label_encoders[col].fit_transform(processed_df[col])
            else:
                processed_df[col + '_encoded'] = self.label_encoders[col].transform(processed_df[col])
        
        # Select features for modeling
        self.feature_columns = [
            'amount_log', 'hour', 'is_weekend', 'failed_attempts', 
            'account_age_days', 'transaction_frequency_last_24h',
            'merchant_category_encoded', 'device_type_encoded', 'location_risk_encoded',
            'is_high_amount', 'is_night_transaction', 'high_frequency', 'new_account'
        ]
        
        X = processed_df[self.feature_columns]
        y = processed_df['is_fraud']
        
        print(f"✅ Features prepared: {len(self.feature_columns)} features")
        return X, y, processed_df
    
    def train_model(self, X, y):
        """
        Train the fraud detection model
        """
        print("🤖 Training fraud detection model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model (good for beginners)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\n📈 Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Top 5 Most Important Features:")
        print(feature_importance.head())
        
        return X_test_scaled, y_test, y_pred, y_pred_proba, feature_importance
    
    def predict_transaction(self, transaction_data):
        """
        Predict if a single transaction is fraudulent
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert to DataFrame for consistent processing
        df = pd.DataFrame([transaction_data])
        
        # Add dummy is_fraud column for preprocessing (won't be used)
        df['is_fraud'] = 0
        
        # Preprocess (this will create the features we need)
        processed_df = df.copy()
        
        # Feature engineering (same as in preprocess_data)
        processed_df['amount_log'] = np.log1p(processed_df['amount'])
        processed_df['is_high_amount'] = (processed_df['amount'] > processed_df['amount'].quantile(0.95)).astype(int)
        processed_df['is_night_transaction'] = ((processed_df['hour'] < 6) | (processed_df['hour'] > 22)).astype(int)
        processed_df['high_frequency'] = (processed_df['transaction_frequency_last_24h'] > 10).astype(int)
        processed_df['new_account'] = (processed_df['account_age_days'] < 30).astype(int)
        
        # Encode categorical variables
        categorical_columns = ['merchant_category', 'device_type', 'location_risk']
        for col in categorical_columns:
            processed_df[col + '_encoded'] = self.label_encoders[col].transform(processed_df[col])
        
        # Select features
        X = processed_df[self.feature_columns]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        fraud_probability = self.model.predict_proba(X_scaled)[0, 1]
        is_fraud = self.model.predict(X_scaled)[0]
        
        return {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_probability),
            'risk_level': 'High' if fraud_probability > 0.7 else 'Medium' if fraud_probability > 0.3 else 'Low'
        }
    
    def save_model(self, filepath='upi_fraud_model.pkl'):
        """
        Save trained model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='upi_fraud_model.pkl'):
        """
        Load trained model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        print(f"✅ Model loaded from {filepath}")

def main():
    """
    Main function to run the fraud detection system
    """
    print("🚀 UPI Fraud Detection System - Beginner Project")
    print("=" * 50)
    
    # Initialize detector
    detector = UPIFraudDetector()
    
    # Step 1: Generate sample data
    df = detector.generate_sample_data(10000)
    
    # Step 2: Preprocess data
    X, y, processed_df = detector.preprocess_data(df)
    
    # Step 3: Train model
    X_test, y_test, y_pred, y_pred_proba, feature_importance = detector.train_model(X, y)
    
    # Step 4: Test with sample transactions
    print("\n🧪 Testing with sample transactions:")
    print("-" * 40)
    
    # Test case 1: Normal transaction
    normal_transaction = {
        'amount': 500,
        'hour': 14,
        'is_weekend': 0,
        'failed_attempts': 0,
        'account_age_days': 180,
        'transaction_frequency_last_24h': 3,
        'merchant_category': 'Food',
        'device_type': 'Android',
        'location_risk': 'Low'
    }
    
    result = detector.predict_transaction(normal_transaction)
    print(f"Normal Transaction: {result}")
    
    # Test case 2: Suspicious transaction
    suspicious_transaction = {
        'amount': 75000,
        'hour': 2,
        'is_weekend': 1,
        'failed_attempts': 5,
        'account_age_days': 7,
        'transaction_frequency_last_24h': 25,
        'merchant_category': 'Transfer',
        'device_type': 'Web',
        'location_risk': 'High'
    }
    
    result = detector.predict_transaction(suspicious_transaction)
    print(f"Suspicious Transaction: {result}")
    
    # Step 5: Save model
    detector.save_model()
    
    print("\n✅ Project completed successfully!")
    print("\nNext steps for you:")
    print("1. Try modifying transaction parameters and test predictions")
    print("2. Add more features like merchant history, user behavior")
    print("3. Experiment with different ML algorithms")
    print("4. Create a simple web interface using Flask/FastAPI")

if __name__ == "__main__":
    main()
    # Step 6: YOUR EXPERIMENTS
    print("\n🧪 YOUR TURN - Test Different Transactions:")
    print("-" * 50)
    
    # Experiment 1: Try a midnight transaction
    midnight_transaction = {
        'amount': 15000,
        'hour': 1,  # 1 AM
        'is_weekend': 1,
        'failed_attempts': 0,
        'account_age_days': 100,
        'transaction_frequency_last_24h': 2,
        'merchant_category': 'Transfer',
        'device_type': 'Web',
        'location_risk': 'Medium'
    }
    
    result = detector.predict_transaction(midnight_transaction)
    print(f"Midnight Transaction: {result}")
    
    # Experiment 2: Try changing the amount
    amounts_to_test = [100, 5000, 25000, 100000]
    for amount in amounts_to_test:
        test_tx = {
            'amount': amount,
            'hour': 14,
            'is_weekend': 0,
            'failed_attempts': 0,
            'account_age_days': 200,
            'transaction_frequency_last_24h': 3,
            'merchant_category': 'Shopping',
            'device_type': 'Android',
            'location_risk': 'Low'
        }
        result = detector.predict_transaction(test_tx)
        print(f"Amount ₹{amount}: Risk Level = {result['risk_level']} (Probability: {result['fraud_probability']:.3f})")