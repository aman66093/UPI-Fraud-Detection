import joblib
import pandas as pd
import numpy as np
from upi_fraud_detection import UPIFraudDetector

def interactive_fraud_tester():
    print("🔍 Interactive UPI Fraud Detector")
    print("=" * 40)
    
    # Load the saved model
    detector = UPIFraudDetector()
    try:
        detector.load_model('upi_fraud_model.pkl')
    except:
        print("❌ Model not found! Run upi_fraud_detection.py first")
        return
    
    while True:
        print("\nEnter transaction details:")
        
        try:
            amount = float(input("Amount (₹): "))
            hour = int(input("Hour (0-23): "))
            is_weekend = int(input("Is weekend? (0=No, 1=Yes): "))
            failed_attempts = int(input("Failed attempts: "))
            account_age = int(input("Account age (days): "))
            frequency = int(input("Transactions in last 24h: "))
            
            print("\nSelect merchant category:")
            print("1. Food  2. Shopping  3. Bills  4. Transfer  5. Entertainment")
            cat_choice = int(input("Choice (1-5): "))
            categories = ['Food', 'Shopping', 'Bills', 'Transfer', 'Entertainment']
            merchant_category = categories[cat_choice - 1]
            
            print("\nSelect device:")
            print("1. Android  2. iOS  3. Web")
            device_choice = int(input("Choice (1-3): "))
            devices = ['Android', 'iOS', 'Web']
            device_type = devices[device_choice - 1]
            
            print("\nSelect location risk:")
            print("1. Low  2. Medium  3. High")
            risk_choice = int(input("Choice (1-3): "))
            risks = ['Low', 'Medium', 'High']
            location_risk = risks[risk_choice - 1]
            
            # Test the transaction
            transaction = {
                'amount': amount,
                'hour': hour,
                'is_weekend': is_weekend,
                'failed_attempts': failed_attempts,
                'account_age_days': account_age,
                'transaction_frequency_last_24h': frequency,
                'merchant_category': merchant_category,
                'device_type': device_type,
                'location_risk': location_risk
            }
            
            result = detector.predict_transaction(transaction)
            
            print("\n" + "="*50)
            print("🎯 FRAUD DETECTION RESULT:")
            print(f"🚨 Fraud Status: {'⚠️ FRAUD DETECTED' if result['is_fraud'] else '✅ LEGITIMATE'}")
            print(f"📊 Risk Level: {result['risk_level']}")
            print(f"🎲 Fraud Probability: {result['fraud_probability']:.1%}")
            print("="*50)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        continue_test = input("\nTest another transaction? (y/n): ").lower()
        if continue_test != 'y':
            break
    
    print("👋 Thanks for testing!")

if __name__ == "__main__":
    interactive_fraud_tester()