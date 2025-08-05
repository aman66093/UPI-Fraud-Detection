# Improved UPI Fraud Detector - More Realistic Version
import joblib
import pandas as pd
import numpy as np
from upi_fraud_detection import UPIFraudDetector

def calculate_location_risk(user_city, transaction_city, user_state, transaction_state):
    """
    Automatically calculate location risk based on geographic data
    This is what a real system would do!
    """
    # Define high-risk locations (in real system, this comes from fraud database)
    high_risk_cities = ['Unknown', 'Border_City', 'High_Crime_Area']
    medium_risk_cities = ['Tourist_Spot', 'Commercial_Hub']
    
    # Same city = Low risk
    if user_city.lower() == transaction_city.lower():
        return 'Low'
    
    # Same state, different city = Medium risk
    elif user_state.lower() == transaction_state.lower():
        return 'Medium'
    
    # Different state or high-risk location = High risk
    elif transaction_city in high_risk_cities:
        return 'High'
    
    # Different state = Medium-High risk
    else:
        return 'Medium'

def calculate_merchant_risk(merchant_name, merchant_type):
    """
    Calculate merchant risk automatically
    """
    # In real system, this comes from merchant fraud database
    high_risk_merchants = ['Unknown_Merchant', 'New_Merchant', 'Previously_Flagged']
    
    if merchant_name in high_risk_merchants:
        return 'High'
    elif merchant_type in ['Transfer', 'Cash_Withdrawal']:
        return 'Medium'
    else:
        return 'Low'

def realistic_fraud_tester():
    print("🔍 Realistic UPI Fraud Detector")
    print("=" * 40)
    print("💡 Now the system calculates risks automatically!")
    print()
    
    # Load the saved model
    detector = UPIFraudDetector()
    try:
        detector.load_model('upi_fraud_model.pkl')
    except:
        print("❌ Model not found! Run upi_fraud_detection.py first")
        return
    
    while True:
        print("\n" + "="*50)
        print("📱 Enter Transaction Details:")
        print("="*50)
        
        try:
            # Basic transaction info (what user actually knows)
            amount = float(input("💰 Transaction Amount (₹): "))
            
            print("\n🕐 When is this transaction happening?")
            hour = int(input("   Hour (0-23): "))
            is_weekend = int(input("   Is it weekend? (0=No, 1=Yes): "))
            
            print("\n👤 Account Information:")
            account_age = int(input("   How old is your account? (days): "))
            frequency = int(input("   How many transactions today?: "))
            failed_attempts = int(input("   Any failed attempts before this? (0-5): "))
            
            print("\n🏪 Transaction Details:")
            print("   Merchant Categories:")
            print("   1. Food/Restaurant  2. Online Shopping  3. Bill Payment")
            print("   4. Money Transfer   5. Entertainment")
            cat_choice = int(input("   Select category (1-5): "))
            categories = ['Food', 'Shopping', 'Bills', 'Transfer', 'Entertainment']
            merchant_category = categories[cat_choice - 1]
            
            print("\n📱 Device Information:")
            print("   1. Android Phone  2. iPhone  3. Web Browser")
            device_choice = int(input("   Your device (1-3): "))
            devices = ['Android', 'iOS', 'Web']
            device_type = devices[device_choice - 1]
            
            print("\n🌍 Location Information:")
            user_city = input("   Your home city: ").strip() or "Mumbai"
            transaction_city = input("   Transaction city (press Enter if same): ").strip() or user_city
            user_state = input("   Your home state: ").strip() or "Maharashtra"
            transaction_state = input("   Transaction state (press Enter if same): ").strip() or user_state
            
            # 🎯 HERE'S THE MAGIC - System calculates risks automatically!
            print("\n🤖 System is calculating risks...")
            
            location_risk = calculate_location_risk(user_city, transaction_city, user_state, transaction_state)
            print(f"   📍 Location Risk: {location_risk}")
            print(f"      Reason: User from {user_city}, {user_state} → Transaction in {transaction_city}, {transaction_state}")
            
            # Build the transaction data
            transaction = {
                'amount': amount,
                'hour': hour,
                'is_weekend': is_weekend,
                'failed_attempts': failed_attempts,
                'account_age_days': account_age,
                'transaction_frequency_last_24h': frequency,
                'merchant_category': merchant_category,
                'device_type': device_type,
                'location_risk': location_risk  # ✅ System calculated this!
            }
            
            # Get prediction
            result = detector.predict_transaction(transaction)
            
            # Show detailed results
            print("\n" + "🎯" + "="*48 + "🎯")
            print("           FRAUD DETECTION ANALYSIS")
            print("🎯" + "="*48 + "🎯")
            
            # Risk factors analysis
            print("\n📊 RISK FACTORS DETECTED:")
            risk_factors = []
            
            if amount > 50000:
                risk_factors.append(f"💰 High Amount: ₹{amount:,.0f}")
            elif amount < 10:
                risk_factors.append(f"💰 Suspiciously Low Amount: ₹{amount}")
                
            if hour < 6 or hour > 22:
                risk_factors.append(f"🌙 Unusual Time: {hour}:00")
                
            if failed_attempts > 2:
                risk_factors.append(f"❌ Multiple Failed Attempts: {failed_attempts}")
                
            if account_age < 30:
                risk_factors.append(f"🆕 New Account: {account_age} days old")
                
            if frequency > 15:
                risk_factors.append(f"⚡ High Frequency: {frequency} transactions today")
                
            if location_risk == 'High':
                risk_factors.append(f"🌍 High Location Risk: {user_city} → {transaction_city}")
            elif location_risk == 'Medium':
                risk_factors.append(f"🌍 Medium Location Risk: Cross-state transaction")
                
            if merchant_category == 'Transfer':
                risk_factors.append("🏦 Money Transfer (Higher Risk Category)")
                
            if risk_factors:
                for factor in risk_factors:
                    print(f"   ⚠️  {factor}")
            else:
                print("   ✅ No major risk factors detected")
            
            # Final verdict
            print(f"\n🎲 FINAL VERDICT:")
            if result['is_fraud']:
                print(f"   🚨 ⚠️  TRANSACTION BLOCKED - FRAUD DETECTED")
                print(f"   📊 Confidence: {result['fraud_probability']:.1%}")
                print(f"   🎯 Risk Level: {result['risk_level']}")
                print(f"   💡 Recommendation: Manual review required")
            else:
                print(f"   ✅ TRANSACTION APPROVED - APPEARS LEGITIMATE")
                print(f"   📊 Fraud Probability: {result['fraud_probability']:.1%}")
                print(f"   🎯 Risk Level: {result['risk_level']}")
                print(f"   💡 Recommendation: Process normally")
            
            print("🎯" + "="*48 + "🎯")
            
        except ValueError:
            print("❌ Please enter valid numbers!")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "="*50)
        continue_test = input("🔄 Test another transaction? (y/n): ").lower().strip()
        if continue_test not in ['y', 'yes']:
            break
    
    print("\n🎉 Thanks for testing the Realistic Fraud Detector!")
    print("💡 Key Insight: The system now calculates risks automatically,")
    print("   just like real fraud detection systems do!")

if __name__ == "__main__":
    realistic_fraud_tester()