import os
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import Increment
from google.cloud.firestore_v1.base_query import FieldFilter

from dotenv import load_dotenv

load_dotenv()


cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "firebase_service_account.json")
if not os.path.exists(cred_path):
    print(f"[X] Error: {cred_path} not found!")
    exit(1)

print("[*] Initializing Firebase Admin SDK...")
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Testing with your hardwareId
TEST_ID = "fc5085107cf2f409"
TEST_REWARD = 15.0

print("\n==================================================")
print(f" TESTING REWARD LOGIC FOR DEVICE: {TEST_ID}")
print("==================================================")

try:
    print("[1] Searching 'registered_devices' collection...")
    
    # THE FIX: Search the 'hardwareId' field, not 'macAddress'
    docs = db.collection("registered_devices").where(filter=FieldFilter("hardwareId", "==", TEST_ID)).limit(1).stream()
    
    email = None
    for doc in docs:
        data = doc.to_dict()
        email = data.get("email")
        print(f"    -> Found Document ID: {doc.id}")
        print(f"    -> Extracted Email:   {email}")

    if not email:
        print(f"\n[X] FAILURE: No device found with hardwareId '{TEST_ID}'")
        exit(1)

    print(f"\n[2] Depositing {TEST_REWARD} MBs to 'users' collection...")
    user_ref = db.collection("users").document(email)
    
    doc_before = user_ref.get()
    balance_before = doc_before.to_dict().get("liquid_mbs", 0.0) if doc_before.exists else 0.0
    print(f"    -> Current Balance: {balance_before} MBs")

    user_ref.set({"liquid_mbs": Increment(TEST_REWARD)}, merge=True)
    print(f"    -> Successfully executed Increment(+{TEST_REWARD})!")

    doc_after = user_ref.get()
    balance_after = doc_after.to_dict().get("liquid_mbs", 0.0)
    print(f"    -> New Balance:     {balance_after} MBs")
    
    print("\n[+] TEST COMPLETE: Logic executed perfectly.")

except Exception as e:
    print(f"\n[X] A CRITICAL ERROR OCCURRED: {e}")