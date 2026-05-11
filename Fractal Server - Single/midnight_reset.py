import time
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import pytz
from datetime import datetime

# 1. Setup
cred = credentials.Certificate("firebase_service_account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def reset_global_users():
    """
    Checks all registered devices, identifies which ones are currently 
    in their midnight hour, and resets their linked user accounts.
    """
    print(f"[*] Starting Global Hourly Sweep: {datetime.now(pytz.utc)} UTC")
    
    # Map to keep track of who we've already reset this hour (to avoid double-writes)
    processed_emails = set()
    reset_count = 0
    
    # 1. Get all devices that have a timezone set
    # Note: In a massive production app, you'd index this or use a Cloud Function.
    devices_ref = db.collection("registered_devices").stream()

    batch = db.batch()

    for device_doc in devices_ref:
        device_data = device_doc.to_dict()
        user_tz_str = device_data.get("timeZone")
        user_email = device_data.get("email")

        if not user_tz_str or not user_email or user_email in processed_emails:
            continue

        try:
            # 2. Check if it is currently Midnight (00:xx) in the user's timezone
            user_tz = pytz.timezone(user_tz_str)
            user_now = datetime.now(user_tz)

            if user_now.hour == 0:
                # This user is in their midnight window! 
                user_ref = db.collection("users").document(user_email)
                
                # We only reset if they actually have a balance
                user_doc = user_ref.get()
                if user_doc.exists and user_doc.to_dict().get("liquid_mbs", 0) > 0:
                    batch.update(user_ref, {"liquid_mbs": 0.0})
                    processed_emails.add(user_email)
                    reset_count += 1

                # Firestore batch limit
                if reset_count % 500 == 0:
                    batch.commit()
                    batch = db.batch()
        except Exception as tz_err:
            print(f"[!] Invalid timezone {user_tz_str} for {user_email}: {tz_err}")

    if reset_count % 500 != 0 and reset_count > 0:
        batch.commit()

    print(f"[+] Sweep Complete. {reset_count} accounts cleared for their local midnight.")

def main():
    print("[*] Global Midnight Sweeper Active (Production Mode)")
    
    while True:
        # Run the reset logic
        try:
            reset_global_users()
        except Exception as e:
            print(f"[X] Critical Sweep Error: {e}")

        # Sleep until the start of the next hour
        # This is more efficient than sleeping 60 seconds
        now = datetime.now()
        seconds_until_next_hour = 3600 - (now.minute * 60 + now.second)
        
        print(f"[*] Next sweep in {round(seconds_until_next_hour / 60, 1)} minutes...")
        time.sleep(seconds_until_next_hour + 5) # +5s to ensure we cross the hour mark

if __name__ == "__main__":
    main()