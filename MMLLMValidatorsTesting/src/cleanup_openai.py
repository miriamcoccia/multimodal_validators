import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# 1. Setup
load_dotenv("secrets.env")

try:
    client = OpenAI()
except Exception as e:
    print(f"❌ Error initializing client: {e}")
    exit(1)

def cleanup():
    print("🧹 STARTING OPENAI CLEANUP...")
    print("--------------------------------------------------")
    
    # --- STEP 1: Cancel Active Batches ---
    print("\n1️⃣  Checking for Active/Pending Batches...")
    try:
        # Get recent batches (limit 100)
        batches = client.batches.list(limit=100)
        count_cancelled = 0
        
        for batch in batches:
            # We only need to cancel unfinished ones
            if batch.status in ['validating', 'in_progress', 'finalizing']:
                print(f"   🚫 Cancelling batch {batch.id} (Status: {batch.status})...")
                try:
                    client.batches.cancel(batch.id)
                    count_cancelled += 1
                except Exception as e:
                    print(f"      Could not cancel {batch.id}: {e}")
            else:
                # We can't "delete" the history record of a completed/failed batch 
                # via API, but we will delete its files in Step 2.
                pass
        
        if count_cancelled == 0:
            print("   ✅ No active batches found to cancel.")
        else:
            print(f"   ⚠️ Cancelled {count_cancelled} active batches.")

    except Exception as e:
        print(f"   ❌ Error listing batches: {e}")

    # --- STEP 2: Delete Batch Files (Keep Images) ---
    print("\n2️⃣  Cleaning up Files (Keeping Images)...")
    
    try:
        files = client.files.list()
        deleted_count = 0
        kept_count = 0
        
        # Extensions to PROTECT
        image_exts = ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff')

        for file in files:
            # SAFETY CHECK: Is it an image?
            if file.filename.lower().endswith(image_exts):
                print(f"   🛡️  Keeping Image: {file.filename}")
                kept_count += 1
                continue
            
            # If it's not an image, delete it (mostly .jsonl batch files)
            try:
                # Optional: Filter strictly for batch purpose if you have other non-image files
                # if file.purpose in ['batch', 'batch_output']:
                
                print(f"   🗑️  Deleting: {file.filename} (ID: {file.id})")
                client.files.delete(file.id)
                deleted_count += 1
                
                # Sleep briefly to avoid rate limits if you have hundreds
                time.sleep(0.1) 
                
            except Exception as e:
                print(f"      ❌ Failed to delete {file.filename}: {e}")

        print("--------------------------------------------------")
        print(f"✨ Cleanup Summary:")
        print(f"   🛡️  Images Preserved: {kept_count}")
        print(f"   🗑️  Junk Files Deleted: {deleted_count}")

    except Exception as e:
        print(f"   ❌ Error listing files: {e}")

if __name__ == "__main__":
    print("⚠️  DANGER ZONE ⚠️")
    print("This will cancel ALL active batches and delete ALL non-image files (e.g., .jsonl).")
    confirm = input("Are you sure you want to proceed? (yes/no): ")
    
    if confirm.lower() == "yes":
        cleanup()
    else:
        print("❌ Operation cancelled.")