import os
import glob
import requests
import time
import shutil
from dotenv import load_dotenv

# --- Initialize Environment ---
# Assuming secrets.env is in MMLLMValidatorsTesting/
load_dotenv("secrets.env")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1"

# --- Precise Path Configuration ---
BASE_DIR = "/Users/miriam/projects/multimodal_validators/MMLLMValidatorsTesting"
DATA_DIR = os.path.join(BASE_DIR, "data")
# Look inside the data/ folder for the nebius single files
TARGET_PATTERN = os.path.join(DATA_DIR, "*nebius*single*.jsonl")
LOG_FILE = os.path.join(BASE_DIR, "submitted_batches_log.txt")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_originals")

CHUNK_SIZE_MB = 90 

def log_status(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def split_large_file(input_path):
    # Get just the name without the .jsonl extension
    file_name_only = os.path.splitext(os.path.basename(input_path))[0]
    max_bytes = CHUNK_SIZE_MB * 1024 * 1024
    chunks = []
    file_num = 1
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines, curr_size = [], 0
        for line in f:
            line_bytes = len(line.encode('utf-8'))
            if curr_size + line_bytes > max_bytes:
                # Save chunks in the data folder
                chunk_name = os.path.join(DATA_DIR, f"{file_name_only}_chunk_{file_num:03d}.jsonl")
                with open(chunk_name, 'w', encoding='utf-8') as cf:
                    cf.writelines(lines)
                chunks.append(chunk_name)
                file_num += 1
                lines, curr_size = [], 0
            lines.append(line)
            curr_size += line_bytes
        if lines:
            chunk_name = os.path.join(DATA_DIR, f"{file_name_only}_chunk_{file_num:03d}.jsonl")
            with open(chunk_name, 'w', encoding='utf-8') as cf:
                cf.writelines(lines)
            chunks.append(chunk_name)
    return chunks

def submit_to_nebius(file_path):
    headers = {"Authorization": f"Bearer {NEBIUS_API_KEY}"}
    try:
        with open(file_path, 'rb') as f:
            r = requests.post(f"{NEBIUS_BASE_URL}/files", headers=headers, 
                              files={"file": f}, data={"purpose": "batch"}, timeout=300)
        r.raise_for_status()
        file_id = r.json().get('id')
        
        payload = {"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"}
        r = requests.post(f"{NEBIUS_BASE_URL}/batches", headers=headers, json=payload)
        r.raise_for_status()
        return r.json().get('id')
    except Exception as e:
        return f"Error: {e}"

def run_rescue():
    if not NEBIUS_API_KEY:
        log_status("❌ CRITICAL: NEBIUS_API_KEY missing from secrets.env")
        return

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # glob will now look specifically in /data/
    files_to_fix = [f for f in glob.glob(TARGET_PATTERN) if "_chunk_" not in f]

    if not files_to_fix:
        log_status(f"✅ No nebius_single files found in {DATA_DIR}")
        return

    log_status(f"🚀 Rescue started for {len(files_to_fix)} files.")

    for big_file in files_to_fix:
        log_status(f"📂 Processing: {os.path.basename(big_file)}")
        chunks = split_large_file(big_file)
        
        success_count = 0
        for i, chunk in enumerate(chunks, 1):
            batch_id = submit_to_nebius(chunk)
            if "Error" in str(batch_id):
                log_status(f"   ❌ Part {i} Failed: {batch_id}")
            else:
                log_status(f"   ✅ Part {i} Launched: {batch_id}")
                success_count += 1
                os.remove(chunk)

        if success_count == len(chunks):
            # Move the big 1.36GB file to the processed folder
            dest = os.path.join(PROCESSED_DIR, os.path.basename(big_file))
            shutil.move(big_file, dest)
            log_status(f"📦 Moved original to {PROCESSED_DIR}")

    log_status("✨ All done. You can safely close your terminal.")

if __name__ == "__main__":
    run_rescue()