"""
NirmanAI — One-click Setup & Run Script
Run this first: python setup.py
Then run dashboard: streamlit run app.py
"""
import subprocess, sys, os

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        sys.exit(1)

print("\n" + "="*55)
print("  NirmanAI — Setup & Training Pipeline")
print("  Team Aim-Nexus | IIT Madras")
print("="*55)

# 1. Install dependencies
print("\n[1/4] Installing dependencies...")
run(f"{sys.executable} -m pip install -q -r requirements.txt")

# 2. Generate datasets
print("\n[2/4] Generating synthetic datasets...")
run(f"{sys.executable} generate_data.py")

# 3. Train models
print("\n[3/4] Training delay prediction model...")
run(f"{sys.executable} train_delay_model.py")

print("\n[3/4] Training wastage estimation model...")
run(f"{sys.executable} train_wastage_model.py")

# 4. Done
print("\n" + "="*55)
print("  [✓] Setup complete!")
print("="*55)
print("\nNext step — Launch the dashboard:")
print("  streamlit run app.py")
print("\nDashboard will open at: http://localhost:8501")
print("="*55 + "\n")
