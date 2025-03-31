import subprocess
import time
import os

def run_model():
    print("Running soil analysis model...")
    start_time = time.time()
    
    try:
        # Run model.py and capture output
        result = subprocess.run(
            ["python", "model.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        if "Predictions saved to" in result.stdout:
            print(f"\nModel completed in {time.time()-start_time:.1f} seconds")
            return True
        else:
            print("Model run may not have completed successfully")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Model failed with error:\n{e.stderr}")
        return False

def run_ui():
    print("\nLaunching Streamlit UI...")
    try:
        # Launch Streamlit UI
        subprocess.run(["streamlit", "run", "ui.py"])
    except KeyboardInterrupt:
        print("\nUI closed by user")

if __name__ == "__main__":
    if run_model():
        run_ui()
    else:
        print("Cannot launch UI due to model errors")