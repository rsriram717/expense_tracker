#!/usr/bin/env python
"""
Launcher for Transaction Categorizer Streamlit App
"""

import os
import sys
import subprocess

def main():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure the root directory is in the Python path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    script_path = os.path.join(current_dir, "scripts", "review_transactions.py")
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return 1
        
    print("Launching Transaction Categorizer App...")
    
    try:
        # Add environment variables to ensure root directory is in path
        env = os.environ.copy()
        env["PYTHONPATH"] = current_dir + os.pathsep + env.get("PYTHONPATH", "")
        
        # Launch streamlit with the updated app
        subprocess.run(["streamlit", "run", script_path], check=True, env=env)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        return 1
    except FileNotFoundError:
        print("Streamlit not found. Please install it with 'pip install streamlit'")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 