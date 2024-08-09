import subprocess
import time
import os

def run_streamlit():
    # Obtain the path to the main.py file
    main_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
    # Run Streamlit
    cmd = ["streamlit", "run", main_py_path]
    subprocess.Popen(cmd)
    # Wait for the server to start
    time.sleep(2)

if __name__ == "__main__":
    run_streamlit()
