import subprocess
import time
import os
import signal

def run_streamlit():
    # Obtain the path to the main.py file
    main_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace('run', 'assets'), 'CatGPT_app.py')
    
    # Run Streamlit as a subprocess
    cmd = ["streamlit", "run", main_py_path]
    process = subprocess.Popen(cmd)

    try:
        # Wait for the server to start
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Terminate the Streamlit process on Ctrl+C
        process.send_signal(signal.SIGINT)
        process.wait()

if __name__ == "__main__":
    run_streamlit()