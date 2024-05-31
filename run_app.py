# run_chat_ui.py
import subprocess

def run_streamlit_app():
    subprocess.run(["streamlit", "run", "chat_ui.py"])

if __name__ == "__main__":
    run_streamlit_app()
