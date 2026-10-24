# Resistor Lab

A Streamlit app for AI-powered resistor detection and resistor color-band calculations.

## Run locally

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

The detector uses `runs/detect/resistor_detector_fast/weights/best.pt`. The
project's `.gitignore` keeps other training artifacts out of Git while allowing
this checkpoint to be included for deployment.

## Deploy from GitHub

1. Create a GitHub repository and upload this project, including
   `runs/detect/resistor_detector_fast/weights/best.pt`.
2. Connect the repository to Streamlit Community Cloud and select `app.py` as
   the app's main file.
3. Deploy. The generated `*.streamlit.app` URL is the shareable website link.

GitHub Pages does not run Python apps; use a Streamlit-compatible host for this
project.
