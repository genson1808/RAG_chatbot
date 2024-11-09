### create virtualenv
```
python3 -m venv .venv && source .venv/bin/activate
```

### Install packages
```
pip install -r requirements.txt
```

### Environment variables
All env variables goes to .env ( cp `example.env` to `.env` and paste required env variables)

### Run the python files (following the video to run step by step is recommended)
```
python3 ingest.py
chainlit run app.py
```
pip install "onnxruntime<1.20"
