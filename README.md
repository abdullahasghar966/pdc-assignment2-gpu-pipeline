# Assignment 2 - GPU model with an automatic retrain pipeline

## What it does
- `data/*.csv`: training data. Every CSV in this folder is merged for training.
- `train.py`: trains a digit classifier with PyTorch. Uses the GPU if available, else CPU. Writes `models/model.pt` and `models/metrics.json`.
- `api.py`: web API.
  - `GET /accuracy` returns the model's accuracy.
  - `POST /predict` with body `{"features": [64 numbers]}` returns the digit.
  - `POST /reload` loads the newest model after a retrain.
- `Jenkinsfile`: pipeline that installs dependencies, trains, and reloads the API.
- A GitHub push webhook triggers Jenkins, which runs the pipeline.

## Run locally
```powershell
python -m pip install -r requirements.txt
python make_dataset.py   # only once
python train.py
python api.py
```
Then open http://localhost:5000/accuracy

## Example request
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:5000/predict -ContentType 'application/json' -Body (@{features=(1..64 | ForEach-Object {0})} | ConvertTo-Json)
```
