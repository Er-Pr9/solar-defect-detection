# Solar Panel Defect Detection - Streamlit App

## Files
- `app.py` -> Streamlit user interface
- `model_utils.py` -> model loading, preprocessing, prediction, top-3 inference, Grad-CAM
- `config.py` -> constants and class descriptions
- `requirements.txt` -> Python dependencies
- `solar_defect_model_final.pth` -> trained model checkpoint (place this in the same folder)

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Your notebook-to-file split
Keep these in the notebook:
- XML parsing
- dataframe creation
- train/test split
- DataLoader creation
- training loop
- evaluation and confusion matrix
- class weighted experiments

Move these into the app files:
- model checkpoint loading
- inference transform
- single image prediction
- top-3 prediction
- Grad-CAM visualization

## Important note
Your checkpoint must contain:
- `model_state_dict`
- `class_names`
