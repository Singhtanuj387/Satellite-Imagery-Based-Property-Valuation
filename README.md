# Satellite-Imagery-Based-Property-Valuation

**Predicting house prices using tabular and image data** ✅

---

## Overview
This project trains and evaluates models that predict property price using:
- A multi-input model that uses property images + tabular features (images branch + tabular branch merged).
- Tabular-only models (Linear Regression, XGBoost, LightGBM with multi-output capability).

The notebooks you asked about:
- `model_traning_withimages.ipynb` — Multi-input model using images and tabular metadata.
- `model_traning_tabular.ipynb` — Tabular-only experiments and production-ready regression models.

---

## Quick setup (Linux)
1. If you prefer to create a new venv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

2. Install required packages (minimal list):
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn opencv-python pillow tensorflow keras xgboost lightgbm rasterio jupyterlab
   ```
   - Note: `rasterio` may require OS-level GDAL packages or use conda for easier install: `conda install -c conda-forge rasterio`.

3. (Optional) If you want GPU TensorFlow, install the appropriate `tensorflow` (GPU) package and CUDA drivers for your system.

---

## Files & Important Outputs
- `train(1)(train(1)).csv` — training CSV used in notebooks
- `test2(test(1)).csv` — dataset used for inference in notebooks
- `res_images/` — directory containing images (named like `property_<id>.jpg` / `.tif`)
- `best_model.h5` — saved Keras model from image+tabular training
- `output.csv` — inference outputs

---

## How to run
1. Start JupyterLab / Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
2. Open the notebooks and run cells sequentially. Recommended order:
   - `model_traning_tabular.ipynb` (explore tabular-only baselines)
   - `model_traning_withimages.ipynb` (train multi-input model using images + tabular)

3. For automated runs / reproducible runs you can use `nbconvert` or `papermill`:
   ```bash
   jupyter nbconvert --to notebook --execute model_traning_tabular.ipynb --output executed_tabular.ipynb
   ```

---

## Architecture summary
### model_traning_withimages.ipynb (images + tabular)
- **Preprocessing**
  - Images: support for `.jpg`, `.png`, and `.tif` (uses `rasterio` for TIFFs). Images are read, normalized and resized to a common `IMAGE_SIZE` (e.g., 224x224).
  - Tabular features: engineered features like `age`, `is_renovated`, `basement_present`, `living_lot_ratio`, `years_since_renovation`. Features are scaled (e.g., `StandardScaler`).
  - Target price is log-transformed using `np.log1p` for stability during training.
- **Model architecture**
  - **Image branch:** CNN backbone that extracts features from images (e.g., Conv -> Pool -> Flatten or transfer-learning backbone). Produces an image embedding vector.
  - **Tabular branch:** Dense layers that process numeric/categorical features to get a tabular embedding.
  - **Concatenation:** Image and tabular embeddings concatenated, followed by Dense layers to predict the target (regression output in log-space).
  - **Loss & training:** Regression with MSE or MAE; training callbacks may save the best model (`best_model.h5`).
- **Inference:** Predictions from model come back in log1p space; invert using `np.expm1(preds)` to obtain original price units.

### model_traning_tabular.ipynb (tabular-only)
- **Preprocessing**
  - Same engineered features as above.
  - Target price transforms: `FunctionTransformer(np.log1p)` applied to `y` during training.
  - `StandardScaler` for features; optional PCA for dimensionality reduction.
- **Models explored**
  - Linear Regression (baseline)
  - XGBoost (`XGBRegressor`) with hyperparameters tuned manually or via RandomizedSearchCV
  - LightGBM (`LGBMRegressor`) trained with early stopping and optionally as a `MultiOutputRegressor` if multiple targets
- **Evaluation**
  - Report metrics: MSE, MAE, R² on test data (metrics calculated on transformed scale or optionally inverted to original scale).
- **Inference**
  - Predict on unseen `df2` (e.g., `test2(test(1)).csv`), invert target transform: `np.expm1(y_pred)` and store in output dataframe.

---

## Common issues & troubleshooting
- KeyError when dropping columns (e.g., empty string in the list): ensure the columns list contains valid column names or use `df.drop(columns=cols, errors='ignore')`.
- TIFF read failures: ensure `rasterio` installed and GDAL available; otherwise images may be replaced with black placeholder images during preprocessing.
- Predicted values look odd: check whether the model outputs log-space values — if so invert with `np.expm1`.
- GPU/TF install problems: verify CUDA/cuDNN versions match TensorFlow GPU build.

---

## Useful snippets
- Inverse log1p transform:
  ```python
  y_pred = model.predict(X)
  y_pred_orig = np.expm1(y_pred.flatten())
  ```
- Save predictions to CSV:
  ```python
  df2['predicted_price'] = np.expm1(y_pred.flatten())
  df2.to_csv('output.csv', index=False)
  ```

---

## Next steps & suggestions
- Add a `requirements.txt` (I can generate this for you if you want).
- Add a small `run_inference.py` script to run production inference outside notebooks.
- Add tests for the preprocessing functions and a simple CI job.

---

## Contact / license
- Author: (your name)
- License: MIT (change as desired)

---

If you'd like, I can also:
- Generate `requirements.txt` for the environment, or
- Add a `run_inference.py` script and an example CLI for quick inference.

Feel free to tell me any edits you'd like to the README.
