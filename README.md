# MLP Car Price Prediction

## Description

This project builds a regression model using a **Multilayer Perceptron (MLP)** to predict the **selling price of used cars** based on several numerical and categorical features. The model is trained on a dataset downloaded via `kagglehub`.

---

## Features

- Regression using MLP Neural Network
- Data preprocessing and normalization
- One-hot encoding of categorical variables
- Scaled input features using StandardScaler
- Visualization of training performance (MAE and Loss)

---

## Technologies Used

- Python
- Pandas
- NumPy
- TensorFlow
- Scikit-learn
- Matplotlib

---

## Installation

```bash
pip install tensorflow pandas scikit-learn matplotlib kagglehub
```

---

## Dataset

**Source:** [`vijayaadithyanvg/car-price-predictionused-cars`](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars)  
**Total Entries:** 301  
**Features:**
- `Car_Name`, `Year`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`
- **Target Variable:** `Selling_Price`

---

## Project Tasks

1. **Data Preprocessing**  
   - Dropped irrelevant columns (`Car_Name`)  
   - Converted categorical variables (`Fuel_Type`, `Selling_type`, `Transmission`) using `pd.get_dummies`  
   - Scaled numerical features using `StandardScaler`  
   - Split the dataset into training and testing sets (80/20)  

2. **Model Architecture**  
   A deep MLP model with the following layers:
   ```python
   model = Sequential([
       Input(shape=(X_train.shape[1],)),
       Dense(128, activation='relu'),
       Dense(64, activation='relu'),
       Dense(64, activation='relu'),
       Dense(32, activation='relu'),
       Dense(1, activation='linear')
   ])
   ```

3. **Compilation & Training**  
   - Optimizer: `Adam`  
   - Loss Function: `Mean Squared Error`  
   - Metrics: `Mean Absolute Error (MAE)`  
   - Trained for 200 epochs with a batch size of 32  
   - Used 20% of training data for validation  

4. **Performance**  
   - Final Test MAE: ~0.54  
   - Final Test Loss (MSE): ~0.70  
   - Plotted Training vs Validation curves for both MAE and Loss  

5. **Visualization**  
   - Generated plots for Training/Validation MAE and Loss using `matplotlib`  

6. **Sample Predictions**

| Actual Price | Predicted Price |
|--------------|------------------|
| 0.35         | 0.353            |
| 10.11        | 10.76            |
| 4.95         | 5.67             |
| 0.15         | 0.15             |
| 6.95         | 7.44             |


