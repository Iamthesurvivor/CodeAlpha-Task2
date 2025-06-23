# Car Price Prediction Task 🚗💰

## 🎯 Objective

Build a machine learning system that predicts the **selling price of a used car** using key features such as brand goodwill, mileage, horsepower, and other attributes. The goal is to understand which factors most influence a car’s resale value and use regression techniques to model this relationship.*selling price of a used car** using key features such as brand goodwill, mileage, horsepower, and other attributes. The goal is to understand which factors most influence a car’s resale value and use regression techniques to model this relationship.

This task is part of the **CodeAlpha** internship and is included in the `codealpha-tasks` repository.nd is included in the `codealpha-tasks` repository.

---

## 📌 Problem Statement

> **CAR PRICE PREDICTION**  
>  
> Create a model that predicts the selling price of cars based on features like brand, age, fuel type, mileage, horsepower, and more.   cars based on features like brand, age, fuel type, mileage, horsepower, and more.  
> The solution must include proper data preprocessing, feature engineering, and performance evaluation using Python libraries such as Pandas, Scikit-learn, and Matplotlib.  ng, feature engineering, and performance evaluation using Python libraries such as Pandas, Scikit-learn, and Matplotlib.  
> Also, demonstrate real-world ML applications in pricing systems.ricing systems.

---

## 🛠 Technologies Used

- **Python 3.x**
- **Pandas**, **NumPy** – data handling and cleaningg
- **Scikit-learn** – preprocessing, model training, evaluation evaluation
- **Random Forest Regressor** – core regression modelel
- **Matplotlib**, **Seaborn** – visualizations
- **Joblib** – saving the trained model

---

## 🔍 Features Used for Prediction

| Feature           | Description                                         |
|-------------------|-----------------------------------------------------|
| `Car_Name`        | Used to extract the brand of the car                |
| `Present_Price`   | Original showroom price in ₹ lakh                   |
| `Kms_Driven`      | Total kilometers driven                             |
| `Fuel_Type`       | Type of fuel used (Petrol, Diesel, CNG, etc.)       |
| `Seller_Type`     | Whether the seller is a dealer or individual        |
| `Transmission`    | Type of transmission (Manual/Automatic)             |
| `Owner`           | Number of previous owners                           |


---
## 🧪 Model Training & Evaluation

- **Split**: 80/20 train-test split  
- **Preprocessing**: OneHotEncoding, ColumnTransformer  
- **Model**: RandomForestRegressor  
- **Evaluation Metrics**:
  - R² Score
  - Mean Absolute Error (MAE)
- **Visualization**:
  - Predicted vs Actual price scatter plot
  - Feature importance bar chart


---

## 📊 Results

| Metric | Value        |
|--------|--------------|
| R²     | ~0.96        |
| MAE    | ~0.63 ₹ lakh |

## 📦 Output

- `model_price.pkl`: Trained model pipeline  
- `predicted_prices.csv`: Price predictions  
- Visual plots: Feature importance and validation results  


---

## 🧪 How to Run the Project

Open your terminal in the project folder and run the following:

```bash
cd "car price prediction"

# 1. Create and activate a virtual environment
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# 2. Generate the requirements.txt file (if not already created)
pip freeze > requirements.txt

# 3. Install required libraries
pip install -r requirements.txt

# 4. Run the main Python script
python codealpha-tasks/"car price prediction.py"