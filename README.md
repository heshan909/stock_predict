# 📈 Physics-Based Stock Price Predictor (with GUI)

This project is a Python-based GUI application for stock market analysis and forecasting. It integrates both **machine learning regression models** and **numerical physics techniques** (differentiation, integration, interpolation, root solving) to provide rich insights on stock price behavior.

---

## 🔧 Features

- ✅ Interactive GUI with Tkinter
- ✅ Fetch live historical stock data using `yfinance`
- ✅ Predict next 7 and 21 days using linear regression
- ✅ Visualize stock trends, moving averages, and returns
- ✅ (Upcoming) Plot numerical methods: 
  - Numerical differentiation & acceleration
  - Integration (Trapezoidal, Simpson’s)
  - Interpolation (Lagrange, Cubic Spline)
  - Non-linear root finding (Bisection)

---

## 🗂 Project Structure


├── project_GUI.py          # Main GUI application (Tkinter)  
├── stock_predict.py        # Backend for data processing and regression    
├── numerical_stock_gui.py  # (Optional) Physics-based numerical  plotting module  
├── README.md               # You are here.  

## ▶ How to Run

1. 🔧 Install Requirements

`pip install yfinance numpy pandas matplotlib tkcalendar scikit-learn scipy`

You must have Python 3.7+ installed.

---

2. 🚀 Launch the App

`python project_GUI.py`

You’ll be able to:

- Select a stock ticker

- Choose a date range

- Click Predict to see trendlines

- Click Physics Predict (after full setup) to visualize numerical analysis

## 📘 Modules Explained

 stock_predict.py

- Fetches and prepares historical data

- Applies MinMax scaling and builds linear regression

- Predicts future stock prices and returns residuals

project_GUI.py

- Fully-featured Tkinter interface

- Displays historical and predicted stock prices

- Shows summary boxes and graphical results

numerical_stock_gui.py (optional)

- Contains physics-focused numerical methods

- Generates plots for differentiation, integration, and interpolation

- To be integrated via Physics Predict button

## 📊 Example Use Case

Predict next week's price trend of AAPL and compare it with cubic spline interpolation curve or numerical derivatives. Use both machine learning and classical physics-inspired techniques to cross-validate behavior.

## 💡 Future Ideas

- Integrate neural networks (LSTM)

- Export predictions to Excel

- Deploy as a web app with Streamlit

- Add parameter tuning and error visualization

## 📘 License

MIT License.
© 2025 heshan909

## 🤝 Contributions

🤝 Contributions
Pull requests and suggestions are welcome!

