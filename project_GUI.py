#project_GUI.py

import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import numpy as np
import threading
import stock_predict as Pp
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


def pulse(widget, times=6, color1="#FFD600", color2="#1976d2", delay=180):
    def _pulse(i):
        if i >= times:
            widget.configure(bg=color2)
            return
        widget.configure(bg=color1 if i % 2 == 0 else color2)
        widget.after(delay, _pulse, i + 1)
    _pulse(0)


class CustomStockPredictor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("📈 Custom Stock Price Predictor")
        self.geometry("1100x760")
        self.configure(bg="#F3F6FB")
        self.minsize(1000 , 800)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background="#F3F6FB", font=("Segoe UI", 11))
        style.configure("Header.TLabel", font=("Segoe UI", 24, "bold"), foreground="#1976d2", background="#F3F6FB")
        style.configure("TButton", font=("Segoe UI", 11, "bold"), background="#1976d2", foreground="white")
        style.map("TButton", background=[('active', '#388e3c')])

        # --- Top Bar ---
        self.header = ttk.Label(self, text="📈 Custom Stock Price Predictor", style="Header.TLabel")
        self.header.grid(row=0, column=0, columnspan=4, pady=(14, 6), padx=16)

        # --- Input Section ---
        input_frm = ttk.Frame(self)
        input_frm.grid(row=1, column=0, sticky="ew", padx=20, pady=5, columnspan=4)
        input_frm.columnconfigure(10, weight=1)

        ttk.Label(input_frm, text="Ticker:").grid(row=0, column=0, padx=3, pady=3, sticky="w")
        TICKER_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "PEP", "NVDA", "IBM", "ORCL","BABA","PYPL","INTC","UBER","ADBE","ZM","NKE","PINS","MCD"]
        self.ticker_var = tk.StringVar(value=TICKER_LIST[0])
        self.ticker_combo = ttk.Combobox(input_frm, textvariable=self.ticker_var, values=TICKER_LIST,
                                         width=10, state="readonly", font=("Consolas", 12))
        self.ticker_combo.grid(row=0, column=1, padx=4)

        ttk.Label(input_frm, text="Look-back:").grid(row=0, column=2, padx=(18, 0))
        self.lookback_var = tk.IntVar(value=60)
        self.lookback_spin = ttk.Spinbox(input_frm, from_=5, to=365, textvariable=self.lookback_var, width=6)
        self.lookback_spin.grid(row=0, column=3, padx=7)

        ttk.Label(input_frm, text="Start Date:").grid(row=0, column=4, padx=(21, 0))
        self.start_date = DateEntry(input_frm, width=11, year=datetime.now().year - 2, date_pattern='yyyy-mm-dd')
        self.start_date.grid(row=0, column=5, padx=7)

        ttk.Label(input_frm, text="End Date:").grid(row=0, column=6, padx=(16, 0))
        self.end_date = DateEntry(input_frm, width=11)
        self.end_date.set_date(datetime.now())
        self.end_date.grid(row=0, column=7, padx=7)

        self.predict_btn = ttk.Button(input_frm, text="Predict ▶", command=self.predict)
        self.predict_btn.grid(row=0, column=8, padx=(23, 2))

        # --- Info Strip ---
        info_frm = tk.Frame(self, bg="#1976d2", height=20)
        info_frm.grid(row=2, column=0, pady=10, sticky="ew", padx=0, columnspan=4)
        info_frm.columnconfigure(0, weight=1)

        self.info_strip = tk.Label(
            info_frm,
            text="Last Price: -- | 7-Day Return: -- | 1-Year Return: --",
            font=("Segoe UI", 14, "bold"),
            bg="#1976d2", fg="#fff", anchor="center", pady=13
        )
        self.info_strip.grid(row=0, column=0, sticky="ew", padx=0)

        # --- Matplotlib Plot 1 ---
        self.fig, self.ax = plt.subplots(figsize=(9.2, 4.9), dpi=96)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=4, column=0, columnspan=4, padx=20, pady=(4, 2), sticky="nsew")
        self.rowconfigure(4, weight=1)
        self.columnconfigure(0, weight=1)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.90, bottom=0.20, left=0.08, right=0.99)

        # --- Matplotlib Plot 2 ---  
        self.fig2, self.ax2 = plt.subplots(figsize=(9.2,3.0), dpi=96)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self)
        self.plot_widget2 = self.canvas2.get_tk_widget()
        self.plot_widget2.grid(row=5, column=0, columnspan=4, padx=20, pady=(4, 2), sticky="nsew")
        self.rowconfigure(5, weight=0)
        self.fig2.tight_layout()
        self.fig2.subplots_adjust(top=0.90, bottom=0.2, left=0.08, right=0.99)
        
    
        # --- Bottom panel ---
        bottom_frame = tk.Frame(self, bg="#F3F6FB")
        bottom_frame.grid(row=6, column=0, pady=(14, 13), columnspan=4, sticky="ew")
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)

        # Previous 7 days (left)
        prevdays_frame = tk.Frame(bottom_frame, bg="#F3F6FB")
        prevdays_frame.grid(row=0, column=0, padx=(40, 12), sticky="nsew")

        prev_label = tk.Label(
            prevdays_frame, text="Previous 7 Days Stock Prices", font=("Segoe UI", 12, "bold"),
            bg="#1976d2", fg="#fff", padx=8, pady=4, bd=2, relief="solid"
        )
        prev_label.pack(anchor="center", pady=(1, 6))

        self.prev_7days_box = tk.Label(
            prevdays_frame, text="--", font=("Consolas", 15, "bold"),
            bg="#FFE082", fg="#333", padx=8, pady=8, relief="groove", width=25, height=7, anchor="n", justify="left"
        )
        self.prev_7days_box.pack(anchor="center")

        # Next 7 day predictions (right)
        sevendays_frame = tk.Frame(bottom_frame, bg="#F3F6FB")
        sevendays_frame.grid(row=0, column=1, padx=(12, 40), sticky="nsew")

        seven_label = tk.Label(
            sevendays_frame, text="Next 7 Days Predictions", font=("Segoe UI", 12, "bold"),
            bg="#1976d2", fg="#fff", padx=8, pady=4, bd=2, relief="solid"
        )
        seven_label.pack(anchor="center", pady=(1, 6))

        self.seven_preds = tk.Label(
            sevendays_frame, text="--", font=("Consolas", 15, "bold"),
            bg="#FFE082", fg="#333", padx=8, pady=8, relief="groove", width=25, height=7, anchor="n", justify="left"
        )
        self.seven_preds.pack(anchor="center")

    def predict(self):
        self.predict_btn.config(text="Predicting...", state="disabled")
        threading.Thread(target=self._predict_main).start()

    def _predict_main(self):
        ticker = self.ticker_var.get().strip().upper()
        try:
            look_back = int(self.lookback_var.get())
        except ValueError:
            self._show_error("Look-back must be a number.")
            return

        try:
            start_date = self.start_date.get_date()
            end_date = self.end_date.get_date()
            if start_date >= end_date:
                self._show_error("Start date must be before end date.")
                return

            data = Pp.fetch_stock_data(ticker, start_date, end_date)
            if data is None or data.empty:
                self._show_error("No data for that ticker/dates.")
                return

            Xtr, Xts, ytr, yts, scaler, df, dates_test = Pp.prepare_data(data, look_back)
            model = Pp.train_model(Xtr, ytr)
            predictions = Pp.make_predictions(model, Xts, scaler).flatten()

            last_actual_date = df['Date'].iloc[-1]
            last_actual_price = float(df['Close'].iloc[-1])

            # Calc returns
            try:
                price_7d_ago = float(df['Close'].iloc[-8])
                r7d = ((last_actual_price - price_7d_ago) / price_7d_ago) * 100
            except Exception:
                r7d = float('nan')

            try:
                price_1y_ago = float(df['Close'].iloc[-252])
                r1y = ((last_actual_price - price_1y_ago) / price_1y_ago) * 100
            except Exception:
                r1y = float('nan')

            r7d_text = f"{r7d:+.2f}%" if r7d == r7d else "n/a"
            r1y_text = f"{r1y:+.2f}%" if r1y == r1y else "n/a"

            info = f"Last Price: ${last_actual_price:.2f} ({last_actual_date.date()}) | 7-Day Return: {r7d_text} | 1-Year Return: {r1y_text}"
            self._set_label_animated(self.info_strip, info)

            # Previous 7 days info box
            prev_7_dates = df['Date'].iloc[-7:]
            prev_7_prices = df['Close'].iloc[-7:]
            prev_text = "\n".join([f"{dt:%a %Y-%m-%d}: ${pr:.2f}" for dt, pr in zip(prev_7_dates, prev_7_prices)])
            self.prev_7days_box.config(text=prev_text)
            pulse(self.prev_7days_box, color1="#FFD600", color2="#FFE082", times=6, delay=120)

            # Next 7 days predictions for the message box (always 7 days)
            seven_preds, seven_days = [], []
            curr_seq = scaler.transform(df[['Close']].values)[-look_back:].reshape(1, -1)
            last_date = df['Date'].iloc[-1]

            for i in range(7):
                pred_scaled = model.predict(curr_seq)
                pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                next_day = last_date + timedelta(days=i + 1)
                seven_preds.append(pred)
                seven_days.append(next_day)
                curr_seq = np.roll(curr_seq, -1)
                curr_seq[0, -1] = scaler.transform([[pred]])[0][0]

            pred_text = "\n".join([f"{date:%a %Y-%m-%d}: ${price:.2f}" for date, price in zip(seven_days, seven_preds)])
            self.seven_preds.config(text=pred_text)
            pulse(self.seven_preds, color1="#FFD600", color2="#FFE082", times=6, delay=120)

            # Longer future forecast for plotting (e.g., 14 or 21 days)
            num_future_days = 21 
            future_preds, future_days = [], []
            curr_seq = scaler.transform(df[['Close']].values)[-look_back:].reshape(1, -1)

            for i in range(num_future_days):
                pred_scaled = model.predict(curr_seq)
                pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                next_day = last_date + timedelta(days=i + 1)
                future_preds.append(pred)
                future_days.append(next_day)
                curr_seq = np.roll(curr_seq, -1)
                curr_seq[0, -1] = scaler.transform([[pred]])[0][0]

            # Plot graph with extended future forecast dashed line
            self._plot_graph(df, predictions, dates_test, future_days, future_preds)
            self._plot_seven_day_forecast(future_days, future_preds)

        except Exception as e:
            self._show_error(f"An unexpected error occurred:\n{e}")

        finally:
            self.predict_btn.config(text="Predict ▶", state="normal")

    def _set_label_animated(self, label, text):
        self.after(0, lambda: label.config(text=text))
        pulse(label, times=4, color1="#FFD600", color2=label['bg'], delay=110)

    def _plot_graph(self, df, predictions, dates_test, forecast_days=None, multi_preds=None):
        self.ax.clear()
        self.ax.plot(df['Date'], df['Close'], label="Actual", lw=2.1, color="#1976d2")
        self.ax.plot(dates_test, predictions, label="Predicted (Test Set)", color="#e65100", lw=2, ls="--")

        if forecast_days is not None and multi_preds is not None:
            self.ax.plot(
                forecast_days, multi_preds,
                label=f"Forecast (Next {len(forecast_days)} Days)",
                color="#005811", lw=2, ls="--", marker='o'
            )

        self.ax.set_title(f"Stock Prediction: {self.ticker_var.get()}", fontsize=15, pad=10)
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price ($)")
        self.ax.grid(linestyle=":", alpha=0.24)
        self.ax.legend()

        all_dates = list(df['Date']) + (forecast_days if forecast_days else [])
        if all_dates:
            self.ax.set_xlim(min(all_dates), max(all_dates))
        self.fig.autofmt_xdate()
        self.canvas.draw_idle()

    def _show_error(self, msg):
        self.predict_btn.config(text="Predict ▶", state="normal")
        messagebox.showerror("Error", msg)

    def _plot_seven_day_forecast(self, forecast_days, forecast_prices):
        self.ax2.clear()
        self.ax2.plot(forecast_days, forecast_prices, marker='o', color="#005811", lw=2.5, label=f"Next {len(forecast_days)}-Day Forecast")
        self.ax2.set_title(f"Next {len(forecast_days)}-Day Predicted Prices: {self.ticker_var.get()}", fontsize=13, pad=8)
        self.ax2.set_xlabel("Date")
        self.ax2.set_ylabel("Predicted Price ($)")
        self.ax2.grid(linestyle=":", alpha=0.25)
        self.ax2.legend()
        self.fig2.autofmt_xdate()
        self.canvas2.draw_idle()


if __name__ == "__main__":
    app = CustomStockPredictor()
    app.mainloop()
