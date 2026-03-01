# Systematic Portfolio Lab: Momentum vs. Contrarian Strategies 

**[Click here to open the Live App]**

An interactive Python-based web application built with Streamlit to backtest, analyze, and visualize dynamic asset allocation strategies. 

This project explores the intersection of financial theory and data science by comparing static portfolios against systematic rules-based strategies using real-world market data.

##  Overview

The application fetches historical market data via the Yahoo Finance API and runs a backtest simulating portfolio drift and transaction costs. It visualizes the performance, risk metrics, and underlying signals of three distinct allocation models:

1. **Static Benchmark (80/20):** A traditional buy-and-hold approach.
2. **Momentum (Volatility Targeting):** Scales equity exposure inversely to market volatility. It reduces risk during turbulent regimes and increases it during calm markets to maintain a constant target volatility.
3. **Contrarian (RSI Mean-Reversion):** A tactical strategy that buys into fear (RSI < Buy Threshold) and reduces exposure during greed phases (RSI > Sell Threshold).

##  System Architecture

The code is structured using Object-Oriented Programming (OOP) to cleanly separate data, logic, and execution—following best practices for maintainability:

* `DataHandler`: Responsible for fetching, cleaning, and formatting raw API data.
* `StrategyEngine`: Calculates technical indicators (Rolling Volatility, RSI) and generates target portfolio weights while strictly preventing look-ahead bias.
* `Backtester`: Simulates the actual investment process, calculating daily portfolio drift, turnover, and geometric transaction costs (in basis points).

##  Tech Stack

* **Language:** Python
* **Frontend UI:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Financial Data:** yfinance (Yahoo Finance API)
* **Visualization:** Plotly (interactive charts & subplots)

##  How to Run Locally

If you want to run this project on your local machine, follow these steps:


Install the required dependencies:

Bash
pip install -r requirements.txt
Run the Streamlit app:

Bash
streamlit run portfolio_backtester.py
