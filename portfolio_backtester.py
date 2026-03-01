import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MarketData:
    """Fetches and cleans financial data from Yahoo Finance."""
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start = start_date
        self.end = end_date
    
    def download(self):
        raw_data = yf.download(
            list(self.tickers.values()), 
            start=self.start, 
            end=self.end, 
            auto_adjust=True, 
            progress=False
        )
        
        # Handle potential MultiIndex from yfinance
        if isinstance(raw_data.columns, pd.MultiIndex):
            try:
                df_close = raw_data['Close'].copy()
            except KeyError:
                df_close = raw_data.xs('Close', axis=1, level=0).copy()
        else:
            df_close = raw_data['Close'].copy()

        # Map ticker symbols back to custom names (e.g. 'Stock', 'Gold')
        inverse_map = {v: k for k, v in self.tickers.items()}
        df_close.rename(columns=inverse_map, inplace=True)
        
        prices = df_close.dropna()
        returns = prices.pct_change().dropna()
        
        return prices, returns

class AllocationStrategies:
    """Computes target weights for different portfolio strategies."""
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns
        
    def get_volatility_target_weights(self, target_vol=0.12, lookback=21):
        stock_returns = self.returns['Stock']
        
        # Calculate annualized rolling volatility
        rolling_vol = stock_returns.rolling(window=lookback).std() * np.sqrt(252)
        
        # Calculate weight to match target vola, cap at 1.0 (100% equity)
        w_stock = (target_vol / rolling_vol).replace([np.inf, -np.inf], 0)
        w_stock = w_stock.clip(upper=1.0).fillna(0.8) 
        
        # Shift by 1 day to prevent look-ahead bias
        w_stock = w_stock.shift(1).fillna(0.8)
        
        return w_stock, rolling_vol

    def get_rsi_contrarian_weights(self, window=14, buy_threshold=30, sell_threshold=70):
        stock_prices = self.prices['Stock']
        price_diff = stock_prices.diff()
        
        # Separate gains and losses
        gains = price_diff.where(price_diff > 0, 0.0)
        losses = -price_diff.where(price_diff < 0, 0.0)
        
        # Exponential moving average for standard RSI calculation
        avg_gain = gains.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1/window, adjust=False).mean()
        
        # Calculate RSI, handling edge cases where avg_loss is 0
        relative_strength = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + relative_strength.fillna(np.inf))) 
        
        # Generate signals based on thresholds
        signals = pd.Series(np.nan, index=stock_prices.index)
        signals.iloc[0] = 0.80  # Initial weight
        
        signals[rsi < buy_threshold] = 1.0  # Buy signal
        signals[rsi > sell_threshold] = 0.5 # Sell signal
        
        # Forward fill to hold the position until the next signal
        signals = signals.ffill() 
        w_stock = signals.shift(1).fillna(0.8)
        
        return w_stock, rsi

class PortfolioBacktest:
    """Runs the backtest simulation including portfolio drift and transaction costs."""
    def __init__(self, returns, weights_stock, cost_bps=10):
        self.returns = returns
        self.w_stock = weights_stock
        self.w_gold = 1.0 - weights_stock
        self.cost_bps = cost_bps / 10000
        
    def run(self):
        ret_stock = self.returns['Stock']
        ret_gold = self.returns['Gold']
        
        # Gross return of the portfolio
        gross_return = (self.w_stock * ret_stock) + (self.w_gold * ret_gold)
        
        # Calculate actual weight at the end of the day (portfolio drift)
        drifted_w_stock = (self.w_stock * (1 + ret_stock)) / (1 + gross_return)
        
        # Turnover is the difference between today's target and yesterday's drifted weight
        turnover_stock = abs(self.w_stock - drifted_w_stock.shift(1).fillna(0.80))
        total_turnover = turnover_stock * 2  # Accounting for both buy and sell
        
        # Deduct transaction costs geometrically
        costs = total_turnover * self.cost_bps
        net_return = ((1 + gross_return) * (1 - costs)) - 1
        
        net_return = net_return.dropna()
        equity_curve = 100 * (1 + net_return).cumprod()
        
        return equity_curve, net_return

@st.cache_data(ttl=3600)
def load_data(tickers, start, end):
    handler = MarketData(tickers, start, end)
    return handler.download()

def calculate_metrics(series, strategy_name):
    """Helper function to compute standard performance metrics."""
    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    cagr = (series.iloc[-1] / series.iloc[0])**(252/len(series)) - 1
    
    daily_returns = series.pct_change().dropna()
    mean_return = daily_returns.mean()
    volatility = daily_returns.std() * np.sqrt(252)
    
    ann_arithmetic_ret = mean_return * 252
    sharpe = ann_arithmetic_ret / volatility if volatility > 0 else 0
    
    drawdown = series / series.cummax() - 1.0
    max_dd = drawdown.min()
    
    return [strategy_name, f"{total_return:.1%}", f"{cagr:.1%}", f"{volatility:.1%}", f"{sharpe:.2f}", f"{max_dd:.1%}"]

def main():
    st.set_page_config(layout="wide", page_title="Portfolio Allocation Analysis")

    st.markdown("""
    <style>
        .reportview-container { background: #ffffff }
        h1 { color: #0f172a; font-family: 'Helvetica', sans-serif; }
        .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #e9ecef; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Portfolio Allocation Analysis: Vol-Targeting vs. Mean-Reversion")
    st.markdown("A quantitative backtest comparing dynamic asset allocation strategies.")

    with st.sidebar:
        st.header("1. Parameters")
        ticker_stock = st.text_input("Risk Asset", "SPY") 
        ticker_gold = st.text_input("Safe Asset", "GLD")
        start_date = st.date_input("Start Date", pd.to_datetime("2000-01-01"))
        
        st.header("2. Strategy Settings")
        st.subheader("Volatility Targeting")
        vol_target = st.slider("Target Volatility (Ann.)", 0.05, 0.25, 0.12, 0.01)
        
        st.subheader("RSI Mean-Reversion")
        rsi_window = st.number_input("RSI Window", 10, 30, 14)
        rsi_buy = st.slider("Buy Threshold", 10, 40, 30)
        rsi_sell = st.slider("Sell Threshold", 60, 90, 70)
        
        st.header("3. Market Frictions")
        cost_bps = st.number_input("Transaction Costs (bps)", 0, 50, 10)

    try:
        tickers = {'Stock': ticker_stock, 'Gold': ticker_gold}
        prices, returns = load_data(tickers, start_date, pd.to_datetime("today"))
        
        if prices.empty:
            st.warning("No data found for the selected tickers or date range.")
            return

        strategy = AllocationStrategies(prices, returns)
        
        w_static = pd.Series(0.80, index=prices.index)
        w_vol, vol_series = strategy.get_volatility_target_weights(target_vol=vol_target)
        w_rsi, rsi_series = strategy.get_rsi_contrarian_weights(window=rsi_window, buy_threshold=rsi_buy, sell_threshold=rsi_sell)
        
        bt_static = PortfolioBacktest(returns, w_static, cost_bps=cost_bps)
        bt_vol = PortfolioBacktest(returns, w_vol, cost_bps=cost_bps)
        bt_rsi = PortfolioBacktest(returns, w_rsi, cost_bps=cost_bps)
        
        eq_static, _ = bt_static.run()
        eq_vol, _ = bt_vol.run()
        eq_rsi, _ = bt_rsi.run()
        
        metrics = [
            calculate_metrics(eq_static, "Static (80/20)"),
            calculate_metrics(eq_vol, "Vol-Targeting"),
            calculate_metrics(eq_rsi, "RSI Contrarian")
        ]

        df_metrics = pd.DataFrame(metrics, columns=["Strategy", "Total Return", "CAGR", "Volatility", "Sharpe Ratio", "Max DD"])
        st.table(df_metrics.set_index("Strategy"))

        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.30, 0.15, 0.15, 0.20, 0.20],
            subplot_titles=("Equity Curve", "Equity Allocation (%)", "Drawdowns", "Rolling Volatility", "RSI Indicator")
        )

        fig.add_trace(go.Scatter(x=eq_static.index, y=eq_static, name="Static", line=dict(color="grey")), row=1, col=1)
        fig.add_trace(go.Scatter(x=eq_vol.index, y=eq_vol, name="Vol-Targeting", line=dict(color="#2ca02c")), row=1, col=1)
        fig.add_trace(go.Scatter(x=eq_rsi.index, y=eq_rsi, name="Contrarian", line=dict(color="#d62728")), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=w_vol.index, y=w_vol, name="Weight (Vol)", line=dict(width=1, dash='dot', color="#2ca02c"), fill='tozeroy'), row=2, col=1)
        fig.add_trace(go.Scatter(x=w_rsi.index, y=w_rsi, name="Weight (RSI)", line=dict(width=1, color="#d62728")), row=2, col=1)
        fig.update_yaxes(range=[0, 1.1], row=2, col=1)

        dd_vol = eq_vol / eq_vol.cummax() - 1
        dd_rsi = eq_rsi / eq_rsi.cummax() - 1
        fig.add_trace(go.Scatter(x=dd_vol.index, y=dd_vol, name="DD Vol-Target", line=dict(width=0), fill='tozeroy', fillcolor='rgba(44, 160, 44, 0.3)'), row=3, col=1)
        fig.add_trace(go.Scatter(x=dd_rsi.index, y=dd_rsi, name="DD RSI", line=dict(width=0), fill='tozeroy', fillcolor='rgba(214, 39, 40, 0.3)'), row=3, col=1)

        fig.add_trace(go.Scatter(x=vol_series.index, y=vol_series, name="Volatility", line=dict(color="#eab308", width=1), fill='tozeroy', fillcolor='rgba(234, 179, 8, 0.2)'), row=4, col=1)
        fig.add_hline(y=vol_target, line_dash="dot", line_color="black", annotation_text="Target", row=4, col=1)

        fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI", line=dict(color="#06b6d4", width=1)), row=5, col=1)
        fig.add_hline(y=rsi_buy, line_color="#ef4444", line_dash="dash", annotation_text="Buy", row=5, col=1)
        fig.add_hline(y=rsi_sell, line_color="#ef4444", line_dash="dash", annotation_text="Sell", row=5, col=1)
        fig.add_hrect(y0=rsi_buy, y1=rsi_sell, fillcolor="grey", opacity=0.1, line_width=0, row=5, col=1)

        fig.update_layout(height=1200, template="simple_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Market Regime Analysis")
        corr_window = st.slider("Rolling Correlation Window (Days)", 30, 252, 90)
        rolling_corr = returns['Stock'].rolling(corr_window).corr(returns['Gold'])
        
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, fill='tozeroy', name=f'{corr_window}d Corr', line=dict(color='black', width=1)))
        fig_corr.update_layout(title=f"Stock-Gold Correlation (Rolling {corr_window} days)", height=300, template="simple_white")
        st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"Execution error: {e}")

if __name__ == "__main__":
    main()
