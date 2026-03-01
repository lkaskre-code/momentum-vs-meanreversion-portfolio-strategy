import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. CLASS DEFINITIONS (Tidy Finance Style)
# ==========================================

class DataHandler:
    """Holt und bereinigt Finanzdaten. Trennung von Logik und Datenhaltung."""
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start = start_date
        self.end = end_date
        self.data = None
        self.returns = None
    
    def fetch_data(self):
        raw = yf.download(list(self.tickers.values()), start=self.start, end=self.end, auto_adjust=True, progress=False)
        
        # Flatten MultiIndex columns if necessary
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                df = raw['Close'].copy()
            except KeyError:
                df = raw.xs('Close', axis=1, level=0).copy()
        else:
            df = raw['Close'].copy()

        # Rename columns based on map
        inv_map = {v: k for k, v in self.tickers.items()}
        df.rename(columns=inv_map, inplace=True)
        
        self.data = df.dropna()
        self.returns = self.data.pct_change().dropna()
        return self.data, self.returns

class StrategyEngine:
    """Berechnet Signale und Gewichte. Verhindert Look-Ahead Bias."""
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns
        
    def calc_momentum_weights(self, vol_target=0.12, lookback=21):
        """Volatility Targeting (Risk-Managed Momentum)."""
        stock_ret = self.returns['Stock']
        vol = stock_ret.rolling(window=lookback).std() * np.sqrt(252)
        
        # FIX: Wenn Vola noch nicht berechenbar ist, bleiben wir beim Startgewicht (0.8)
        w_stock = (vol_target / vol).replace([np.inf, -np.inf], 0)
        w_stock = w_stock.clip(upper=1.0).fillna(0.8) 
        
        # Shift(1) verhindert Look-Ahead Bias
        w_stock = w_stock.shift(1).fillna(0.8)
        return w_stock, vol

    def calc_contrarian_weights(self, rsi_window=14, buy_thresh=30, sell_thresh=70):
        """Contrarian Strategy basierend auf RSI Mean-Reversion."""
        prices = self.prices['Stock']
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # FIX: Wilder's Smoothing (Exponential Moving Average) für korrekten RSI
        avg_gain = gain.ewm(alpha=1/rsi_window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_window, adjust=False).mean()
        
        # FIX: Verhindert Division durch 0 bei perfekten Aufwärtsphasen
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs.fillna(np.inf))) 
        
        # FIX: State-Machine (Gedächtnis) für die Signale
        signals = pd.Series(np.nan, index=prices.index)
        signals.iloc[0] = 0.80  # Startgewicht
        
        signals[rsi < buy_thresh] = 1.0  # Panik -> Kaufen
        signals[rsi > sell_thresh] = 0.5 # Gier -> Reduzieren
        
        # ffill() hält das Signal, bis ein neues ausgelöst wird
        signals = signals.ffill() 
        w_stock = signals.shift(1).fillna(0.8)
        
        return w_stock, rsi

class Backtester:
    """Führt den Backtest inkl. Portfolio-Drift und geom. Transaktionskosten durch."""
    def __init__(self, returns, stock_weights, cost_bps=10):
        self.returns = returns
        self.w_stock = stock_weights
        self.w_gold = 1.0 - stock_weights
        self.cost_bps = cost_bps / 10000
        
    def run(self):
        stock_ret = self.returns['Stock']
        gold_ret = self.returns['Gold']
        
        # 1. Brutto-Portfoliorendite
        port_ret_gross = (self.w_stock * stock_ret) + (self.w_gold * gold_ret)
        
        # 2. FIX: Portfolio-Drift berechnen (Gewichte am ENDE des Tages)
        drifted_w_stock = (self.w_stock * (1 + stock_ret)) / (1 + port_ret_gross)
        
        # 3. Turnover basierend auf dem Drift des VORTAGES
        turnover_stock = abs(self.w_stock - drifted_w_stock.shift(1).fillna(0.80))
        total_turnover = turnover_stock * 2  # Kauf und Verkauf
        
        # 4. FIX: Geometrische Transaktionskosten
        costs = total_turnover * self.cost_bps
        net_ret = ((1 + port_ret_gross) * (1 - costs)) - 1
        
        net_ret = net_ret.dropna()
        equity = 100 * (1 + net_ret).cumprod()
        
        return equity, net_ret

# ==========================================
# 2. STREAMLIT UI
# ==========================================
st.set_page_config(layout="wide", page_title="Scientific Portfolio Lab")

st.markdown("""
<style>
    .reportview-container { background: #ffffff }
    h1 { color: #0f172a; font-family: 'Helvetica', sans-serif; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #e9ecef; }
</style>
""", unsafe_allow_html=True)

st.title("Empirical Finance: Momentum vs. Contrarian Framework")
st.markdown("A Tidy Finance implementation for dynamic asset allocation analysis.")

with st.sidebar:
    st.header("1. Data Parameters")
    # HIER GEÄNDERT: SPY als Default-Wert
    ticker_s = st.text_input("Risk Asset (Stock)", "SPY") 
    ticker_g = st.text_input("Safe Asset (Gold)", "GLD")
    st.date_input("Start Date", pd.to_datetime("2005-01-01"), key="start_date")
    
    st.header("2. Strategy Logic")
    st.subheader("Momentum (Vol-Target)")
    vol_target = st.slider("Target Volatility (Ann.)", 0.05, 0.25, 0.12, 0.01)
    
    st.subheader("Contrarian (Mean-Rev)")
    rsi_period = st.number_input("RSI Period", 10, 30, 14)
    rsi_buy = st.slider("Buy Threshold (Panic)", 10, 40, 30)
    rsi_sell = st.slider("Sell Threshold (Greed)", 60, 90, 70)
    
    st.header("3. Frictions")
    cost_bps = st.number_input("Trans. Costs (bps)", 0, 50, 10)

@st.cache_data(ttl=3600)
def load_market_data(tickers_dict, start, end):
    dh = DataHandler(tickers_dict, start, end)
    return dh.fetch_data()

try:
    tickers_dict = {'Stock': ticker_s, 'Gold': ticker_g}
    prices, returns = load_market_data(tickers_dict, st.session_state.start_date, pd.to_datetime("today"))
    
    if prices.empty:
        st.error("Error: No data found. Please check tickers.")
        st.stop()

    strat = StrategyEngine(prices, returns)
    
    w_static = pd.Series(0.80, index=prices.index)
    w_mom, vol_series = strat.calc_momentum_weights(vol_target=vol_target)
    w_con, rsi_series = strat.calc_contrarian_weights(rsi_window=rsi_period, buy_thresh=rsi_buy, sell_thresh=rsi_sell)
    
    bt_static = Backtester(returns, w_static, cost_bps=cost_bps)
    bt_mom = Backtester(returns, w_mom, cost_bps=cost_bps)
    bt_con = Backtester(returns, w_con, cost_bps=cost_bps)
    
    eq_static, ret_static = bt_static.run()
    eq_mom, ret_mom = bt_mom.run()
    eq_con, ret_con = bt_con.run()
    
    # ==========================================
    # 3. SCIENTIFIC VISUALIZATION
    # ==========================================
    def get_metrics(series, name):
        total_ret = (series.iloc[-1] / series.iloc[0]) - 1
        cagr = (series.iloc[-1] / series.iloc[0])**(252/len(series)) - 1
        
        # Wissenschaftlich saubere Metriken für Sharpe
        daily_mean_ret = series.pct_change().mean()
        vol = series.pct_change().std() * np.sqrt(252)
        ann_arithmetic_ret = daily_mean_ret * 252
        sharpe = ann_arithmetic_ret / vol if vol > 0 else 0
        
        roll_max = series.cummax()
        drawdown = series / roll_max - 1.0
        max_dd = drawdown.min()
        
        return [name, f"{total_ret:.1%}", f"{cagr:.1%}", f"{vol:.1%}", f"{sharpe:.2f}", f"{max_dd:.1%}"]

    metrics_data = [
        get_metrics(eq_static, "Static (80/20)"),
        get_metrics(eq_mom, "Momentum (Vol-Target)"),
        get_metrics(eq_con, "Contrarian (RSI)")
    ]

    columns = ["Strategy", "Total Return", "CAGR", "Volatility", "Sharpe Ratio", "Max DD"]
    metrics_df = pd.DataFrame(metrics_data, columns=columns)
    st.table(metrics_df.set_index("Strategy"))

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.30, 0.15, 0.15, 0.20, 0.20],
                        subplot_titles=("Portfolio Value (Base 100)", "Asset Allocation (Stock %)", "Drawdowns", "Metric: Volatility Regime", "Metric: RSI Signal"))

    fig.add_trace(go.Scatter(x=eq_static.index, y=eq_static, name="Static", line=dict(color="grey")), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_mom.index, y=eq_mom, name="Momentum", line=dict(color="#2ca02c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_con.index, y=eq_con, name="Contrarian", line=dict(color="#d62728")), row=1, col=1)
    fig.update_yaxes(title="Index", tickformat="d", row=1, col=1)

    fig.add_trace(go.Scatter(x=w_mom.index, y=w_mom, name="W_Stock (Mom)", line=dict(width=1, dash='dot', color="#2ca02c"), fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=w_con.index, y=w_con, name="W_Stock (Con)", line=dict(width=1, color="#d62728")), row=2, col=1)
    fig.update_yaxes(range=[0, 1.1], title="Stock %", row=2, col=1)

    dd_mom = eq_mom / eq_mom.cummax() - 1
    dd_con = eq_con / eq_con.cummax() - 1
    fig.add_trace(go.Scatter(x=dd_mom.index, y=dd_mom, name="DD Mom", line=dict(width=0), fill='tozeroy', fillcolor='rgba(44, 160, 44, 0.3)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=dd_con.index, y=dd_con, name="DD Con", line=dict(width=0), fill='tozeroy', fillcolor='rgba(214, 39, 40, 0.3)'), row=3, col=1)
    fig.update_yaxes(title="Drawdown", row=3, col=1)

    fig.add_trace(go.Scatter(x=vol_series.index, y=vol_series, name="Market Volatility", line=dict(color="#eab308", width=1), fill='tozeroy', fillcolor='rgba(234, 179, 8, 0.2)'), row=4, col=1)
    fig.add_hline(y=vol_target, line_dash="dot", line_color="black", annotation_text="Target", row=4, col=1)
    fig.update_yaxes(title="Ann. Vola", row=4, col=1)

    fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI (14)", line=dict(color="#06b6d4", width=1)), row=5, col=1)
    fig.add_hline(y=rsi_buy, line_color="#ef4444", line_dash="dash", annotation_text="Buy", row=5, col=1)
    fig.add_hline(y=rsi_sell, line_color="#ef4444", line_dash="dash", annotation_text="Sell", row=5, col=1)
    fig.add_hrect(y0=rsi_buy, y1=rsi_sell, fillcolor="grey", opacity=0.1, line_width=0, row=5, col=1)
    fig.update_yaxes(range=[0, 100], title="RSI", row=5, col=1)

    fig.update_layout(height=1200, template="simple_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regime Check: Correlations")
    corr_window = st.slider("Rolling Correlation Window (Days)", 30, 252, 90)
    roll_corr = returns['Stock'].rolling(corr_window).corr(returns['Gold'])
    
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, fill='tozeroy', name=f'{corr_window}d Corr', line=dict(color='black', width=1)))
    fig_corr.update_layout(title=f"Stock-Gold Correlation (Rolling {corr_window} days)", height=300, template="simple_white")
    st.plotly_chart(fig_corr, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.exception(e)
