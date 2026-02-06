import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Alpha Quant Desk", layout="wide")

# Fix per gli avvisi di versione 2026
WIDTH_MODE = 'stretch' 

st.title("📊 Alpha Execution & Risk Dashboard")
st.caption("Status: Mercato Chiuso (Dati Last Close) | Analisi Intraday & Overnight")

# --- DATA ENGINE ---
@st.cache_data(ttl=60) # Cache di 2 minuti per non sovraccaricare Yahoo
def get_data():
    tickers = ['ES=F', 'EURUSD=X', 'GBPUSD=X', '^GDAXI', 'ORCL', 'ADBE', 'SNPS', 'PAYX', 'AIR', '^GSPC', '^IRX']
    df = yf.download(tickers, period="1y", interval="1d")['Close'].ffill().dropna()
    return df

try:
    data = get_data()
    returns = data.pct_change().dropna()
except Exception as e:
    st.error(f"Errore nel download dati: {e}")
    st.stop()

# --- INPUT & POSIZIONI ---
# (Uso i tuoi dati originali)
positions = {
    'ES=F': {'qty': 1, 'mult': 50, 'margin': 12000, 'entry': 6907.00},
    'EURUSD=X': {'qty': 2, 'mult': 100000, 'margin': 5000, 'entry': 1.1770},
    'GBPUSD=X': {'qty': -3, 'mult': 100000, 'margin': 7500, 'entry': 1.3365},
    '^GDAXI': {'qty': 4, 'mult': 1, 'margin': 7500, 'entry': 24110.00},
    'ORCL': {'qty': 200, 'mult': 1, 'margin': 0, 'entry': 192.06},
    'ADBE': {'qty': 50, 'mult': 1, 'margin': 0, 'entry': 341.42},
    'SNPS': {'qty': 50, 'mult': 1, 'margin': 0, 'entry': 469.93},
    'PAYX': {'qty': 100, 'mult': 1, 'margin': 0, 'entry': 117.51},
    'AIR': {'qty': 400, 'mult': 1, 'margin': 0, 'entry': 91.00}
}

# --- CALCOLI QUANTITATIVI ---
current_prices = data.iloc[-1]
# Definiamo il capitale reale (Equity) per calcolare il rischio sul conto
account_balance = 300000  # <--- Cambia questo valore con il tuo capitale reale

notional_vals = {t: current_prices[t] * v['qty'] * v['mult'] for t, v in positions.items()}
total_notional = sum(abs(v) for v in notional_vals.values())
weights = np.array([notional_vals[t] / total_notional for t in positions.keys()])

# Rendimento Portafoglio vs Mercato
port_rets = returns[list(positions.keys())].dot(weights)
mkt_rets = returns['^GSPC']

# Calcolo P&L monetario storico
daily_dollar_returns = pd.DataFrame()
for t, v in positions.items():
    daily_dollar_returns[t] = returns[t] * (current_prices[t] * v['qty'] * v['mult'])

port_pnl_history = daily_dollar_returns.sum(axis=1)

# IL FIX: Calcoliamo il VaR rispetto al CAPITALE, non al Nozionale
var_95_monetary = np.percentile(port_pnl_history, 5)
# Qui dividiamo per account_balance per vedere l'impatto REALE sul portafoglio
var_95_percentage = abs(var_95_monetary / account_balance) 

# Beta (Sensibilità al mercato)
beta = np.cov(port_rets, mkt_rets)[0, 1] / np.var(mkt_rets)

# --- LAYOUT DASHBOARD ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Exposure", f"${total_notional:,.0f}")
c2.metric("Portfolio Beta", f"{beta:.2f}")
# Adesso questo valore rifletterà la leva: se sei a leva 3x, il VaR sarà 3 volte più alto
c3.metric("VaR 95% Daily", f"{var_95_percentage:.2%}")
c4.metric("Risk-Free (T-Bill)", f"{data['^IRX'].iloc[-1]:.2f}%")

# --- LIVE INVENTORY & PnL MONITOR ---
st.subheader("Live Inventory Monitor")

inventory_data = []
total_live_pnl = 0

for t, v in positions.items():
    p_last = current_prices[t]
    p_entry = v['entry']
    qty = v['qty']
    mult = v['mult']
    
    # Calcolo P&L: (Prezzo Attuale - Prezzo Entrata) * Qty * Moltiplicatore
    # Per gli Short (qty negativa), il calcolo si inverte correttamente in automatico
    unrealized_pnl = (p_last - p_entry) * qty * mult
    total_live_pnl += unrealized_pnl
    
    inventory_data.append({
        "Ticker": t,
        "Side": "LONG" if qty > 0 else "SHORT",
        "Quantity": abs(qty),
        "Entry Price": round(p_entry, 4 if "USD" in t else 2),
        "Last Price": round(p_last, 4 if "USD" in t else 2),
        "Unrealized PnL ($)": unrealized_pnl,
        "Unrealized PnL (%)": (p_last / p_entry - 1) if qty > 0 else (p_entry / p_last - 1)
    })

df_inv = pd.DataFrame(inventory_data)

# Visualizzazione Tabella con Formattazione Condizionale
st.table(df_inv.style.format({
    "Entry Price": "{:,.4f}",
    "Last Price": "{:,.4f}",
    "Unrealized PnL ($)": "{:+,.2f}",
    "Unrealized PnL (%)": "{:+.2%}"
}).applymap(lambda x: 'color: #2ca02c' if (isinstance(x, float) and x > 0) else 'color: #d62728' if (isinstance(x, float) and x < 0) else '', 
            subset=['Unrealized PnL ($)', 'Unrealized PnL (%)']))

st.metric("Total Unrealized P&L", f"${total_live_pnl:,.2f}", delta=f"{(total_live_pnl/account_balance):.2%}")
st.divider()

# --- MACHINE LEARNING (Regime Detection Ordinato) ---
st.subheader("🤖 Market State (Ordered K-Means Clustering)")

# 1. Preparazione dati
ml_data = pd.DataFrame({
    'Ret': mkt_rets, 
    'Vol': mkt_rets.rolling(20).std()
}).dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(ml_data)

# 2. Fit K-Means
km = KMeans(n_clusters=3, n_init=10, random_state=42)
ml_data['Raw_Regime'] = km.fit_predict(scaled_features)

# 3. LOGICA DI ORDINAMENTO (L'Edge Quant)
# Calcoliamo la volatilità media per ogni cluster assegnato casualmente
cluster_vols = ml_data.groupby('Raw_Regime')['Vol'].mean().sort_values()

# Creiamo una mappa: il cluster con Vol più bassa -> 0, media -> 1, alta -> 2
mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_vols.index)}
ml_data['Regime'] = ml_data['Raw_Regime'].map(mapping)

# 4. Plot con colori fissi
# Mappa colori: Green = Stable, Yellow = Volatile, Red = Crash
color_map = {'0': '#2ca02c', '1': '#ff7f0e', '2': '#d62728'}

fig_ml = px.scatter(ml_data, x=ml_data.index, y=data['^GSPC'].reindex(ml_data.index), 
                    color=ml_data['Regime'].astype(str),
                    color_discrete_map=color_map,
                    title="Regimi Ordinati: 0 = Stable, 1 = Volatile, 2 = Crash")

st.plotly_chart(fig_ml, width=WIDTH_MODE)

# --- ANALISI DELLE CORRELAZIONI ---
st.subheader("🔗 Dynamic Correlations")
st.info("Nota: Se le azioni tech (ADBE, ORCL, SNPS) mostrano correlazione > 0.7, la diversificazione è bassa.")
fig_corr = px.imshow(returns[list(positions.keys())].corr(), text_auto=True, color_continuous_scale='RdBu_r')
st.plotly_chart(fig_corr, width=WIDTH_MODE)

import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# --- 7. ANALISI 3D: VOLATILITY SURFACE (VERSIONE CORRETTA) ---
# st.subheader("🧊 3D Risk Surface (Volatility & Time)")

# # Calcoliamo la volatilità rolling
# rolling_vol = returns[list(positions.keys())].rolling(window=20).std().dropna()

# # TRUCCO: Usiamo i nomi reali per gli assi
# z_data = rolling_vol.values
# x_assets = rolling_vol.columns  # Nomi dei Ticker
# y_dates = rolling_vol.index     # Date reali

# fig_3d = go.Figure(data=[go.Surface(
#     z=z_data, 
#     x=x_assets, 
#     y=y_dates, 
#     colorscale='Viridis'
# )])

# fig_3d.update_layout(
#     title='Volatility Surface per Asset',
#     scene=dict(
#         xaxis_title='Asset Name',
#         yaxis_title='Timeline',
#         zaxis_title='Daily Vol'
#     ),
#     width=900, height=700
# )
# st.plotly_chart(fig_3d, use_container_width=True)

# --- 7. ANALISI 3D: VOLATILITY SURFACE (Z-SCORE EDITION) ---
st.subheader("🧊 3D Risk Surface (Relative Volatility Z-Score)")
st.write("Questo grafico normalizza la volatilità: i picchi indicano anomalie statistiche rispetto alla media storica di ogni asset.")

# 1. Calcolo Volatilità Rolling a 20 giorni
rolling_vol = returns[list(positions.keys())].rolling(window=20).std().dropna()

# 2. Trasformazione in Z-Score (Normalizzazione)
# Questo rende confrontabili asset diversi (es. Oro vs Bitcoin)
z_score_vol = (rolling_vol - rolling_vol.mean()) / rolling_vol.std()

# 3. Creazione superficie 3D
fig_3d = go.Figure(data=[go.Surface(
    z=z_score_vol.values, 
    x=z_score_vol.columns,   # Nomi degli Asset
    y=z_score_vol.index,     # Date reali (Timeline)
    colorscale='Hot',        # Colori: Nero -> Rosso -> Giallo -> Bianco
    colorbar=dict(title='Z-Score')
)])

fig_3d.update_layout(
    title='Z-Score Volatility Surface (Rischio Relativo)',
    scene=dict(
        xaxis_title='Assets',
        yaxis_title='Timeline',
        zaxis_title='Z-Score (Sigma)',
        # Migliora la prospettiva iniziale
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    width=1000, height=800
)

st.plotly_chart(fig_3d, use_container_width=True)

# --- 8. PROBABILITY DENSITY FUNCTION (PDF) ---
st.subheader("📊 Distribuzione Reale del Rischio (Non-Normalità)")
c1, c2 = st.columns(2)

with c1:
    # Calcolo KDE (Kernel Density Estimation) per i rendimenti del portafoglio
    kde = gaussian_kde(port_rets)
    x_range = np.linspace(port_rets.min(), port_rets.max(), 1000)
    
    fig_pdf = go.Figure()
    # Curva Reale
    fig_pdf.add_trace(go.Scatter(x=x_range, y=kde(x_range), name="Distribuzione Reale", fill='tozeroy'))
    # Curva Teorica Normale (Gauss)
    norm_pdf = norm.pdf(x_range, port_rets.mean(), port_rets.std())
    fig_pdf.add_trace(go.Scatter(x=x_range, y=norm_pdf, name="Distribuzione Teorica (Gauss)", line=dict(dash='dash')))
    
    fig_pdf.update_layout(title="Reale vs Teorica: Cerca le 'Fat Tails'", xaxis_title="Rendimento Giornaliero")
    st.plotly_chart(fig_pdf, width='stretch')

with c2:
    st.write("**Analisi del Kurtosis (Skewness)**")
    kurt = port_rets.kurtosis()
    skew = port_rets.skew()
    st.metric("Kurtosis (Eccesso)", f"{kurt:.2f}")
    st.write("Se > 3, hai code grasse: i 'Cigni Neri' sono più probabili di quanto pensi.")
    st.metric("Skewness (Asimmetria)", f"{skew:.2f}")

# --- 9. ROLLING SHARPE RATIO (CON PROTEZIONE T-BILL) ---
st.subheader("📈 Rolling Sharpe Ratio (60 Giorni)")

# Recupero tasso risk-free con fallback al 4% se NaN
rf_rate = data['^IRX'].iloc[-1]
if np.isnan(rf_rate):
    rf_rate = 4.0 

# Calcoliamo lo Sharpe Ratio usando i rendimenti reali del portafoglio (quelli che hanno generato il VaR 3.11%)
# Trasformiamo il P&L monetario in rendimento percentuale per il calcolo
rolling_port_rets = port_pnl_history / account_balance
rolling_sharpe = (rolling_port_rets.rolling(60).mean() * 252 - (rf_rate/100)) / (rolling_port_rets.rolling(60).std() * np.sqrt(252))

fig_sharpe = px.line(rolling_sharpe, title=f"Efficienza del Portfolio (Basata su Equity {account_balance}$)", labels={'value': 'Sharpe Ratio', 'Date': 'Tempo'})
st.plotly_chart(fig_sharpe, use_container_width=True)

import plotly.figure_factory as ff

# --- 10. SCENARIO SIMULATOR (WAR ROOM) ---
st.header("🕵️ Intelligence & Stress Simulator")
st.write("Simula un evento macro combinato partendo dal Mark-to-Market (MtM) attuale.")

# PUNTO CHIAVE: L'Equity attuale tiene conto del P&L aperto (-11k nel tuo caso)
current_equity = account_balance + total_live_pnl 
portfolio_beta = beta

c_sim1, c_sim2, c_sim3 = st.columns(3)
with c_sim1:
    s_mkt = st.slider("S&P 500 Move (%)", -20.0, 20.0, -5.0)
with c_sim2:
    s_tech = st.slider("Tech Sector Extra Move (%)", -20.0, 20.0, -2.0)
with c_sim3:
    s_usd = st.slider("USD Strength (%)", -10.0, 10.0, 1.0)

# Calcolo impatto pesato (Lo shock mangia sull'esposizione totale, non sul capitale)
impact_mkt = (account_balance * portfolio_beta * (s_mkt / 100))
impact_tech_extra = ((notional_vals['ORCL'] + notional_vals['ADBE'] + notional_vals['SNPS']) * (s_tech / 100))
impact_fx = (notional_vals['EURUSD=X'] * (-s_usd / 100)) + (notional_vals['GBPUSD=X'] * (s_usd / 100))

total_stress_impact = impact_mkt + impact_tech_extra + impact_fx

# Il saldo finale ora sottrae lo shock dal valore GIA' DECURTATO del P&L attuale
final_stressed_balance = current_equity + total_stress_impact

# Calcolo della variazione percentuale rispetto all'Equity attuale
pct_impact_on_equity = (total_stress_impact / current_equity) if current_equity != 0 else 0

st.metric(
    "P&L Stimato Scenario (Shock Marginale)", 
    f"${total_stress_impact:,.2f}", 
    delta=f"{pct_impact_on_equity:.2%} (vs Equity)",
    delta_color="inverse"
)

col_res1, col_res2 = st.columns(2)
with col_res1:
    st.write(f"Equity Attuale (MtM): **${current_equity:,.2f}**")
with col_res2:
    st.write(f"Capitale Finale Stimato: **${final_stressed_balance:,.2f}**")

# Alert Visivo per il Risk Manager
if final_stressed_balance < (sum(v['margin'] for v in positions.values())):
    st.error("⚠️ CRITICAL: Lo scenario simulato porta il capitale sotto il margine richiesto (MARGIN CALL).")

# --- 11. DRAWDOWN DURATION ANALYSIS ---
st.subheader("📉 Analisi dei Drawdown (Resilienza)")
# Calcoliamo la curva cumulata
cum_ret = (1 + port_rets).cumprod()
peak = cum_ret.cummax()
drawdown = (cum_ret - peak) / peak

fig_dd = px.area(drawdown, title="Storico dei Drawdown (Quanto tempo restiamo sott'acqua?)", color_discrete_sequence=['red'])
st.plotly_chart(fig_dd, width='stretch')

# --- 12. RISK CLUSTER MAP (Z-SCORE) ---
st.subheader("📍 Asset Risk Profiling")
# Creiamo un grafico a bolle: Beta vs Volatilità vs Peso
risk_profile = pd.DataFrame({
    'Asset': list(positions.keys()),
    'Beta': [np.cov(returns[t], mkt_rets)[0,1]/np.var(mkt_rets) for t in positions.keys()],
    'Vol': [returns[t].std() * np.sqrt(252) for t in positions.keys()],
    'Exposure': [abs(v) for v in notional_vals.values()]
})

fig_bubble = px.scatter(risk_profile, x="Beta", y="Vol", size="Exposure", color="Asset",
                 hover_name="Asset", log_x=False, size_max=60,
                 title="Mappa del Rischio: Dimensione = Esposizione Monetaria")
st.plotly_chart(fig_bubble, width='stretch')

# --- AGGIUNTA ALLA SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.header("🎯 Target Risk Control")
target_beta = st.sidebar.slider("Target Portfolio Beta", 0.0, 1.0, 0.3)
inception_days = st.sidebar.number_input("Giorni dall'Inizio Portafoglio", value=30)

# --- FILTRO PERFORMANCE (INCEPTION) ---
# Consideriamo solo gli ultimi N giorni per drawdown e performance reale
real_port_rets = port_rets.tail(inception_days)

# --- 8. REBALANCING ENGINE (HEDGING CALC) ---
st.header("⚖️ Rebalancing & Hedging Optimizer")
st.write(f"Per portare il Beta attuale ({portfolio_beta:.2f}) al Target ({target_beta:.2f}):")

# Calcolo del valore del contratto Future ES (S&P 500)
es_contract_value = current_prices['ES=F'] * 50
beta_gap = portfolio_beta - target_beta
# Formula per l'Hedging: (Beta Gap * Valore Portafoglio) / Valore Contratto Future
hedge_contracts = (beta_gap * total_notional) / es_contract_value

c_reb1, c_reb2 = st.columns(2)
with c_reb1:
    if hedge_contracts > 0:
        st.warning(f"ACTION: SELL (Short) {abs(round(hedge_contracts, 2))} contracts ES=F")
    else:
        st.success(f"ACTION: BUY (Long) {abs(round(hedge_contracts, 2))} contracts ES=F")
    st.caption("Questo bilancia l'esposizione sistematica al mercato.")

# --- 9. EXECUTION SIGNALS: DIRECTIONAL TAKE PROFIT / STOP LOSS ---
st.subheader("🎯 Intelligence di Uscita (Directional Z-Score)")
st.write("I target sono invertiti automaticamente per le posizioni SHORT.")

execution_table = []
tickers_portfolio = list(positions.keys())
positions_config = positions
for t in tickers_portfolio:
    qty = positions_config[t]['qty']
    direction = 1 if qty > 0 else -1  # 1 per Long, -1 per Short
    
    last_ret = returns[t].iloc[-1]
    std_dev = returns[t].std()
    z_score = last_ret / std_dev
    
    # Calcolo Target Direzionali (2 Deviazioni Standard)
    # Se Long: TP = Prezzo + 2sd | SL = Prezzo - 2sd
    # Se Short: TP = Prezzo - 2sd | SL = Prezzo + 2sd
    tp_level = current_prices[t] * (1 + (direction * 2 * std_dev))
    sl_level = current_prices[t] * (1 - (direction * 2 * std_dev))
    
    # Logica di Azione corretta per la direzione
    # Per uno Short, un Z-Score negativo è buono (prezzo scende)
    effective_performance = z_score * direction
    
    status = "HOLD"
    if effective_performance > 1.5: 
        status = "TAKE PROFIT 🟢"
    elif effective_performance < -1.5: 
        status = "STOP LOSS / REVIEW 🔴"
    
    execution_table.append({
        "Asset": t,
        "Side": "LONG" if direction == 1 else "SHORT",
        "Z-Score": round(z_score, 2),
        "Curr Price": round(current_prices[t], 4 if "USD" in t else 2),
        "TP Target (2σ)": round(tp_level, 4 if "USD" in t else 2),
        "SL Target (2σ)": round(sl_level, 4 if "USD" in t else 2),
        "Action": status
    })

st.table(pd.DataFrame(execution_table))

# --- 10. REAL DRAWDOWN (POST-INCEPTION) ---
st.subheader("📉 Real Performance (Last 30 Days)")
cum_real_ret = (1 + real_port_rets).cumprod()
peak_real = cum_real_ret.cummax()
dd_real = (cum_real_ret - peak_real) / peak_real

fig_real_dd = px.area(dd_real, title="Drawdown Reale dall'Inizio Operatività", color_discrete_sequence=['red'])
st.plotly_chart(fig_real_dd, width=WIDTH_MODE)

# --- 11. GREEKS & SENSITIVITY (PERCENTAGE BASED) ---
st.header("📐 Sensitivity Analysis (Institutional Greeks)")

greeks_data = []
for t in tickers_portfolio:
    asset_beta = np.cov(returns[t], mkt_rets)[0, 1] / np.var(mkt_rets)
    beta_weight_pct = (notional_vals[t] * asset_beta) / total_notional
    
    greeks_data.append({
        "Asset": t,
        "Asset Beta": round(asset_beta, 2),
        # Salva il numero puro, senza trasformarlo in stringa con %
        "Weight (%)": notional_vals[t] / total_notional, 
        "Beta-Weighted Contribution": round(beta_weight_pct, 4),
        "Delta ($)": round(notional_vals[t], 2)
    })

df_greeks = pd.DataFrame(greeks_data)
st.table(df_greeks.style.format({
    "Weight (%)": "{:.2%}",
    "Delta ($)": "{:,.2f}",
    "Beta-Weighted Contribution": "{:.2%}"
}))

# --- 12. SURGICAL REBALANCING (ACTIONABLE) ---
st.subheader("⚖️ Surgical Rebalancing Tool")
st.write("Se il Beta-Adjusted Contribution è troppo alto, riduci la size dell'asset specifico.")

# Identifichiamo l'asset che "spinge" di più il rischio
max_risk_asset = df_greeks.loc[df_greeks['Beta-Weighted Contribution'].idxmax()]
st.info(f"L'asset che contribuisce maggiormente al rischio sistemico è **{max_risk_asset['Asset']}**. " 
        f"Riducendo questa posizione del 20%, il Beta del portafoglio scenderebbe significativamente.")

# --- 12. TAIL RISK: HISTORICAL CVaR (EXPECTED SHORTFALL) ---
st.subheader("⚠️ Tail Risk Analysis")

# 1. Creiamo la serie dei rendimenti reali basata sul P&L monetario / Capitale
# balance è il valore che hai usato per il simulatore (300.000)
port_rets_real = port_pnl_history / account_balance 

# 2. Calcolo VaR e CVaR (Expected Shortfall)
var_95_daily_real = np.percentile(port_rets_real, 5)
# Media dei rendimenti nel peggior 5% dei casi
cvar_95_daily_real = port_rets_real[port_rets_real <= var_95_daily_real].mean()

c1_tail, c2_tail = st.columns(2)
with c1_tail:
    # Mostriamo i valori assoluti per pulizia grafica
    st.metric("VaR 95% (Daily)", f"{abs(var_95_daily_real):.2%}", 
              help="Nel 95% dei giorni la perdita non supererà questo valore.")
    
    st.metric("CVaR 95% (Expected Shortfall)", f"{abs(cvar_95_daily_real):.2%}", 
              delta_color="inverse",
              help="Se superiamo il VaR, questa è la perdita media attesa (misura della gravità del crash).")

# --- 13. DISTANCE TO FAILURE (Z-SCORE OF ACCOUNT) ---
with c2_tail:
    # Calcoliamo quanto è "vicino" il Margin Call (Fallimento)
    # total_margin è la somma dei margini richiesti dai broker per i tuoi contratti
    total_margin = sum(v['margin'] for v in positions.values())
    safety_buffer = account_balance - total_margin # Cash libero prima della liquidazione
    
    # Deviazione Standard monetaria del portafoglio (quanti $ oscilla il conto al giorno)
    daily_vol_dollars = port_pnl_history.std() 
    
    # Distance to Failure: Quanti Sigma (deviazioni standard) di perdita servono 
    # per azzerare il cash libero e finire in Margin Call?
    distance_to_failure = safety_buffer / daily_vol_dollars if daily_vol_dollars > 0 else 0
    
    st.metric("Distance to Failure (Sigma)", f"{distance_to_failure:.2f} σ")
    
    if distance_to_failure < 2:
        st.error(f"RISCHIO ELEVATO: Sei a soli {distance_to_failure:.2f} σ dal Margin Call!")
    elif distance_to_failure < 3:
        st.warning("ATTENZIONE: Buffer di sicurezza ridotto (sotto 3 σ).")
    else:
        st.success("Buffer di Margine Adeguato (Solidità istituzionale)")

# --- 14. COMPONENT VaR (REBALANCING INTELLIGENTE) ---
st.subheader("⚖️ Component VaR: Chi domina il rischio?")
st.write("Questo grafico mostra quanto ogni singola posizione contribuisce al VaR totale del 3.11%.")

# 1. Calcoliamo la matrice di covarianza dei rendimenti monetari (dollari)
cov_matrix_dollars = daily_dollar_returns.cov()

# 2. Calcoliamo la varianza totale del portafoglio (monetaria)
port_variance_dollars = np.dot(np.ones(len(positions)), np.dot(cov_matrix_dollars, np.ones(len(positions))))
port_std_dollars = np.sqrt(port_variance_dollars)

# 3. Calcolo del Component VaR: (Covarianza Asset-Portafoglio / Volatilità Portafoglio)
# Questo ci dice quanti dollari di rischio apporta ogni asset
marginal_contribution = cov_matrix_dollars.sum(axis=1) / port_std_dollars
component_var_dollars = marginal_contribution # Contributo in $

# 4. Trasformiamo in percentuale sul Capitale Totale (per coerenza con il 3.11%)
component_var_pct = (component_var_dollars * 1.645) / account_balance 

# Creazione Grafico
fig_comp_var = px.bar(
    x=component_var_pct.index, 
    y=component_var_pct.values, 
    title="Contributo Percentuale al VaR Totale",
    labels={'x': 'Asset', 'y': 'Contributo al VaR (%)'},
    color=component_var_pct.values,
    color_continuous_scale='Reds'
)

fig_comp_var.update_layout(showlegend=False)
st.plotly_chart(fig_comp_var, use_container_width=True)

# Tip per l'utente
max_risk_ticker = component_var_pct.idxmax()
st.info(f"💡 **Insight:** L'asset **{max_risk_ticker}** è il principale motore del tuo rischio. "
        f"Una riduzione della size su questo ticker abbasserebbe il VaR più velocemente rispetto agli altri.")

# --- 1. DOWNLOAD DATI VOLUME PER ADV ---
@st.cache_data(ttl=10)
def load_volume_data():
    tickers = list(positions_config.keys())
    # Scarichiamo i dati storici inclusi i volumi
    df_vol = yf.download(tickers, period="60d", interval="1d")['Volume'].ffill()
    return df_vol

volume_data = load_volume_data()

# --- 2. LIQUIDITY STRESS TEST (ADV BASED) ---
st.header("💧 Institutional Liquidity Analysis (ADV)")
st.write("Analisi basata sull'Average Daily Volume (30gg). Obiettivo: liquidazione < 10% del volume giornaliero.")

adv_30 = volume_data.tail(30).mean()
liquidity_stats = []

for t in tickers_portfolio:
    shares = positions_config[t]['qty'] * (50 if t == 'ES=F' else 1) # Moltiplicatore per futures
    asset_adv = adv_30[t]
    
    # % di partecipazione necessaria per chiudere in 1 giorno
    pct_of_adv = (abs(shares) / asset_adv) if asset_adv > 0 else 0
    
    # Days to Liquidate (assumendo di non superare il 10% del volume giornaliero per non fare slippage)
    days_to_liquidate = (abs(shares) / (0.10 * asset_adv)) if asset_adv > 0 else 0
    
    liquidity_stats.append({
        "Asset": t,
        "Avg Volume (30d)": f"{int(asset_adv):,}",
        "Your Size (Shares/Contr)": abs(shares),
        "% of Daily ADV": f"{pct_of_adv:.4%}",
        "Days to Liquidate (10% limit)": round(days_to_liquidate, 2)
    })

st.table(pd.DataFrame(liquidity_stats))

# --- 3. WHAT-IF REBALANCING SIMULATOR ---
st.header("🧪 What-If Rebalancing Simulator")
st.write("Simula la chiusura parziale di una posizione per vedere l'impatto sul VaR reale del conto.")

col_w1, col_w2 = st.columns(2)
with col_w1:
    asset_to_cut = st.selectbox("Seleziona Asset da ridurre", tickers_portfolio, key="whatif_asset")
    reduction_pct = st.slider(f"Riduci size di {asset_to_cut} (%)", 0, 100, 0)

# LOGICA MONETARIA: Creiamo un P&L virtuale riducendo solo l'asset scelto
virtual_pnl_history = port_pnl_history.copy()
# Sottraiamo la quota di P&L dell'asset selezionato in base alla riduzione
asset_pnl_contribution = daily_dollar_returns[asset_to_cut]
virtual_pnl_history = port_pnl_history - (asset_pnl_contribution * (reduction_pct / 100))

# Calcolo nuove metriche virtuali sul capitale (account_balance)
v_var_95_monetary = np.percentile(virtual_pnl_history, 5)
v_var_95_pct = abs(v_var_95_monetary / account_balance)
v_cvar_95_pct = abs(virtual_pnl_history[virtual_pnl_history <= v_var_95_monetary].mean() / account_balance)

with col_w2:
    st.write("### Impatto Previsto sul Capitale")
    delta_var = v_var_95_pct - var_95_percentage
    st.metric("Nuovo VaR 95% (Daily)", f"{v_var_95_pct:.2%}", delta=f"{delta_var:.2%}", delta_color="inverse")
    st.metric("Nuovo CVaR (95%)", f"{v_cvar_95_pct:.2%}", delta=f"{(v_cvar_95_pct - abs(cvar_95_daily_real)):.2%}", delta_color="inverse")

# --- 15. HIERARCHICAL CLUSTERING (DENDROGRAM) ---
st.header("🌳 Risk Taxonomy (Hierarchical Clustering)")
st.write("Questo grafico raggruppa gli asset che 'respirano' insieme. Se due rami si chiudono subito, quegli asset sono lo stesso rischio.")

# Calcolo della matrice di linkage
corr_matrix = returns[tickers_portfolio].corr().values
# Trasformiamo la correlazione in "distanza" (distanza = sqrt(2 * (1 - rho)))
distance_matrix = sch.distance.pdist(returns[tickers_portfolio].T, metric='correlation')
linkage = sch.linkage(distance_matrix, method='ward')

# Creazione del plot con Matplotlib (Streamlit lo supporta bene)
fig_den, ax_den = plt.subplots(figsize=(10, 5))
dendro = sch.dendrogram(linkage, labels=tickers_portfolio, ax=ax_den, orientation='top')
plt.xticks(rotation=45)
plt.title("Portfolio Correlation Dendrogram")

st.pyplot(fig_den)

# --- NOTA SULLE VOLATILITÀ (PER TUA CHIAREZZA NELLA DASHBOARD) ---
st.sidebar.info(f"""
**Legenda Volatilità:**
- **Grafico 3D:** Daily Vol (0.01 = 1%)
- **Bubble Chart:** Annual Vol (0.60 = 60%)
- **Relazione:** Daily * 15.8 = Annual
""")