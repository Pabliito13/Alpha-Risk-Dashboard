import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Alpha Quant Desk", layout="wide")

# Fix per gli avvisi di versione 2026
WIDTH_MODE = 'stretch' 

st.title("📊 Alpha Execution & Risk Dashboard")
st.caption("Status: Mercato Chiuso (Dati Last Close) | Analisi Intraday & Overnight")

# --- DATA ENGINE ---
@st.cache_data(ttl=60) # Cache di un'ora per non sovraccaricare Yahoo
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
    'ES=F': {'qty': 1, 'mult': 50, 'margin': 12000},
    'EURUSD=X': {'qty': 2, 'mult': 100000, 'margin': 5000},
    'GBPUSD=X': {'qty': -3, 'mult': 100000, 'margin': 7500},
    '^GDAXI': {'qty': 4, 'mult': 1, 'margin': 7500},
    'ORCL': {'qty': 200, 'mult': 1, 'margin': 0},
    'ADBE': {'qty': 50, 'mult': 1, 'margin': 0},
    'SNPS': {'qty': 50, 'mult': 1, 'margin': 0},
    'PAYX': {'qty': 100, 'mult': 1, 'margin': 0},
    'AIR': {'qty': 400, 'mult': 1, 'margin': 0}
}

# --- CALCOLI QUANTITATIVI ---
current_prices = data.iloc[-1]
notional_vals = {t: current_prices[t] * v['qty'] * v['mult'] for t, v in positions.items()}
total_notional = sum(abs(v) for v in notional_vals.values())
weights = np.array([notional_vals[t] / total_notional for t in positions.keys()])

# Rendimento Portafoglio vs Mercato
port_rets = returns[list(positions.keys())].dot(weights)
mkt_rets = returns['^GSPC']

# Beta (Sensibilità al mercato)
beta = np.cov(port_rets, mkt_rets)[0, 1] / np.var(mkt_rets)

# --- LAYOUT DASHBOARD ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Exposure", f"${total_notional:,.0f}")
c2.metric("Portfolio Beta", f"{beta:.2f}")
c3.metric("VaR 95% Daily", f"{np.percentile(port_rets, 5):.2%}")
c4.metric("Risk-Free (T-Bill)", f"{data['^IRX'].iloc[-1]:.2f}%")

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

# --- 7. ANALISI 3D: VOLATILITY SURFACE ---
st.subheader("🧊 3D Risk Surface (Volatility & Time)")
st.write("Questo grafico mostra l'intensità del rischio (Z-Score) per ogni asset nell'ultimo periodo.")

# Calcoliamo la volatilità rolling a 20 giorni per tutti i ticker
rolling_vol = returns[list(positions.keys())].rolling(window=20).std().dropna()

# Creazione superficie 3D
z_data = rolling_vol.values
x_data = np.arange(len(rolling_vol.columns)) # Asset
y_data = np.arange(len(rolling_vol.index))   # Tempo

fig_3d = go.Figure(data=[go.Surface(z=z_data, colorscale='Viridis')])
fig_3d.update_layout(
    title='Volatility Surface per Asset',
    scene=dict(
        xaxis_title='Assets (Index)',
        yaxis_title='Time (Days)',
        zaxis_title='Volatility'
    ),
    width=900, height=700
)
st.plotly_chart(fig_3d, width='stretch')

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

# --- 9. ROLLING SHARPE RATIO ---
st.subheader("📈 Rolling Sharpe Ratio (60 Giorni)")
# Calcoliamo lo Sharpe Ratio su una finestra mobile di 60 giorni
rolling_sharpe = (port_rets.rolling(60).mean() * 252 - 0.04) / (port_rets.rolling(60).std() * np.sqrt(252))
fig_sharpe = px.line(rolling_sharpe, title="Efficienza del Portfolio nel Tempo")
st.plotly_chart(fig_sharpe, width='stretch')

import plotly.figure_factory as ff

# --- 10. SCENARIO SIMULATOR (WAR ROOM) ---
st.header("🕵️ Intelligence & Stress Simulator")
st.write("Simula un evento macro combinato per vedere l'impatto sul capitale reale.")

balance = 300000  # Capitale iniziale
portfolio_beta = beta

c_sim1, c_sim2, c_sim3 = st.columns(3)
with c_sim1:
    s_mkt = st.slider("S&P 500 Move (%)", -20.0, 20.0, -5.0)
with c_sim2:
    s_tech = st.slider("Tech Sector Extra Move (%)", -20.0, 20.0, -2.0)
with c_sim3:
    s_usd = st.slider("USD Strength (%)", -10.0, 10.0, 1.0)

# Calcolo impatto pesato
# Usiamo il Beta per l'S&P, e i pesi specifici per tech e forex
impact_mkt = (balance * portfolio_beta * (s_mkt / 100))
impact_tech_extra = ((notional_vals['ORCL'] + notional_vals['ADBE'] + notional_vals['SNPS']) * (s_tech / 100))
# Se USD sale (+), EURUSD e GBPUSD scendono (negativo per i long, positivo per gli short)
impact_fx = (notional_vals['EURUSD=X'] * (-s_usd / 100)) + (notional_vals['GBPUSD=X'] * (s_usd / 100))

total_impact = impact_mkt + impact_tech_extra + impact_fx
new_balance = balance + total_impact

st.metric("P&L Stimato Scenario", f"${total_impact:,.2f}", delta=f"{(total_impact/balance):.2%}")
st.write(f"Capitale Finale Stimato: **${new_balance:,.2f}**")

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
        st.warning(f"ACTION: VENDERE (Short) {abs(round(hedge_contracts, 2))} contratti ES=F")
    else:
        st.success(f"ACTION: ACQUISTARE (Long) {abs(round(hedge_contracts, 2))} contratti ES=F")
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
var_95_daily = np.percentile(port_rets, 5)
# CVaR: Media delle perdite che superano il VaR
cvar_95_daily = port_rets[port_rets <= var_95_daily].mean()

c1_tail, c2_tail = st.columns(2)
with c1_tail:
    st.metric("VaR 95% (Daily)", f"{var_95_daily:.2%}", help="La perdita massima nel 95% dei casi")
    st.metric("CVaR 95% (Expected Shortfall)", f"{cvar_95_daily:.2%}", delta_color="inverse",
              help="Se buchiamo il VaR, questa è la perdita media che dobbiamo aspettarci")

# --- 13. DISTANCE TO FAILURE (Z-SCORE OF ACCOUNT) ---
with c2_tail:
    # Calcoliamo quanto è "vicino" il Margin Call (Fallimento)
    total_margin = sum(v['margin'] for v in positions_config.values())
    safety_buffer = balance - total_margin # Quanto cash 'libero' abbiamo
    daily_vol_dollars = total_notional * port_rets.std()
    
    # Quante deviazioni standard di perdita servono per azzerare il buffer?
    distance_to_failure = safety_buffer / daily_vol_dollars if daily_vol_dollars > 0 else 0
    
    st.metric("Distance to Failure (Sigma)", f"{distance_to_failure:.2f} σ")
    if distance_to_failure < 2:
        st.error("RISCHIO ELEVATO: Sei a meno di 2 deviazioni standard dal Margin Call!")
    else:
        st.success("Buffer di Margine Adeguato")

# --- 14. COMPONENT VaR (REBALANCING INTELLIGENTE) ---
st.subheader("⚖️ Component VaR: Chi ruba più rischio?")
# Mostra quanto ogni asset contribuisce al rischio totale
cov_matrix = returns[tickers_portfolio].cov()
marginal_var = (cov_matrix.dot(weights)) / port_rets.std()
component_var = weights * marginal_var

fig_comp_var = px.bar(x=tickers_portfolio, y=component_var, 
                      title="Contributo al Rischio (Se vuoi ridurre il VaR, taglia i picchi più alti)")
st.plotly_chart(fig_comp_var, width=WIDTH_MODE)

# --- 1. DOWNLOAD DATI VOLUME PER ADV ---
@st.cache_data(ttl=60)
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
st.write("Simula la chiusura parziale di una posizione per vedere l'impatto sui rischi di coda.")

col_w1, col_w2 = st.columns(2)
with col_w1:
    asset_to_cut = st.selectbox("Seleziona Asset da ridurre", tickers_portfolio)
    reduction_pct = st.slider(f"Riduci size di {asset_to_cut} (%)", 0, 100, 0)

# Calcolo "Virtuale"
virtual_weights = weights.copy()
asset_idx = tickers_portfolio.index(asset_to_cut)
virtual_weights[asset_idx] = virtual_weights[asset_idx] * (1 - reduction_pct/100)
# Rinormalizziamo i pesi (anche se in realtà se vendi e tieni cash il totale cambia)
virtual_port_rets = returns[tickers_portfolio].dot(virtual_weights)

# Metriche Virtuali
v_beta = np.cov(virtual_port_rets, mkt_rets)[0, 1] / np.var(mkt_rets)
v_var_95 = np.percentile(virtual_port_rets, 5)
v_cvar_95 = virtual_port_rets[virtual_port_rets <= v_var_95].mean()

with col_w2:
    st.write("### Impatto Previsto")
    st.metric("Nuovo Portfolio Beta", f"{v_beta:.2f}", delta=f"{v_beta - portfolio_beta:.2f}", delta_color="inverse")
    st.metric("Nuovo CVaR (95%)", f"{v_cvar_95:.2%}", delta=f"{v_cvar_95 - cvar_95_daily:.2%}", delta_color="normal")
    
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

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
#update
