import streamlit as st
import pandas as pd
import clickhouse_connect
import plotly.express as px
import os

st.set_page_config(page_title="Insurance Pipeline Dashboard", page_icon="📊", layout="wide")

st.title("📊 Insurance Data Pipeline - Monitoring Dashboard")
st.markdown("Dashboard analytique en temps réel branché sur **ClickHouse**.")

# Configuration ClickHouse
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")

@st.cache_data(ttl=60)
def load_data():
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
        )
        # Check if table exists and has data
        result = client.query("EXISTS TABLE insurance.claims_transformed")
        if result.result_rows[0][0] == 0:
            return pd.DataFrame()
            
        df = client.query_df("SELECT * FROM insurance.claims_transformed")
        return df
    except Exception as e:
        st.error(f"Erreur de connexion à ClickHouse: {e}")
        return pd.DataFrame()

with st.spinner('Chargement des données depuis ClickHouse...'):
    df = load_data()

if df.empty:
    st.warning("⚠️ Aucune donnée trouvée dans la table `insurance.claims_transformed`. Le pipeline Airflow a-t-il bien tourné ?")
else:
    # --- KPIs ---
    st.header("📈 Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    
    total_claims = len(df)
    total_amount = df['claim_amount'].sum()
    avg_amount = df['claim_amount'].mean()
    regions_count = df['region'].nunique()
    
    col1.metric("Total Lignes Chargées", f"{total_claims:,}")
    col2.metric("Montant Total ($)", f"${total_amount:,.2f}")
    col3.metric("Montant Moyen ($)", f"${avg_amount:,.2f}")
    col4.metric("Régions Couvertes", regions_count)
    
    st.divider()

    # --- Charts ---
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Répartition par Type d'Assurance")
        fig_type = px.pie(df, names='insurance_type', hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig_type, use_container_width=True)
        
    with colB:
        st.subheader("Montant des Réclamations par Région")
        df_region = df.groupby('region')['claim_amount'].sum().reset_index()
        fig_region = px.bar(df_region, x='region', y='claim_amount', color='region', 
                            color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_region, use_container_width=True)

    st.divider()
    
    colC, colD = st.columns(2)
    
    with colC:
        st.subheader("Catégorisation des Montants")
        df_cat = df['amount_category'].value_counts().reset_index()
        df_cat.columns = ['amount_category', 'count']
        fig_cat = px.bar(df_cat, x='amount_category', y='count', color='amount_category')
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with colD:
        st.subheader("Évolution dans le temps (Année/Mois)")
        # Assuming claim_date is available and parsed correctly
        if 'claim_date' in df.columns:
            df_time = df.groupby(['claim_year', 'claim_month']).size().reset_index(name='count')
            df_time['date_str'] = df_time['claim_year'].astype(str) + '-' + df_time['claim_month'].astype(str).str.zfill(2)
            df_time = df_time.sort_values(['claim_year', 'claim_month'])
            fig_time = px.line(df_time, x='date_str', y='count', markers=True)
            st.plotly_chart(fig_time, use_container_width=True)

    st.divider()
    
    # --- Data Quality Metrics ---
    st.header("🛡️ Métriques de Qualité (Mock)")
    q1, q2, q3 = st.columns(3)
    q1.metric("SLA Pipeline", "✅ Respecté (< 1h)")
    q2.metric("Taux de Valeurs Nulles", "0.00%", delta="-0.05% vs hier", delta_color="inverse")
    q3.metric("Taux d'Invalides", "0.00%")
    
    st.divider()
    
    st.subheader("Aperçu des Données (100 premières lignes)")
    st.dataframe(df.head(100), use_container_width=True)
