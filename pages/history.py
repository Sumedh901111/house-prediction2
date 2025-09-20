import streamlit as st
import pandas as pd

st.title("📜 Prediction History")

if 'history' not in st.session_state or len(st.session_state.history) == 0:
    st.info("No previous predictions yet.")
else:
    # Convert session state history to DataFrame
    df = pd.DataFrame(st.session_state.history)
    
    # Fix dictionary of lists to columns
    for col in df.columns:
        if isinstance(df[col][0], list):
            df[col] = df[col].apply(lambda x: x[0])
    
    st.dataframe(df)
    
    # Download button
    st.download_button(
        "Download History as CSV",
        df.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv"
    )
