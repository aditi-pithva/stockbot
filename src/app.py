import streamlit as st

st.title("StockBot Test App")

st.write("Hello! This is a basic Streamlit app to test deployment.")

user_input = st.text_input("Say something to the bot:")

if user_input:
    st.write(f"Echo: {user_input}")
