import streamlit as st
import pandas as pd
from pymongo import MongoClient
import subprocess

# Function to execute selected Streamlit script
def run_streamlit_script(script_name):
    subprocess.Popen(["streamlit", "run", f"{script_name}.py"])

# Home button with icon in top-right corner
if st.button("Home 🏠", key='home_button', help="Home"):
    run_streamlit_script("DashB")


# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['Company_Portfolio']
collection = db['Companies']

# Get list of companies
companies = collection.distinct('company')

# Streamlit app
st.title('Company Portfolio Viewer')

# Dropdown to select company
selected_company = st.selectbox('Select Company', companies)

# Fetch data for selected company
company_data = collection.find_one({'company': selected_company})

# Convert data to DataFrame for display
df = pd.DataFrame(company_data, index=[0])

# Transpose DataFrame for vertical display
df_vertical = df.T

# Display data vertically with adjusted width using st.dataframe
st.dataframe(df_vertical, width=10000)
