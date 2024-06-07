import streamlit as st
import subprocess

# Function to execute selected Streamlit script
def run_streamlit_script(script_name):
    subprocess.Popen(["streamlit", "run", f"{script_name}.py"])

# Home button with icon in top-right corner
if st.button("Home 🏠", key='home_button', help="Home"):
    run_streamlit_script("DashB")



# Title and Hero Section
st.title("Stock Market Prediction with LSTM")
st.write("Harnessing the power of LSTM networks and Explainable AI (XAI) for informed investment decisions.")

# Load an image from a file path
image_path = 'LSTM.jpg'
st.image(image_path, caption='LSTM Architecture', use_column_width=True)

# Data Acquisition
st.header("Data Acquisition: Feeding the Machine")
st.write("The system gathers data from two primary sources to understand market movements:")
col1, col2 = st.columns(2)
with col1:
    st.write("**Financial Data**")
    st.write("- Historical stock prices")
    st.write("- Relevant financial indicators (e.g., P/E ratio, dividend yield)")
with col2:
    st.write("**Financial News & Sentiment**")
    st.write("- News articles and social media sentiment")
    st.write("- Quantifies positive or negative influence on the market")

# Data Preprocessing: Cleaning and Shaping
st.header("Data Preprocessing: Getting Ready for Training")
st.write("The raw data undergoes several transformations before feeding into the LSTM model:")
st.write("* **Cleaning:** Removing irrelevant data points or formatting inconsistencies.")
st.write("* **Normalization:** Scaling all data points to a similar range for better model training.")
st.write("* **Sentiment Analysis:** Extracting sentiment from financial news to quantify its market influence.")

# Data Splitting: Training and Testing
st.header("Data Splitting: Dividing and Conquering")
st.write("The preprocessed data is strategically divided into two sets:")
st.write("* **Training Set (70%)** : Used to train the LSTM model, helping it learn patterns.")
st.write("* **Test Set (30%)** : Used to evaluate the model's performance on unseen data.")

# LSTM Model Training: The Core Engine
st.header("LSTM Model Training: Unveiling Market Secrets")
st.write("A Long Short-Term Memory (LSTM) network is employed for its expertise in sequential data:")
st.write("  * A type of Recurrent Neural Network (RNN) that excels at handling time series data like stock prices.")
st.write("  * The model is trained on the training set, learning to identify hidden patterns and relationships within the data.")
st.write("  * This knowledge empowers it to predict future stock prices based on historical trends and market influences.")

# Model Testing and Evaluation: Gauging Performance
st.header("Model Testing and Evaluation: Putting It to the Test")
st.write("Once trained, the model's performance is evaluated on the unseen test set:")
st.write("  * The model predicts stock prices for the test data.")
st.write("  * The accuracy of these predictions is measured using metrics like Mean Squared Error (MSE).")
st.write("  * This evaluation helps assess the model's effectiveness and identify areas for improvement.")

# Explainable AI (XAI): Demystifying the Black Box
st.header("Explainable AI (XAI): Peeking Inside the Machine")
st.write("XAI techniques shed light on the LSTM model's decision-making process:")
st.write("  * By explaining the factors influencing the model's predictions.")
st.write("  * This transparency builds trust and allows users to understand what drives the model's recommendations.")
st.write("  * It also helps identify the most crucial factors impacting the model's predictions.")

# Visualization: Making Predictions Clear
st.header("Visualization: A Picture is Worth a Thousand Numbers")
st.write("The system generates a clear visualization of the predicted stock prices:")
st.write("  * This visualization helps users to interpret the model's predictions with ease.")
st.write("  * By visualizing trends, users can make informed investment decisions based on the model's insights.")

# Disclaimer
st.subheader("Disclaimer")
st.write("The stock market is inherently complex, and no model can guarantee perfect accuracy.")
st.write("This system is intended as a tool to support investment decisions, not a replacement for financial expertise. Always conduct thorough research and consult with a financial advisor before making investment decisions.")
