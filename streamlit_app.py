import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pymongo import MongoClient
import pymongo
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def app():
    st.title('Reliance Portfolio Overview')
    
    st.write("""Reliance Industries Limited is India's largest private sector conglomerate, with business interests spanning petrochemicals, refining, oil & gas exploration, retail, telecommunications, and digital services. Reliance's diversified portfolio and focus on disruptive innovation have positioned it as a key player in driving India's economic growth and transformation. The company's retail arm, Reliance Retail, is the largest and most profitable retailer in India, operating across multiple formats and categories. Reliance Jio Infocomm, the telecom subsidiary, has revolutionized India's digital landscape with its affordable data and voice services. Through its commitment to sustainability and inclusive growth, Reliance continues to create long-term value for stakeholders while contributing to India's socio-economic development.
    """)
    
    ##########################################
    
    #Key events
    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb+srv://jamalceg123:p16QLbUdxhXHU0vb@stocksenseatlas.387yu8u.mongodb.net/")
    db = client.Stock_data_Prices_NEW_2  # Assuming the database name is Stock_data_Prices_NEW_2
    collection = db["RELIANCE.NS"]  # Assuming the collection name is RELIANCE.NS
    
    # Fetch historical stock price data from MongoDB collection
    rel_data = collection.find({}, {"Date": 1, "Close": 1}).sort("Date", 1)  # Assuming 'Date' and 'Close' are fields in the collection
    
    # Convert MongoDB cursor to pandas DataFrame
    rel_df = pd.DataFrame(list(rel_data))
    
    # Create a Streamlit app
    st.title('Reliance Stock Prices with Key Events')
    
    # Plot the stock prices with a light blue color
    plt.figure(figsize=(12, 6))
    plt.plot(rel_df['Date'], rel_df['Close'], label='RELIANCE.NS Stock Price', color='lightblue')
    
    # Add event markers
    events = {
        pd.to_datetime('2014-05-17'): 'Indian General Elections Result',
        pd.to_datetime('2017-07-01'): 'GST Implementation',
        pd.to_datetime('2019-05-23'): 'Indian Lok Sabha Election Result',
        pd.to_datetime('2024-02-02'): 'UB 24',
        pd.to_datetime('2023-02-02'): 'UB 23',
        pd.to_datetime('2022-02-02'): 'UB 22',    
        pd.to_datetime('2021-02-02'): 'UB 21',
        pd.to_datetime('2020-02-02'): 'UB 20',
        pd.to_datetime('2019-07-06'): 'UB 19',
        pd.to_datetime('2018-02-02'): 'UB 18',
        pd.to_datetime('2008-09-15'): 'Global Financial Crisis', 
        pd.to_datetime('2020-02-20'): 'COVID-19 Pandemic Crash',
        pd.to_datetime('2016-11-08'): 'Demonetization',  
        pd.to_datetime('2020-11-02'): 'M.Ambani\'s Health Rumour',
        pd.to_datetime('2020-03-25'): 'FB\'s investment in RIL gains 14%'
    }
    
    for event_date, event_label in events.items():
        closest_date = rel_df['Date'].iloc[(rel_df['Date'] - event_date).abs().argsort()[0]]  # Find closest date
        event_price = rel_df.loc[rel_df['Date'] == closest_date, 'Close'].values[0]
        plt.scatter(closest_date, event_price, color='red', marker='o', label=event_label)
        plt.text(closest_date, event_price, event_label, fontsize=10, ha='right', va='bottom', color='black')
    
    # Add title, labels, legend, and grid
    plt.xlabel('Date')
    plt.ylabel('Stock Price (INR)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Display the plot using Streamlit
    st.pyplot(plt)
    
    #########################################
    
    # Actual and Prediction for whole graph
    # Fetch historical stock price data from MongoDB collection
    RELIANCE_data = collection.find({}, {"Date": 1, "Close": 1}).sort("Date", 1)  # Assuming 'Date' and 'Close' are fields in the collection
    
    # Convert MongoDB cursor to pandas DataFrame
    RELIANCE_df = pd.DataFrame(list(RELIANCE_data))
    
    # Data preprocessing
    data = RELIANCE_df[['Date', 'Close']]  # Extract 'Date' and 'Close' prices
    data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime format
    data.set_index('Date', inplace=True)  # Set 'Date' as index
    scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize data to range [0, 1]
    scaled_data = scaler.fit_transform(data)
    
    # Split data into training and testing sets
    training_data_len = int(len(scaled_data) * 0.8)  # 80% of data for training, 20% for testing
    
    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len:]
    
    # Function to create sequences and labels
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            X.append(seq)
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    # Define sequence length (number of time steps to look back)
    seq_length = 60  # You can adjust this parameter based on your preference
    
    # Create sequences and labels for training and testing sets
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=2, batch_size=32)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse scaling to get actual prices
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Streamlit App
    st.title('Reliance Limited (RELIANCE.NS) Stock Price Prediction using LSTM')
    st.write('Actual vs. Predicted Stock Prices')
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index[-len(y_test):], y_test, label='Actual Stock Price')
    ax.plot(data.index[-len(predictions):], predictions, label='Predicted Stock Price')
    ax.set_title('Reliance Limited (RELIANCE.NS) Stock Price Prediction using LSTM')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.legend()
    st.pyplot(fig)
    
    ############################################
    
    # Stock Price Forecast
    # Fetch historical stock price data for Reliance Limited (Reliance.NS) from MongoDB
    data_cursor = collection.find({}, {'_id': 0, 'Date': 1, 'Close': 1}).sort('Date', 1)
    mongo_data = list(data_cursor)
    mongo_df = pd.DataFrame(mongo_data)
    mongo_df['Date'] = pd.to_datetime(mongo_df['Date'])
    
    # Data preprocessing
    data = mongo_df['Close'].values.reshape(-1, 1)  # Extract 'Close' prices as numpy array
    scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize data to range [0, 1]
    scaled_data = scaler.fit_transform(data)
    
    # Define sequence length (number of time steps to look back)
    seq_length = 60  # You can adjust this parameter based on your preference
    
    # Create sequences and labels for the entire dataset
    X, y = create_sequences(scaled_data, seq_length)
    
    # Reshape input data to fit LSTM model (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X, y, epochs=2, batch_size=32)
    
    # Forecast future stock prices
    num_days = 30  # Number of days to forecast
    forecast = []
    last_sequence = scaled_data[-seq_length:]  # Last observed sequence
    
    for _ in range(num_days):
        last_sequence = np.reshape(last_sequence, (1, seq_length, 1))  # Reshape input to fit LSTM model
        next_value = model.predict(last_sequence)  # Predict next value
        forecast.append(next_value[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], next_value, axis=1)  # Update sequence with predicted value
    
    # Inverse scaling to get actual prices
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    
    # Streamlit App
    st.title('Reliance Limited (Reliance.NS) Stock Price Forecast')
    st.write('Next 30 Days Forecast')
    
    # Plot the forecasted results
    forecast_dates = pd.date_range(start=mongo_df['Date'].iloc[-1] + timedelta(days=1), periods=num_days)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mongo_df['Date'], mongo_df['Close'], label='Historical Stock Price')
    ax.plot(forecast_dates, forecast, label='Forecasted Stock Price')
    ax.set_title('Reliance Limited (Reliance.NS) Stock Price Forecast')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.legend()
    st.pyplot(fig)

