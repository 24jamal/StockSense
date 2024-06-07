import streamlit as st
import Reliance
import app1
import LSTMArch
import nlp
import Portfolio
import Documentation
import feedback
import ChatBot
import Infosys
import TCS
import Cipla

# Define UI elements
st.title('Stock Sense : A DL Powered Stock Recommender 📈')

# Add checkboxes for selecting companies
selected_companies = st.sidebar.multiselect('Select Companies', ['Infosys', 'TCS', 'Reliance', 'Cipla'])

# Navigation sidebar
nav_selection = st.sidebar.radio('Navigation', ['Home 🏠', 'LSTM 📶', 'NLP 😊😔', 'News Stand📰', 'Portfolio 🏢', 'Documentation 📁', 'ChatBot 💬', 'Feedback 📋'])

# Function to render different pages
def render_page(nav_selection):
    if nav_selection == 'Home 🏠':
        st.write('Welcome to the home page! 😄')
    elif nav_selection == 'LSTM 📶':
        LSTMArch.app()
    elif nav_selection == 'News Stand📰':
        try:
            app1.app()
        except Exception as e:
            st.error(f"Error loading News Stand: {e}")
    elif nav_selection == 'NLP 😊😔':
        nlp.app()
    elif nav_selection == 'Portfolio 🏢':
        Portfolio.app()
    elif nav_selection == 'Documentation 📁':
        Documentation.app()
    elif nav_selection == 'Feedback 📋':
        feedback.app()
    elif nav_selection == 'ChatBot 💬':
        ChatBot.app()

# Render selected page
render_page(nav_selection)

# Display predictions for selected companies
if selected_companies:
    st.subheader('Predictions for Selected Companies')
    for company in selected_companies:
        st.write(f'Predictions for {company}')
        if company == 'Infosys':
            Infosys.app()
        elif company == 'TCS':
            TCS.app()
        elif company == 'Reliance':
            Reliance.app()
        elif company == 'Cipla':
            Cipla.app()

# Add images in grid format
col1, col2, col3, col4 = st.columns(4)
# Row 1
col1.image("RIL.jpg", use_column_width=True)
button1 = col1.button("Reliance")
col2.image("Cipla.jpg", use_column_width=True)
button2 = col2.button("Cipla")
col3.image("BRIT.jpg", use_column_width=True)
button3 = col3.button("Britannia")
col4.image("TCS.jpg", use_column_width=True)
button4 = col4.button("TCS")
# Row 2
col5, col6, col7, col8 = st.columns(4)
col5.image("Axis.jpg", use_column_width=True)
button5 = col5.button("Axis Bank")
col6.image("KOTAK.jpg", use_column_width=True)
button6 = col6.button("Kotak Mahindra")
col7.image("LT.jpg", use_column_width=True)
button7 = col7.button("Larsen & Toubro")
col8.image("SBI.jpg", use_column_width=True)
button8 = col8.button("State Bank of India")
# Row 3
col9, col10, col11, col12 = st.columns(4)
col9.image("Infosys.jpg", use_column_width=True)
button9 = col9.button("Infosys")
col10.image("UNIV.jpg", use_column_width=True)
button10 = col10.button("Hindustan Unilever")
col11.image("godrej.jpg", use_column_width=True)
button11 = col11.button("Godrej CP")
col12.image("hdfc.jpg", use_column_width=True)
button12 = col12.button("HDFC Bank")
# Row 4
col13, col14, col15, col16 = st.columns(4)
col13.image("ONGC.jpg", use_column_width=True)
button13 = col13.button("ONGC Ltd")
col14.image("ttmt.jpg", use_column_width=True)
button14 = col14.button("TATA Motors")
col15.image("sun.jpg", use_column_width=True)
button15 = col15.button("Sun Pharma")
col16.image("MRF.jpg", use_column_width=True)
button16 = col16.button("MRF")
# Check if a button is clicked and launch the corresponding Streamlit app
if button1:
    Reliance.app()
elif button2:
    Cipla.app()
elif button3:
    Britannia.app()
elif button4:
    TCS.app()
elif button5:
    Axis.app()
elif button6:
    Kotak.app()
elif button7:
    LT.app()
elif button8:
    SBI.app()
elif button9:
    Infosys.app()
elif button10:
    Unilever.app()
elif button11:
    GODREJCP.app()
elif button12:
    HDFCBank.app()
elif button13:
    ONGC.app()
elif button14:
    TATAMotors.app()
elif button15:
    SunPharma.app()
elif button16:
    MRF.app()
