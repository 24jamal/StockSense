import streamlit as st
import subprocess

# Define UI elements
st.title('Stock Sense : A DL Powered Stock Recommender 	📈 ')

# Add checkboxes for selecting companies
selected_companies = st.sidebar.multiselect('Select Companies', ['Infosys', 'TCS', 'Reliance', 'Cipla'])

nav_selection = st.sidebar.radio('Navigation', ['Home 🏠', 'LSTM 📶', 'NLP 😊😔','News Stand📰','Portfolio 🏢','Documentation 📁','ChatBot 💬','Feedback 📋'])

# Function to execute selected Streamlit script
def run_streamlit_script(script_name):
    subprocess.Popen(["streamlit", "run", f"{script_name}.py"])

# Function to render different pages
def render_page(nav_selection):
    if nav_selection == 'Home 🏠':
        st.write('Welcome to the home page! 😄')
    elif nav_selection == 'LSTM 📶':
        st.write('You are now on the LSTM page.')
        run_streamlit_script("LSTMArch")
    elif nav_selection == 'News Stand📰':
        st.write('You are now on the News page.')
        run_streamlit_script("app")
    elif nav_selection == 'NLP 😊😔':
        st.write('You are now on the NLP page.')
        run_streamlit_script("nlp")
    elif nav_selection == 'Portfolio 🏢':
        st.write('You are now on the Portfolio page.')
        run_streamlit_script("Portfolio")
    elif nav_selection == 'Documentation 📁':
        st.write('You are now on the Documentation page.')
        run_streamlit_script("Documentation")
    elif nav_selection == 'Feedback 📋':
        st.write('You are now on the Feedback page.')
        run_streamlit_script("feedback")
    elif nav_selection == 'ChatBot 💬':
        st.write('You are now on the ChatBot page.')
        run_streamlit_script("ChatBot")
#


    # Display predictions for selected companies
    if selected_companies:
        st.subheader('Predictions for Selected Companies')
        for company in selected_companies:
            st.write(f'Predictions for {company}')

    # Run scripts for selected companies
    for company in selected_companies:
        st.write(f"Running {company} script...")
        run_streamlit_script(company)

# Render selected page
render_page(nav_selection)

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
    run_streamlit_script("Reliance")
elif button2:
    run_streamlit_script("Cipla")
elif button3:
    run_streamlit_script("Britannia")
elif button4:
    run_streamlit_script("TCS")
elif button5:
    run_streamlit_script("Axis")
elif button6:
    run_streamlit_script("Kotak")
elif button7:
    run_streamlit_script("LT")
elif button8:
    run_streamlit_script("SBI")
elif button9:
    run_streamlit_script("Infosys")
elif button10:
    run_streamlit_script("Unilever")
elif button11:
    run_streamlit_script("GODREJCP")
elif button12:
    run_streamlit_script("HDFCBank")
elif button13:
    run_streamlit_script("ONGC")
elif button14:
    run_streamlit_script("TATAMotors")
elif button15:
    run_streamlit_script("SunPharma")
elif button16:
    run_streamlit_script("MRF")


