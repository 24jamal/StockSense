import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import subprocess

# Function to execute selected Streamlit script
def run_streamlit_script(script_name):
    subprocess.Popen(["streamlit", "run", f"{script_name}.py"])

# MongoDB connection
client = MongoClient("mongodb+srv://jamalceg123:p16QLbUdxhXHU0vb@stocksenseatlas.387yu8u.mongodb.net/")
db = client["News_DataBase_Atlas"]

# Function to retrieve articles for a given company within a date range and optionally filtered by search query
def retrieve_articles_within_date_range(company, start_date, end_date, page_index, search_query=None):
    # Convert start_date and end_date to ISO 8601 format strings
    start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Query MongoDB for articles within the specified date range
    articles_collection = db[company.lower() + "_articles"]
    query = {
        "publishedAt": {"$gte": start_date_str, "$lte": end_date_str}
    }

    # Filter articles by search query if provided
    if search_query:
        search_query = search_query.lower()  # Convert search query to lowercase
        articles = articles_collection.find(query).sort("publishedAt", -1).skip(page_index * 5).limit(5)
        filtered_articles = []
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            author = article.get('author', '')  # Get the author value
            if author:  # Check if author value is not None
                author = author.lower()  # Convert author value to lowercase
            if (search_query in title or
                search_query in description or
                (author and search_query in author)):  # Check if author is not None before using lower()
                filtered_articles.append(article)
        return filtered_articles
    else:
        # No search query provided, return all articles within the date range
        articles = articles_collection.find(query).sort("publishedAt", -1).skip(page_index * 5).limit(5)
        return list(articles)

# Home button with icon in top-right corner
if st.button("Home 🏠", key='home_button', help="Home"):
    run_streamlit_script("DashB")

# Streamlit app
st.title('News Stand 📰')

# Companies list
companies = ['Infosys', 'TCS', 'Reliance', 'Kotak', 'Cipla',"HDFC",
    "HINDUNILVR",
    "BHARTIARTL",
    "ITC",
    "LT", "TTMT", "SBIN", "SUNPHARMA", "BRITANNIA", "AXISBANK", "GODREJCP", "ONGC"
    ]

# Option bar
start_date, end_date = st.columns(2)
page_index, company_selection, search_query = st.columns([1, 2, 3])

start_date = start_date.date_input("Start Date")
end_date = end_date.date_input("End Date")

page_index = page_index.number_input("Page Index", min_value=0, value=0)

company = company_selection.selectbox("Select Company", companies)

search_query = search_query.text_input("Search by Title, Description, or Author")

# Display data for the selected company within the specified date range
articles = retrieve_articles_within_date_range(company, start_date, end_date, page_index, search_query)
if articles:
    table_data = []
    for article in articles:
        title = article.get("title", "N/A")
        description = article.get("description", "N/A")
        author = article.get("author", "N/A")
        
        # Highlight the specific word in the data if search query is provided
        if search_query:
            title = title.replace(search_query, f'<mark>{search_query}</mark>')
            description = description.replace(search_query, f'<mark>{search_query}</mark>')
            author = author.replace(search_query, f'<mark>{search_query}</mark>')
        
        table_row = {
            "Title": title,
            "Description": description,
            "Author": author,
            "Published At": article.get("publishedAt", "N/A"),
            "Source": article.get("source", {}).get("name", "N/A"),
            "URL": article.get("url", "N/A")
        }
        table_data.append(table_row)
    st.table(table_data)
else:
    st.write("No articles found for", company)
