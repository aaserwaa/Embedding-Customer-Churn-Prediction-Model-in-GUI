import streamlit as st
import pandas as pd
import pyodbc
import _thread

# Define a custom hash function for _thread.RLock objects
def hash_thread_rlock(obj):
    # Return a hashable value (e.g., 0)
    return 0

# Title of the page
st.title("Data Page")

# Define a function to establish a database connection and cache it
@st.cache(hash_funcs={_thread.RLock: hash_thread_rlock})
def connect_to_database():
    # Load database credentials from secrets.toml file
    environment_variables = st.secrets["connection"]

    # Extract credentials
    server = environment_variables["SERVER"]
    database = environment_variables["DATABASE"]
    username = environment_variables["USERNAME"]  # Changed from "UID" to "USERNAME"
    password = environment_variables["PASSWORD"]

    # Connection string
    connection_string = f"DRIVER=SQL Server;SERVER={server};DATABASE={database};UID={username};PWD={password}"

    # Establish a connection
    connection = pyodbc.connect(connection_string)
    
    query = "SELECT * FROM LP2_Telco_churn_first_3000"

    data = pd.read_sql(query, connection)
    return data

# Create selection option
column1, column2 = st.columns(2)
with column2:
        option = st.selectbox('Choose columns to be viewed',
                              ('All Columns','Numeric Columns','Categorical Columns'))
 
df = connect_to_database()
 
# Display based on selection
if option == 'Numeric Columns':
    st.subheader('Numeric Columns')
    st.write(df.select_dtypes(include='number'))
elif option == 'Categorical Columns':
    st.subheader('Categorical Columns')
    st.write(df.select_dtypes(include='object'))
else:
    st.subheader('Entire Dataset')
    st.write(df)
