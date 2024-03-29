import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Display Title and Description
st.title("Data Baru dari Spreadsheet")
st.markdown("Data dibawah ini berasal dari pemilik kos yang terhubung ke spreadsheet.")

# Establishing a Google Sheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Fetch existing vendors data
existing_data = conn.read(worksheet="Data", usecols=list(range(20)), ttl=5)
existing_data = existing_data.dropna(how="all")

# Map values in 'jenis' column
existing_data['jenis'] = existing_data['jenis'].map({'Putra': 1, 'Putri': 2, 'Campur': 3})

# Map values in specified columns
columns_to_map = ['listrik', 'akseskos', 'ac', 'kasur', 'kamarmandidalam', 'klosetduduk', 
                  'penjagakos', 'penguruskos', 'cctv', 'wifi', 'ibadah', 'bank', 'rumahsakit', 
                  'universitas']

for column in columns_to_map:
    existing_data[column] = existing_data[column].map({'Ada': 1, 'Tidak ada': 0})

# Sidebar for Data Mining
st.sidebar.title("Data Mining")
selected_tab = st.sidebar.selectbox("Choose an option", ["Data Spreadsheet", "Data Mining"])

if selected_tab == "Data Spreadsheet":
    st.dataframe(existing_data)
elif selected_tab == "Data Mining":
    st.subheader("Multiple Linear Regression for Data Mining")

    # Remove irrelevant columns
    columns_to_remove = ['id', 'name', 'description', 'number']
    existing_data = existing_data.drop(columns=columns_to_remove, errors='ignore')

    # Assume the last column is the target variable and the rest are features
    X = existing_data.drop(columns=['price'])
    y = existing_data['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("Mean Squared Error:", mse)
    st.write("R-squared Score:", r2)
