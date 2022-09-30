# importing required libraries

from json import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import warnings

from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
# %matplotlib inline

# footer template
footer_temp = """
<!-- CSS  -->
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" 
type="text/css" rel="stylesheet" media="screen,projection"/>
<link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" 
integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
<footer class="page-footer grey darken-4">
<div class="container" id="aboutapp">
<div class="row">
<div class="col l6 s12">
<h5 class="white-text">Nairobi Securities Stock Exchange</h5>
<h6 class="grey-text text-lighten-4">This is a Streamlit Class practical.</h6>
<p class="grey-text text-lighten-4">September 2022</p>
</div>
<div class="col l3 s12">
<h5 class="white-text">Connect With Us</h5>
<ul>
<a href="#" target="_blank" class="white-text">
<i class="fab fa-facebook fa-4x"></i>
</a>
<a href="#" target="_blank" class="white-text">
<i class="fab fa-linkedin fa-4x"></i>
</a>
<a href="#" target="_blank" class="white-text">
<i class="fab fa-youtube-square fa-4x"></i>
</a>
<a href="#" target="_blank" class="white-text">
<i class="fab fa-github-square fa-4x"></i>
</a>
</ul>
</div>
</div>
</div>
<div class="footer-copyright">
<div class="container">
Made by <a class="white-text text-lighten-3" href="#">Jijo </a><br/>
<a class="white-text text-lighten-3" href="#"></a>
</div>
</div>
</footer>
"""

# dom settings
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_icon="$$$", page_title="TestingDash")

# load data
data_path = "G:\progressus\ADS_JULY_assignments\ADS_assignment2\data.csv"

# design view
st.sidebar.header("Nairobi Stock Exchange")
data = st.sidebar.file_uploader("Upload dataset", type=['csv','xlsx','txt'])

# logic
def clean_currency(curr):
    return float(curr.replace(",", "")) 

if data is not None:
    pass
else:
    loaded_dt = pd.read_csv(data_path)
    # cleaning
    loaded_dt["price"] = loaded_dt['price'].apply(clean_currency)
    loaded_dt['date'] = pd.to_datetime(loaded_dt['date'],dayfirst=True)
    # generating new columns with types
    # convert column dtypes
    loaded_dt["price"] = pd.to_numeric(loaded_dt["price"])
    loaded_dt['ticker'] = loaded_dt.ticker.astype('string')
    loaded_dt['company'] = loaded_dt.company.astype('string')
    loaded_dt["date"] = pd.to_datetime(loaded_dt["date"])
    loaded_dt['month'] = loaded_dt['date'].dt.month
    loaded_dt = loaded_dt.set_index("date", inplace=False)
    # create month column
    #loaded_dt['month'] = loaded_dt['date'].dt.month
    # get specific company names
    company_names = list(loaded_dt.company.unique())




# select box
menu = ["Business Snapshot","Analysis","About"]
selection = st.sidebar.selectbox("Key Performance Indicator: KPI", menu)

st.sidebar.write('''The Nairobi Securities Exchange (NSE) is a leading African Exchange,
                    based in Kenya – one of the fastest-growing economies in Sub-Saharan Africa. 
                    Founded in 1954, NSE has a six decade heritage in listing equity and debt securities. 
                    It offers a world class trading facility for local and international investors looking to 
                    gain exposure to Kenya and Africa’s economic growth.''')

if selection == "Business Snapshot":
    
    st.subheader("Display data")
    st.dataframe(loaded_dt.head(8))
    st.subheader(" ")
    user_option = st.selectbox("Choose company to show trend: ", company_names )

    per_company = loaded_dt[loaded_dt['company'] == user_option]
    month10_company = loaded_dt[(loaded_dt['company'] == user_option) & (loaded_dt['month'] == 10)]
    month11_company = loaded_dt[(loaded_dt['company'] == user_option) & (loaded_dt['month'] == 11)]
   
    # trend for each company
    
        #special_col = st.container()
    with st.container():
        plt.style.use('ggplot')
        plt.figure(figsize=(12,7))        
        # use_index=True
        ax = per_company['price'].plot(marker='o', xticks=per_company.index, rot=90, grid=True)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title(f"Price variation for {user_option}", fontsize="15")
        #plt.show()
        st.pyplot(plt)
    
      # get the records for each month for each company
    col1, col2 = st.columns(2)
    # column 1
    with col1:

        # checking trend for 10 month
        st.subheader(f"First month price trends for company {user_option}")
        plt.figure(figsize=(9,5))
        ax = month10_company['price'].plot(color="#EDB120", marker='o', xticks=month10_company['price'].index, rot=90, grid=True)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        #ax.set_title(f"Price variation for company {company_names[0]} in the {11} month", fontsize="20")
        #plt.show()
        st.pyplot(plt)     
      
    
    with col2:

        # checking trend for 11 month
        st.subheader(f"Second month price trends for company {user_option}")
        plt.figure(figsize=(9,5))
        ax = month11_company['price'].plot(color="#7E2F8E", marker='o', xticks=month11_company['price'].index, rot=90, grid=True)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        #ax.set_title(f"Price variation for company {company_names[0]} in the {11} month", fontsize="20")
        #plt.show()
        st.pyplot(plt)     


elif selection == "Analysis":

    st.subheader("Perform Summary Statistics")
    st.write(loaded_dt.head())

    cola, colb, colc = st.columns(3)
    with cola:
        if st.checkbox("show shape"):
            st.write("data shape")
            st.write("{:,} rows; {:,} columns".format(loaded_dt.shape[0], loaded_dt.shape[1]))

            # data description
            # st.markdown("More Info")           
            # loaded_dt.info()
    with colb:
        if st.checkbox("show descriptive stats"):
            # data description
            st.markdown("More Info")
            b = loaded_dt.describe()
            st.write(b)
    with colc:
        if st.checkbox("show missing values"):
            st.write("missing values in data")
            objto = loaded_dt.isna().sum()
            st.write(objto)

            # data description
            # st.markdown("Duplicates")
            # dups = loaded_dt.duplicated().value_counts()
            # st.write(dups)
    

    # finding missing values

elif selection == "About":
    st.header("About")
    components.html(footer_temp,height=400)




