# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import streamlit as st
import streamlit.components.v1 as components
import warnings

# modelling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# grid search
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")
np.random.seed(42)

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
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_icon="$$$", page_title="Bank Churn")

# load data
data_path = "./banking_churn.csv"

# design view
st.sidebar.header("Banking Churn")
data = st.sidebar.file_uploader("Upload dataset", type=['csv','xlsx','txt'])


# user_opt = st.sidebar.file_uploader("Load saved model",)

if data is not None:
    pass
else:
    bank_dt = pd.read_csv(data_path)

   # feature selection
    churn_dt = bank_dt[['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts',
                    'HasCrCard','IsActiveMember','EstimatedSalary','Exited']]

    # encoding
    churn_dt['Geography'] = churn_dt['Geography'].astype('category')
    churn_dt['Gender'] = churn_dt['Gender'].astype('category')

    # one hot encoding
    encode_geo = pd.get_dummies(churn_dt['Geography'])
    encode_gen = pd.get_dummies(churn_dt['Gender'])
    churn_dt = churn_dt.join(encode_geo)
    churn_dt = churn_dt.join(encode_gen)
    # trimming
    churn_dt.drop(['Geography','Gender'], axis=1, inplace=True)


# select box
menu = ["Features Snapshot","Train&Predict","About"]
selection = st.sidebar.selectbox("Menu Options", menu)

st.sidebar.write('''\n
                    : RowNumber—corresponds to the record (row) number and has no effect on the output. \n
                    : CustomerId—contains random values and has no effect on customer leaving the bank. \n
                    : Surname—the surname of a customer has no impact on their decision to leave the bank. \n
                    : CreditScore—can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank. \n
                    : Geography—a customer’s location can affect their decision to leave the bank. \n
                    : Gender—it’s interesting to explore whether gender plays a role in a customer leaving the bank. \n
                    : Age—this is certainly relevant, since older customers are less likely to leave their bank than younger ones. \n
                    : Tenure—refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank. \n
                    : Balance—also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances. \n
                    : NumOfProducts—refers to the number of products that a customer has purchased through the bank. \n
                    : HasCrCard—denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank. \n
                    : IsActiveMember—active customers are less likely to leave the bank. \n
                    : EstimatedSalary—as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries. \n
                    : Exited—whether or not the customer left the bank. \n''')

if selection == "Features Snapshot":
    
    st.subheader("Main data")
    st.dataframe(bank_dt.head(8))

    st.subheader("Important Features")
    st.dataframe(churn_dt.head(8))    
    churned_dt = churn_dt.drop(['France','Germany','Spain','Female','Male'], axis=1)
    # histogram columns
    with st.container():
        st.subheader("Features Histogram")
        # looking for outliers
        # plt.style.use('ggplot')
        # plt.figure(figsize=(20,10))
        ax = churned_dt.hist(figsize=(20,10))       
        #plt.show()
        st.pyplot(plt)   
    
    # outliers boxplot
    with st.container():
        st.subheader("Features boxplot")
        for z in churned_dt.columns:
            if churned_dt[z].dtype != object:
                plt.figure(figsize=(5,1))
                ax = sns.boxplot(data=churned_dt, x=z)
                # churn_dt.boxplot([z])
                # ax = churn_dt[z].plot(kind='box')
                # plt.show()
                st.pyplot(plt) 
      
    # correlation heatmap
    with st.container():
        st.subheader("Feature's Correlation Heatmap")
        # heatmap
        plt.figure(figsize=(16,8))
        ax = sns.heatmap(churned_dt.corr(), cmap='cividis', linewidths=2,fmt='.3f', annot=True)
        #plt.show()
        st.pyplot(plt)   

    # pairplot
    with st.container():
        plt.figure(figsize=(20,10))
        ax = sns.pairplot(churned_dt)
        # plt.show()
        st.pyplot(plt)  


elif selection == "Train&Predict":

    # training and testing data
    X_dt = churn_dt.drop('Exited',axis=1)
    y_dt = churn_dt['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X_dt,y_dt, test_size=0.2, random_state=17, stratify=y_dt)
    
    # loading saved model
    model_obj = joblib.load("bank_churn_md")
    model_preds = model_obj.predict(X_test)
    rfc_score = model_obj.score(X_test,y_test)
    acc_score = accuracy_score(y_test,model_preds)
    cl_report = classification_report(y_test, model_preds)
    conf_matrix = confusion_matrix(y_test, model_preds)

    # load tuned model
    tuned_modelobj = joblib.load("tuned_bank_churn_md")
    
    st.subheader("Please wait while: model is training")   

   

    # model instantiate
    #rfc_obj = RandomForestClassifier()

    # fitting the model
    # rfc_obj.fit(X_train,y_train)
    # targ_predictions = rfc_obj.predict(X_test)

    # rfc_score = rfc_obj.score(X_test,y_test)

    # cola, colb, colc = st.columns(3)
    with st.container():
        # evaluate the model
        # score method or accuracy_score
        # print(accuracy_score(y_test, targ_predictions))
        if st.checkbox("Show Accuracy_Score"):
            st.subheader("Model Score")
            st.success(rfc_score)
            st.subheader("With Accuracy Score")
            st.success(acc_score)

    with st.container():
        if st.checkbox("Show Classification Report?"):
            st.subheader("Classification report")
            st.text(cl_report)

    with st.container():
        if st.checkbox("show Confusion Matrix?"):
            st.subheader("Confusion Matrix")
            # cm_display = confusion_matrix(y_test, targ_predictions)
            # st.write(cm_display)
            plt.figure(figsize=(20,10))
            ax = ConfusionMatrixDisplay(conf_matrix).plot()
            # plt.show()
            st.pyplot(plt)

    # grid search
    st.subheader("HyperParameter tuning with GridSearch")  
    
    # Define the parameters to search over
    # #param_grid = {'n_estimators': [i for i in range(10, 201, 10)], 'max_depth': [0,5,10,15,20], 'min_samples_split':[1,2,3],
    #                 'min_samples_leaf':[1,2]}

    # Setup the grid search
    # # grid = GridSearchCV(RandomForestClassifier(),
    #                 param_grid,
    #                 cv=5,
    #                 scoring='recall')

    # Fit the grid search to the data
    #grid.fit(X_train, y_train)

    # Find the best parameters
    # best_pr = grid.best_params_
    

    # # Set the model to the best estimator
    # rfc_grid = grid.best_estimator_    

    # Fit the best model
    # rfc_grid.fit(X_train, y_train)

    # Find the best model scores
    st.subheader("Our best hyperparameters are:")
    best_pr = {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 40}
    st.json(best_pr)
    st.subheader("Model score after tuning")
    tuned_score =  tuned_modelobj.score(X_test,y_test)
    st.success(tuned_score)
    
    # select box
    # choicez = {"PureModel":rfc_score,"GridsModel":tuned_score}
    # sel = st.sidebar.selectbox("Save Model", choicez)
    # filename = "bank_churn_md"
    # filenames = "tuned_bank_churn_md"
    # if sel == 'PureModel':
    #     joblib.dump(rfc_obj, filename)
    # elif sel == 'GridsModel':
    #     joblib.dump(rfc_grid, filenames)

    


elif selection == "About":
    st.header("About")
    components.html(footer_temp,height=400)




