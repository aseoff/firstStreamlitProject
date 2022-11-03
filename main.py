from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import pandas as pd

#Creating blocks for each of the following sections
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

#UI Customizations - this is the basic concept!
st.markdown(
"""
<style>
.main {
    background-color: #AADBA6
}
</style>
""",
unsafe_allow_html=True
)


@st.cache #Doesn't keep doing this over and over again!
def get_data(filename):
    data = pd.read_csv(filename)
    return data

with header:
    st.title("Welcome to Jesse's awesome data science project!")
    st.text('In this project I look into the house market in Orange County, California. ...')

with dataset:
    st.header('OC Houses dataset')
    st.text('I found this dataset on blablabla.com,...')

    oc_data = get_data('data/data_OC_sample.csv')
    st.write(oc_data.head())

    st.subheader('OC Houses - Year built distribution')
    year_built_dist = pd.DataFrame(oc_data['yearbuilt'].value_counts())
    st.bar_chart(year_built_dist)

with features:
    st.header('The features I created')

    st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic...')
    st.markdown('* **second feature:** I created this feature because of this... I calculated it using this logic...')


with model_training:
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance changes')

    #Creates 2 column formatting for selection and diplay sides
    sel_col, disp_col = st.columns(2)

    #Input parameters
    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No limit'], index = 0)
   
    sel_col.text('Here is a list of features in my data:')
    sel_col.write(oc_data.columns)

    input_feature = sel_col.text_input('Which feature should be used as the input feature?', 'yearbuilt')

    # apply the model
    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators = n_estimators)
    
    #Define input and output
    X = oc_data[[input_feature]]
    y = oc_data[['taxvaluedollarcnt']]

    #Fit the model
    regr.fit(X,y)

    #Predict
    prediction = regr.predict(y)

    #Get and display error metrics
    disp_col.subheader('Mean absolute error of the model is: ')
    disp_col.write(mean_absolute_error(y, prediction))
    disp_col.subheader('Mean squared error of the model is: ')
    disp_col.write(mean_squared_error(y, prediction))
    disp_col.subheader('R squared score of the model is: ')
    disp_col.write(r2_score(y, prediction))


