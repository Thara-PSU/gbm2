import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

def main():
    from PIL import Image
    image_hospital = Image.open('Neuro1.png')
    image_ban = Image.open('Neuro2.png')
    st.image(image_ban, use_column_width=False)
    st.sidebar.image(image_hospital)
if __name__ == '__main__':
    main()


st.write("""
# 2-year survival prediction of glioblastoma App
This app predicts the **2-year survival of glioblastoma**!
Data obtained from the [Thara library](https://github.com/allisonhorst/palmerpenguins) 
in R by Thara Tunthanathip.
Surgery:0=Bx,1=Partial resection 
""")

st.write("""
Label
Surgery:0=Bx,1=Partial resection, 2=Total resection 
""")
st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if  uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        sex = st.sidebar.selectbox('sex', ('male', 'female'))
        Eloquent = st.sidebar.selectbox('Eloquent', ('NE', 'EE'))
        Surgery = st.sidebar.slider('Surgery', 1, 3, 2)
        Location = st.sidebar.slider('Location ', 0, 2, 1)
        CCRT_gr = st.sidebar.slider('CCRT_gr', 0, 1, 1)
        data = {'Eloquent': Eloquent,
                'Surgery': Surgery,
                'Location': Location,
                'CCRT_gr': CCRT_gr,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
GBM_raw = pd.read_csv('GBM2020surgery1.csv')
GBM = GBM_raw.drop(columns=['type'])
df = pd.concat([input_df,GBM],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['Eloquent','sex']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)


# Reads in saved classification model
load_clf = pickle.load(open('gbm_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
two_year_survival = np.array(['no','death'])
st.write(two_year_survival[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Class Labels and their corresponding index number')

label_name = np.array(['no','death'])
st.write(label_name)
# labels -dictionary
names ={0:'no',
1: 'death'}

