import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(page_title="Prediction of Disease Outbreaks",
                   layout="wide")

working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_trained_model.sav','rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_trained_model.sav','rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_trained_model.sav','rb'))

with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreaks System',
                           
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                            menu_icon='hospital-fill',
                            icons=['activity','heart','person'],
                            default_index=0)
    
if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1: 
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2: 
        Glucose = st.text_input('Glucose Level')

    with col3: 
        BLoodPressure = st.text_input('Blood Pressure Value')

    with col1: 
        SkinThickness = st.text_input('Skin Thickness Value')

    with col2: 
        Insulin = st.text_input('Insulin Value')

    with col3: 
        BMI = st.text_input('BMI Value')

    with col1: 
        DiabetesPedigreePrediction = st.text_input('Diabetes Pedigree Function Value')

    with col2: 
        Age = st.text_input('Age of the Person')

    #code for Prediction

    diab_diagnosis = ''

    #creating a button for prediction

    if st.button('Diabetes Test Result'):

        user_input = np.array([[Pregnancies, Glucose, BLoodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreePrediction, Age]])

        # Load the pre-fitted scaler
        scaler_path = f'{working_dir}/saved_models/diabetes_scaler1.sav'
        scaler = pickle.load(open(scaler_path, 'rb'))

        # Transform the input data using the pre-fitted scaler
        std_input = scaler.transform(user_input)
         
        print(std_input)
        # Make prediction
        diab_prediction = diabetes_model.predict(std_input)

        print(diab_prediction)

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The Person is Diabetic'
        else:
            diab_diagnosis = 'The Person is Not Diabetic'

    st.success(diab_diagnosis)


if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1: 
        age = st.text_input('Enter Age')

    with col2: 
        sex = st.text_input('Enter Sex')

    with col3: 
        cp = st.text_input('Chest Pain Types')

    with col1: 
        trestbps = st.text_input('Resting Blood Pressure')

    with col2: 
        chol = st.text_input('Serum Cholestrol in mg/dl')

    with col3: 
        fbs = st.text_input('Fasting BLood Sugar > 120 mg/dl')

    with col1: 
        restecg = st.text_input('Resting Electrocardiographic Results')

    with col2: 
        thalach = st.text_input('Maximum Heart Rate Achieved')

    with col3: 
        exang = st.text_input('Exercise Induced Angina')

    with col1: 
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2: 
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3: 
        ca = st.text_input('Major vessels colored by flouroscopy')

    with col1: 
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    #code for Prediction

    heart_diagnosis = ''

    #creating a button for prediction

    if st.button('Heart Disease Test Result'):

        user_input = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

        scaler_path = f'{working_dir}/saved_models/Heart_scaler.sav'
        scaler = pickle.load(open(scaler_path, 'rb'))

        # Transform the input data using the pre-fitted scaler
        std_input = scaler.transform(user_input)
         
        print(std_input)
        # Make prediction
        heart_prediction = heart_disease_model.predict(std_input)

        print(heart_prediction)

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The Person is having Heart Disease'
        else:
            heart_diagnosis = 'The Person is not having Heart Disease'

    st.success(heart_diagnosis)



if selected == 'Parkinsons Prediction':

    st.title('Parkinsons Prediction using ML')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1: 
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2: 
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3: 
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4: 
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5: 
        jitter_abs = st.text_input('MDVP:Jitter(Abs)')

    with col1: 
        RAP = st.text_input('MDVP:RAP')

    with col2: 
        PPQ = st.text_input('MDVP:PPQ')

    with col3: 
        DDP = st.text_input('Jitter:DDP')

    with col4: 
        shimmer = st.text_input('MDVP:Shimmer')

    with col5: 
        shimmer_db = st.text_input('MDVP:Shimmer(dB)')

    with col1: 
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2: 
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3: 
        APQ = st.text_input('MDVP:APQ')

    with col4: 
        DDA = st.text_input('Shimmer:DDA')

    with col5: 
        NHR = st.text_input('NHR')

    with col1: 
        HNR = st.text_input('HNR')

    with col2: 
        RPDE = st.text_input('RPDE')

    with col3: 
        DFA = st.text_input('DFA')

    with col4: 
        spread1 = st.text_input('spread1')

    with col5: 
        spread2 = st.text_input('spread2')

    with col1: 
        D2 = st.text_input('D2')

    with col2: 
        PPE = st.text_input('PPE')

    #code for Prediction

    parkinsons_diagnosis = ''

    #creating a button for prediction

    if st.button('parkinsons Test Result'):

        user_input1 = [[fo, fhi, flo, Jitter_percent, jitter_abs, RAP, PPQ, DDP, shimmer, shimmer_db, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]

        # Load the pre-fitted scaler
        scaler_path1 = f'{working_dir}/saved_models/Parkinsons_scaler.sav'
        scaler1 = pickle.load(open(scaler_path1, 'rb'))

        # Transform the input data using the pre-fitted scaler
        std_input1 = scaler1.transform(user_input1)
         
        print(std_input1)
        # Make prediction
        parkinsons_prediction = parkinsons_model.predict(std_input1)

        print( parkinsons_prediction)


        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = 'The Person is having Parkinsons'
        else:
            parkinsons_diagnosis = 'The Person is not having Parkinsons'

    st.success(parkinsons_diagnosis)