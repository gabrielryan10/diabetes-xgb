#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import streamlit as st
import pickle
import pandas as pd
filename = 'xgb_diabetes_model_v2.sav'
load_model = pickle.load(open(filename, 'rb'))


# %%


def diabetes_pipe(in_age, in_bmi, in_smoker, in_physactivity,
                  in_fruits, in_veggies, in_hvyalcoholconsump, in_genhlth):
    
    if in_age<=24:
        age=1
    elif in_age<=29:
        age=2
    elif in_age<=34:
        age=3
    elif in_age<=39:
        age=4
    elif in_age<=44:
        age=5
    elif in_age<=49:
        age=6
    elif in_age<=54:
        age=7
    elif in_age<=59:
        age=8
    elif in_age<=64:
        age=9
    elif in_age<=79:
        age=10
    elif in_age<=74:
        age=11
    elif in_age<=79:
        age=12
    else:
        age=13
        
    if in_bmi<=15:
        bmi=0
    elif in_bmi<=20:
        bmi=1
    elif in_bmi<=25:
        bmi=2
    elif in_bmi<=30:
        bmi=3
    else:
        bmi=4
        
     
  
    if in_smoker=='Yes':
        smoker=1
    else:
        smoker=0
     
    if in_physactivity=='Yes':
        physactivity=1
    else:
        physactivity=0

    if in_fruits=='Yes':
        fruits=1
    else:
        fruits=0

    if in_veggies=='Yes':
        veggies=1
    else:
        veggies=0
        
    if in_hvyalcoholconsump=='Yes':
        hvyalcoholconsump=1
    else:
        hvyalcoholconsump=0

    if in_genhlth=='Excellent':
        genhlth=1
    elif in_genhlth=='Good':
        genhlth=2
    elif in_genhlth=='Fair':
        genhlth=3
    elif in_genhlth=='Relatively Poor':
        genhlth=4
    else:
        genhlth=5

    return age, bmi, smoker, physactivity,fruits, veggies, hvyalcoholconsump, genhlth

def diabetes_pred(model_diabetes,age,bmi, smoker, physactivity,
                  fruits, veggies, hvyalcoholconsump, genhlth):
    df=pd.DataFrame()
    df['age']=[age]
    df['bmi']=[bmi]
    df['smoker']=[smoker]
    df['physactivity']=[physactivity]
    df['fruits']=[fruits]
    df['veggies']=[veggies]
    df['hvyalcoholconsump']=[hvyalcoholconsump]
    df['genhlth']=[genhlth]
    
    pred=model_diabetes.predict_proba(df)[:,1]
    

    
    return pred


# %%


def main():
    try:
        st.title("Diabetes Prediction")
        st.write('(XGBoost Model)')

        in_age = int(st.number_input("Age"))
        in_bmi=int(st.number_input("BMI"))
        in_smoker = st.selectbox("Do you smoke?", ["Yes","No"])
        in_physactivity = st.selectbox("Do you exercise?", ["Yes","No"])
        in_fruits = st.selectbox("Do you eat fruits regularly?", ["Yes","No"])
        in_veggies = st.selectbox("Do you eat vegetables regularly?", ["Yes","No"])
        in_hvyalcoholconsump = st.selectbox("Are you a heavy alcohol consumer?", ["Yes","No"])
        in_genhlth = st.selectbox("Rate your general health", ["Excellent","Fair","Relatively Poor","Very Poor"])
    

        x1,x2,x3,x4,x5,x6,x7,x8=diabetes_pipe(in_age, in_bmi, in_smoker,in_physactivity,in_fruits,
                                              in_veggies, in_hvyalcoholconsump, in_genhlth)
        prob=diabetes_pred(load_model,x1,x2,x3,x4,x5,x6,x7,x8)
        
        
    

    
    except Exception as e:
        print("Error occured. Error type is",type(e).__name__)
    
    else:
        st.write('<p style="font-size:30px; color:red;">***Result:***</p>',unsafe_allow_html=True)
        st.write(f'<p style="font-size:24px; color:blue;">Probability of getting diabetes is *{prob}%*</p>',unsafe_allow_html=True)
        st.write('<p style="font-size:30px; color:red;">*************</p>',unsafe_allow_html=True)
       




if __name__ == "__main__":
    main()


# %%




