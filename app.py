import pandas as pd
import  streamlit as st
import numpy as np
import  pandas as pd

import  pickle

pipe= pickle.load(open('pipe.pkl','rb'))

st.header("car price predictor")

year=st.number_input("make year")

kms=st.number_input("KMs driven")

fule=st.selectbox("Fule type",("Diesel","Petrol"))

seller=st.selectbox("Seller type",("Individual","Dealer"))

transmission=st.selectbox("Transmission type",("Manual","Automatic"))

owner=st.selectbox("Owner type",("First Owner","Second Owner","Third Owner"))

mileage=st.number_input("Mileage")

engine=st.number_input("Engine")

power=st.number_input("Max power")

seat=st.number_input("Seat")

brand=st.selectbox("Brand",("Maruti","Hyundai","Mahindra","Tata","Ford","Honda","Toyota","Renault","Chervolet","Volkswagen"))

if st.button('price predict'):
    input=np.array([[year,kms,fule,seller,transmission,owner,mileage,engine,power,seat,brand]])
    input=pd.DataFrame(input,columns=["year","km_driven","fuel","seller_type","transmission","owner","mileage","engine","max_power","seats","brand"])
    y_pred=pipe.predict(input)
    st.title("RS "+str(np.round(y_pred[0])))


