import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform

plt = platform.system()
if plt == 'Linux':pathlib.WindowsPath = pathlib.PosixPath

st.button('Rerun') 

# title
st.title('Aralash klassifikatsiya qiluvchi model')
st.text('Racket Helmet Fish Door Bed shu rasimlarga ishleydigan model')

#rasimni joylash
file = st.file_uploader('Rasm yuklash',type=['png','jpeg','gif','svg','jpg'])
if file:
    st.image(file)

    #PIL convert
    img = PILImage.create(file)

    

    #model
    model = load_learner('Aralash_model.pkl')

    #pridikshin
    pred,pred_id,probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Extimollik: {probs[pred_id]*100:.1f}%")

    # plotlin
    fig = px.bar(x = probs*100,y = model.dls.vocab)
    st.plotly_chart(fig)

  
st.date_input('Your birthday')