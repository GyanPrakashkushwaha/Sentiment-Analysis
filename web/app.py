import streamlit as st
from predict import preprocess_input,tokenize_vectorize_input
from keras.models import load_model
from streamlit.components.v1 import components

model = load_model(r'sentimentAnalysisModel')


st.title("Sentiment Analysis 😊😐😕😡")
st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333; margin-bottom: 10px;margin-top: 1px" />""",unsafe_allow_html=True)

txt = st.text_area(label='input')
def add_text(txt):
    if txt:
        preprocesssed_txt = preprocess_input(text=txt)
        txt_to_predict = tokenize_vectorize_input(text=preprocesssed_txt)
        return txt_to_predict

txt_to_predict = add_text(txt=txt)
if st.button('predict'):
    pred = model.predict(txt_to_predict)[0][0]
    st.subheader(f'negative: {1- round(pred,ndigits=3)} %')
    st.subheader(f'positive: {round(pred,ndigits=3)} %')

    # st.markdown(f' > ## {model.predict(txt_to_predict)[0][0]}')
    
        



