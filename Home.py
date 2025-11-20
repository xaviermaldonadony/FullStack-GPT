import streamlit as st
from langchain.prompts import PromptTemplate

# 7.4
st.set_page_config(
   page_title="Fullstack GPT Home",
   page_icon="🤖"
)

st.title("Fullstack GPT")

st.markdown(
    """
# Hello!
            
Welcome to my FullstackGPT Portfolio!
            
Here are the apps I made:
            
- [x] [DocumentGPT](/DocumentGPT)
- [x] [PrivateGPT](/PrivateGPT)
- [ ] [QuizGPT](/QuizGPT)
- [ ] [SiteGPT](/SiteGPT)
- [ ] [MeetingGPT](/MeetingGPT)
- [ ] [InvestorGPT](/InvestorGPT)
"""
)