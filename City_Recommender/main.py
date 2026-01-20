import os
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["GOOGLE_API_KEY"] = "AIzaSyDubs-CxPBUKKhkwYJU2NgQA6Exxuaw1Qk" # Update this with your Google AI Studio API Key

if not os.getenv("GOOGLE_API_KEY"):
    st.warning("Missing GOOGLE_API_KEY env var. Set it before running.")
    st.stop()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",   # stable model id is supported in Gemini API docs :contentReference[oaicite:2]{index=2}
    temperature=0.1,
    max_output_tokens=256,
)

st.title("City Recommender")

budget = st.sidebar.selectbox(
    "Your Weekly Budget is:",
    ("Less than $1000", "Between $1000 and $2000", "Between $2000 and $5000", "More than $5000"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant."),
    ("user",
     "I want to spend a nice vacation for a week. My budget is {budget}. "
     "Suggest a list of 10 cities to visit that would fit this budget. "
     "Return ONLY the city names as a comma-separated list. No explanations.")
])

chain = prompt | model | StrOutputParser()

if budget:
    try:
        result = chain.invoke({"budget": budget})
        st.write(result)
    except Exception as e:
        st.exception(e)
