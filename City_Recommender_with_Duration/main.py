import streamlit as st
import os
import vertexai
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./key.json" # place the key JSON file in the same folder as your notebook

PROJECT_ID = "cmu-class" # use your project id
REGION = "us-central1"  #

vertexai.init(project=PROJECT_ID, location=REGION)

llm = VertexAI(
    model_name="gemini-2.5-pro",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

st.title("City Recommender with Duration and Budget")


my_budget = st.sidebar.selectbox("Your Budget is:", ("Less than $1000", "Between $1000 and $2000", "Between $2000 and $5000", "More than $5000"))
my_duration = st.sidebar.number_input("Enter the Number of Weeks for Your Vacation", step=1)
col1, col2, col3 = st.sidebar.columns(3)
generate_result = col2.button("Tell Me!")
if generate_result:
    prompt_template_name = PromptTemplate(
        input_variables=['budget','duration'],
        template="I want to spend a nice vacation for {duration} week(s)https://www.twitch.tv/viper. My budget for the entire trip is {budget}. Suggest a list of 10 cities to visit that would fit this budget. Display the list of cities as a comma separated list. Only display the cities without any explanation or description."
    )
    chain = LLMChain(llm=llm, prompt=prompt_template_name)
    result = chain.run(budget=my_budget,duration=my_duration)
    st.write(result)
