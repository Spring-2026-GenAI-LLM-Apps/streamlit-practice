import streamlit as st
import os
import vertexai
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from vertexai.preview.vision_models import ImageGenerationModel

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./key.json" # place the key JSON file in the same folder as your notebook

PROJECT_ID = "cmu-class" # use your project id
REGION = "us-central1"  #



vertexai.init(project=PROJECT_ID, location=REGION)

llm = VertexAI(
    model_name="gemini-2.5-pro",
    max_output_tokens=256,
    temperature=0.5,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

st.title("City Recommender with Sightseeing List")
model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")



my_budget = st.sidebar.selectbox("Your Budget is:", ("Less than $1000", "Between $1000 and $2000", "Between $2000 and $5000", "More than $5000"))
my_duration = st.sidebar.number_input("Enter the Number of Weeks for Your Vacation", step=1)
col1, col2, col3 = st.sidebar.columns(3)
generate_result = col2.button("Tell Me!")
if generate_result:
    prompt_template_budget_duration = PromptTemplate(
        input_variables=['budget','duration'],
        template="I want to spend a nice vacation for {duration} week(s). My budget for the entire trip is {budget}. Suggest exactly one city to visit that would fit this budget. Only display the city without any explanation or description."
    )
    city_chain = LLMChain(llm=llm, prompt=prompt_template_budget_duration, output_key="city_name")
    prompt_template_sightseeing_list = PromptTemplate(
        input_variables=['city_name', 'budget'],
        template="Print the most important two sightseeings in {city_name}. My budget for visiting this city is {budget}. Return the output as a comma-separated string. Don't include any special characters to the output."
    )
    sightseeing_chain = LLMChain(llm=llm, prompt=prompt_template_sightseeing_list, output_key="sightseeing_list")
    chain = SequentialChain(
        chains=[city_chain, sightseeing_chain],
        input_variables=['budget','duration'],
        output_variables=['city_name', 'sightseeing_list']
    )
    result = chain({'budget': my_budget, 'duration': my_duration})
    #st.write(result)
    st.header(result['city_name'].strip())
    places_list = result['sightseeing_list'].strip().split(",")
    st.write("**Places to Visit:**")
    for place in places_list:
        st.write("-", place)
        image_prompt = "Generate an image for " + place + " located in " + result['city_name'].strip()
        images = model.generate_images(
            prompt=image_prompt,
            # Optional parameters
            number_of_images=1,
            language="en",
            # You can't use a seed value and watermark at the same time.
            # add_watermark=False,
            # seed=100,
        )
        images[0].save(location="output.png", include_generation_parameters=False)
        st.image("output.png")
    os.remove("output.png")


