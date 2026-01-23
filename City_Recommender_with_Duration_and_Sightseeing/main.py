import os
import streamlit as st
from PIL import Image

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from google import genai

st.set_page_config(page_title="City Recommender", layout="centered")
st.title("City Recommender with Sightseeing List")

os.environ["GOOGLE_API_KEY"] = "AIzaSyDubs-CxPBUKKhkwYJU2NgQA6Exxuaw1Qk" # Update this with your Google AI Studio API Key

# ---- API Key (Google AI Studio) ----
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Missing GOOGLE_API_KEY. Set it in your environment, restart your terminal, then rerun Streamlit.")
    st.stop()

# ---- LLM (Google Generative AI / AI Studio) ----
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.5,
    max_output_tokens=256,
)

# ---- Image client (Google GenAI SDK) ----
genai_client = genai.Client(api_key=api_key)

# ---- UI ----
my_budget = st.sidebar.selectbox(
    "Your Budget is:",
    ("Less than $1000", "Between $1000 and $2000", "Between $2000 and $5000", "More than $5000"),
)
my_duration = st.sidebar.number_input(
    "Enter the Number of Weeks for Your Vacation",
    min_value=1,
    step=1,
)

col1, col2, col3 = st.sidebar.columns(3)
generate_result = col2.button("Tell Me!")

# ---- Prompt 1: pick exactly one city ----
city_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant."),
    ("user",
     "I want to spend a nice vacation for {duration} week(s). "
     "My budget for the entire trip is {budget}. "
     "Suggest EXACTLY ONE city to visit that would fit this budget. "
     "Return ONLY the city name, no punctuation, no explanation.")
])
city_chain = city_prompt | llm | StrOutputParser()

# ---- Prompt 2: 2 key sights in that city ----
sights_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant."),
    ("user",
     "Print the most important TWO sightseeing spots in {city_name}. "
     "My budget for visiting this city is {budget}. "
     "Return ONLY a comma-separated list of the two places. "
     "No numbering, no special characters, no extra text.")
])
sights_chain = sights_prompt | llm | StrOutputParser()


def safe_split_places(text: str) -> list[str]:
    # Robust splitting: handles extra spaces and accidental double commas
    parts = [p.strip() for p in (text or "").split(",")]
    parts = [p for p in parts if p]
    return parts[:2]  # enforce exactly 2


def generate_place_image(place: str, city: str):
    """
    Uses Gemini API image generation (Imagen) via google-genai.
    Returns a PIL Image or None.
    """
    prompt = f"High-quality travel photograph of {place} in {city}, natural lighting, realistic, no text, no watermark."

    # Model names can vary by account/region. If this model name fails for you,
    # I’ll swap it to the correct one for your enabled models.
    resp = genai_client.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt=prompt,
        config={"number_of_images": 1},
    )

    if not getattr(resp, "generated_images", None):
        return None

    img_bytes = resp.generated_images[0].image.image_bytes
    return Image.open(io.BytesIO(img_bytes))


if generate_result:
    with st.spinner("Picking a city..."):
        city_name = city_chain.invoke({"budget": my_budget, "duration": my_duration}).strip()

    if not city_name:
        st.error("City generation returned empty output. Try again or switch to gemini-2.5-flash.")
        st.stop()

    st.header(city_name)

    with st.spinner("Getting top sights..."):
        sights_text = sights_chain.invoke({"city_name": city_name, "budget": my_budget}).strip()

    places_list = safe_split_places(sights_text)

    st.write("**Places to Visit:**")

    for place in places_list:
        st.write("-", place)

        # Generate image for each place
        try:
            with st.spinner(f"Generating an image for {place}..."):
                # Use images only if your account has image generation enabled.
                # If it errors, we show the text list without images.
                import io
                img = None

                prompt = f"High-quality travel photograph of {place} in {city_name}, realistic, natural lighting, no text."
                resp = genai_client.models.generate_images(
                    model="imagen-4.0-generate-001",
                    prompt=prompt,
                    config={"number_of_images": 1},
                )

                if getattr(resp, "generated_images", None):
                    img_bytes = resp.generated_images[0].image.image_bytes
                    img = Image.open(io.BytesIO(img_bytes))

                if img:
                    st.image(img, caption=f"{place} — {city_name}", width='stretch')
                else:
                    st.info(f"No image returned for {place}.")
        except Exception as e:
            st.warning(f"Couldn't generate image for {place}. Showing text only.")
            with st.expander("Image generation error"):
                st.exception(e)