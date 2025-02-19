import os
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
from gpt4all import Embed4All

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the key is loaded successfully
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Make sure it's in the .env file.")

# Initialize the Groq client
client = Groq(api_key=groq_api_key)

def get_response(question):
    # Detect the language of the question
    lang = detect(question)

    # Set the prompt template with the URL directly included
    template_en = """
    You are a dedicated financial data assistant specializing in delivering accurate, concise, and up-to-date information exclusively based on Mirae Asset Mutual Fund's official website: https://www.miraeassetmf.co.in/
    
    Guidelines for Responses:
    1. Format: Always structure your answers in a point-wise format with bullet points for clarity and ease of reading.
    
    2. Focus Areas:
    - Fund performance: NAV, historical returns, benchmarks, and risk metrics.
    - Investment objectives: Fund goals, asset allocation strategies, and target sectors.
    - Portfolio insights: Sector weightage, top holdings, and portfolio diversification.
    - Market analysis: Current trends, financial commentary, and Mirae Asset's market stance.
    - Regulatory updates: Key compliance or policy changes affecting mutual funds.
    
    3. Content Restrictions: Ensure all information is:
    - Directly relevant to Mirae Asset Mutual Fund offerings.
    - Based on verified, official, and current sources.
    - Free from references to unrelated financial products, companies, or speculative content.
    
    4. Boundary Conditions:
    - For unrelated queries (not about Mirae Asset Mutual Funds), respond with:
        "I am designed to provide information solely about Mirae Asset Mutual Fund and its associated products."
    - Avoid excessive details and keep responses succinct while maintaining value.
    
    Example Question: {question}
    
    Answer:
    """
    
    # Choose prompt template based on detected language
    prompt_text = template_en.format(question=question)

    # Groq API to post-process and improve the answer quality
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=1,  
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Capture streamed response
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""

        return response

    except Exception as e:
        return f"An error occurred: {e}"

# Initialize session state for Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello and welcome! 🎉 You're in the right place to explore the Mirae asset mutual fund. Just ask your question and let’s dive into the details!"}
    ]

# Chat input field at the top
user_input = st.chat_input("𝖠𝗌𝗄 𝖺𝗇𝗒𝗍𝗁𝗂𝗇𝗀 𝖺𝖻𝗈𝗎𝗍 𝖬𝗂𝗋𝖺𝖾 𝖠𝗌𝗌𝖾𝗍....!💸")

# Streamlit title
st.header("𝖬𝖨𝖱𝖠𝖤 𝖠𝖲𝖲𝖤𝖳")

# Process user input
if user_input:
    # Add the user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response
    response = get_response(user_input)
   
    # Add the assistant response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display the full chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
