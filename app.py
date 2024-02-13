import os
import streamlit as st

st.set_page_config(
    page_title="Cloudilic BOT",
    page_icon="./logo/CLOUDILIC.png",
    initial_sidebar_state="auto",
)
st.sidebar.image("./logo/widelogo.png", width=250)

def hhem_score(input_list):
    """
    HHEM (Hughes Hallucination Evaluation Model) Function.

    :param input_list: list contains document & resonse from llm.
    :return: score
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder('vectara/hallucination_evaluation_model')
    
    score =  model.predict([input_list])
    
    return score[0]

def average_hhem_score(input_list):
    
    return sum(input_list) / len(input_list) 

def Chatbot_Vectara_RAG(query, file_path, chat_history, temperature, model_selection):
    """
    Chatbot with RAG using Vectara vectorestore Function.


    :param query: input query.
    :param file_path: pdf file path.
    :param temperature: llm temperature.
    :param model_selection: llm model selected from on UI.
    
    :return: bot answer
    """
    # imports
    
    import openai
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models.fireworks import ChatFireworks
    # from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.vectorstores import Vectara
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    
    #environment keys
    
    fireworks_api_key = st.secrets["FIREWORKS_API_KEY"]

    openai_api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    
    vectara_customer_id = st.secrets["VECTARA_CUSTOMER_ID"]
    vectara_corpus_id = st.secrets["VECTARA_CORPUS_ID"]
    vectara_api_key = st.secrets["VECTARA_API_KEY"]
    
    # file load
    
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    
    vectara = Vectara(
                vectara_customer_id=vectara_customer_id,
                vectara_corpus_id=vectara_corpus_id,
                vectara_api_key=vectara_api_key
            )
    vs = Vectara.from_documents(documents, embedding=None)
    similarity_search_result = vectara.similarity_search_with_score(query)
    # print(similarity_search_result)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # choose model based on UI
    if model_selection == "OpenAI: GPT 3.5":        
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=temperature)
    elif model_selection == "LLama 2: 70B":
        # llm = Anyscale(model_name="meta-llama/Llama-2-70b-chat-hf", temperature=temperature)
        llm = ChatFireworks(fireworks_api_key=fireworks_api_key,model="accounts/fireworks/models/llama-v2-70b-chat", temperature=temperature)
    # elif model_selection == "Gemini Pro":
    #     llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature, convert_system_message_to_human=True)
        
        
    retriever = vs.as_retriever()

    bot = ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory, verbose=False
    )
    
    result = bot.invoke({"question": query, "chat_history": chat_history})
    
    
    
    return result, similarity_search_result[0]


import tempfile
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Create a sidebar
st.sidebar.title("Model Configuration")

# File uploader moved to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Option menu for model selection
model_selection = st.sidebar.selectbox("Model Selection", ["OpenAI: GPT 3.5", "LLama 2: 70B"])

# Slider for selecting model temperature
model_temperature = st.sidebar.slider("Select model temperature", value=0.5, min_value=0.0, step=0.1, max_value=1.0)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

reset_button = st.sidebar.button("Reset")

if reset_button:
    st.session_state.chat_history = []
    st.rerun()
    

if uploaded_file:
    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    
    query = st.chat_input("Ask your question")
            
    if query:
        Chatbot_Vectara_RAG(query, temp_pdf_path, st.session_state.chat_history, model_temperature, model_selection)
        response, semantic_search_result = Chatbot_Vectara_RAG(query, temp_pdf_path, st.session_state.chat_history, model_temperature, model_selection)
        answer = response['answer']
        chat_history = response['chat_history']
        document, similarity_search_score = semantic_search_result
        
        st.session_state.chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=answer),
            ]
        )
        formatted_chat_history = []
    
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                formatted_chat_history.append((message.content, "user"))
            elif isinstance(message, AIMessage):
                formatted_chat_history.append((message.content, "assistant"))


        # Display the messages
        for content, role in formatted_chat_history:
            with st.chat_message(role):
                st.markdown(content)
         
         
            
        if response:
            
            # if semantic_search_result:
            #     st.sidebar.write('Semantic Search Score: ', similarity_search_score)
            st.sidebar.write("*Note: Reset button resect chat\nResults updated with every input & output*")  
            st.sidebar.markdown("## RAG Results")  
            hhem_score_on_rag_response = hhem_score([query, document.page_content])
            st.sidebar.write('HHEM between user input & RAG results: \n Score: ', hhem_score_on_rag_response)
            
            st.sidebar.markdown("## LLM Results")  
            hhem_score_on_llm_response = hhem_score([document.page_content, answer])
            st.sidebar.write('HHEM between Response & RAG results: \n Score: ', hhem_score_on_llm_response)
