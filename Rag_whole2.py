# Import necessary modules from llama_index and external libraries
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext,get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core import load_index_from_storage
import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
import faiss
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
import torch
import nest_asyncio
import streamlit as st


nest_asyncio.apply()
# Initialize the embedding model (BAAI/bge-base-en) to be used for text embedding generation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en", device=device)



if os.path.exists('./vectors_stored/'):
    # Load the persisted vector store
    vector_store = FaissVectorStore.from_persist_dir('./vectors_stored/')
    
    # Reload the storage context to use the persisted vectors
    storage_context = StorageContext.from_defaults(vector_store=vector_store,persist_dir='./vectors_stored/')
    
    # Reload the storage context to use the persisted vectors
    index = load_index_from_storage(storage_context=storage_context,embed_model=embed_model)
    
    
else:
    chat_store = SimpleChatStore()

    # Load documents from the "./data/" directory
    documents = SimpleDirectoryReader("./data/PDF/").load_data()
    
    # Initialize Faiss index with 768-dimensional embeddings using L2 distance
    faiss_index = faiss.IndexFlatL2(768)
    
    # Create a FaissVectorStore object to store the embeddings in the index
    vector_store = FaissVectorStore(faiss_index)
    
    # Create a StorageContext, which handles the storage of the index and embeddings
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create an index from the documents, embedding them using the embed_model
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    
    # Persist the index and vector store to disk
    if not os.path.exists('./vectors_stored'):
        os.mkdir('./vectors_stored')  # Create a directory to store the vectors
        index.storage_context.persist(persist_dir="./vectors_stored/")  # Save vector data


# Load persisted chat history from the stored JSON file
if os.path.exists('./Chat_history/chat_store.json'):
    chat_store = SimpleChatStore.from_persist_path(
        persist_path="./Chat_history/chat_store.json"
    )

    # Reload the chat memory buffer with the persisted chat history
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user1",
    )
else:
    chat_store = SimpleChatStore()
    os.mkdir('./Chat_history')
    chat_store.persist(persist_path="./Chat_history/chat_store.json")
    # Create a chat memory buffer to limit the number of tokens per session
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,  # Limit to 3000 tokens per user session
        chat_store=chat_store,
        chat_store_key="user1",  # Store user chat history with this key
    )
# Initialize the language model (LLM) with Ollama, setting model and timeout configurations
streaming_enabled = True
llm = Ollama(model="phi3:mini", request_timeout=360.0,device=device)

query_type = st.radio(
    "Choose the query type",
    ('Use RetrieverQueryEngine (all PDFs)', 'Use Simple Index Query Engine'),
    index=1,
)
# Configure the query engine based on the user's selection
if query_type == 'Use RetrieverQueryEngine (all PDFs)':
    streaming_enabled = False  # Disable streaming for this option
else:
    streaming_enabled = True  # Enable streaming for the simple index option
    
if query_type == 'Use RetrieverQueryEngine (all PDFs)':
    retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
    response_synthesizer = get_response_synthesizer(response_mode="compact_accumulate", llm=llm)
    q_e = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
else:
    q_e = index.as_query_engine(llm=llm, memory=chat_memory, similarity_top_k=2,streaming=True)

# Function to handle chat inputs and return LLM responses
def chat(inp):
    query=f"""Answer the question as truthfully as possible using the provided documents.
    Question: {inp}"""
    
    if not streaming_enabled:
        response_stream = q_e.query(query)
        response_text = str(response_stream)  # Get the full response directly if not streaming
        response_placeholder = st.empty()
        response_placeholder.markdown(response_text)
    else:
        response_stream=q_e.query(query)
        response_placeholder = st.empty()
        response_text = ""  # Store the progressively generated response
        
        for chunk in response_stream.response_gen:  # Iterate through the response chunks
            response_text += chunk  # Accumulate the response
            response_placeholder.markdown(response_text)  # Update the placeholder with the latest text
        
    chat_store.add_message(key="user1", message={'role':'user','content':inp})  # Add input and response to memory
    chat_store.add_message(key="user1", message={'role':'assistant','content':str(response_text)})  # Add input and response to memory
    chat_store.persist("./Chat_history/chat_store.json")
    
    return str(response_text)

# Create a Gradio chat interface for user interaction with the model
# demo = gr.ChatInterface(fn=chat,examples=["hello", "hola", "merhaba"], title="Echo Bot")
# Streamlit UI Setup
st.title("NanoScience Project")

# Initialize chat history
if "messages" not in st.session_state:
    messages = chat_store.get_messages("user1")
    if messages is None:  # Handle the case where no messages exist
        st.session_state.messages = []
    else:
        st.session_state.messages = messages

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)

# Accept user input
prompt = st.chat_input("Enter your prompt here:")

# If there's a new prompt, process and display it
if prompt:
    # Add user message to chat history
    # st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in the Streamlit chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the model and display it
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response = chat(prompt)  # Call the modified chat function with streaming
        response_placeholder.markdown(response)  # Display the final response

    # Append assistant's response to the chat history
    # st.session_state.messages.append({"role": "assistant", "content": response})
    
''' official  prompt
Extract the following data from the provided PDF and present it in a table: 
 (1) Input Data: switching layer material (TYM_Class), Synthesis method (SM_Class), Top electrode (TE_Class), Thickness of Top electrode (TTE in nm), Bottom electrode (BE_Class), Thickness of bottom electrode (TBE in nm), Thickness of switching layer (TSL in nm); (2) Output Data: Type of Switching (TSUB_Class), Endurance (Cycles) (EC), Retention Time (RT in seconds), Memory Window (MW in V), No. of states (MRS_Class), Conduction mechanism type (CM_Class), Resistive Switching mechanism (RSM_Class);
 (3) Reference Information: Name of the paper, DOI, Year. Ensure all data is extracted in the specified categories and format.

'''