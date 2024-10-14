from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from scraping import get_medication_info
from langchain.schema import SystemMessage
from langchain.prompts import HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


llm = OllamaLLM(model="llama3.1")

llm.temperature = 0.5

few_shot_prompt = """
1. Question : Quel est l'effet secondaire de l'Ibuprofène ?
   Réponse : L'ibuprofène peut causer des douleurs d'estomac, des nausées et des vertiges.

2. Question : Comment prendre du Paracétamol ?
   Réponse : Le paracétamol doit être pris avec de l'eau, généralement toutes les 4 à 6 heures, sans dépasser la dose recommandée.

3. Question : Quelles sont les interactions du Doliprane ?
   Réponse : Le Doliprane peut interagir avec certains médicaments comme les anticoagulants. Consultez toujours un médecin avant de le prendre.

4. Question : Quels sont les usages du Vicks ?
   Réponse : Le Vicks est utilisé pour soulager la toux et le rhume. Appliquez-le sur la poitrine et le dos.

5. Question : Peut-on donner de l'Aspirine à un enfant ?
   Réponse : L'aspirine ne doit pas être donnée aux enfants sans avis médical, en raison du risque de syndrome de Reye.

Utilisez ces exemples pour répondre aux questions comme si vous les saviez déjà, sans mentionner que vous utilisez un contexte.
"""

# Define the chat prompt template
# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=("You are an AI assistant specialized in the medical field.")),
        SystemMessage(content=("Respond based on the information you know without referencing any external context. Do not invent information.")),
        # Format and structure the question and context
        HumanMessagePromptTemplate.from_template(
        """
        {few_shot_prompt}
        Answer the question based on the following information:
        {context}
        Question: {question}
        """
        ),
    ]
)


# Define the chain
chain = (
    prompt
    | llm
)

# Function to extract medication name using LLM
def extract_medication_name_with_llm(message):
    # Use the LLM to process the message and extract the medication name
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """Identify the name of the medication in the following message, even if there are typos.
         If the medication is identified, respond with the name of the medication and only the name of the medication,
         even if there is a typo. If the name of the medication is not found, respond with 'null'. 
         Do not add '.' at the end of the response."""),
        ("human", message)
    ])
    
    extraction_chain = (
        extraction_prompt
        | llm
    )
    
    result = extraction_chain.invoke({"input": message})
    return result.strip()

# Modify the chatbot function
def chatbot(message, history, session_id):
    # Step 1: Use LLM to extract medication name
    medication_name = extract_medication_name_with_llm(message)

    print(medication_name)

    if medication_name == "null":
        augmented_input = f"{message}\n\nContext: No context found."
    else:
        # Step 2: Retrieve medication info
        medication_info = get_medication_info(medication_name)["medication_info"]

        documents = [Document(page_content=medication_info)]


        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        # split_documents = text_splitter.split_documents(documents)
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # vectorstore = FAISS.from_documents(split_documents, embeddings)
        # retriever = vectorstore.as_retriever()
        # relevant_docs = retriever.get_relevant_documents(message)

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        relevant_docs = query_engine.query(message)

        print(relevant_docs)

        print("Length of relevant docs:", len(relevant_docs))

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        print("Context:", context)
    
    # Step 3: Use the LLM to generate a response
    response = chain.invoke(
        {
            "context": context,
            "question": message,
            "few_shot_prompt": few_shot_prompt
        },
        config={"configurable": {"session_id": session_id}}
    )
    
    return response

# Create the Gradio chat interface
iface = gr.ChatInterface(
    fn=chatbot,
    title="Ordotype AI",
    description="Bonjour ! Je suis votre assistant ordotype, ici pour vous aider.",
    theme="soft",
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()