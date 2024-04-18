import os
from io import BytesIO
from urllib.parse import quote

import streamlit as st
from azure.storage.blob import BlobServiceClient

import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Azure Blob Storage Setup
# Azure storage account url
storage_account_url = "https://meddocsearchsa.blob.core.windows.net/"
container_name = "med-docs"
# BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient(account_url=storage_account_url)
# Get a client to interact with a specific container - though it won't do anything yet
container_client = blob_service_client.get_container_client(container_name)

# Enter openai_api_key
api_key = os.getenv('OPEN_AI_API_KEY')

# Define your download directory
download_dir = '/app/downloads/med-docs'
# Create chromadb client with file path to where to store database
persist_dir = '/app/VecDB'

# Check if the database is already initialized
if not os.path.exists(os.path.join(persist_dir, 'chromadb.db')):
    client = chromadb.PersistentClient(path = persist_dir)
else:
    st.write("The database is already initialized.")

# Name the collection and choose the embeddings model (OpenAI in this case)
db_collection_name = 'Guide_collection'
embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)
vectordb_guidelines = Chroma(client=client, collection_name=db_collection_name, embedding_function=embeddings_model)

# Use Chroma client to interrogate collection
collection = client.get_collection(name = db_collection_name, embedding_function = embeddings_model)

# Wrap the code in functions that can be called from Streamlit's interface
def list_of_pdfs(collection):
    all_docs = collection.get()
    all_docs_list = []
    for i in range(len(all_docs['ids'])):
        guideline_name = all_docs['metadatas'][i]['source'].split('/')[-1]
        if guideline_name not in all_docs_list:
            all_docs_list.append(guideline_name)
    # st.write(f"There are {len(all_docs_list)} items/PDFs in the index.")  # Add this line
    return all_docs_list

def add_new_pdf(local_file_path, db_collection_name, client):
    try:
        # Load the PDF data with PyPDFLoader
        loader = PyPDFLoader(local_file_path)
        raw_documents = loader.load()
        text_splitterR = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 300, length_function = len)
        documents = text_splitterR.split_documents(raw_documents)

        # Check if documents is empty
        if not documents:
            # st.write(f"No documents generated from raw_documents for file: {local_file_path}")
            return

        # Add the document to the collection
        Chroma.from_documents(documents, embeddings_model, collection_name=db_collection_name, client = client)
        # st.write(f"Successfully added documents from {local_file_path} to the collection.")
    except Exception as e:
        st.write(f"An error occurred while processing the file: {local_file_path}")
        st.write(str(e))

@st.cache_resource
def download_blobs(_blob_service_client, container_name, download_dir):
    # Check if the download directory exists, if not, create it
    os.makedirs(download_dir, exist_ok=True)
    # Get a client to interact with a specific container
    container_client = blob_service_client.get_container_client(container_name)
    # List all blobs in the container
    blob_list = list(container_client.list_blobs())  # Convert blob_list to a list here
    # # Debug: Print the type and length of blob_list
    # st.write(f"Type of blob_list: {type(blob_list)}")
    # st.write(f"Length of blob_list: {len(blob_list)}")
    # Get the names of the blobs in the container
    blob_names = [blob.name for blob in blob_list]
    # Get the names of the files in the download directory
    local_files = os.listdir(download_dir)

    # Check if the blobs in the Blob Storage are the same as in the download_dir
    if set(blob_names) == set(local_files):
        # st.write("The blobs in the Blob Storage are the same as in the download_dir.")
        return local_files

    # Create a list to store the names of the downloaded blobs
    downloaded_blobs = []

    # Download each blob to a local file
    for blob in blob_list:
        # Generate the local file path
        local_file_path = os.path.join(download_dir, blob.name)
        # st.write(f"Downloading blob: {blob.name}")
        # Create a blob client for the blob
        blob_client = blob_service_client.get_blob_client(container_name, blob.name)
        # Download the blob data into a BytesIO object
        blob_data = BytesIO()
        try:
            download_stream = blob_client.download_blob()
            blob_data.write(download_stream.readall())
            blob_data.seek(0)
            # st.write(f"Downloaded blob: {blob.name}")
        except Exception as e:
            st.write(f"Failed to download blob: {blob.name}")
            st.write(f"Error: {str(e)}")
            continue

        # Write the blob data to the local file
        with open(local_file_path, 'wb') as f:
            f.write(blob_data.read())
            # st.write(f"Wrote blob data to file: {local_file_path}")

        # Add the name of the downloaded blob to the list
        downloaded_blobs.append(blob.name)

    # st.write("All blobs downloaded successfully.")
    return downloaded_blobs

@st.cache_data
def process_blobs(_blob_list, download_dir, db_collection_name, _client):
    # If collection does not exist then add new pdf, else create list of 'source' docs (from Metadata dictionary)
    for blob in blob_list:
        # Check if collection exists
        # st.write(f"Processing blob: {blob}")
        try:
            collection = client.get_collection(name = db_collection_name, embedding_function = embeddings_model)
            screen = collection.get(where = {'source': os.path.join(download_dir, blob)})
            if screen['metadatas'] == []:
                # st.write(f"Vectordb does not exist for file: {blob}")
                add_new_pdf(local_file_path=os.path.join(download_dir, blob), db_collection_name=db_collection_name, client=client)
            # else:
            #     st.write(f"Vectordb already created for file: {blob}")
        except ValueError:
            # st.write("Collection does not exist so creating it.")
            add_new_pdf(local_file_path=os.path.join(download_dir, blob), db_collection_name=db_collection_name, client=client)

def query_db(query):
    if query == 'list':
        st.write('\n')
        coll_list = list_of_pdfs(collection)
        for l in range(len(coll_list)):
            st.write(coll_list[l])
    else:
        # Print the query
        st.write(f"Results for: {query}")
        # Initialize docs and seen
        docs = []
        seen = set()
        # Keep searching until we find 4 unique results
        k = 4
        while len(docs) < 4:
            new_docs = vectordb_guidelines.similarity_search(query, k=k)
            if not new_docs:  # Add this line
                st.write("No more documents found.")
                break  # Add this line
            # Remove duplicates from new_docs based on page_content
            new_docs = [x for x in new_docs if not (x.page_content in seen or seen.add(x.page_content))]
            # Add the new unique docs to docs
            docs.extend(new_docs)
            # Increase k for the next search
            k += 4

        # Create a list of tuples, each containing a doc and its score
        docs_with_scores = []
        for doc in docs:
            doc_name = doc.metadata['source'].split('/')[-1].split('\\')[-1].replace(' ', '%20').lower()
            score = doc_name.count(query.lower())
            docs_with_scores.append((doc, score))

        # Sort the docs by their scores in descending order
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        st.subheader("\nThe top 4 results are:")
        for i in range (min(len(docs_with_scores), 4)):  # Change this line
            # Get the document from the tuple
            doc = docs_with_scores[i][0]
            # Get the document name
            doc_name = doc.metadata['source'].split('/')[-1]
            # Remove the directory name from the document name
            doc_name = doc_name.split('\\')[-1]
            # Replace spaces in the document name with %20
            doc_url_name = doc_name.replace(' ', '%20')
            # Create the document URL
            doc_url = f"{storage_account_url}{container_name}/{doc_url_name}"
            # Display the document name as a hyperlink and the page number
            st.markdown(f"[{doc_name}]({doc_url}) - Page: {doc.metadata['page']}", unsafe_allow_html=True)

        # Display the excerpt from the top result
        st.subheader("\nExcerpt from the top result:")
        st.write(docs_with_scores[0][0].page_content)
        return [doc for doc, score in docs_with_scores]

# Read the version from the environment variable
app_version = os.getenv('APP_VERSION', 'N/A')
# Display the version in the title
st.title(f"Guideline Search - V {app_version}")

# Call the function to download blobs
blob_list = download_blobs(blob_service_client, container_name, download_dir)
blob_list = list(blob_list)
st.write(f"Number of documents: {len(blob_list)}")

# Process the blobs
process_blobs(blob_list, download_dir, db_collection_name, client)

# Get the query from the user
query = st.text_input('Enter your query (enter "list" to see full list of guidelines):')

# Check if the query is not empty
if query:
    # Run the query and get the results
    docs = query_db(query)

    # Display the results
    if docs:
        st.markdown("---")
        st.header("Search Results")
        for i in range(len(docs)):
            # Get the document name
            doc_name = docs[i].metadata['source'].split('/')[-1]
            # Remove the directory name from the document name
            doc_name = doc_name.split('\\')[-1]
            # Replace spaces in the document name with %20
            doc_url_name = doc_name.replace(' ', '%20')
            # Create the document URL
            doc_url = f"{storage_account_url}{container_name}/{doc_url_name}"
            # Display the document name as a hyperlink and the page number
            st.markdown(f" ##### [{doc_name}]({doc_url}) - Page: {docs[i].metadata['page']}", unsafe_allow_html=True)
            # Display the document content
            st.write(docs[i].page_content)