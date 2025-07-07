import os
import PyPDF2
import streamlit as st
import pickle
import tempfile
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from pymongo import MongoClient
import gridfs
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings("ignore")

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="RAG Chat with History", layout="wide")

# Load keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")  # Add this to your .env file
# st.write(MONGODB_URI)

# MongoDB setup
@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection"""
    try:
        client = MongoClient(MONGODB_URI)
        db = client.rag_chatbot
        fs = gridfs.GridFS(db)
        
        # Test connection
        client.admin.command('ping')
        return client, db, fs
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None, None, None

# HuggingFace embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize resources
try:
    client, db, fs = init_mongodb()
    embeddings = load_embeddings()
    mongodb_connected = client is not None and db is not None and fs is not None
except Exception as e:
    st.error(f"Error initializing resources: {str(e)}")
    client, db, fs, embeddings = None, None, None, None
    mongodb_connected = False

# Streamlit App
st.title("PDF ChatGPT")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Database operations
class MongoDBVectorStore:
    def __init__(self, db, fs, embeddings):
        self.db = db
        self.fs = fs
        self.embeddings = embeddings
        self.collection = db.documents
        self.faiss_collection = db.faiss_indices
        
    def save_faiss_index(self, vector_store, index_name="default"):
        """Save FAISS index to MongoDB GridFS"""
        try:
            # Create temporary file to save FAISS index
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, "temp_faiss")
                vector_store.save_local(temp_path)
                
                # Read all FAISS files
                faiss_files = {}
                for filename in os.listdir(temp_path):
                    filepath = os.path.join(temp_path, filename)
                    with open(filepath, 'rb') as f:
                        faiss_files[filename] = f.read()
                
                # Remove existing index with same name
                existing = self.faiss_collection.find_one({"name": index_name})
                if existing:
                    for file_id in existing.get("file_ids", []):
                        try:
                            self.fs.delete(file_id)
                        except:
                            pass
                    self.faiss_collection.delete_one({"name": index_name})
                
                # Save files to GridFS
                file_ids = {}
                for filename, content in faiss_files.items():
                    file_id = self.fs.put(content, filename=f"{index_name}_{filename}")
                    file_ids[filename] = file_id
                
                # Save metadata
                self.faiss_collection.insert_one({
                    "name": index_name,
                    "file_ids": file_ids,
                    "created_at": datetime.now(),
                    "doc_count": len(vector_store.docstore._dict) if hasattr(vector_store, 'docstore') else 0
                })
                
                return True
        except Exception as e:
            st.error(f"Error saving FAISS index: {str(e)}")
            return False
    
    def load_faiss_index(self, index_name="default"):
        """Load FAISS index from MongoDB GridFS"""
        try:
            # Find index metadata
            index_meta = self.faiss_collection.find_one({"name": index_name})
            if not index_meta:
                return None
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download and save files
                for filename, file_id in index_meta["file_ids"].items():
                    try:
                        content = self.fs.get(file_id).read()
                        filepath = os.path.join(temp_dir, filename)
                        with open(filepath, 'wb') as f:
                            f.write(content)
                    except Exception as e:
                        st.error(f"Error downloading file {filename}: {str(e)}")
                        return None
                
                # Load FAISS index
                vector_store = FAISS.load_local(
                    temp_dir, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                return vector_store
                
        except Exception as e:
            st.error(f"Error loading FAISS index: {str(e)}")
            return None
    
    def save_document(self, filename, content, chunks):
        """Save document metadata and chunks"""
        try:
            # Create document hash for deduplication
            doc_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Check if document already exists
            existing = self.collection.find_one({"hash": doc_hash})
            if existing:
                st.warning(f"Document '{filename}' already exists in database.")
                return False
            
            # Save document
            doc_data = {
                "filename": filename,
                "hash": doc_hash,
                "content": content,
                "chunks": chunks,
                "uploaded_at": datetime.now(),
                "chunk_count": len(chunks)
            }
            
            self.collection.insert_one(doc_data)
            return True
            
        except Exception as e:
            st.error(f"Error saving document: {str(e)}")
            return False
    
    def get_all_documents(self):
        """Get all documents from database"""
        try:
            return list(self.collection.find({}, {"content": 0}))  # Exclude content for performance
        except Exception as e:
            st.error(f"Error fetching documents: {str(e)}")
            return []
    
    def delete_document(self, doc_id):
        """Delete a document"""
        try:
            self.collection.delete_one({"_id": doc_id})
            return True
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False
    
    def get_all_chunks(self):
        """Get all chunks from all documents"""
        try:
            chunks = []
            for doc in self.collection.find({}, {"chunks": 1}):
                chunks.extend(doc.get("chunks", []))
            return chunks
        except Exception as e:
            st.error(f"Error fetching chunks: {str(e)}")
            return []

# Check if MongoDB is available before proceeding
if mongodb_connected and embeddings is not None:
    # Initialize MongoDB vector store
    mongo_store = MongoDBVectorStore(db, fs, embeddings)
    
    # Sidebar: PDF upload and management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # Upload section
        file = st.file_uploader("Upload a PDF", type="pdf")
        
        # Display existing documents
        st.subheader("📚 Existing Documents")
        docs = mongo_store.get_all_documents()
        
        if docs:
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc['filename']}")
                    st.caption(f"Uploaded: {doc['uploaded_at'].strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"Chunks: {doc.get('chunk_count', 0)}")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['_id']}"):
                        if mongo_store.delete_document(doc['_id']):
                            st.success("Document deleted!")
                            st.rerun()
        else:
            st.info("No documents uploaded yet.")
        
        # Control buttons
        st.subheader("🔧 Controls")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")
        
        if st.button("🔄 Rebuild Index"):
            with st.spinner("Rebuilding FAISS index..."):
                chunks = mongo_store.get_all_chunks()
                if chunks:
                    vector_store = FAISS.from_texts(chunks, embeddings)
                    if mongo_store.save_faiss_index(vector_store):
                        st.session_state.vector_store = vector_store
                        st.success("Index rebuilt successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to rebuild index.")
                else:
                    st.warning("No documents to index.")

    # PDF text extractor
    def extract_text_from_pdf(file_path):
        try:
            reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Load FAISS index on startup
    if st.session_state.vector_store is None:
        with st.spinner("Loading FAISS index from MongoDB..."):
            vector_store = mongo_store.load_faiss_index()
            if vector_store:
                st.session_state.vector_store = vector_store
                st.sidebar.success("🔁 FAISS index loaded from MongoDB.")
            else:
                st.sidebar.info("No existing FAISS index found.")

    # Upload and process new PDF
    if file:
        with st.spinner("Processing PDF..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file.getbuffer())
                    tmp_file_path = tmp_file.name

                # Extract text
                text = extract_text_from_pdf(tmp_file_path)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                if not text.strip():
                    st.error("No text extracted from PDF. The PDF might be image-based or corrupted.")
                else:
                    # Split into chunks
                    chunks = splitter.split_text(text)
                    
                    if not chunks:
                        st.error("No chunks created from the PDF text.")
                    else:
                        # Save document to MongoDB
                        if mongo_store.save_document(file.name, text, chunks):
                            st.sidebar.success(f"✅ '{file.name}' saved to MongoDB.")
                            
                            # Update FAISS index
                            try:
                                if st.session_state.vector_store:
                                    st.session_state.vector_store.add_texts(chunks)
                                else:
                                    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                                
                                # Save updated index
                                if mongo_store.save_faiss_index(st.session_state.vector_store):
                                    st.sidebar.success("📚 FAISS index updated in MongoDB.")
                                    st.rerun()
                                else:
                                    st.sidebar.error("Failed to update FAISS index.")
                            except Exception as e:
                                st.sidebar.error(f"Error updating FAISS index: {str(e)}")
                        else:
                            st.sidebar.error("Failed to save document to MongoDB.")
                        
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    # Main chat area
    if st.session_state.vector_store:
        st.subheader("💬 Ask a question about your PDFs")
        
        # Create columns for better layout
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input("Your question:", key="question_input")
        
        with col2:
            search_button = st.button("🔍 Search", type="primary")

        if question and (search_button or question):
            with st.spinner("Searching for relevant information..."):
                try:
                    # Perform similarity search
                    results = st.session_state.vector_store.similarity_search(question, k=3)
                    
                    if not results:
                        st.warning("No relevant documents found for your query.")
                    else:
                        # Initialize LLM
                        llm = ChatGroq(
                            groq_api_key=GROQ_API_KEY,
                            model_name="llama3-70b-8192",
                            temperature=0,
                            max_tokens=1000,
                        )

                        # Create retrieval QA chain
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
                            return_source_documents=True
                        )
                        
                        # Get answer
                        result = qa_chain({"query": question})
                        answer = result["result"]
                        source_docs = result["source_documents"]

                        # Save to chat history
                        st.session_state.chat_history.append((question, answer))
                        
                        # Display current answer
                        st.markdown("### 🤖 Answer:")
                        st.markdown(answer)
                        
                        # Show source chunks
                        with st.expander("📖 Source Information"):
                            for i, doc in enumerate(source_docs, 1):
                                st.markdown(f"**Source {i}:**")
                                content = doc.page_content
                                if len(content) > 500:
                                    st.markdown(content[:500] + "...")
                                else:
                                    st.markdown(content)
                                st.markdown("---")
                        
                except Exception as e:
                    st.error(f"Error processing your question: {str(e)}")
                    st.info("Please try rephrasing your question or check if documents are properly loaded.")
    else:
        st.info("📤 Please upload a PDF file to start chatting!")

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 🗃️ Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{i}: {q[:50]}..." if len(q) > 50 else f"Q{i}: {q}"):
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")

    # Database stats
    with st.sidebar:
        st.subheader("📊 Database Stats")
        if docs:
            total_docs = len(docs)
            total_chunks = sum(doc.get('chunk_count', 0) for doc in docs)
            st.metric("Total Documents", total_docs)
            st.metric("Total Chunks", total_chunks)
            
            # Show recent uploads
            recent_docs = sorted(docs, key=lambda x: x.get('uploaded_at', datetime.min), reverse=True)[:3]
            st.markdown("**Recent Uploads:**")
            for doc in recent_docs:
                st.caption(f"📄 {doc['filename']}")
        else:
            st.metric("Total Documents", 0)
            st.metric("Total Chunks", 0)

else:
    st.error("❌ Failed to connect to MongoDB. Please check your connection string in the .env file.")
    st.info("Add MONGODB_URI to your .env file with your MongoDB connection string.")
    st.code("""
# Add this to your .env file:
GROQ_API_KEY=your_groq_api_key_here
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/rag_chatbot?retryWrites=true&w=majority
""")