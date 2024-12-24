# Chatbot-Project-1
The project I’m going to talk about is a *Chatbot for PDF files* built using *Streamlit* and *LangChain*. This chatbot lets users upload a PDF file and ask questions about its content. It processes the PDF’s text and provides answers based on the information extracted.

### How I built the project:

1. **User Interface with Streamlit**  
   I used *Streamlit* to create a simple and user-friendly interface for the chatbot. On the sidebar, I included a brief description of the app, highlighting its main features and the technologies used, like Streamlit and LangChain, which are great tools for building machine learning applications.

2. **Reading the PDF**  
   Users can upload their PDF files through the app. I used a library called *PyPDF2* to read the text from the PDF pages. Extracting this text is essential so the chatbot can later answer questions based on the content.

3. **Splitting the Text**  
   PDFs can have a lot of text, which is too much for the chatbot to handle all at once. To fix this, I used *LangChain’s RecursiveCharacterTextSplitter* to break the text into smaller chunks of 1,000 characters, with a 200-character overlap to maintain context. This makes it easier for the chatbot to process the text.

4. **Vector Embeddings and FAISS**  
   After splitting the text, I converted it into numerical representations (called embeddings) using the *sentence-transformers/all-MiniLM-L6-v2* model from HuggingFace. These embeddings help the chatbot understand the meaning of the text. To store and search these embeddings quickly, I used a tool called *FAISS*. If the embeddings for a PDF already exist, the app loads them to save time. Otherwise, it creates and saves new ones.

5. **Searching the Document**  
   Once the PDF is processed, users can ask questions about the document. The app uses FAISS to search for the most relevant text chunks that match the user’s query. This ensures the chatbot retrieves the right information.

6. **Answering the Questions**  
   The chatbot displays the most relevant chunks of text related to the user’s query, making it easy to interact with the document and find answers.

### Challenges  
I also tried using *large language models (LLMs)* from the HuggingFace Hub to improve the chatbot’s performance. However, these models required a lot of computing power and took too much time to run on my system. Additionally, the HuggingFace Hub offers limited free access tokens, which was a restriction for my project.

### Conclusion  
The goal of this project was to make it easy for users to "talk" to their PDF files, whether it’s a research paper, a report, or a textbook. By combining natural language processing and vector-based search, the chatbot can answer questions accurately, making it a helpful tool for working with large amounts of text.
