# Yoga Assistant Bot

Yoga Assistant Bot is an AI-powered chatbot designed to provide accurate, respectful, and constructive responses related to yoga concepts, benefits, and financial knowledge. The bot uses advanced natural language processing techniques, including vector databases and retrieval-based question answering, to deliver reliable and context-aware answers.

---

## Features

- **PDF Document Parsing**: Load and process PDF documents to extract relevant information.
- **Vector Database**: Use FAISS for efficient document storage and retrieval.
- **Custom Prompting**: Tailored prompts to ensure ethical, unbiased, and accurate responses.
- **Retrieval-Based QA**: Combines document retrieval with OpenAI's GPT-3.5-turbo for context-aware answers.
- **Streamlit Interface**: A user-friendly web interface for interacting with the bot.
- **Chainlit Integration**: Support for conversational workflows using Chainlit.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/yoga-assistant-bot.git
   cd yoga-assistant-bot
   ```
# Install the required dependencies:
'''pip install -r requirements.txt'''
Set your OpenAI API key:
 ```export OPENAI_API_KEY="your_openai_api_key" ```
Usage
1. Create the Vector Database
Run the script to process PDF documents and create a FAISS vector database:
 ```python create_vector_db.py ```

3. Run the Streamlit Interface
Launch the Yoga Assistant Bot web interface:
 ```streamlit run yoga_assistant.py  ```



5. Run the Chainlit Chatbot
Start the chatbot using Chainlit:
```chainlit run yoga_assistant.py```

Project Structure
create_vector_db.py: Script to process PDF documents and create a FAISS vector database.
yoga_assistant.py: Main script for the Streamlit-based chatbot interface.
chainlit_integration.py: Chainlit integration for conversational workflows.
vectorstores/: Directory to store the FAISS vector database.
DATA articles/: Directory containing the PDF documents to be processed.
Technologies Used
LangChain: For document loading, text splitting, and retrieval-based QA.
FAISS: Vector database for efficient document retrieval.
HuggingFace Transformers: For generating embeddings using sentence-transformers/all-MiniLM-L6-v2.
OpenAI GPT-3.5-turbo: For generating conversational responses.
Streamlit: For building the web interface.
Chainlit: For conversational chatbot workflows.
Custom Prompt
The bot uses a custom prompt to ensure ethical, unbiased, and accurate responses:

Example Queries
"What are the benefits of yoga for mental health?"
"How does yoga improve flexibility?"
"Can you explain the financial benefits of yoga retreats?"
Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
LangChain
HuggingFace Transformers
FAISS
Streamlit
Chainlit


