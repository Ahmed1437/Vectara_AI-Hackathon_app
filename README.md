# Chat with PDF Document App (RAG based)

Welcome to the official repository for the "Chat with PDF Document App," a cutting-edge solution developed for the Vectara AI Hackathon by the Cloudilic organization. This app is designed to revolutionize the way we interact with PDF documents by enabling a conversational interface powered by advanced AI technologies.

## Overview

The "Chat with PDF Document App" leverages the Retrieval-Augmented Generation (RAG) approach, combining the strengths of Vectara's highly efficient document indexing and search capabilities with the powerful language understanding and generation models of OpenAI GPT-3.5 and LLaMA 2. This unique combination allows users to engage in natural language conversations with any PDF document, extracting information, insights, and answers directly from the text.

## Features

- **PDF Document Conversations**: Chat in natural language with any PDF document to get answers, summaries, and insights.
- **Advanced AI Integration**: Utilizes Vectara's indexing and search, OpenAI's GPT-3.5, and LLaMA 2 models for understanding and generating human-like responses.
- **Seamless User Experience**: Easy-to-use interface for uploading PDF documents and starting conversations right away.
- **Insightful Analytics**: Gain valuable insights into the content of your PDFs with advanced analytics powered by AI.

## Getting Started

To get started with the "Chat with PDF Document App," follow these steps:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/Ahmed1437/Vectara_AI-Hackathon_app.git
    ```

2. **Set Up Your Environment**

    Ensure you have Python 3.8 or newer installed. Set up a virtual environment and install the required dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Run the Application**

    Follow the instructions in the deployment guide to start the app and begin chatting with your PDF documents.

## How It Works

The app processes PDF documents by first extracting the text using PDF parsing libraries. It then uses Vectara's indexing to organize the content, making it searchable. When a user queries the chat interface, the app retrieves relevant information from the document using Vectara's search capabilities. It then uses RAG, combining this retrieved information with the generative powers of GPT-3.5 and LLaMA 2, to create informative, contextually relevant responses.

## Acknowledgments

This project was created for the Vectara AI Hackathon to showcase the potential of integrating Vectara's document indexing and search capabilities with state-of-the-art language models from OpenAI and LLaMA. Special thanks to the Cloudilic organization for their support and collaboration in making this project a reality.
