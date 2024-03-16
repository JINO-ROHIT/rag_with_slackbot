# RAG with SlackBOT

This project shows how to hook up a RAG pipeline and connect it with a slackbot to receive questions and generate answers in the channel

## Project Overview

This project receives messages from a slackbot and then does RAG to return the generated response back to the slackbot channel. It uses the Vlite numpy vector database to store the embeddings of the simple chunked document.

## How to run

1. ### Install the dependencies

```bash
pip install requirements.txt
```

2. ### Create a .env file and add your slack bot oath token

   SLACK_OATH_TOKEN = "xxxx"
   SLACK_APP_TOKEN = "xxxx"

3. ### Start the bolt server

```bash
python app.py
```