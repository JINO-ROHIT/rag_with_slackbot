import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from vlite_db.main import VLite
from vlite_db.utils import *

from loguru import logger

from generation import load, generate

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.simplefilter("ignore")

system_prompt = """[INST] <<SYS>>
You are a helpful python assistant. Given the context , answer the user's query ONLY with the current context. DONOT make up anything and if you dont know the answer, just say you dont know.
<</SYS>>
"""

app = App(token = os.environ.get('SLACK_OATH_TOKEN', default = ''))

db = VLite('vlite_20240315_191416.npz')
logger.info("Database has been loaded")

_, _, _, pipe = load()
logger.info("Model has been loaded")

@app.message(".*")
def message_handler(message, say):
    #print(message)

    extracted_chunks, _ = db.remember(message['text'], top_k = 3)

    context = ""
    for _text in extracted_chunks:
        context += _text
        
    message = system_prompt + "\n\n USER QUERY \n" + message['text'] + "\n\n CONTEXT: \n" + context
    message += ' [/INST]'

    #print(message)

    final_response = generate(pipe, message, max_new_tokens = 512)
    final_response = final_response.split("[/INST]")[-1].strip().split(':')[-1].strip()
    logger.info("Response has been generated")

    say(final_response)



if __name__ == "__main__":
    SocketModeHandler(app, os.environ.get('SLACK_APP_TOKEN', default='')).start()