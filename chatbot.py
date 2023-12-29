import telebot
import sys
import logging
from telebot import types
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding

# Initialize the Telegram bot
telegram_api_token = ""
bot = telebot.TeleBot(telegram_api_token)

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load documents and initialize models
documents = SimpleDirectoryReader("data").load_data()

llm = LlamaCPP(
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="thenlper/gte-large")
)

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Function to handle user messages
@bot.message_handler(func=lambda message: True)
def handle_messages(message):
    # Get the user's question
    user_question = message.text

    # Use the index to find a response
    query_engine = index.as_query_engine()
    response = query_engine.query(user_question)

    # Send the response back to the user
    bot.reply_to(message, response)

    # Ask for another question
    bot.send_message(message.chat.id, "Ask another question:")

# Start the bot
if __name__ == "__main__":
    bot.polling(none_stop=True)
