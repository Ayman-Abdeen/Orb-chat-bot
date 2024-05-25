from flask import Flask, Response
from main_embedding import chat_gen

app = Flask(__name__)


@app.route('/<query>')
def hello_world(query):
    chat= chat_gen()
    chat.ask_Bot(query,)
    return Response(chat.ask_Bot(query,))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port =8123)