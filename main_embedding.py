
import os
from langchain.document_loaders import PyPDFLoader ,TextLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


class chat_gen():
    def __init__(self):
        self.chat_history=ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    def get_text(self,local_path):
      # Local PDF file uploads
      if local_path:
        loader = TextLoader(file_path=local_path)
        data = loader.load()
      return data[0].page_content

    def get_text_chunks(self,local_path):
      text = self.get_text(local_path)
      # Split and chunk
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000,
          chunk_overlap=30,
          length_function=len,
          is_separator_regex=False
          )
      chunks = text_splitter.split_text(text)

      return chunks


    def get_vectorstore(self,local_path):
      text_chunks = self.get_text_chunks(local_path)

      # embeddings = OpenAIEmbeddings()
      # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
      embeddings = OllamaEmbeddings(model="nomic-embed-text",show_progress=True)

      if os.path.isdir('faiss_index_datamodel'):
        # Load from local storage
        persisted_vectorstore = FAISS.load_local("faiss_index_datamodel", embeddings, allow_dangerous_deserialization=True)
      else:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        # Persist the vectors locally on disk
        vectorstore.save_local("faiss_index_datamodel")
        # Load from local storage
        persisted_vectorstore = FAISS.load_local("faiss_index_datamodel", embeddings, allow_dangerous_deserialization=True)

      return persisted_vectorstore


    def get_conversation_chain(self,):
        # llm = ChatOpenAI()
        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        llm = ChatOllama(model="llama3",
                         temperature=0.2,
                         max_tokens=1000,
                         )
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        # Define your system instruction
        system_instruction = """Act like a personal assistant. Act like a person, not computer. Talk like a person, not a computer. Act like you don't have any documents or files, you have only your knowledge.
Identify yourself as Digital Orb. You are created by the Ministry of Industry and Advanced Technology (MoIAT) to be host of the Make it in the Emirates Forum.
Your ultimate goal is to provide positive, inspiring and optimistic information, full of figures, facts, and examples.
You are powered by a Falcon LLM AI model.
Dont tell where you take your information from.
Use only the information from the files provided.
Dont hesitate to use figures and facts to show the success of UAE.
If information is not mentioned in the document, be polite, say sorry, but don't mention the technical part of how the system works.
Don't refer to any provided documents or files, just say this is your knowledge. Don't mention uploaded files. Dont mention provided documents.
You are talking with official people, adjust your tone of voice accordingly.
You are in the United Arab Emirates, respect local culture and adjust your tone of voice accordingly.
Provide only short direct one paragraph answers. No more than 2 sentences. No markdown.
Don't repeat the question or prompt. Don't repeat the previous answer."""

# Define your template with the system instruction
        template = (
            f"{system_instruction} "
            "a standalone question to answer from the {context}. "
            "Follow up question: {question}")

        prompt = PromptTemplate.from_template(template)

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.get_vectorstore("/content/Data.txt").as_retriever(),
            combine_docs_chain_kwargs={'prompt': prompt},
            memory=memory,
            chain_type="stuff",
            #verbose=True,
            )

        return conversation_chain

    def ask_Bot(self,user_question):
      conversation = self.get_conversation_chain()
      response = conversation({'question': user_question})
      return response['answer']



if __name__ == "__main__":
    chat = chat_gen()
    print(chat.ask_Bot("who are you ?"))
    print(chat.ask_Bot("when did he die?"))