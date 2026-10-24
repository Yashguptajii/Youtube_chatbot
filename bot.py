from langchain_openai import OpenAIEmbeddings
from EmbeddedModels.documents_similarity import embedding
from h11._abnf import chunk_size
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

video_id = "PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0"

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi"])
    
    transcript=" ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available fro this video")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

len(chunks)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever= vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})

print(retriever)

retriever.invoke('')

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.2)

prompt = PromptTemplate(
    template="""You are a helpful assistant.
    Answer only from the provided transcript context.
    If the context is insufficient, just say you don't know.
    {context}
    Question: {question}
    """,
    input_variables=['context','question']
)

question = "is the topic of aliens discussed in this video? if yes then what was discussed?"

retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context":context_text,"question": question})

answer = llm.invoke(final_prompt)
print(answer.content)


