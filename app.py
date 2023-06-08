from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
import requests
from bs4 import BeautifulSoup

def search_statmuse(query: str) -> str:
    URL = f'https://www.statmuse.com/nba/ask/{query}'
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, "html.parser")
    return soup.find("h1", class_="nlg-answer").text

def main():
    load_dotenv()
    st.set_page_config(page_title="Basketball Stats")
    st.header("Search for NBA stats")

    llm = OpenAI(temperature=0)

    statmuse_tool = Tool(
            name = "Statmuse",
            func = search_statmuse,
            description = "A sports search engine. Use this more than normal search if the question is about NBA basketball, like 'who is the highest scoring player in the NBA?'. Always specify a year or timeframe with your search. Only ask about one player or team at a time, don't ask about multiple players at once."
        )

    tools = load_tools(["serpapi", "llm-math"], llm=llm) + [statmuse_tool]

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    input = st.text_input(label='Enter the statistic you would like to find', max_chars=200)

    if input is not None:
        st.write(agent.run(input))


if __name__ == '__main__':
    main()