#from openai import OpenAI
import streamlit as st
import google.generativeai as genai
import os
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
GOOGLE_API_KEY='AIzaSyCNcF7j6PuLOmXIKV0kpopcAHZwbB25WqQ'
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
from langchain_community.document_loaders import PyMuPDFLoader
import time
st.title("Resume Analyzer Chatbot")


if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = "gemini-1.5-flash"


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    if i == 0:
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if not st.session_state.file_uploaded:
    placeholder = st.empty()

    with placeholder.container():
        pdf_file = st.file_uploader("Choose a file", "pdf")
        st.header("""Upload Your Resume and Let Me Guide You! 🚀
I'll analyze your resume and provide personalized career and academic advice to help you reach your goals. Let’s unlock your potential together!""")
    if pdf_file is not None:
        st.session_state.file_uploaded = True
        placeholder.empty()
        newplaceholder = st.empty()
        with newplaceholder.container():
            st.success("Resume uploaded successfully! Analyzing...")
        time.sleep(0.5)
        newplaceholder.empty()
        ### saving the uploaded resume to folder
        save_image_path = pdf_file.name
        pdf_name = pdf_file.name
        with open(pdf_file.name, "wb") as f:
            f.write(pdf_file.getbuffer())
        loader = PyMuPDFLoader(save_image_path)
        docs = loader.load()
        res = docs[0].page_content
        llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0,
                max_tokens=None,
                timeout=None,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
                # other params...
            )
        from langchain_core.prompts import PromptTemplate
        template = """You are a Resume Analyzer assistant. Upon extracting the candidate's name, you will initiate the interaction with a personalized greeting. Based on the content of the resume, you will formulate a thoughtful and supportive career-related question, demonstrating your intention to assist the candidate in advancing their professional journey.
Note: If you take find any relevant information in the resume reply "no information found".
#####
Resume:
{res}
        """

        prompt_template = PromptTemplate.from_template(template)
        chain = prompt_template | llm | StrOutputParser()
        # Display user message in chat message container
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt_template.invoke({"res": res}).text})
        def response_generator(prompt,chain):
            for event in chain.stream({"res": prompt}):
                yield event
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(res,chain))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    # Accept user input
if st.session_state.file_uploaded:
    if prompt := st.chat_input("What is up?"):
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
            # other params...
        )
        from langchain_core.prompts import PromptTemplate
        template = """{content}"""

        prompt_template = PromptTemplate.from_template(template)
        chain = prompt_template | llm | StrOutputParser()
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        print(st.session_state.messages)
        def response_generator(prompt,chain):
            prompt = "\n\n".join(["user: " + m['content'] if m['role'] == "user" else "assistant: " + m['content'] for m in st.session_state.messages])
            for event in chain.stream({"content": prompt}):
                yield event
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt,chain))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
