import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser, output_parser
from langchain.retrievers import WikipediaRetriever


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="💩",
)

st.title("💩 QuizGPT")
st.markdown(
    """
            #### 제가 내는 문제를 한번 맞춰보시렵니까?
            *퀴즈 진행을 위해 아래 순서를 따라주세요*
            1. 왼쪽 설정 창에 OpenAPI API키를 입력해주세요 
            2. 난이도를 선택해주세요
            """
)


@st.cache_data(show_spinner="위키피디아 검색중...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    st.title("설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("API Key를 입력해주세요", icon="🗝️")
    st.markdown("***")

    difficulty = st.selectbox(
        "난이도 선택",
        ["쉬움", "보통", "어려움"],
    )
    st.markdown("***")
    topic = st.text_input("퀴즈 주제를 정해주세요")
    if topic:
        docs = wiki_search(topic)
    st.markdown("***")

    st.link_button("Github Repo 바로가기", "https://github.com/asuracoder91/quizgpt")


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# Prompt

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    You MUST use only Korean language on questions and answers.
    Use (o) to signal the correct answer.

    Question examples:
         
    Question: 바다의 색깔은 무엇입니까?
    Answers: 빨강|노랑|초록|파랑(o)
         
    Question: 대한민국의 수도는?
    Answers: 방콕|서울(o)|뉴욕|도쿄
         
    Question: 영화 아바타가 개봉한 해는?
    Answers: 2007|2001|2009(o)|1998
         
    Question: 이순신은 누구입니까?
    Answers: 해군장군(o)|화가|배우|모델
         
    Your turn!
         
    Context: {context}
    """,
        )
    ]
)


formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: 바다의 색깔은 무엇입니까?
    Answers: 빨강|노랑|초록|파랑(o)
         
    Question: 대한민국의 수도는?
    Answers: 방콕|서울(o)|뉴욕|도쿄
         
    Question: 영화 아바타가 개봉한 해는?
    Answers: 2007|2001|2009(o)|1998
         
    Question: 이순신은 누구입니까?
    Answers: 해군장군(o)|화가|배우|모델
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "바다의 색깔은 무엇입니까?",
                "answers": [
                        {{
                            "answer": "빨강",
                            "correct": false
                        }},
                            "answer": "노랑",
                            "correct": false
                        }},
                        {{
                            "answer": "초록",
                            "correct": false
                        }},
                        {{
                            "answer": "파랑",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "대한민국의 수도는?",
                "answers": [
                        {{
                            "answer": "방콕",
                            "correct": false
                        }},
                        {{
                            "answer": "서울",
                            "correct": true
                        }},
                        {{
                            "answer": "뉴욕",
                            "correct": false
                        }},
                        {{
                            "answer": "도쿄",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "영화 아바타가 개봉한 해는?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "이순신은 누구입니까?",
                "answers": [
                        {{
                            "answer": "해군장군",
                            "correct": true
                        }},
                        {{
                            "answer": "화가",
                            "correct": false
                        }},
                        {{
                            "answer": "배우",
                            "correct": false
                        }},
                        {{
                            "answer": "모델",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)


@st.cache_data(show_spinner="퀴즈 생성중")
def run_quiz_chain(_docs, topic):
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.1)
    questions_chain = {"context": format_docs} | questions_prompt | llm
    formatting_chain = formatting_prompt | llm
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


# 실행
if openai_api_key and difficulty and topic:
    if not docs:
        st.markdown(
            """
        안녕하세요, QuizGPT 입니다.
        """
        )
    else:
        response = run_quiz_chain(docs, topic)
        with st.form("questions_form"):
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                elif value is not None:
                    st.error("Wrong!")
            button = st.form_submit_button()


# 오류 알림 처리
elif not openai_api_key:
    st.warning("OpenAI API 키를 입력해주세요.")
elif not difficulty:
    st.info("난이도를 선택해주세요.")
elif not topic:
    st.info("퀴즈 주제를 입력해주세요.")
