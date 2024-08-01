from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# LLM setup
model_name = "gpt-4o-mini"
metadata = "CRAG, gpt-4o"
llm = ChatOpenAI(model_name=model_name, temperature=0)

# RAG chain setup
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 
    
    Use the following documents to answer the question. 
    
    If you don't know the answer, just say that you don't know. 
    
    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Documents: {documents} 
    Answer: 
    """,
    input_variables=["question", "documents"],
)

rag_chain = prompt | llm | StrOutputParser()

# Grading prompt setup
system_prompt = """You are a teacher grading a quiz. You will be given: 
1/ a QUESTION 
2/ a set of comma separated FACTS provided by the student

You are grading RELEVANCE RECALL:
A score of 1 means that ANY of the FACTS are relevant to the QUESTION. 
A score of 0 means that NONE of the FACTS are relevant to the QUESTION. 
1 is the highest (best) score. 0 is the lowest score you can give. 

Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

human_prompt = "FACTS: \n\n {documents} \n\n QUESTION: {question}"

grade_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
