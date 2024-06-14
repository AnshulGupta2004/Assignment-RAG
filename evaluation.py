import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
# from groq import Groq
# from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from dotenv import load_dotenv
import pandas as pd

# load_dotenv()
# GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def process_pdfs():
    pdf_docs = r"policy-booklet-0923.pdf"
    
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    
    
    return vectorstore

def evaluate_question(vectorstore):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    retriever = vectorstore.as_retriever()
    # llm = ChatGroq(temperature=0,
    #                model_name="llama3-70b-8192",
    #                api_key=GROQ_API_KEY)
    llm = Ollama(model="llama3")
    template = """You are a helpful, respectful and honest assistant.
                Always answer as helpfully as possible, while being safe.
                Your answers should not include any harmful, unethical, racist, sexist, toxic,
                dangerous, or illegal content. Please ensure that your responses are socially
                unbiased and positive in nature.
                If a question does not make any sense, or is not factually coherent,
                explain why instead of answering something not correct.
                If you don't know the answer to a question, please don't share false information.
                Question: {question}
                Context: {context}
                Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    questions = [
        "What should you do if you have a car accident and need to claim?",
        "What is covered under the Comprehensive Plus policy for windscreen damage?",
        "What does the Comprehensive policy include for in-car entertainment fitted after the car was made?",
        "How much is covered for personal belongings under the Comprehensive Plus policy?",
        "What are the contact numbers for claiming and windscreen claims?",
        "What does the DriveSure policy include?",
        "What is the cover amount for theft of car keys under the Comprehensive policy?",
        "What should you do if your car is written off?",
        "Are mechanical or electrical failures covered under this policy?",
        "What happens if you leave your car unlocked or the keys in the car?",
        "Does the policy cover damage to tyres caused by braking or punctures?",
        "What does the policy provide if your car is stolen and not recovered?",
        "What is the market value coverage for fire damage under the Comprehensive policy?",
        "How much does the policy cover for child car seat replacement?",
        "What is the excess amount for windscreen replacement if using an approved supplier?",
        "What is included under Section 1: Liability for personal injuries?",
        "What is the cover limit for property damage under Section 1: Liability?",
        "Does the policy include cover for vandalism?",
        "What is the policy's stance on using non-approved repairers?",
        "Are accessories like charging cables covered under the policy?",
        "What is the duration for which a courtesy car is provided?",
        "Is cover provided for misfuelling?",
        "What does the policy cover under Section 6: Personal benefits for hotel expenses?",
        "What are the travel cost cover limits if a hire car is not available?",
        "Is the car's battery covered under the policy?",
        "What happens if you refuse an offer to settle your claim?",
        "What is the cover limit for medical expenses under the policy?",
        "What should you do if you receive a communication from the court?",
        "What is the difference between commuting and business use?",
        "Are modifications to the car covered under the policy?"
    ]

    ground_truths = [
        ["Call 0345 878 6261 and provide details such as your personal information, policy number, car registration, and a description of the damage."],
        ["Comprehensive Plus covers the replacement or repair of accidentally broken glass in the windscreen, sunroof, or windows."],
        ["Comprehensive policy covers up to £1,000 for in-car entertainment fitted after the car was made."],
        ["Comprehensive Plus covers up to £500 for personal belongings."],
        ["For claims, call 0345 878 6261; for windscreen claims, call 0800 328 9150."],
        ["DriveSure includes telematics insurance that captures driving data to adjust premiums based on driving style."],
        ["Comprehensive policy covers up to £1,000 for theft of car keys."],
        ["If your car is written off, the policy covers the market value and ends once the claim is settled."],
        ["No, mechanical or electrical failures are not covered."],
        ["Claims for theft or attempted theft are not covered if the car is left unlocked or with keys inside."],
        ["No, damage to tyres caused by braking, punctures, cuts, or bursts is not covered."],
        ["The policy provides the market value of your car if it is stolen and not recovered."],
        ["The Comprehensive policy covers the market value for fire damage."],
        ["The policy covers unlimited amounts for child car seat replacement."],
        ["The excess amount for windscreen replacement with an approved supplier is shown in your car insurance details."],
        ["Unlimited cover for personal injuries to other people is included."],
        ["The cover limit for property damage is £20,000,000 per accident."],
        ["Yes, vandalism is covered under the Comprehensive and Comprehensive Plus policies."],
        ["Repairs by non-approved repairers are not guaranteed by the policy."],
        ["Yes, accessories like charging cables are covered under fire, theft, and accidental damage sections."],
        ["A courtesy car is provided for the duration of the repair by an approved repairer."],
        ["Yes, damage caused by misfuelling is covered under the Comprehensive and Comprehensive Plus policies."],
        ["Hotel expenses up to £250 (Comprehensive) or £300 (Comprehensive Plus) are covered if you cannot drive your car after an accident."],
        ["Travel costs are covered up to £50 per day, up to a total of £500 per claim."],
        ["Yes, the car's battery is covered if damaged as a result of an insured incident, whether owned or leased."],
        ["If you refuse an offer or payment that is deemed reasonable, further costs may not be covered."],
        ["Medical expenses are covered up to £200 (Essentials) or £400 (Comprehensive and Comprehensive Plus)."],
        ["Contact the insurance provider immediately to handle the communication or follow their instructions."],
        ["Commuting is driving to and from a permanent place of work, while business use includes driving in connection with business or employment."],
        ["Yes, modifications are covered but must be declared to the insurance provider."]
    ]

    answers = []
    contexts = []

    for query in questions:
        answers.append(rag_chain.invoke(query))
        contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])


    data = {
        "question" : questions,
        "answer" : answers,
        "contexts" : contexts,
        "ground_truths" : ground_truths
    } 
    dataset = Dataset.from_dict(data)
    result = evaluate(
            dataset = dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ],
            llm = llm,
            embeddings = embeddings
        )
    df = result.to_pandas()
    print(df)

def main():
    vectorstore = process_pdfs()
    evaluate_question(vectorstore)

if __name__ == "__main__":
    main()
