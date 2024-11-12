import requests
from bs4 import BeautifulSoup
import os
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader, ListIndex, Document
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()
##openai_api_key = os.getenv("OPENAI_API_KEY")
##os.environ["OPENAI_API_KEY"] = openai_api_key

class JobInsights:
    def __init__(self, job_title):
        self.job_title = job_title
        self.openai_api_key = userdata.get('openai_api_key')
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        self.top3_jobs = self.scrape_job()
        self.summary_3jobs = self.job_description()
        self.prompt = "Describe in 5 paragraphs: the job role/key responsibilities, the qualifications needed for the job and the key skills required for the job (both technical and soft skills), the domain and the seniority level. "
        self.result = self.semantic_search()


    def scrape_job(self):
        url = f"https://www.jobly.fi/tyopaikat?search={self.job_title}"
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        job_listings = soup.find_all("a", class_="recruiter-job-link")
        job_links = [listing['href'] for listing in job_listings]
        return job_links[::2][:3]

    def job_description(self):
        dic = {}
        for i, link in enumerate(self.top3_jobs):
            response = requests.get(link)
            soup = BeautifulSoup(response.content, "html.parser")
            job_description_element = soup.find("div", class_="l-main")
            job_description = job_description_element.get_text(strip=True)
            text = "Hae paikkaaTallenna työpaikka"
            first_occurrence = job_description.find(text)
            second_occurrence = job_description.find(text, first_occurrence + 1)
            extracted_text = job_description[first_occurrence+len(text):second_occurrence]
            dic[i] = extracted_text
        return dic

    def semantic_search(self):
        insights = {}
        for item, summary in self.summary_3jobs.items():
            documents = [Document(text=summary)]
            index = GPTVectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()
            response = query_engine.query(self.prompt)
            insights[item] = response.response
        return insights

    def get_insights(self):
      company_insights = {}
      llm_qa = pipeline("question-answering")
      llm_summarize  = pipeline("summarization", model="facebook/bart-large-cnn")
      question1 = "What is the name of the company/team?"
      question2 = "What is the domain of the company?"
      question3 = "What is the seniority level?"
      for index in self.result:
          q_context1 = self.result[index].split('\n\n')[0]
          q_context2 = self.result[index].split('\n\n')[3]
          q_context3 = self.result[index].split('\n\n')[4]
          company_name = llm_qa(question=question1, context=q_context1)
          domain = llm_qa(question=question1, context=q_context2)
          seniority = llm_qa(question=question1, context=q_context3)
          responsibility = llm_summarize(self.result[index].split('\n\n')[0], max_length=115, clean_up_tokenization_spaces=True)[0]['summary_text']
          qualification = llm_summarize(self.result[index].split('\n\n')[1], max_length=115, clean_up_tokenization_spaces=True)[0]['summary_text']
          skills = llm_summarize(self.result[index].split('\n\n')[2], max_length=115, clean_up_tokenization_spaces=True)[0]['summary_text']
          company_insights[index] = {'company_name': company_name, 'responsibility': responsibility, 'qualification': qualification, 'skills': skills, 'domain':domain,'seniority':seniority}
      return company_insights
