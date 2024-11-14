#Setting up the SSL Certificate
import ssl
import warnings

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

#Stores the link to a given news article
article_url = 'https://www.cnn.com/2024/11/05/us/tropical-storm-rafael-forecast-hurricane-hnk/index.html'

import requests
response = requests.get(article_url, verify=False)

#Imports for making the agents
import os
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import ChatOpenAI
from newspaper import Article

#Imports all the tools the agents will use
from tools import (
    ArticleAnalysisTools,
    ResearchTools,
)

#Loads environment variables from the .env file
load_dotenv()

#Creates an article object using the contents from the link
article = Article(article_url)

#Initializes the tools
article_analysis_tools = ArticleAnalysisTools(article)
research_tools = ResearchTools(article)

#Defines the agents that will work together to help detect fake news
class CustomAgents:
    def __init__(self):
        #Reference the API key for accessing OpenAI from the .env file
        api_key = os.getenv("OPENAI_API_KEY")
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, api_key=api_key)
    
    #Function for creating the Article Analysis Agent
    def article_analysis_agent(self):
        return Agent(
            role="Article Analysis Agent",
            goal="""
            Your goal is to summarize the contents of the news article, determine the emotions conveyed in the news article, 
            identify any biases present in the news article, and identify any logical fallacies
            present in the news article.
            """,
            backstory="""
            You are very detail oriented and attentive when reading articles. You are able to pick out the small details while being able to understand how these details
            fit into the larger picture. Additionally, you have a lot of experience working for various news agencies. As a result, you know how to 
            easily identify the agenda behind a news article. By being able to easily identify the agenda of a news article, you can determine whether the information is biased.
            Additionally, you have a doctorate in philosophy and have a strong bakground in logical reasoning.
            You have spent over 10 years studying and teaching logical fallacies at the university level.
            Thus, you are able to easily and accurately identify various logical fallacies conveyed through
            writing.
            """,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
            tools=[article_analysis_tools.summarize_article(), article_analysis_tools.analyze_sentiment(), article_analysis_tools.emotion_detection(), article_analysis_tools.bias_detection(), 
                   article_analysis_tools.partToWhole_detection(), article_analysis_tools.cherrypicking_detection(), 
                   article_analysis_tools.genetic_detection()]
        )
    
    #Function for creating the Research Agent
    def research_agent(self):
        return Agent(
            role="Research Agent",
            goal="""
            Your goal is to research the information online that is described in the news article, as well as cross-reference that information with
            the information in the news article.
            """,
            backstory="""
            You are a skilled researcher who is able to think critically about the information being researched.
            Hence, you are able to make rational conclusions about the truthfulness of the information being researched.
            """,
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
            tools=[research_tools.research_information(), research_tools.cross_reference()]
        )