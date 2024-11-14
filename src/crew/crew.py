#Setting up the SSL Certificate
import ssl
import warnings

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')


#Stores the link to a given news article
article_url = 'https://www.cnn.com/2024/11/05/us/tropical-storm-rafael-forecast-hurricane-hnk/index.html'

import requests
response = requests.get(article_url, verify=False)

from newspaper import Article
from crewai import Crew, Process

# Imports all the agents from the CustomAgents class from the agents.py file
from agents import CustomAgents

# Imports all the tools from the tools.py file
from tools import (
    ArticleAnalysisTools,
    ResearchTools
)

# Imports all the tasks from the tasks.py file
from tasks import (
    ArticleAnalysisTasks,
    ResearchTasks
)

#Creates an article object using the contents from the link
article = Article(article_url)

#Initializes the agents
article_analysis_agent = CustomAgents().article_analysis_agent()
research_agent = CustomAgents().research_agent()

#Initializes the tools
article_analysis_tools = ArticleAnalysisTools(article)
research_tools = ResearchTools(article)

#Initializes the tasks
article_analysis_tasks = ArticleAnalysisTasks(article, article_analysis_agent, article_analysis_tools)
research_tasks = ResearchTasks(article, research_agent, research_tools)

#Creates a crew to manage the agents and tasks
FakeNewsDetectionCrew = Crew(
    agents=[
        article_analysis_agent,
        research_agent
    ],
    tasks=[
        article_analysis_tasks,
        research_tasks
    ],
    process=Process.sequential,
    verbose=True,
)