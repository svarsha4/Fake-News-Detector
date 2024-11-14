#Imports for completing the tasks
from crewai import Task
from textwrap import dedent

# Defines the tasks the Article Analysis Agent will accomplish to help detect fake news
# These tasks will use the tools from the ArticleAnalysisTools class
class ArticleAnalysisTasks:
    def __init__(self, article, agent, tools):
        self.article = article
        self.agent = agent
        self.tools = tools
    
    ### TEXT SUMMARIZATION TASK ---------------------------------------------------
    #Function for completing the text summarization task
    def text_summarization(self):
        return Task(
            description=dedent(
                f"""
                Summarize the content of the news article using the summarize_article function from the ArticleAnalysisTools class.
                Article text: {self.article.text}
                """
            ),
            expected_output="""
                Provide a two-paragraph summary of the news article.
            """,
            agent=self.agent,
            tools=self.tools,
        )
    
    ### SENTIMENT ANALYSIS & EMOTION DETECTION TASK ---------------------------------------------------
    #Function for completing the emotion detection task
    def emotions_conveyed(self):
        return Task(
            description=dedent(
                f"""
                Determine the kinds of emotions conveyed through the news article 
                using the emotion_detection function from the ArticleAnalysisTools class. 
                Classify these emotions as positive or negative using the analyze_sentiment function
                from the ArticleAnalysisTools class.
                Article text: {self.article.text}
                """
            ),
            expected_output="""
                Create a table listing the emotions conveyed in the news article and their corresponding
                types (Positive, Negative). 
            """,
            agent=self.agent,
            tools=self.tools,
        )
    
    ### BIAS DETECTION TASK ---------------------------------------------------
    #Function for completing the bias detection task
    def bias_detection(self):
        return Task(
            description=dedent(
                f"""
                Identify any biases that may be present in the news article using the bias_detection function
                from the ArticleAnalysisTools class.
                Article text: {self.article.text}
                """
            ),
            expected_output="""
                Provide a 3-sentence to a paragraph summary describing any biases
                that may be present in the news article.
            """,
            agent=self.agent,
            tools=self.tools,
        )
    
    ### PART-TO-WHOLE FALLACY DETECTION TASK ---------------------------------------------------
    #Function for completing the logical fallacy detection task
    def logical_fallacies_detection(self):
        return Task(
            description=dedent(
                f"""
                Identify whether there are any logical fallacies present in the news article
                using the logical_fallacy_detection function from the ArticleAnalysisTools class.
                Article text: {self.article.text}
                """
            ),
            expected_output="""
            Provide a bulleted list of each of the logical fallacies that were found in the news article.
            Each bullet should contain 1-2 sentences briefly describing the instance of the fallacy, followed by the name
            of the fallacy in parentheses.
            """,
            agent=self.agent,
            tools=self.tools,
        )




   
# Defines the tasks the Research Agent will accomplish to help detect fake news
# These tasks will use the tools from the ResearchTools class
class ResearchTasks:
    def __init__(self, article, agent, tools):
        self.article = article
        self.agent = agent
        self.tools = tools
    
    ### RESEARCH INFORMATION TASK ---------------------------------------------------
    #Function for completing the research information task
    def research_information(self):
        return Task(
            description=dedent(
                f"""
                Research the information provided in the news article online using the
                research_information function from the ResearchTools class.
                Article text: {self.article.text}
                """
            ),
            expected_output="""
                Provide a bulleted list of the information that was researched online.
                Each bullet should contain 3-5 sentences summarizing the information, 
                followed by the source and date in parentheses.
            """,
            agent=self.agent,
            tools=self.tools,
        )
        
    ### CROSS-REFERENCING TASK ---------------------------------------------------
    #Function for completing the cross-referencing task
    def cross_reference_information(self):
        return Task(
            description=dedent(
                f"""
                Cross-reference the information provided in the news article with the online research
                pertaining to that information. Use the cross_reference function from the 
                ResearchTools class to accomplish this task.
                Article text: {self.article.text}
                """
            ),
            expected_output="""
                Provide a bulleted list of the information from online that was cross-referenced with
                the news article. Each bullet should contain 3-5 sentences
                summarizing the information, followed by any
                discrepancies noted in parentheses.
            """,
            agent=self.agent,
            tools=self.tools,
        )
        
    