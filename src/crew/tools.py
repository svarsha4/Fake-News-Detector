#Setting up the SSL Certificate
import ssl
import warnings

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

#Imports for text summarization
import nltk
from transformers import pipeline

#Imports for sentiment analysis
from textblob import TextBlob
from typing import List, Dict, Any

#Imports for bias detection
from collections import Counter

#Imports for extracting key information from article
from requests_html import HTMLSession
from rake_nltk import Rake

#Imports for researching information
import webbrowser
from bs4 import BeautifulSoup

#Imports for cross-referencing research information with the given news article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#The punkt NLP model will be used for the text summarization and sentiment analysis tools
nltk.download('punkt')

#Creating an Article object
class Article:
    def __init__(self, text: str):
        self.text = text

###TOOLS FOR THE ARTICLE ANALYSIS AGENT ---------------------------------------------------
class ArticleAnalysisTools:
    def __init__(self, article: Article):
        self.article = article
    
    ### TEXT SUMMARIZATION ---------------------------------------------------
    #Function for summarizing the article
    def summarize_article(self) -> None:
        summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6', use_auth_token=False, trust_remote_code=True, verify=False)
        summary = summarizer(self.article, max_length=130, min_length=30, do_sample=False)
        print(f"Summary: {summary}")
    
    ### SENTIMENT ANALYSIS ---------------------------------------------------
    #Function for analyzing the sentiment of the article
    def analyze_sentiment(self) -> None:
        blob = TextBlob(self.article.text)
        for sentence in blob:
            sentiment_result = 'positive' if sentence.sentiment.polarity > 0 else 'negative' if sentence.sentiment.polarity < 0 else 'neutral'
            print(f"Sentiment: {sentiment_result}")
    
    ### EMOTION DETECTION ---------------------------------------------------
    #Function for detecting the emotions conveyed in the article
    def emotion_detection(self) -> None:
        emotion_model = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa', use_auth_token=False, trust_remote_code=True, verify=False)
        emotions = emotion_model(self.article.text)
        if emotions:
            emotion_counts = Counter([emotion for emotion in emotions])
            print("Emotion Analysis:")
            for emotion, count in emotion_counts.items():
                print(f"{emotion}: {count} instances")
        else:
            print("No emotions detected.")
    
    ### BIAS DETECTION ---------------------------------------------------
    #Function for retrieving sentences that show bias within some text
    def get_biased_sentences(self) -> List[str]:
        blob = TextBlob(self.article.text)
        biased_sentences = [sentence for sentence in blob if sentence.sentiment.subjectivity > 0.5]
        return biased_sentences
    
    #Function for identifying common biased keywords/phrases that may appear in a news article
    def identify_bias_keywords_phrases(self) -> List[List[str]]:
        bias_identify = []
        bias_keywords = ["always", "never", "clearly", "obviously", "undoubtedly", "everyone knows", "in fact"]
        bias_identify.append(bias_keywords)
        bias_phrases = [sentence for sentence in nltk.sent_tokenize(self.article.text) if any(keyword in sentence.lower() for keyword in bias_keywords)]
        bias_identify.append(bias_phrases)
        return bias_identify
    
    #Function for detecting the bias present in the article
    def bias_detection(self) -> None:
        biased_sentences = self.get_biased_sentences()
        bias_keywords = self.identify_bias_keywords_phrases()[0]
        bias_phrases = self.identify_bias_keywords_phrases()[1]
        bias_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", use_auth_token=False, trust_remote_code=True, verify=False)
        bias_results = bias_classifier(self.article.text, bias_keywords)
        if bias_results:
            print("\nBias Detection:")
            for result in bias_results:
                print(f"- {result}")
                print("Reasoning:")
            print(f"- Number of subjective sentences: {len(biased_sentences)}")
            print(f"- Persuasive phrases: {len(bias_phrases)}")
    
    ### LOGICAL FALLACIES DETECTION ---------------------------------------------------
    #Function for detecting what kinds of logical fallacies may be present in the article
    def logical_fallacy_detection(self) -> None:
        sentences = nltk.sent_tokenize(self.article.text)
        logical_fallacies_model = pipeline('text-classification', model='q3fer/distilbert-base-fallacy-classification', use_auth_token=False, trust_remote_code=True, verify=False)
        for i in range(0, len(sentences) - 1, 2):
            sentence_pair = sentences[i] + " " + sentences[i + 1]
            logical_fallacies_result = logical_fallacies_model(sentence_pair)
            print(f"Sentence Pair: {sentence_pair}")
            print(f"- Logical Fallacy: {logical_fallacies_result}\n")
    
    
    



### TOOLS FOR THE RESEARCH AGENT ---------------------------------------------------
class ResearchTools:
    def __init__(self, article):
        self.article = article
    
    ### RESEARCHING INFORMATION ---------------------------------------------------
    #Function for extracting keywords from the article
    def extract_keywords(self):
        nltk.download('stopwords')
        r = Rake()
        article_keywords = r.extract_keywords_from_text(self.article.text)
        return article_keywords
    
    #Function that uses the keywords extracted from the article to know what to search through the web
    def get_search_results(self, keywords: List[str]) -> List[str]:
        session = HTMLSession()
        query = "+".join(keywords)
        response = session.get(f"https://www.google.com/search?q={query}")
        response.html.render()
        links = response.html.xpath('//a[@href]', first=False)
        urls = [link.attrs['href'] for link in links if 'url?q=' in link.attrs['href']]
        return urls
    
    #Function to scrape the contents of a given URL
    def scrape_url(self, url):
        session = HTMLSession()
        webbrowser.open(url)
        response = session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all(['h1', 'p'])
        text = [result.text for result in results]
        source = ' '.join(text)
        return source
    
    #Function for conveting sentences into chunks
    def chunk_sentences(self, sentences):
        max_chunk = 500
        current_chunk = 0 
        chunks = []
        for sentence in sentences:
            if len(chunks) == current_chunk + 1: 
                if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                    chunks[current_chunk].extend(sentence.split(' '))
                else:
                    current_chunk += 1
                    chunks.append(sentence.split(' '))
            else:
                print(current_chunk)
                chunks.append(sentence.split(' '))
        for chunk_id in range(len(chunks)):
            chunks[chunk_id] = ' '.join(chunks[chunk_id])
        return chunks
    
    #Function for summarizing the a given research source obtained from the web
    def summarize_research_source(self, source, summarizer):
        source = source.replace('.', '.<eos>')
        source = source.replace('?', '?<eos>')
        source = source.replace('!', '!<eos>')
        sentences = source.split('<eos>')
        chunks = self.chunk_sentences(sentences)
        res = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
        text = ' '.join([summ['summary_text'] for summ in res])
        return text

    #Function for retrieving the research obtained from the web
    def research_information(self):
        summarizer = pipeline("summarization", model='sshleifer/distilbart-cnn-12-6', use_auth_token=False, trust_remote_code=True, verify=False)
        article_keywords = self.extract_keywords()
        research_urls = self.get_search_results(article_keywords)
        research_sources = []
        for url in research_urls:
            research_sources.append(self.scrape_url(url))
        res_summaries = []
        for source in research_sources:
            source_summary = self.summarize_research_source(source, summarizer)
            res_summaries.append(source_summary)
        for summary in res_summaries:
            print(f"Research Summary: {summary}")
        return res_summaries
    
    ### CROSS-REFERENCING RESEARCH INFORMATION ---------------------------------------------------
    #Helper function for cross-referencing the research information with the given news article
    def cross_ref_helper(self, text, research):
        info = [text, research]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(info)
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity_matrix[0][0]
    
    #Function for cross-referencing the research information with the given news article
    def cross_reference(self):
        res_summaries = self.research_information()
        for summary in res_summaries:
            similarity_score = self.cross_ref_helper(self.article.text, summary)
            if similarity_score > 0.5:
                print(f"The research and the news article contain similar information. {summary}")


