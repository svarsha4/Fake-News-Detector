
<!-- README.md is generated from README.Rmd. Please edit that file -->

# Fake News Detector

## By: Saul Varshavsky

<!-- badges: start -->
<!-- badges: end -->


### Project Setup

- **Installations**:
  - pip install nltk
  - pip install textblob
  - pip install newspaper3k
  - pip install transformers
  - pip install requests_html
  - pip install rake_nltk
  - pip install crewai
  - pip install 'crewai[tools]' (install various tools that you can use for your agents without having to manually create them)
  - pip install python-dotenv
  - pip install scikit-learn
  - pip install lxml_html_clean

- **Setting up the OpenAI API**:
  - curl https://api.openai.com/v1/engines -H "Authorization: Bearer OPEN_AI_KEY"

- **Setting up an Ollama model**:
  - ollama pull MODEL_NAME

- **Activating Virtual Environment**:
  - source /Users/sxv1284/Documents/Fake\ News\ Detector/myenv/bin/activate

- **Running the Code**:
  - cd /Users/sxv1284/Documents/Fake\ News\ Detector/myenv
  - cd src
  - cd crew
  - python /Users/sxv1284/Documents/Fake\ News\ Detector/myenv/src/crew/main.py
  - If you get an SSL Certificate issue, run the following:
    - pip install --upgrade certifi
    - /Applications/Python\ 3.11/Install\ Certificates.command

- **Sources**:
  - https://youtu.be/z4DQYprjPSs?si=rivAzwxiOPNdBO9K (text summarization and sentiment analysis)
  - https://youtu.be/b9-0GpCqAQw?si=3ZjErYgy2J04-BCL (emotion classification)
  - https://youtu.be/O43XPHC_DEs?si=VWeV-qrCOb5Q-PFx (keyword extraction)
  - https://youtu.be/ij1btJBsfkY?si=d5_8TBVFUzTNDCqd (creating search engine to research information)
  - https://youtu.be/JctmnczWg0U?si=IVm5wlJ3CHX9shDE (scraping and summarizing the research information)
  - https://huggingface.co/q3fer/distilbert-base-fallacy-classification (Hugging Face model for identifying logical fallacies)
  - https://textblob.readthedocs.io/en/dev/ (TextBlob documentation)
  - https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524 (TextBlob documentation)
