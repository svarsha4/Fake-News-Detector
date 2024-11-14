#Imports
import sys
import os
from dotenv import load_dotenv

# Imports the crew from the crew.py file
from crew import FakeNewsDetectionCrew

# Ensures the parent directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#Loads environment variables from the .env file
load_dotenv()

#Executing the crew
def run():
    inputs = {
        'article_text': "Your news article text goes here",
    }
    FakeNewsDetectionCrew.kickoff(inputs=inputs)

if __name__ == "__main__":
    run()