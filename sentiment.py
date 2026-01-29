import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(symbol):
    """
    Fetches recent news for a symbol and returns a consolidated sentiment score (-1 to 1).
    """
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return 0.0 # Neutral if no news
        
        scores = []
        for item in news:
            title = item.get('title', '')
            # analyze title sentiment
            ts = analyzer.polarity_scores(title)
            scores.append(ts['compound'])
            
        if not scores:
            return 0.0
            
        avg_score = sum(scores) / len(scores)
        return round(avg_score, 4)
    except Exception as e:
        print(f"Sentiment Error for {symbol}: {e}")
        return 0.0

if __name__ == "__main__":
    # Test
    test_symbol = "ADANIENT.NS"
    print(f"Sentiment for {test_symbol}: {get_sentiment_score(test_symbol)}")
