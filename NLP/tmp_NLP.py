from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

sentiment_list = analyser.polarity_scores(str(doc))

tmp = 1