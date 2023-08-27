from newspaper import Article
from newspaper.article import ArticleException
import requests
from datetime import date


def get_todays_news_batch():
    
    # Send a GET request to the NewsAPI endpoint with our API key
    response = requests.get('https://newsapi.org/v2/top-headlines?country=us&apiKey=c6df40cacc6a407ab6520c84cc9eae31')

    print(response.json())

    # Convert the response to JSON
    data = response.json()

    # Loop over all the articles
    for i, article in enumerate(data['articles']):
        # Each article is a dictionary. We are interested in the 'url' field.
        article_url = article['url']

        # Use the Newspaper3k library to extract the full text of the article
        news_article = Article(article_url)
        news_article.download()

        try:
            news_article.parse()

            # Get the title and full text of the article
            title = news_article.title
            full_text = news_article.text

            # Create a filename using today's date and the article index
            filename = f"data/articles/news_{date.today()}_{i+1}.txt"

            # Open the file in write mode
            with open(filename, 'w') as f:
                # Write the title and full text to the file
                f.write(f"Title: {title}\n\n{full_text}")
        except ArticleException as e:
            print("failed to download an article:", e)
if __name__=="__main__":
    get_todays_news_batch()
