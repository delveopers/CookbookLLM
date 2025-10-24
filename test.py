from data import Britannica, Wikipedia

scraper = Britannica(filepath="../results/britannica_articles.txt", max_limit=3, metrics=True)
scraper(["Black Hole", "Relativity"])
