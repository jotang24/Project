import requests
from bs4 import BeautifulSoup
import collections
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import concurrent.futures

# Download NLTK stopwords
nltk.download('stopwords')

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

class WordsParser:
    search_tags = ['p', 'div', 'span', 'a', 'h1', 'h2', 'h3', 'h4']

    def __init__(self):
        self.common_words = collections.Counter()
        self.summary_sentences = []

    def handle_data(self, data):
        if self.current_tag in self.search_tags:
            for word in data.strip().split():
                common_word = word.lower().translate(str.maketrans('', '', '.,:"'))

                if (
                        len(common_word) > 2 and
                        common_word not in stopwords.words('english') and
                        common_word[0].isalpha()
                ):
                    self.common_words[common_word] += 1

            # Collecting potential summary sentences from paragraphs and headers
            if self.current_tag in ['p', 'h1', 'h2', 'h3']:
                self.summary_sentences.append(data.strip())

def is_similar(word1, word2):
    return stemmer.stem(word1.lower()) == stemmer.stem(word2.lower())

def get_keywords_and_summary_from_url(url, query_words):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text

        words_parser = WordsParser()
        soup = BeautifulSoup(html, 'html.parser')

        # Limit the number of tags processed to avoid large data
        tags = soup.find_all(words_parser.search_tags, limit=100)

        for tag in tags:
            words_parser.current_tag = tag.name
            words_parser.handle_data(tag.get_text())

        # Exclude query words and similar words using stemming and case normalization
        for query_word in query_words:
            query_word_stem = stemmer.stem(query_word.lower())
            words_parser.common_words = {word: count for word, count in words_parser.common_words.items() if not is_similar(word, query_word)}

        return words_parser.common_words, words_parser.summary_sentences
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return None, None

def get_google_results_count(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        result_stats = soup.find(id='result-stats')

        if result_stats:
            count_text = result_stats.text.split()[1].replace(',', '')
            return int(count_text), soup.select('.tF2Cxc a')
        else:
            return None, None
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    query = "James Dyson Journey "  # INPUT

    results_count, result_urls = get_google_results_count(query)

    if results_count is not None:
        print(f"Number of results for '{query}': {results_count}")

        # Collect keyword data and summaries from multiple sources
        keyword_data_list = []
        all_summary_sentences = []

        # Use concurrent requests to fetch multiple URLs in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(get_keywords_and_summary_from_url, url['href'], query.split()): url for url in result_urls[:5]}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    keywords, summaries = future.result()
                    if keywords is not None:
                        keyword_data_list.append(keywords)
                        all_summary_sentences.extend(summaries)
                except Exception as e:
                    print(f"Error retrieving data from {url['href']}: {e}")

        # Aggregate keyword data
        keyword_data_aggregated = collections.Counter()
        for keyword_data in keyword_data_list:
            keyword_data_aggregated.update(keyword_data)

        # Find top 10 most common words
        top_words = keyword_data_aggregated.most_common(10)

        # Output top 10 most frequent words
        print("\nTop 10 Most Frequent Words:")
        for word, _ in top_words:
            print(f"{word}")

        # Combine summary sentences to form a coherent summary
        summary = ' '.join(all_summary_sentences[:10])  # Limit the number of sentences to 5 for the summary

        # Display summary of the input term
        print("\nSummary:")
        if summary:
            print(summary)
        else:
            print(f"Failed to generate summary for '{query}'.")

        # Display popularity/number of searches
        print(f"\nPopularity/Number of searches: {results_count}")

    else:
        print("Failed to retrieve results count.")
