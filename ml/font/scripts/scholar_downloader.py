"""
Downloads random PDFs from the internet via Semantic Scholar search.
"""


import requests, re, uuid, json, time
from pathlib import Path


DOWNLOAD_FOLDER = Path("samples/scholar")
INDEX_PATH = Path("samples/scholar/index.json")


def run(seconds_between_scholar_requests=2.0):
    """
    :param seconds_between_scholar_requests: Minimum time between requests to Semantic Scholar API.
    """
    index = read_index()
    DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    semantic_scholar_time = None

    while True:
        topic = get_random_topic()
        print("Topic:", topic)

        if semantic_scholar_time and time.time() - semantic_scholar_time < seconds_between_scholar_requests:
            time.sleep(seconds_between_scholar_requests - (time.time() - semantic_scholar_time))
        semantic_scholar_time = time.time()

        for url in iter_semantic_scholar_pdfs(topic):
            if url in index:
                print("Skipped", url)
                continue
            filename = download_pdf(url)
            index[url] = filename
            save_index(index)


def read_index():
    """Returns a dictionary mapping URLs to filenames."""
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return {}


def save_index(index: dict):
    """Writes a dictionary mapping URLs to filenames."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def get_random_topic():
    """Returns a random topic using Wikipedia's random article."""
    url = "https://en.wikipedia.org/wiki/Special:Random"
    response = requests.get(url)
    html = response.text

    match = re.search(r"<title>(.+?) - Wikipedia</title>", html)
    return match[1]


def iter_semantic_scholar_pdfs(query: str):
    """
    Iterate over PDF links from Semantic Scholar API search results.
    NOTE: Request frequency limit seems to vary. Sometimes exits with "Endpoint timeout" error. Just wait and restart.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&openAccessPdf=&fields=openAccessPdf&offset=0&limit=10"
    while (response := requests.get(url)).status_code == 429:
        print("Rate limit exceeded, waiting 10 seconds...")
        time.sleep(10)

    if response.status_code != 200:
        exit(f"Bad Semantic Scholar status: {response.status_code} => {response.json()}")
    data = response.json()

    if "data" not in data:
        return

    for item in data["data"]:
        yield item["openAccessPdf"]["url"]


def download_pdf(url):
    try:
        response = requests.get(url, timeout=20)
    except OSError:
        return
    if response.status_code != 200:
        return
    if "application/pdf" not in response.headers["Content-Type"]:
        return
    filename = f"{uuid.uuid4()}.pdf"
    pdf_bytes = response.content
    with open(DOWNLOAD_FOLDER / filename, "wb") as f:
        f.write(pdf_bytes)
    print("Downloaded", url)
    return filename


if __name__ == "__main__":
    run()
