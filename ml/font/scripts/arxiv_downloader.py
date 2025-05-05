"""
Downloads random PDFs from arxiv.
"""


import json, requests, random, re, datetime
from pathlib import Path


DOWNLOAD_FOLDER = Path("samples/arxiv")
INDEX_PATH = Path("samples/arxiv/index.json")


def run():
    index = read_index()
    DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    while True:
        while (date := get_random_date_str()) in index["dates"]:
            pass

        print("DATE", date)
        query = f"submittedDate:[{date}0000+TO+{date}2400]"
        url = "https://export.arxiv.org/api/query?search_query=" + query

        response = requests.get(url)
        search_result = response.text

        for match in re.finditer(r'href="http://arxiv.org/pdf/(.+?)"', search_result):
            document_id = match[1]
            sanitized_document_id = document_id.replace("/", "_")  # e.g. astro-ph/0301157v1
            local_path = DOWNLOAD_FOLDER / f"{sanitized_document_id}.pdf"

            if local_path.exists():
                continue

            url = f"http://arxiv.org/pdf/{document_id}"
            print("Downloading document", document_id)
            response = requests.get(url)

            if response.status_code == 403:
                print("We're banned.")
                return

            pdf_bytes = response.content
            with open(local_path, "wb") as f:
                f.write(pdf_bytes)

            index["documents"].add(document_id)

        index["dates"].add(date)
        save_index(index)
        print()


def read_index():
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        index = {k: set(v) for k, v in data.items()}
        return index
    except OSError:
        groups = ["documents", "dates"]
        return {g: set() for g in groups}


def save_index(index: dict):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        data = {k: list(v) for k, v in index.items()}
        json.dump(data, f, indent=2)


def get_random_date() -> datetime.date:
    start = datetime.date(2000, 1, 1)
    end = datetime.datetime.now().date()
    delta = end - start
    delta_seconds = (delta.days * 24 * 60 * 60) + delta.seconds
    random_seconds = random.random() * delta_seconds
    return start + datetime.timedelta(seconds=random_seconds)


def get_random_date_str():
    date = get_random_date()
    return date.strftime("%Y%m%d")


if __name__ == "__main__":
    run()
