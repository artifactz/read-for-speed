import pdfplumber
from pdf_overlay import group_words


if __name__ == "__main__":
    pdf_path = "sample3.pdf"
    html_path = "sample3.html"

    with open(html_path, "w") as html:
        html.write("<html><body>\n")

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                words = group_words(page.chars)
                for word in words:
                    word_str = "".join(char["text"] for char in word)

                    font_size = word[0]["size"]
                    font_name = word[0]["fontname"]

                    html.write(f'<span style="font-size: {font_size}pt; font-family: {font_name}" title="{font_name} {font_size:.0f}pt">{word_str}</span>')

                html.write("<br>\n")

        html.write("</body></html>\n")
