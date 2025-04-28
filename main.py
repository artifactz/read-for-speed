"""
Google Cloud Run script.
"""

import io
from flask import Flask, request, Response
import msgpack
import google.cloud.logging
from pdf_overlay import add_text_overlay_file


app = Flask(__name__)

@app.route("/overlay", methods=["POST"])
def overlay():
    """Receives a PDF, overlays it, and returns the modified PDF."""
    # Initialize Google Cloud Logging
    client = google.cloud.logging.Client()
    client.setup_logging()

    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    uploaded_file = request.files["file"]
    input_pdf_file = io.BytesIO()
    uploaded_file.save(input_pdf_file)
    input_pdf_file.seek(0)

    output_pdf_file = io.BytesIO()
    metadata = add_text_overlay_file(input_pdf_file, output_pdf_file)

    output_pdf_file.seek(0)
    response_data = {
        "metadata": metadata,
        "pdf": output_pdf_file.read(),
    }

    return Response(msgpack.packb(response_data), content_type="application/x-msgpack")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
