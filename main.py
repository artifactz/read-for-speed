"""
Google Cloud Run script.
"""

import io
from flask import Flask, request, Response
import msgpack
import google.cloud.logging
import read_for_speed


app = Flask(__name__)

@app.route("/overlay", methods=["POST"])
def overlay():
    """Receives a PDF, overlays it, and returns the modified PDF."""
    # Initialize Google Cloud Logging
    client = google.cloud.logging.Client()
    client.setup_logging()

    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    # return _overlay_bytesio(request.files["file"])
    return _overlay_files(request.files["file"])


def _overlay_bytesio(uploaded_file) -> Response:
    input_pdf_file = io.BytesIO()
    uploaded_file.save(input_pdf_file)
    input_pdf_file.seek(0)

    output_pdf_file = io.BytesIO()
    metadata = read_for_speed.run(input_pdf_file, output_pdf_file)

    output_pdf_file.seek(0)
    output_pdf_bytes = output_pdf_file.read()

    response_data = {
        "metadata": metadata,
        "pdf": output_pdf_bytes,
    }

    return Response(msgpack.packb(response_data), content_type="application/x-msgpack")


def _overlay_files(uploaded_file_storage) -> Response:
    import os, tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as input_pdf_file:
        uploaded_file_storage.save(input_pdf_file.name)

    output_pdf_path = tempfile.mktemp(suffix=".pdf")
    metadata = read_for_speed.run(input_pdf_file.name, output_pdf_path)
    os.remove(input_pdf_file.name)

    with open(output_pdf_path, "rb") as output_pdf_file:
        output_pdf_bytes = output_pdf_file.read()
    os.remove(output_pdf_path)

    response_data = {
        "metadata": metadata,
        "pdf": output_pdf_bytes,
    }

    return Response(msgpack.packb(response_data), content_type="application/x-msgpack")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
