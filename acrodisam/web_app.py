"""
Code largely taken from:
http://runnable.com/UiPcaBXaxGNYAAAL/how-to-upload-a-file-to-the-server-in-flask-for-python
"""
import logging
import mimetypes
import os
import sys
from uuid import uuid4

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask.helpers import send_from_directory
from flask_cors import CORS
import magic
import requests
import textract

from AcroExpExtractors.AcroExpExtractor_FR import AcroExpExtractor_FR
from DatasetParsers import french_wikipedia, FullWikipedia
from acronym_expander import AcronymExpanderFactory, TrainOutDataManager
from inputters import InputArticle, TrainInDataManager
from run_config import RunConfig
from string_constants import (
    FRENCH_WIKIPEDIA_DATASET,
    SDU_AAAI_AI_DATASET,
    file_homepage,
    file_errorpage,
    folder_upload,
    folder_output,
    FULL_WIKIPEDIA_DATASET,
)


LOGGER = logging.getLogger(__name__)
LOGGER.info("Starting server")
APP = Flask(__name__)
cors = CORS(APP)


# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
def select_supported_extension(extension_list):
    """Get the first supported extension.
    Args :
        extension_list (List): List of extension
    Returns :
        (str) : the extension supported from the input list if there is one.
    """
    for extension in extension_list:
        if extension in EXTENSIONS_SUPPORTED:
            return extension
    return None


@APP.route("/")
def index():
    """Display the index page.
    Returns :
        (html) the index html page
    """
    return render_template(file_homepage)


@APP.errorhandler(500)
def internal_server_error(err):
    """Display the server error page.
    Args :
        err (error) : The caught error
    Returns :
        (template): the html page template
        (int) : the error code
    """
    LOGGER.error(err)
    return render_template(file_errorpage), 500


# Route that will process the file upload
@APP.route("/upload", methods=["POST"])
def upload():
    """Get the file.
    Returns :
        the html redirection page
    """
    uploaded_file = request.files["file"]
    if EXPANDER.supportsFile(uploaded_file.filename):
        # Make the filename safe, remove unsupported chars
        # safe_filename = secure_filename(uploaded_file.filename)

        # save filename as guid, helps parallel sessions and accessing info for
        # error analysis
        extension = uploaded_file.filename.rsplit(".", 1)[1]
        safe_filename = str(uuid4()) + "." + extension
        server_file_path = os.path.join(folder_upload, safe_filename)

        # Move the uploaded_file form the temp folder to the upload folder
        uploaded_file.save(server_file_path)

        text = EXPANDER.extractText(safe_filename)
        expanded_acronyms = EXPANDER.processText(text)

        output_file_path = os.path.join(
            folder_output, EXPANDER.getOutputFilename(safe_filename)
        )
        EXPANDER.writeOutputToFile(expanded_acronyms, output_file_path)

        return redirect(
            url_for("output_file", filename=EXPANDER.getOutputFilename(safe_filename))
        )


# This route is expecting a parameter containing the name
# of a uploaded_file. Then it will locate that uploaded_file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@APP.route("/uploads/<filename>")
def uploaded_file(filename):
    """Get the uploaded file.
    Args :
        filename (str): path to the uploaded file
    Returns :
        (file)
    """
    return send_from_directory(folder_upload, filename)


@APP.route("/output/<filename>")
def output_file(filename):
    """Get the output file.
    Args :
        filename (str) : path to the output file
    Returns :
        (file): the output file
    """
    LOGGER.info(folder_output + filename)
    return send_from_directory(folder_output, filename)


@APP.route("/result", methods=["POST", "GET"])
def result():
    """Display the extracted acronyms with their expansion."""
    try:
        if request.method == "POST":
            uploaded_file = request.files["file"]
            uploaded_url = request.form["url"]
            # If uploaded a file
            if uploaded_file.filename != "":
                original_document_name = uploaded_file.filename
                tmp_safe_filename = str(uuid4())
                tmp_server_file_path = os.path.join(folder_upload, tmp_safe_filename)
                # Move the uploaded_file form the temp folder to the upload folder
                uploaded_file.save(tmp_server_file_path)

            # if provided a url
            elif uploaded_url != "":
                original_document_name = uploaded_url
                # local_filename, headers = urllib.request.urlretrieve(uploaded_url)
                response = requests.get(uploaded_url)
                # content_type = response.headers['content-type']
                # extension = mimetypes.guess_extension(content_type)
                # if not extension:
                #    extension = '.html'
                # safe_filename = str(uuid4()) + extension
                # server_file_path = os.path.join(
                #    string_constants.folder_upload, safe_filename)
                # """
                tmp_safe_filename = str(uuid4())
                tmp_server_file_path = os.path.join(folder_upload, tmp_safe_filename)
                with open(tmp_server_file_path, "wb") as new_file:
                    new_file.write(response.content)
            # if neither, send back to main page
            else:
                return render_template("index.html")

            # save filename as guid, helps parallel sessions and accessing info for
            # error analysis
            # extension = uploaded_file.filename.rsplit(".", 1)[1]
            mime_file_type = magic.from_file(tmp_server_file_path, mime=True)
            extensions = mimetypes.guess_all_extensions(mime_file_type)

            extension = select_supported_extension(extensions)

            if not extension:
                # extension = '.txt'
                return render_template(file_errorpage), 500

            safe_filename = tmp_safe_filename + extension
            server_file_path = os.path.join(folder_upload, safe_filename)
            os.rename(tmp_server_file_path, server_file_path)
            content = textract.process(server_file_path)
            content = [content.decode("utf-8")]
            article = InputArticle(raw_text=content[0], preprocesser=TEXT_PRE_PROCESS)
            if extension == ".htm":
                with open(server_file_path, "rb") as text_with_links:
                    expanded_acronyms = EXPANDER.process_article(
                        article, text_with_links=text_with_links, base_url=uploaded_url
                    )
            else:
                expanded_acronyms = EXPANDER.process_article(article)
            # Removes the acronyms with no expansion from the dict
            # Better look for a demo
            exp_acronyms = {
                acro: exp[0]
                for acro, exp in expanded_acronyms.items()
                if exp is not None
            }
            return render_template(
                "result.html", content=exp_acronyms, documentName=original_document_name
            )
    except textract.exceptions.ShellError as textractexcep:
        LOGGER.warning(textractexcep)

    if request.method == "GET":
        return render_template("index.html")


@APP.route("/article_url/<url>")
def get(url):
    """From a given url process the web page to find the pairs (acronym, expansion)
    Args :
        url (str) : the url of the web page
    """
    new_url = url.replace("@@", "/")
    response = requests.get(new_url)
    tmp_safe_filename = str(uuid4())
    tmp_server_file_path = os.path.join(folder_upload, tmp_safe_filename)
    with open(tmp_server_file_path, "wb") as new_file:
        new_file.write(response.content)
    safe_filename = tmp_safe_filename + ".htm"
    server_file_path = os.path.join(folder_upload, safe_filename)
    os.rename(tmp_server_file_path, server_file_path)
    content = textract.process(server_file_path)
    content = [content.decode("utf-8")]
    with open(server_file_path, "rb") as text_with_links:
        results = EXPANDER.process_article(
            content[0], text_with_links=text_with_links, base_url=new_url
        )
    results = {acro: exp for acro, exp in results.items() if exp is not None}
    jsonified_results = jsonify(results)
    return jsonified_results


def main():
    """
    Run the web application accessible to : http://0.0.0.0:5000.
    """
    APP.run(debug=False, host="0.0.0.0", port=5000)
    LOGGER.info("Server is ready")


if __name__ == "__main__":
    LOGGER.info("Initializing Acronym Expander")
    if len(sys.argv) > 1 and sys.argv[1] == "FR":
        OUT_EXPANDER_DATASET_NAME = FRENCH_WIKIPEDIA_DATASET
        in_expander_name = AcroExpExtractor_FR()
        in_expander_args = None
        TEXT_PRE_PROCESS = french_wikipedia.preprocess_text
    else:
        OUT_EXPANDER_DATASET_NAME = FULL_WIKIPEDIA_DATASET
        IN_EXPANDER_DATASET_NAME = SDU_AAAI_AI_DATASET
        train_data_manager_in_expander = TrainInDataManager(
            IN_EXPANDER_DATASET_NAME, storage_type="pickle"
        )
        in_expander_name = "scibert_sklearn"
        in_expander_args = {"epochs": 3, "cuda": True}
        TEXT_PRE_PROCESS = FullWikipedia.text_preprocessing

    train_data_manager_out_expander = TrainOutDataManager(OUT_EXPANDER_DATASET_NAME)

    text_representator_name = "doc2vec"
    text_representator_args = ["50", "CBOW", "25", "8"]

    out_expander_name = "svm"
    out_expander_args = [
        "l2",
        "0.1",
        False,
        text_representator_name,
        text_representator_args,
    ]

    EXTENSIONS_SUPPORTED = textract.parsers._get_available_extensions()

    if len(sys.argv) > 1 and sys.argv[1] == "FR":
        """
        TODO
        EXPANDER = AcronymExpander_Extension(text_extractor=Extract_PdfMiner(),
                                          textPreProcess=TEXT_PRE_PROCESS,
                                          acroExpExtractor=ACRO_EXP_EXTRACTOR,
                                          expanders=[DISAM_EXPANDER],
                                          articleDB=ARTICLE_DB,
                                          acronymDB=ACRONYM_DB)
        """
        pass
    else:
        run_config = RunConfig(
            name=OUT_EXPANDER_DATASET_NAME,
            save_and_load=True,
            persistent_articles="SQLITE",
        )
        expander_fac = AcronymExpanderFactory(
            text_preprocessor=TEXT_PRE_PROCESS,
            in_expander_name=in_expander_name,
            in_expander_args=in_expander_args,
            out_expander_name=out_expander_name,
            out_expander_args=out_expander_args,
            follow_links=True,
            run_config=run_config,
        )
        EXPANDER, _ = expander_fac.create_expander(
            train_data_manager_out_expander, train_data_manager_in_expander
        )
    main()
