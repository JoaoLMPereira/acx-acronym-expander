"""
Takes in a file and extracts the text from within. Should be able to take txt
files, html, and pdf's
"""
import sys
import os
import re
from urllib.request import urlopen
sys.path.append("/Users/leahbracken/Documents/GitHub/Acrodisam/acrodisam_app/acrodisam/pdfminer")
import pdftotext
from io import BytesIO
from io import StringIO
from pdftotext import PdfConverter
import re
import urllib.request
#from AcronymExtractors.AcronymExtractor_v4 import AcronymExtractor_v4
#from AcronymExpanders.Expander_fromText_v3 import Expander_fromText_v
from bs4 import BeautifulSoup
from filter_acronym_parser import new_filter
from filter_acronym_parser import acronyms
from filter_acronym_parser import leah_parser_expansions
from flask import Flask, flash, request, redirect, url_for, render_template
from flask.helpers import send_from_directory
from werkzeug.utils import secure_filename
from filter_acronym_parser import clean_dict
from filter_acronym_parser import acronyms
from filter_acronym_parser import leah_parser_expansions
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
data_path = '.'


def temp_file_function(file):
    #temporary function to display acronym and Expansions
    file.save(secure_filename(file.filename))
    file.stream.seek(0)
    content = file.read()
    content = str(content)
    acro_dict = acronym_dict(content)
    new_dict = clean_dict(acro_dict)
    return new_dict


# Following incomplete function used to deal with pdf and html files
def file_content(file_name, file):
    if file_name.endswith('.txt'):
        file.save(secure_filename(file.filename))
        file.stream.seek(0)
        content = file.read().decode('utf-8')
        content = str(content)
        #f = open(file, "r")
        #content = f.read()
        acronym_list = acronyms(content)
        content_dict = new_filter(acronym_list, content)
        return content_dict
    else:
        other_dict = {}
        file.save(secure_filename(file.filename))
        file.stream.seek(0)
        content = pdf_converter(file)
        #file.save(secure_filename(file.filename))
        #file.stream.seek(0)
        #content = file.read()
        #content = str(content)
        acronym_list = acronyms(content)
        for acronym in acronym_list:
            other_dict[acronym] = content
        return other_dict

def get_text_bs(html):
    # extract contents from html
    tree = BeautifulSoup(urlopen(html), 'lxml')
    body = tree.body
    if body is None:
        return None
    for tag in body.select('script'):
        tag.decompose()
    for tag in body.select('style'):
        tag.decompose()
    text = body.get_text(separator='\n')
    return text


def pdf_converter(file):
    # extracts the content from pdf file
    PDFInstance = PdfConverter(file_path=file)
    text = PDFInstance.convert_pdf_to_txt()
    return text

def acronym_dict(document_name):
    # used to create a dictionary of acronym and their expansions
    acro_dict = {}
    acronyms_list = acronyms(document_name)
    for acronym in acronyms_list:
        expansions = leah_parser_expansions(acronym, document_name)
        acro_dict[acronym] = expansions
    return acro_dict
