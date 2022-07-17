from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
#from io import StringIO
#from cStringIO import StringIO
from io import StringIO
from importlib import reload
import sys
reload(sys)

def name_converter(file_path):
    if file_path[-4:] != '.txt':
        txt_path = file_path[:-4] + '.txt'
    return txt_path


class PdfConverter:

   def __init__(self, file_path):
       self.file_path = file_path
# convert pdf file to a string which has space among words
   def convert_pdf_to_txt(self):
       rsrcmgr = PDFResourceManager()
       retstr = StringIO()
       codec = 'utf8'  # 'utf16','utf-8'
       laparams = LAParams()
       device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
       fp = open(self.file_path, 'rb')
       interpreter = PDFPageInterpreter(rsrcmgr, device)
       password = ""
       maxpages = 0
       caching = True
       pagenos = set()
       for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
           interpreter.process_page(page)
       fp.close()
       device.close()
       str = retstr.getvalue()
       retstr.close()
       return str
# convert pdf file text to string and save as a text_pdf.txt file



   def save_convert_pdf_to_txt(self, file_path):
       txt_name = name_converter(file_path)
       content = self.convert_pdf_to_txt()
       txt_pdf = open(txt_name, 'wb')
       txt_pdf.write(content.encode('utf8', 'replace'))
       txt_pdf.close()
       return txt_name