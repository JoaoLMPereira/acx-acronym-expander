from io import StringIO
import re

from pdfminer.converter import HTMLConverter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage

from TextExtractors.TextExtractor import TextExtractor
import numpy as np
from TextTools import toUnicode


class Extract_PdfMiner(TextExtractor):

    def get_text(self, file_path):
        if file_path[-4:] == '.txt':
            return open(file_path).read()
        txt_path = file_path
        if file_path[-4:] == '.pdf':
            txt_path = file_path[:-4] + '.txt'
        elif file_path[-4:] != '.txt':
            txt_path = file_path + '.txt'

    #    if (os.file_path.isfile(txt_path)):
    #        return open(txt_path).read()
        if file_path[-4:] != '.pdf':
            file_path = file_path + '.pdf'
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(file_path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
            interpreter.process_page(page)
        fp.close()
        device.close()
        result = toUnicode(retstr.getvalue())
        retstr.close()

        return result

    def get_font_filtered_text(self, path):
        txt_path = path
        if path[-4:] == '.txt':
            return open(path).read()
        if path[-4:] == '.pdf':
            txt_path = path[:-4] + '.txt'  # doubt: variable never used!
        elif path[-4:] != '.txt':
            txt_path = path + '.txt'
    #    if (os.path.isfile(txt_path)):
    #        return open(txt_path).read()
        # todo: This needs to be optimized eventually
        htmltext = self.get_html(path)
        return self.html_to_text(htmltext, path, fontfilter=True)

    def get_html(self, path):  # Pulls html from PDF instead of plain text
        if path[-4:] != ".pdf":
            path = path + ".pdf"
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = HTMLConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
            interpreter.process_page(page)
        fp.close()
        device.close()
        result = retstr.getvalue()
        retstr.close()
        return result

    def html_to_text(self, htmltext, path, fontfilter=True):
        def repl_fs(m):  # Munging to create font size tags
            size = m.group(3)
            return "[fontsize__" + size + "]"
        txt = re.sub(r"<(.*)(font-size:)(\d+).*>", repl_fs, htmltext)
        txt = re.sub('<.*>', '', txt)
        if fontfilter:
            fontcounts = {}
            fs = 0
            for w in txt.split():  # Count font sizes
                if w[:11] == "[fontsize__":
                    fs = re.search("\d+", w).group(0)
                else:
                    if fs not in fontcounts:
                        fontcounts[fs] = 0
                    fontcounts[fs] += 1
            main_font = int(max(fontcounts, key=fontcounts.get))
            filtered_text = []
            for w in txt.split():
                if w[:11] == "[fontsize__":
                    fs = int(re.search("\d+", w).group(0))
                # Keep 2 font sizes near main font
                elif np.abs(fs - main_font) < 2:
                    filtered_text.append(w)
            txt = ' '.join(filtered_text)
    #        write_text(path+".txt", txt)
        return txt

    def write_text(self, path, text):
        out_file = open(path, "w")
        out_file.write(text)
        out_file.close()
