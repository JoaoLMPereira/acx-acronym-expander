"""
Created on Jul 21, 2020

@author: jpereira
"""

import os
from uuid import uuid4
import urllib.parse

import textract
import magic
import requests
import mimetypes
import string_constants

import shelve

from bs4 import BeautifulSoup
from Logger import logging

from numpy import absolute

logger = logging.getLogger(__name__)


if os.name == 'nt':
    EXTENSIONS_SUPPORTED = ["pdf"]
else:
    EXTENSIONS_SUPPORTED = textract.parsers._get_available_extensions()
# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation


def selectSupportedExtension(extensionList):
    for extension in extensionList:
        if extension in EXTENSIONS_SUPPORTED:
            return extension
    return None


# tags abbr href


class LinksFollower:
    """
    classdocs
    """

    def __init__(self, in_expander, cache_pages = False):
        """
        Constructor
        """
        self.in_expander = in_expander
        if cache_pages:
            self.cache = shelve.open(string_constants.folder_upload+ "pages_to_file_name")
        else:
            self.cache = None
            
    def _filter_href(self, link):
        if link.startswith("#"):
            return False

        return True

    def _parse_html_file(self, text):
        links = {}
        abbrTagExpansions = {}

        soup = BeautifulSoup(text, "lxml")
        href_tags = soup.find_all("a", href=True)

        links = {
            tag.text.casefold(): (tag.attrs.get("title", None), tag.attrs["href"])
            for tag in href_tags
            if self._filter_href(tag.attrs["href"])
        }

        abbr_tags = soup.find_all("abbr")

        abbrTagExpansions = {
            abbr.text.casefold(): abbr.attrs["title"]
            for abbr in abbr_tags
            if abbr.attrs is not None and abbr.attrs is not None
        }

        return links, abbrTagExpansions

    def _follow_link(self, link):
        if self.cache is None or not link in self.cache:
            # TODO refactor this code and web_apps
            response = requests.get(link)
    
            tmp_safe_filename = str(uuid4())
    
            tmp_server_file_path = os.path.join(
                string_constants.folder_upload, tmp_safe_filename
            )
    
            with open(tmp_server_file_path, "wb") as newFile:
                newFile.write(response.content)
    
            mimeFileType = magic.from_file(tmp_server_file_path, mime=True)
            extensions = mimetypes.guess_all_extensions(mimeFileType)
    
            extension = selectSupportedExtension(extensions)
    
            if not extension:
                return {}
    
            safe_filename = tmp_safe_filename + extension
            server_file_path = os.path.join(string_constants.folder_upload, safe_filename)
    
            os.rename(tmp_server_file_path, server_file_path)

            if self.cache is not None:
                self.cache[link] = server_file_path
        else:
            server_file_path = self.cache[link]
            
        # This is backwards compatible with older cache
        content = textract.process(string_constants.folder_upload + server_file_path.split("/")[-1])
        content = str(content.decode("utf-8"))
        return self.in_expander.get_acronym_expansion_pairs(content)

    def process(self, text, acronym_expansions, base_url):
        links_followed = []

        acronyms_to_expand = {
            acro for acro, exp in acronym_expansions.items() if exp is None
        }

        if len(acronyms_to_expand) < 1:
            return acronym_expansions, links_followed

        links, abbr_tag_expansions = self._parse_html_file(text)

        for acro in acronyms_to_expand:
            followed_link = False
            abbr_exp = abbr_tag_expansions.get(acro.casefold(), None)
            if abbr_exp:
                followed_link = True
                links_followed.append(acro)
                
                expansion = self.in_expander.get_best_expansion(acro, abbr_exp)
                if expansion:
                    acronym_expansions[acro] = expansion
                    continue

            link = links.get(acro.casefold(), None)
            if link:
                if not followed_link:
                    links_followed.append(acro)
                link_title = link[0]
                if link_title:
                    expansion = self.in_expander.get_best_expansion(acro, link_title)
                    if expansion:
                        acronym_expansions[acro] = expansion
                        continue

                relative_url = link[1]
                if base_url:
                    absolute_url = urllib.parse.urljoin(base_url, relative_url)
                else:
                    absolute_url = relative_url

                try:
                    link_exp_dict = self._follow_link(absolute_url)
                    link_exp = link_exp_dict.get(acro)
                    if link_exp:
                        acronym_expansions[acro] = link_exp
                        continue
                except Exception:
                    logger.exception("Failed to follow this link: %s", absolute_url)
                    continue

        return acronym_expansions, links_followed
    
