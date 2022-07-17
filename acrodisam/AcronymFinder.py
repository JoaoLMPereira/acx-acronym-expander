'''
Created on Jul 10, 2018

@author: jpereira
'''
import re
from lxml import html
import requests

def getExpansions(acronym):
    page = requests.get('https://www.acronymfinder.com/'+acronym+'.html')
    tree = html.fromstring(page.content)

    expansion_without_link_list = tree.xpath('//td[@class="result-list__body__meaning"]/text()')
    expansion_with_link_list = tree.xpath('//td[@class="result-list__body__meaning"]/a/text()')

    expansion_list = expansion_without_link_list + expansion_with_link_list

    return [re.sub(r' \([^)]*\)$', '', expansion) for expansion in expansion_list]

#print ('Buyers: ', buyers)

#def main():
#    expansion_list = getExpansions("EMT")
#    print ('Expansions: ', '\n'.join(expansion_list))
#
#    expansion_list = getExpansions("InvalidAcronym")
#    print ('Expansions: ', '\n'.join(expansion_list))
#if __name__ == "__main__":
#    main()
