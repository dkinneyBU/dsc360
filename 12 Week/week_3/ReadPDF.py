#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:18:45 2023

@author: david
"""
# %% Imports
import urllib.request
import PyPDF2
import io

# %% Retrieve and print a PDF with no table
# cite: https://wellsr.com/python/read-pdf-files-with-python-using-pypdf2/
url = 'https://github.com/bellevue-university/dsc360/raw/main/12%20Week/week_3/Week_3_No_Tables.pdf'
req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
remote_file = urllib.request.urlopen(req).read()
remote_file_bytes = io.BytesIO(remote_file)
pdfdoc_remote = PyPDF2.PdfFileReader(remote_file_bytes)

for i in range(pdfdoc_remote.numPages):
    current_page = pdfdoc_remote.getPage(i)
    print("===================")
    print("Content on page:" + str(i + 1))
    print("===================")
    print(current_page.extractText())
    
# %% Read pdf containing a table and print it.
import camelot
from tabula import read_pdf
from tabulate import tabulate

# %% cite: https://www.geeksforgeeks.org/how-to-extract-pdf-tables-in-python/
df = read_pdf("/home/david/Documents/GitHub/dsc360/12 Week/week_3/Week_3_With_Tables.pdf", pages="all")
# print(tabulate(df))

# %% cite: https://www.geeksforgeeks.org/how-to-extract-pdf-tables-in-python/
fp = "/home/david/Documents/GitHub/dsc360/12 Week/week_3/Week_3_With_Tables.pdf"
pdf = camelot.read_pdf(fp)
print(pdf[0].df)
