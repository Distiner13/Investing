'''
Table containing all the competitors names and their metrics from finviz using panda.read_html\
    
(Functional, for now)
31/01/2024 - we corrected the program using requests, then pandas to avoid 403 error
            this makes a list of tables called comparisons and we take the desired table
            from it at index 9. We hope this does not change, but we're not sure. 
'''

#%%
import re
import requests
import pandas as pd
import collections
from openpyxl.reader.excel import load_workbook
collections.Callable = collections.abc.Callable
from bs4 import BeautifulSoup
from ProfileScraping import getProfile

def getCompetitors(symbol):
    pattern = re.compile(r'[^a-zA-Z]')  # Matches any non-alphabetical character
    info = getProfile(symbol)
    sector = re.sub(pattern, '', info[0].lower().replace(" ",""))
    industry = re.sub(pattern, '', info[1].lower().replace(" ",""))
    k = 0
    firsttime = True
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                             "Version/15.4 Safari/605.1.15"}
    #"https://finviz.com/screener.ashx?v=151&f=ind_utilitiesrenewable,sec_utilities"
    while(requests.get("https://finviz.com/screener.ashx?v=152&f=ind_" + industry + ",sec_" + sector + "&r=" + str(k) + "1&c=0,1,2,79,3,4,5,6,7,8,9,10,11,12,13,73,74,75,14,15,16,77,17,18,19,20,21,23,22,82,78,24,25,85,26,27,28,29,30,31,84,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,68,70,80,83,76,60,61,62,63,64,67,69,81,65,66", headers=headers)):
        url = "https://finviz.com/screener.ashx?v=152&f=ind_" + industry + ",sec_" + sector + "&r=" + str(k) + "1&c=0,1,2,79,3,4,5,6,7,8,9,10,11,12,13,73,74,75,14,15,16,77,17,18,19,20,21,23,22,82,78,24,25,85,26,27,28,29,30,31,84,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,68,70,80,83,76,60,61,62,63,64,67,69,81,65,66"
        print(url)
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        # could use pandas to extract the table instead of beautiful soup
        response = requests.get(url, headers=header)
        comparisons = pd.read_html(response.content) #this is a list of tables from the website
        comparison = comparisons[9] #this is the index of the table we want to extract
        len(comparison)

        #comparison = pd.DataFrame(values, columns=col)
        if (len(comparison) == 1):
            break
        print(comparison)
        #this file path should be taken as input in order to make sure it's useful
        file_path = "C:/Users/User/Documents/- Desktop _Archive/PL-/Learn/Programming/Finance-Project/Financial-project-main/Financial-project-main/Output.xlsx"

        # Load the existing workbook
        book = load_workbook(file_path)

        sheet = book['Sheet1']

        start_row = int(comparison.iloc[-1]['No.'])

        # Convert the DataFrame to a list of lists
        data_values = comparison.values.tolist()

        if (firsttime == True):
            header_titles = comparison.columns.tolist()
            sheet.append(header_titles)
            firsttime = False

        # Insert the data into the sheet starting from the specified row
        for i, row in enumerate(data_values, start=start_row):
            sheet.append(row)

        # Save the modified workbook
        book.save(file_path)
        k += 2


getCompetitors("ADN")
