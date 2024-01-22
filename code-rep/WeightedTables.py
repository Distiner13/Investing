'''
Weighted Tables takes the scores we attrbuted to each metric and gives a score to the companies within a list
which we get from 'ProfileScraping.py'
(NOT Functional, for now)
'''

#%%
import pandas as pd
from openpyxl.reader.excel import load_workbook
from scipy import stats
from ComparisonTables import Rewrite, RunComparisonTables
import ProfileScraping

notNecessary = [9, 19, 23, 24, 25, 26, 76]
IncreaseIsGood = [1, 15, 16, 20, 21, 22, 27, 28, 41, 42, 43, 44, 45, 48,
                  49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
IncreaseIsBad = [1, 8, 10, 11, 12, 13, 14, 46, 47, 68]
ValueArr = [8, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 27, 28, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50]
ValueWeight = [1.428571429, 1.428571429, 1.428571429, 1.428571429, 1.428571429, 1.428571429, 5, 5,
               0.2, 0.2, 0.8, 2, 2, 10, 10, 10, 5, 5, 5, 5, 3.333333333, 3.333333333,
               3.333333333]

MomentumArr = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 68]

MomentumWeight = [0.2919708029, 1.167883212, 3.503649635, 7.00729927, 14.01459854, 14.01459854, 8.275862069,
                  8.275862069,
                  0.6896551724, 2.75862069, 1.111111111, 2.777777778, 11.11111111, 15]

def findRemove(df):
    missing_threshold = 3
    columns_to_check = IncreaseIsGood + IncreaseIsBad

    df.replace("-", pd.NA, inplace=True)
    # Count the number of missing metrics for each company
    missing_counts = df.iloc[:, columns_to_check].isnull().sum(axis=1)

    # Filter companies based on missing metric count
    companies_to_keep = missing_counts <= missing_threshold
    print("companies to keep: \n", companies_to_keep)
    filtered_df = df[companies_to_keep]

    print('filteredDF: \n', filtered_df)

    try:
        filtered_df2 = filtered_df.loc[df['Slope'] > 0]
    except:
        filtered_df2 = filtered_df
    print('filteredDF 2: \n', filtered_df2)

    try:
        filtered_df3 = filtered_df2.loc[df['Positive2Years'] > 0]
    except:
        filtered_df3 = filtered_df2

    print('filteredDF 3: \n', filtered_df3)


    print("filtered: \n", filtered_df3)

    return filtered_df3

def getNormalizedDatas(numeric_df):
    Good_columns = numeric_df.iloc[:, IncreaseIsGood]
    Bad_columns = numeric_df.iloc[:, IncreaseIsBad]

    for column in Good_columns.columns:
        for index in Good_columns.index:
            numeric_df.loc[index, column] = (numeric_df.loc[index, column] / max(abs(numeric_df.loc[:, column]))) *100

    for column in Bad_columns.columns:
        for index in Bad_columns.index:
            numeric_df.loc[index, column] = (1-(numeric_df.loc[index, column] / max(abs(numeric_df.loc[:, column])))) *100

    return numeric_df

def SORT(merged_df, C):
    #Sorting the table
    merged_df.sort_values(by=C, inplace=True, ascending=False)
    merged_df.reset_index(drop=True, inplace=True)
    print(f'Sorted Table by {C}: \n', merged_df)

def extract_values(Data):
    Data['Slope'], Data['Positive2Years'] = None, None

    # Iterate over each index and row in the dataframe
    for index, row in Data.iterrows():
        # Define the following values using the value in the column called 'Ticker'
        slope, positive_2_years = ProfileScraping.getFinancials(row['Ticker\n\n'])
        print(f'slope, positive_2_years: {slope, positive_2_years}')

        # For each value defined, we insert it into the appropriate column in that same row
        Data.at[index, 'Slope'] = slope
        Data.at[index, 'Positive2Years'] = positive_2_years

def RunWeightedTables():
    file_path = "C:/Users/User/Documents/- Desktop _Archive/PL-/Learn/Programming/Finance-Project/Financial-project-main/Financial-project-main/Output.xlsx"
    Data = pd.read_excel(file_path, sheet_name='Sheet1')
    print('This is the Data before zip: \n', Data)
    extract_values(Data)
    print("This is Before filter: \n", Data)
    Data = findRemove(Data)
    print("This is after filter: \n", Data)
    Data_replaced = Data.applymap(Rewrite)
    print("This is after remapping: \n", Data_replaced)

    #Put the percentile values directly in the original value
    #so no need to differentiate increase is good or increase is bad
    selected_columns = RunComparisonTables()
    print("Full: \n", Data_replaced)

    # Convert the columns to numeric values, ignoring non-numeric values
    numeric_df = Data_replaced.apply(pd.to_numeric, errors='coerce')
    TickerCol = Data_replaced.iloc[:, 1:2]
    print("TickerCol: \n", TickerCol)

    for column in numeric_df.columns:
        try:
            numeric_df[column].fillna(numeric_df[column].mean(), inplace=True)
        except:
            continue

    print("value columns original: \n", numeric_df.iloc[:, ValueArr])
    # Get the indices from selected_columns
    numeric_df = getNormalizedDatas(numeric_df)

    print("Updated: \n", numeric_df)

    Value_columns = numeric_df.iloc[:, ValueArr].copy()
    Momentum_columns = numeric_df.iloc[:, MomentumArr].copy()
    print("Value_columns \n", Value_columns)
    print("Momentum_columns \n", Momentum_columns)

    Value_columns.loc[:, 'WeightedValueAverage'] = (Value_columns * ValueWeight).sum(axis=1) / sum(ValueWeight)
    print("weighted average: \n", Value_columns)

    Momentum_columns.loc[:, 'WeightedMomentumAverage'] = (Momentum_columns * MomentumWeight).sum(axis=1) / sum(MomentumWeight)
    print("weighted average: \n", Momentum_columns)

    # merge good and bad
    merged_df = pd.merge(Momentum_columns, Value_columns, left_index=True, right_index=True)
    merged_df['Score'] = (merged_df['WeightedValueAverage'] + merged_df['WeightedMomentumAverage'])/2
    merged_df.iloc[:, 0] = TickerCol.iloc[:, 0]
    #we might be losing the organization of the column titles: instead of TICKER for example, we have performance week...
    SORT(merged_df, 'WeightedValueAverage')
    SORT(merged_df, 'WeightedMomentumAverage')
    SORT(merged_df, 'Score')

    #merged_df['Slope'], merged_df['Positive2Years'] = zip(*merged_df['Perf Week'].apply(ProfileScraping.getFinancials))

    file_path = "C:/Users/User/Documents/- Desktop _Archive/PL-/Learn/Programming/Finance-Project/Financial-project-main/Financial-project-main/Scores.xlsx"
    k = 0
    firsttime = True
    # Load the existing workbook
    book = load_workbook(file_path)

    sheet = book['Sheet1']


    # Convert the DataFrame to a list of lists
    data_values = merged_df.values.tolist()

    if (firsttime == True):
        header_titles = merged_df.columns.tolist()
        sheet.append(header_titles)
        firsttime = False

    # Insert the data into the sheet starting from the specified row
    for i, row in enumerate(data_values, start=0):
        sheet.append(row)

    # Save the modified workbook
    book.save(file_path)
    k += 2
