#%%


###########################
import pandas as pd

def Rewrite(string):
    if isinstance(string, str):
        try:
            if (string[len(string)-1] == 'M'):
                string = string[:-1]
                R = float(string) * 1000000
                return R

            elif (string[len(string)-1] == 'B'):
                string = string[:-1]
                R = float(string) * 1000000000
                return R

            elif (string[len(string)-1] == '%'):
                string = string[:-1]
                R = float(string) / 100
                return R

            elif (len(string) == 1 and '-' in string):
                R = 'N/A'
                return R
        except:
            pass
        else:
            return string
    else:
        return string

def RunComparisonTables():
    IncreaseIsGood = [1, 15, 16, 19, 20, 21, 22, 23, 24, 25, 27, 28, 33, 41, 42, 43, 44, 45, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 76]
    IncreaseIsBad = [1, 8, 9, 10, 11, 12, 13, 14, 46, 47, 68]

    ValueArr = [8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 27, 28, 33, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50]
    ValueWeight = [1.428571429, 1.428571429, 1.428571429, 1.428571429, 1.428571429, 1.428571429, 1.428571429, 5, 5, 7.333333333, 0.2, 0.2, 0.8, 0.8, 4, 4, 2, 2, 2, 10, 10, 10, 5, 5, 5, 5, 3.333333333, 3.333333333, 3.333333333]

    MomentumArr = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 68, 76]

    MomentumWeight = [0.2919708029, 1.167883212, 3.503649635, 7.00729927, 14.01459854, 14.01459854, 8.275862069, 8.275862069,
             0.6896551724, 2.75862069, 1.111111111, 2.777777778, 11.11111111, 15, 10]

    file_path = "C:/Users/User/Documents/- Desktop _Archive/PL-/Learn/Programming/Finance-Project/Financial-project-main/Financial-project-main/Output.xlsx"

    Data = pd.read_excel(file_path, sheet_name='Sheet1')
    Data_replaced = Data.applymap(Rewrite)
    print("Full: \n", Data_replaced)

    # Convert the columns to numeric values, ignoring non-numeric values
    numeric_df = Data_replaced.apply(pd.to_numeric, errors='coerce')
    TickerCol = Data_replaced.iloc[:, 1:2]
    print("TickerCol: \n", TickerCol)

    for column in numeric_df.columns:
        try:
            numeric_df[column].fillna(numeric_df[column].mean(), inplace = True)
        except:
            continue

    Good_columns = numeric_df.iloc[:, IncreaseIsGood]
    Bad_columns = numeric_df.iloc[:, IncreaseIsBad]
    print("Good increase \n", Good_columns)
    print("Bad increase \n", Bad_columns)

    #IncGood
    max_values = Good_columns.max()
    IncGoodPer = pd.DataFrame()
    for column in Good_columns.columns:
        IncGoodPer[column] = Good_columns[column] / max_values[column] * 100

    print("Good increase percentile: \n", IncGoodPer)

    #IncBad
    max_values = Bad_columns.max()
    IncBadPer = pd.DataFrame()
    for column in Bad_columns.columns:
        IncBadPer[column] = (1 - (Bad_columns[column] / max_values[column])) * 100

    print("Bad increase percentile: \n", IncBadPer)

    #merge good and bad
    merged_df = pd.merge(IncGoodPer, IncBadPer, left_index=True, right_index=True)
    merged_df['Score'] = merged_df.mean(axis=1)
    merged_df.iloc[:, 0] = TickerCol.iloc[:, 0]

    merged_df.sort_values(by = 'Score', inplace = True, ascending=False)
    merged_df.reset_index(drop = True, inplace = True)
    print("Final Table: \n", merged_df)


