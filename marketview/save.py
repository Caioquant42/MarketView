import pandas as pd

def save_to_excel(dataframe, file_name):
    """
    Saves a DataFrame to an Excel file in xlsx format.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame to be saved.
        file_name (str): The name of the Excel file to save.
    """
    try:
        dataframe.to_excel(file_name, index=False)
        print(f"DataFrame successfully saved to '{file_name}'")
    except Exception as e:
        print(f"Error occurred while saving DataFrame to '{file_name}': {str(e)}")

# Example usage:
# save_to_excel(my_dataframe, "output.xlsx")
