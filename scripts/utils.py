import pandas as pd
import os

#GENERETE UNIQUE CSV

def add_ticker_and_load_csv(file_path):

    file_path = 'data/sp500/SP500.csv'
    folder_path = "data/sp500/csv/"
    output_file_path = 'data/sp500/SP500.csv'

    ticker = os.path.basename(file_path).split('.')[0]
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df.insert(0,"Ticker",ticker)

    return df

def generete_unique_csv():

    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    combined_df = pd.concat((add_ticker_and_load_csv(file) for file in csv_files), ignore_index=True)
    sorted_df = combined_df.sort_values(['Ticker', 'Date'])

    sorted_df.to_csv(output_file_path, index=False)

def create_missing_values_csv(df):
    # Extract the year from the 'Date' column
    df['Year'] = df['Date'].dt.year

    # Count the total number of rows for each ticker
    total_counts = df.groupby('Ticker').size()

    # Count the missing values for 'Adjusted Close' for each ticker
    missing_counts = df[df['Adjusted Close'].isnull()].groupby('Ticker').size()

    # Calculate the overall percentage of missing values for each ticker
    overall_missing_percentage = (missing_counts / total_counts * 100).reset_index(name='Overall Missing Percentage')

    # Calculate the number of missing values for each ticker, each year
    missing_counts_yearly = df[df['Adjusted Close'].isnull()].groupby(['Ticker', 'Year']).size()

    # Calculate the total number of rows for each ticker, each year
    total_counts_yearly = df.groupby(['Ticker', 'Year']).size()

    # Calculate the percentage of missing values for each ticker, each year
    missing_percentage_yearly = (missing_counts_yearly / total_counts_yearly * 100).reset_index(name='Missing Percentage')

    # Merge the overall missing percentage with the yearly statistics
    ticker_missing_stats = pd.merge(overall_missing_percentage, missing_percentage_yearly, on='Ticker', how='right')

    # Display the statistics for each ticker
    ticker_missing_stats = ticker_missing_stats[['Ticker', 'Overall Missing Percentage', 'Year', 'Missing Percentage']]
    ticker_missing_stats['Total Counts'] = total_counts_yearly.values
    ticker_missing_stats.to_csv('data/sp500/missing_values.csv')

def create_number_rows_by_year(df):
    # Count the number of rows for each ticker per year
    ticker_year_distribution = df.groupby(['Ticker', 'Year']).size().unstack().fillna(0)

    ticker_year_distribution.to_csv('data/sp500/numberrows.csv')