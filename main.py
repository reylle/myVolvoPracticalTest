import numpy as np
import os
import pandas as pd
import re
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from urllib import parse


def download_data_set():
    """
    Download the Beijing PM2.5 data set and saves it in the program folder
    """
    # The page used to extract the PM2.5 data set
    url = "https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data"
    # Request the main data set HTML page
    main_page = requests.get(url)
    # Verify if the request was successful
    if main_page.status_code != 200:
        raise requests.HTTPError(f"The request returned the HTTP error code {main_page.status_code} - "
                                 f"{main_page.reason}.")
    # Parse the HTML response using beautifulsoup library
    main_page_parsed = BeautifulSoup(main_page.text, "html.parser")
    # Retrieve the data set download url (<a> tag) that contains the text Data Folder
    try:
        download_url = main_page_parsed.find("a", string="Data Folder")["href"]
    except TypeError:
        raise Exception("Site HTML changed.")
    # Join the main url with the download url
    download_url = parse.urljoin(url, download_url)
    # Request the data set download HTML page
    download_page = requests.get(download_url)
    # Parse the HTML response using beautifulsoup library
    download_page_parsed = BeautifulSoup(download_page.text, "html.parser")
    # Retrieve the data set file name
    # more data sets could be downloaded using a regular expression that searches for ".csv"
    try:
        file_name = download_page_parsed.find("a", string=re.compile("PRSA.+.csv"))["href"]
    except TypeError:
        raise Exception("Site HTML changed.")
    # Join download url with the file name
    download_link = parse.urljoin(download_url, file_name)
    # Request the data set csv HTML page
    data_set_page = requests.get(download_link)
    # Save the page content as a csv
    with open(file_name, 'w') as csv_file:
        csv_file.write(data_set_page.text.replace('\r\n', '\n'))


def main():
    # region READ_DATA

    # Verify if the data set is already downloaded
    if not os.path.exists("PRSA_data_2010.1.1-2014.12.31.csv"):
        download_data_set()

    # Read the data set csv as a DataFrame using pandas
    df = pd.read_csv("PRSA_data_2010.1.1-2014.12.31.csv")

    # endregion READ_DATA

    # region EXPLORATORY_ANALYSIS

    # Describe the dataset
    print(df.describe())

    # Format the timestamp fields
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

    # Set the DataFrame index
    df.set_index('date', inplace=True)

    # Remove the Row Number because it's only an identifier and the timestamp
    df.drop(columns=['No', 'year', 'month', 'day', 'hour'], inplace=True)

    # The first 24 instances got an error in the reading (NaN for the PM2.5)
    # Since removing them will not highly affect the other instances it will be disregarded
    df = df[24:]

    # To maintain the correct order for each reading, when NaN is found it is replaced with the mean
    df['pm2.5'].fillna(df['pm2.5'].mean(), inplace=True)

    # Perform a uni-variate analysis
    # For each column
    for column in df.columns:
        # If is the polynomic variable
        if column == "cbwd":
            # Plot a bar graph
            plt.figure()
            sns.countplot(x=df[column], order=df[column].value_counts().index)
            plt.show()
        else:
            # Plot the histogram
            plt.figure()
            sns.histplot(x=df[column], kde=True)
            plt.show()

            # Perform the box plot
            plt.figure()
            sns.boxplot(x=df[column])
            plt.show()

    # Perform a bi-variate analysis
    # Correlation
    plt.figure()
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True)
    plt.show()

    # Pair plot
    sns.pairplot(df.fillna(0), diag_kind='kde')
    plt.show()

    # endregion EXPLORATORY_ANALYSIS

    # region REGRESSION

    # Encode the cbwd column to be numeric
    le = LabelEncoder().fit(df['cbwd'])
    df['cbwd'] = le.transform(df['cbwd'])

    # Scale all values using the min max scaler
    min_max_scaler = MinMaxScaler()
    scaled_data = min_max_scaler.fit_transform(df.values)

    # Split into the input and output
    x = scaled_data[0:-1, :]
    y = scaled_data[1:, 0]

    # Split the data in train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=False)

    # Reshape
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    # Try to load the model, in case of an error, create, train, and save the model
    try:
        model = load_model('regression_model')
    except (OSError, IOError, ImportError):
        # Create the LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), dropout=0.2))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        # Train the model
        early_stop = EarlyStopping(patience=10)
        error_history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=2,
                                  callbacks=[early_stop], shuffle=False)

        # Plot the training error
        plt.plot(error_history.history['loss'], label='train')
        plt.plot(error_history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        model.save("regression_model")

    # Use the trained model to predict
    predictions = model.predict(x_test)

    # Reshape the true input to its original shape
    x_test_re = x_test.reshape(x_test.shape[0], x_test.shape[2])

    # Concatenate the predictions with the input
    predictions = np.concatenate((predictions, x_test_re[:, 1:]), axis=1)

    # Scale the predictions to the original range
    predictions = min_max_scaler.inverse_transform(predictions)

    # Reshape the true output
    y_test = y_test.reshape(y_test.shape[0], 1)

    # Concatenate the true output with the input
    y_test = np.concatenate((y_test, x_test_re[:, 1:]), axis=1)

    # Scale the true output to the original range
    y_test = min_max_scaler.inverse_transform(y_test)

    # Print the root of the mean squared error
    print(np.sqrt(mean_squared_error(y_test[:, 0], predictions[:, 0])))

    # endregion REGRESSION


if __name__ == '__main__':
    main()
