import requests
import os
import zipfile

MAIN_DIREC = "data_loaders/data/uci/"
os.mkdir("data_loaders/data")
os.mkdir(MAIN_DIREC)

directories = ["wine-quality", "concrete", "yacht", "energy-efficiency", "protein", "kin8nm", "energy"]

simple_urls = ["https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/wine-quality-red/data/data.txt",
               "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
               "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
               "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
               "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
               "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/6eb4497628d12b0f300f4b4f6bdc386bebad565c/UCI_Datasets/kin8nm/data/data.txt",
               "https://github.com/Srceh/DistCal/blob/master/Dataset/Appliances_energy_prediction.csv?raw=true"]

file_names = ["wine_data_new.txt", 
              "Concrete_Data.xls",
              "yacht_hydrodynamics.data", 
              "ENB2012_data.xlsx",
              "CASP.csv",
              "data.txt", 
              "beijing.csv",
              "energydata_complete.csv"]


for i in range(len(directories)):
    print("Downloading {}".format(directories[i]))
    os.mkdir(MAIN_DIREC + directories[i])
    r = requests.get(simple_urls[i], allow_redirects=True)
    open(MAIN_DIREC + "{}/{}".format(directories[i], file_names[i]), "wb").write(r.content)

zip_directories = ["power-plant", "naval", "song"]
zip_urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
           "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
           "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"]

zip_file_names = ["/CCPP.zip", "/UCI_CBM_Dataset.zip", "/YearPredictionMSD.zip"]

for i in range(len(zip_directories)):
    print("Downloading {}".format(zip_directories[i]))
    path = MAIN_DIREC + zip_directories[i]
    os.mkdir(path)
    r = requests.get(zip_urls[i], allow_redirects=True)
    open(path + zip_file_names[i], "wb").write(r.content)
    with zipfile.ZipFile(path + zip_file_names[i], "r") as zip_ref:
        zip_ref.extractall(MAIN_DIREC + zip_directories[i])

directory = "crime"
crime_urls = ["https://raw.githubusercontent.com/ShengjiaZhao/Individual-Calibration/master/data/communities.data", "https://raw.githubusercontent.com/ShengjiaZhao/Individual-Calibration/master/data/names"]
print("Downloading {}".format(directory))
os.mkdir(MAIN_DIREC + directory)
file_names = ["communities.data", "names"]

for i in range(len(crime_urls)):
    r = requests.get(crime_urls[i], allow_redirects=True)
    open(MAIN_DIREC + "{}/{}".format(directory, file_names[i]), "wb").write(r.content)
