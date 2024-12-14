import pandas as pd
from openpyxl import load_workbook
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# Membaca file Excel
data = pd.read_excel('air_quality.xlsx')
print(data)

# Menghitung Missing Values
missing_count = data.isna().sum()
print("Missing Values Count: ")
print(missing_count)

# Mengisi Missing Values dengan Median atau Mode untuk kolom kategorikal
data["Temperature"] = data["Temperature"].fillna(data["Temperature"].median())
data["Humidity"] = data["Humidity"].fillna(data["Humidity"].median())
data["PM2.5"] = data["PM2.5"].fillna(data["PM2.5"].median())
data["PM10"] = data["PM10"].fillna(data["PM10"].median())
data["NO2"] = data["NO2"].fillna(data["NO2"].median())
data["SO2"] = data["SO2"].fillna(data["SO2"].median())
data["CO"] = data["CO"].fillna(data["CO"].median())
data["Proximity_to_Industrial_Areas"] = data["Proximity_to_Industrial_Areas"].fillna(data["Proximity_to_Industrial_Areas"].median())
data["Population_Density"] = data["Population_Density"].fillna(data["Population_Density"].median())

data["Air Quality"]= data["Air Quality"].fillna(data["Air Quality"].mode()[0])
print("Data after handling missing values:")
print(data)

# Memastikan tidak ada missing values lagi
print("Missing values after imputation:")
print(data.isnull().sum())

# Menyusun data untuk fitur (X) dan target (y)
X = data.drop(columns=["Air Quality"])  # Menghapus kolom target
y = data["Air Quality"]  # Kolom target

# Membagi data menjadi data training dan testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menyediakan pilihan model untuk pengguna
while True:
    print("Pilih model yang ingin digunakan:")
    print("1. Gaussian Naive Bayes (GNB)")
    print("2. K-Nearest Neighbors (KNN)")
    print("0. Keluar")
    input_user = input("Masukkan pilihan Anda (1/2/0): ")
    
    if input_user == "1":
        # Model Gaussian Naive Bayes
        GNB = GaussianNB()
        GNB.fit(X_train, y_train)
        pred = GNB.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        print(f"Akurasi model Gaussian Naive Bayes: {accuracy}")
        
    
    elif input_user == "2":
        # Model K-Nearest Neighbors
        KNN = KNeighborsClassifier(n_neighbors=5)
        KNN.fit(X_train, y_train)
        pred = KNN.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        print(f"Akurasi model KNN: {accuracy}")
        
    
    elif input_user == "0":
        print("Keluar dari program.")
        break  
    
    else:
        print("Pilihan tidak valid. Coba lagi.")
