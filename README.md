# UAS-Weather_Prediction
Project ini adalah implementasi metode supervised learning menggunakan algoritma KNN untuk menentukan label atau menilai dari prediksi cuaca
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

df = pd.read_csv('seattle-weather.csv')
df


Data Understanding :
//mempresentasikan mengenai jumlah baris dan kolom pada data
1. print ('data shape :', df.shape)
//melihat tipe data jika ada object dilakukan proses encoding
2. df.info()
//mempresentasikan sebagai nilai penting pada data
3. df.describe()
//memperlihatkan nilai uniq
4. df.temp_max.value_counts()
-  df.weather.value_counts()
//mevisualisasikan cuaca
5. sns.histplot(df['weather'])

Data Cleaning
//missing value untuk mencari nilai kosong
1. df.isnull().sum()
//duplikasi data untuk mencari nilai double
2. df.duplicated().sum()

Exploratory Data Analysis (EDA)
melakukan penjabaran tiap atribut agar peneliti paham
1. sns.set_theme(style='ticks')
	sns.countplot(y='weather', data=df, palette='flare')
	plt.ylabel("cuaca")
	plt.xlabel('terjadi sebanyak')
	plt.show()


Data Preparation
digunakan untuk melakukan preparasi data seperti encodeing
//menampilkan 5 data pertama
1. df.head()
untuk proses ML tidak bisa menggunakan string dilakukan proses encoding
//proses encoding
2. from sklearn.preprocessing import LabelEncoder

	label_encoder = LabelEncoder()

	df['date'] = label_encoder.fit_transform(df['date'])
	df['weather'] = label_encoder.fit_transform(df['weather'])
//melihat data setelah di encoding
3. df.head()
//splitting atau pembagian data
4. x = df.drop(columns = ['weather'])
	y = df['weather']

	print("x : ", x.shape)
	print("y : ", y.shape)
//pembagian data test dan data training
5.x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
//melihat nilai x_train
6. print(f'x_train : {x_train.shape}')
	print(f'y_train : {y_train.shape}')
	print(f'x_test : {x_test.shape}')
	print(f'y_test : {y_test.shape}')


Modeling
1. knn = KNeighborsClassifier(n_neighbors=3)
	knn.fit(x_train, y_train)

	y_pred = knn.predict(x_test)
	KNN_acc = accuracy_score(y_pred, y_test)

	print(classification_report(y_test, y_pred))
	print('Akurasi KNN : {:.2f}%' .format(KNN_acc*100))

2. classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)

3. #testing
	testing = {'date': [1462],
           'precipitation': [12],
           'temp_max': [3.5],
           'temp_min': [2.0],
           'wind': [7.0]}

	testing = pd.DataFrame(testing)
	testing
//prediksi
4. pred_test = knn.predict(testing)
	print("Hasil Prediksi Cuaca Terbaru")
	print(pred_test)
