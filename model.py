import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# Movie dataset
data = pd.read_excel('/content/gdrive/MyDrive/movies.xlsx')


# Missing values
missing_values = data.isnull().sum()
print("Απουσιάζουσες τιμές ανά στήλη:")
print(missing_values)


data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
#data['Oscar Winners'] = pd.to_numeric(data['Oscar Winners'], errors='coerce')
data['Opening Weekend'] = pd.to_numeric(data['Opening Weekend'], errors='coerce').astype('float64')
data['Rotten Tomatoes  critics'] = pd.to_numeric(data['Rotten Tomatoes  critics'], errors='coerce').astype('Int64')
data['Metacritic  critics'] = pd.to_numeric(data['Metacritic  critics'], errors='coerce').astype('Int64')
data['Genre'] = data['Genre'].astype(str)
#data['Average critics '] = data['Average critics '].astype(int)
data['Rotten Tomatoes Audience ']= data['Rotten Tomatoes Audience '].astype('Int64')
#data['Average audience ']= data['Average audience '].astype(int)
#data['Audience vs Critics deviance ']= data['Audience vs Critics deviance '].astype(int)
data['Rotten Tomatoes  critics'].fillna(0, inplace=True)
data['Opening Weekend'].fillna(0, inplace=True)
data['Worldwide Gross ($million)'].fillna(0, inplace=True)
data['Rotten Tomatoes Audience '].fillna(0, inplace=True)
data['Metacritic  critics'].fillna(0, inplace=True)
data['Genre'].fillna(0, inplace=True)
data['IMDb Rating'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
data['Oscar Winners'] = data['Oscar Winners'].replace({'Oscar winner': 1, pd.NA: 0})

# Removing rows
data = data.drop(columns=['Distributor', 'Oscar Detail', 'IMDB vs RT disparity', 'Primary Genre','Release Date (US)'])
print(data.dtypes)

for column in data.columns:
    unique_values = data[column].unique()
    print(f"{column}: {unique_values}")

missing_values = data.isnull().sum()
print("Απουσιάζουσες τιμές ανά στήλη:")
print(missing_values)



data['Oscar Winners'] = data['Oscar Winners'].replace({'Oscar Winner': 1, 0: 0})

# Επιλογή χαρακτηριστικών

print(data["Oscar Winners"])
features = ["Rotten Tomatoes  critics", "Metacritic  critics", "Opening Weekend" , "Genre", "Worldwide Gross ($million)", "Rotten Tomatoes Audience "]
X = data[features]
y = data["Oscar Winners"]


# Μετατροπή κατηγορικών μεταβλητών με one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(handle_unknown='ignore'), ["Genre"])],
    remainder="passthrough"
)

# Κατασκευή μοντέλου
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Διαχωρισμός δεδομένων

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Unique values in y_train:", y_train.unique())
print("Class distribution in y_train:", y_train.value_counts())

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Εκπαίδευση μοντέλου
model.fit(X_train, y_train)

preprocessor.fit(X_train)

# Εκτίμηση απόδοσης
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")


prediction_data = pd.read_excel('/content/gdrive/MyDrive/movies_test _anon.xlsx')
prediction_data.info()

prediction_data['Rotten Tomatoes  critics'] = pd.to_numeric(prediction_data['Rotten Tomatoes  critics'], errors='coerce').astype('Int64')
prediction_data['Worldwide Gross ($million)'] = pd.to_numeric(prediction_data['Worldwide Gross ($million)'], errors='coerce').astype('float64')
prediction_data['Worldwide Gross ($million)'].fillna(0, inplace=True)
prediction_data['Metacritic  critics'] = pd.to_numeric(prediction_data['Metacritic  critics'], errors='coerce').astype('Int64')
prediction_data['Genre'] = data['Genre'].astype(str)
prediction_data['Opening Weekend'] = pd.to_numeric(prediction_data['Opening Weekend'], errors='coerce').astype('float64')
prediction_data['Opening Weekend'].fillna(0, inplace=True)
prediction_data['Rotten Tomatoes  critics'].fillna(0, inplace=True)
prediction_data['Metacritic  critics'].fillna(0, inplace=True)
new_X=prediction_data[features]
missing_values = new_X.isnull().sum()
print("Απουσιάζουσες τιμές ανά στήλη:")
print(missing_values)
predictions = model.predict(new_X)
print(predictions)

predictions_df = pd.DataFrame({
    'Predicted Oscar Winner': predictions
})

predictions_file_path = '/content/gdrive/MyDrive/predictions.csv'

predictions_df.to_csv(predictions_file_path)
# Επιλογή χαρακτηριστικών για τη συσταδοποίηση
clustering_features = ["Rotten Tomatoes  critics", "Metacritic  critics", "Opening Weekend", "Worldwide Gross ($million)"]

X_clustering = data[clustering_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)

# Εφαρμογή του αλγορίθμου K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Εμφάνιση cluster
for cluster in range(3):
    print(f"\nCluster {cluster + 1} Characteristics:")
    print(data[data['Cluster'] == cluster][clustering_features].describe())

# Αξιολόγηση συσταδοποιήσης
silhouette_avg = silhouette_score(X_scaled, data['Cluster'])
print(f"Silhouette: {silhouette_avg}")

