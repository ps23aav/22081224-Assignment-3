### Load some important libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit

### Load the First Dataset
forest_area  = pd.read_csv('Forest Area.csv', skiprows=4)

### Shows the head of the dataset
forest_area.head()

### Checking for null values in the dataset
null_values = forest_area.isnull().sum()

print("Null Values in Each Column:")
print(null_values)



### Removing Null Values
forest_area_cleaned = forest_area.fillna(0)

forest_area_cleaned.head()

#Transpose the DataFrame
forest_area_transposed = forest_area_cleaned.T
print("Transposed DataFrame:")
print(forest_area_transposed.head())



forest_area_numeric = forest_area_cleaned.loc[:, '1990':]

# Transpose the numeric DataFrame
forest_area_transposed = forest_area_numeric.T

#Normalization
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

forest_area_normalized = normalize(forest_area_transposed)


print("\nNormalized DataFrame:")
print(forest_area_normalized.head())


def data_summary(df):
    return df.describe()
summary = data_summary(forest_area_cleaned)
print("\nData Summary:")
print(summary)

### Shows the Elbow Method for K-means clustering
selected_columns = forest_area_cleaned.loc[:, '1960':'2022']

# Check the selected columns before scaling
print("Selected Columns for K-Means:")
print(selected_columns.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_columns)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()


k = 3


kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
forest_area_cleaned['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize the clustering results
plt.figure(figsize=(50, 15))
sns.scatterplot(x='Country Name', y='Indicator Name', hue='Cluster', 
                data=forest_area_cleaned, palette='viridis')
plt.title('K-means Clustering of Forest Area Data')
plt.xticks(rotation=90)
plt.show()



### Shows the Visualization for K-means Clustering of Forest Area 2010
clustering_data = forest_area_cleaned.iloc[:, 4:-1]

num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

forest_area_cleaned['Cluster'] = kmeans.fit_predict(clustering_data)

plt.figure(figsize=(10, 6))
for cluster in range(num_clusters):
    cluster_data = forest_area_cleaned[forest_area_cleaned['Cluster'] == cluster]
    plt.scatter(cluster_data.index, cluster_data['2010'], label=f'Cluster {cluster + 1}')

plt.title('K-means Clustering of Forest Area (2010)')
plt.xlabel('Data Point Index')
plt.ylabel('Forest Area (sq. km) in 2010')
plt.legend()
plt.show()



# Evaluating the clustering model using silhouette score
silhouette_avg = silhouette_score(clustering_data, forest_area_cleaned['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")


# Evaluating the clustering model using inertia
inertia = kmeans.inertia_
print(f"Inertia: {inertia}")








### Load the second dataset
Rural_land  = pd.read_csv('Rural Land.csv', skiprows=4)

### Showing the head of the dataset
Rural_land.head()

### Check Null Values 
null_values = Rural_land.isnull().sum()

print("Null Values in Each Column:")
print(null_values)


### Replaceing null values with 0
Rural_land_filled = Rural_land.fillna(0)

print(Rural_land_filled.head())


### Fitting a Second-Degree Polynomial
country_data = Rural_land_filled[Rural_land_filled['Country Name'] == 'Aruba']
years = country_data.columns[4:-1].astype(int)
attribute_values = country_data.iloc[:, 4:-1].values.flatten()


def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c


params, covariance = curve_fit(polynomial_function, years, attribute_values)

years_future = np.arange(1960, 2031)
predicted_values = polynomial_function(years_future, *params)

### Plotting the original data and the fitted curve
plt.scatter(years, attribute_values, label='Original Data')
plt.plot(years_future, predicted_values, label='Fitted Curve', color='red')
plt.xlabel('Year')
plt.ylabel('Attribute Value')
plt.title('Fitting a Second-Degree Polynomial')
plt.legend()
plt.show()



### Predicting values for the year 2030
prediction_2030 = polynomial_function(2030, *params)
print(f'Predicted value for the year 2030: {prediction_2030:.2f}')


### Define the err_ranges function
def err_ranges(covariance, alpha=0.05):
    alpha = 1 - alpha
    n = len(covariance)
    quantile = 1 - alpha / 2
    err = np.zeros(n)
    for i in range(n):
        err[i] = np.sqrt(covariance[i, i]) * np.abs(np.percentile(np.random.normal(0, 1, 10000), quantile))
    return err


### Extracting data for a specific country
country_data = Rural_land_filled[Rural_land_filled['Country Name'] == 'Aruba']
years = country_data.columns[4:-1].astype(int)
attribute_values = country_data.iloc[:, 4:-1].values.flatten()


### Fitting the data using curve_fit
params, covariance = curve_fit(polynomial_function, years, attribute_values)


### Estimate confidence range using err_ranges
confidence_range = err_ranges(covariance)



### Generate predicted values for the years 1960-2030
years_future = np.arange(1960, 2031)
predicted_values = polynomial_function(years_future, *params)



### Repeat confidence_range for each predicted value
confidence_range_broadcasted = np.tile(confidence_range, (len(years_future), 1))



### Plotting the original data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.scatter(years, attribute_values, label='Original Data')
plt.xlabel('Year')
plt.ylabel('Attribute Value')
plt.title('Original Data')

### Assuming you have previously defined confidence_range_broadcasted
confidence_range_broadcasted_reshaped = confidence_range_broadcasted[:, 0]


### Plotting the fitted curve with confidence range
plt.subplot(2, 1, 2)
plt.plot(years_future, predicted_values, label='Fitted Curve', color='red')
plt.fill_between(years_future, predicted_values - confidence_range_broadcasted_reshaped, predicted_values + confidence_range_broadcasted_reshaped, color='gray', alpha=0.2, label='Confidence Range')
plt.xlabel('Year')
plt.ylabel('Attribute Value')
plt.title('Fitting a Second-Degree Polynomial with Confidence Range')
plt.legend()

plt.tight_layout()
plt.show()

