import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore
from sklearn.impute import KNNImputer

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import chi2, SelectKBest

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Understand the Goal : Data Preprocessing

# Read the csv
df1 = pd.read_csv("/Users/prakash/Downloads/airbnb-original.csv")
df2 = pd.read_csv("/Users/prakash/Downloads/titanic.csv")

# Data exploration : SHID
print(df2.shape)
print(df2.head())
print(df2.info())
print(df2.describe())

# Getting Type of attributes
# Analysing spread and distribution of all numerical attributes

print("\n\n\nData Types")
print(df2.dtypes)
numerical_attributes = df2.select_dtypes(include=['number'])
print("\n\nNumerical attr.")
print(numerical_attributes)

x = numerical_attributes.columns.tolist()
print("\n\n\n")
print(x)

print("\n\n\n")
for attr in x:
    print(df2[attr].describe())
    sns.histplot(df2[attr], kde=True)
    plt.show()
    
    
# Data cleaning / Preprocessing : MOD
# - Missing values
# - Outliers
# - Duplicates
    

print(df2.isnull().sum())

threshold = 0.4 * len(df2)
df2.dropna(axis=1, inplace=True, thresh=threshold)

# REMOVING PARTICULAR COLUMN
df2.drop(columns=['Name'], inplace=True)

print("\n\n\n")
print(df2.isnull().sum())
print("\n\n\n")

# Imputation
df2['Age'] = df2['Age'].fillna(df2['Age'].mean())


print("\n\n\n")
print(df2['Age'].isnull().sum())
print("\n\n\n")

# KNN Imputer
knn_df = df2[x].copy()
imputer  = KNNImputer(n_neighbors=5)
knn_imputed = imputer.fit_transform(knn_df)
df2[x] = knn_imputed[:,:]

print(df2[x].isnull().sum())

# Detecting Outliers
from scipy.stats import zscore

df2['Age_zscore'] = zscore(df2['Age'])
outliers = df2[df2['Age_zscore'].abs() > 3]  # Data points where z-score is greater than 3 or less than -3
print("\n\n\n")
print(outliers)
print("\n\n\n")

df2 = df2[df2['Age_zscore'].abs() <= 3]  # Remove rows where Z-score > 3
print("\n\n\n")
print(len(df2))
print("\n\n\n")

# Duplicates
print("\n\n\n")
print(df2.duplicated().sum())
df2 = df2.drop_duplicates(subset=[''], keep='first')
df2.drop_duplicates(inplace=True)
print("\n\n\n")

# Data Validation:
# - Data Type checking
# - Range checking
# - Cross-Field validation

print("\n\n\nData Types")
print(df2.dtypes)

# Cap Range
df2[''] = df2[''].apply(lambda x : _ if x > _ else x)

# CFV
df2 = df2[df2[''] >= df2['']]

invalid_fare_class = df[(df['Pclass'] == 1) & (df['Fare'] == 0)]
df.loc[(df['Pclass'] == 1) & (df['Fare'] == 0), 'Fare'] = df['Fare'].median()

# Data Transformation

# Overcome any bias towards features with larger absolute values


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# - Min-Max normalization
scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# - z-score normalization
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


# Converting data from one format to other
# - Encoding (Label, One-Hot, Frequency)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)  # Drop first to avoid dummy variable trap
print(df.head())

df['Embarked_freq'] = df['Embarked'].map(df['Embarked'].value_counts())




# Reduce noise
# - Descretizing - Binning (Equal width or Equal Frequency)

# Equal Width
df['Age_Bin'] = pd.cut(df['Age'], bins=3, labels=['Young', 'Middle', 'Old'])

# Equal Frequency
df['Age_Bin'] = pd.qcut(df['Age'], q=3, labels=['Young', 'Middle', 'Old'])


print(df2.isna().sum())
# Feature selction
from sklearn.feature_selection import chi2, SelectKBest

selector_chi2 = SelectKBest(chi2, k=5)
selector_chi2.fit_transform(numerical_attributes, df2['Survived'])

print(df2.columns[selector_chi2.get_support()])





# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# pca = PCA(n_components=3)
# x_df = pca.fit_transform(df2[x])

# x_df = pd.DataFrame(x_df, columns=['PC1', 'PC2', 'PC3'])

# print(x_df)

lda = LDA(n_components=1)
x_lda = lda.fit_transform(df2[x], df2['Survived'])

x_lda = pd.DataFrame(x_lda, columns=['PC1'])

print(x_lda)

# PLOTTINGS

sns.histplot(df2['Age'], kde=True)  # Adding KDE for smoother visualization
plt.show()

sns.boxplot(x='Survived', y='Age', data=df2)  # Box plot to check age distribution for each survival class
plt.show()

sns.violinplot(x='Survived', y='Age', data=df2)
plt.show()

sns.scatterplot(x='Age', y='Fare', data=df2)  # Scatter plot between age and fare
plt.show()

sns.lineplot(x='Age', y='Fare', data=df2)
plt.show()

sns.pairplot(df2[['Age', 'Fare', 'Pclass', 'Survived']])  # Visualizing pairwise relationships
plt.show()

correlation_matrix = numerical_attributes.corr()  # Correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  # Annotates correlations
plt.show()

sns.countplot(x='Survived', data=df2)  # Count of survival status
plt.show()

sns.barplot(x='Pclass', y='Fare', data=df2)  # Average Fare per Pclass
plt.show()

sns.catplot(x='Survived', y='Age', data=df2, kind='bar')  # Box plot for 'Survived' vs 'Age'
plt.show()



    
    