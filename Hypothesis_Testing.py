from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# filepath='C:/noise_music_data.csv'
filepath='C:/drone_music_data.csv'
df=pd.read_csv(filepath)

top_cols=['chroma_0','chroma_1','chroma_2','chroma_3','chroma_4','chroma_5','chroma_6','chroma_7','chroma_8','chroma_10','chroma_9','chroma_11']

X=df.loc[:,top_cols]
y=df.loc[:,'label']#.astype(int)
feature_names = X.columns
# Assuming X is your feature matrix and y is your binary label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a classifier
clf = RandomForestClassifier()
clf.fit(X_train_pca, y_train)

# Get predicted probabilities
y_prob = clf.predict_proba(X_test_pca)[:, 1]

# Compute ROC curve and ROC-AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Drone vs Music')
plt.legend()
plt.show()

# Create DataFrame with principal components and feature names
components_df = pd.DataFrame(pca.components_, columns=feature_names)
# Visualize explained variance ratio for each principal component
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.8, align='center')
plt.title('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='r')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Ratio')

plt.tight_layout()
plt.show()

# Display the DataFrame with principal components and feature names
print("Principal Components and Feature Loadings:")
print(components_df)

##############
# Get the loadings for each feature in the first principal component
loadings = pca.components_[0]

# Create a DataFrame to display feature names and their loadings
loading_df = pd.DataFrame({'Feature': feature_names, 'Loading': loadings})
loading_df = loading_df.reindex(loading_df['Loading'].abs().sort_values(ascending=False).index)

# Display the top 5 features
top_features = loading_df.head(5)
print("Top 5 features explaining the most variance:")
print(top_features)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='r')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Ratio')

plt.tight_layout()
plt.show()
################

# Get the loadings for each feature in the first 5 principal components
loadings_df = pd.DataFrame(pca.components_[:5], columns=feature_names)

# Display the top 5 features for each principal component
for i in range(5):
    print(f"Top 5 features for Principal Component {i + 1}:")
    top_features = loadings_df.iloc[i].abs().sort_values(ascending=False).head(5)
    print(top_features)
    print()


########## Step:4 Hypothesis Testing.... why chroma works for Drone ##########