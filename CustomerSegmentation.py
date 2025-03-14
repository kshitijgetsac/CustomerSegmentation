import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the datasets
# make sure to use the correct path
superstore_path = "/work/SampleSuperstore.csv" 
population_path = "/work/us_pop_by_state.csv"  

superstore = pd.read_csv(superstore_path)
population = pd.read_csv(population_path)

# Display the initial rows of each dataset
print("Superstore Data:\n", superstore.head())
print("Population Data:\n", population.head())

# Preprocessing Superstore Dataset
superstore_agg = superstore.groupby('State').agg(
    total_sales=('Sales', 'sum'),
    total_profit=('Profit', 'sum'),
    transaction_count=('Segment', 'count')
).reset_index()

# Preprocessing Population Dataset
population.rename(columns={'state': 'State', 'Population': 'population'}, inplace=True)

# Merge datasets on State
merged_data = pd.merge(superstore_agg, population, on='State', how='inner')
print(merged_data.head())
# Feature Engineering
merged_data['sales_per_capita'] = merged_data['total_sales'] / merged_data['2020_census']
merged_data['profit_per_capita'] = merged_data['total_profit'] / merged_data['2020_census']
merged_data['sales_potential_score'] = (
    merged_data['sales_per_capita'] * 0.5 + (merged_data['total_sales'] / merged_data['transaction_count']) * 0.5
)

# Exploratory Data Analysis (EDA) with separate colors for each state
plt.figure(figsize=(12, 6))
colors = sns.color_palette('husl', len(merged_data))  # Create a distinct color for each state
sns.barplot(x='State', y='sales_per_capita', 
            data=merged_data.sort_values(by='sales_per_capita', ascending=False), 
            palette=colors)  # Use the colors palette
plt.xticks(rotation=90)
plt.title('Sales Per Capita by State')
plt.show()



# Visualizing Profit Margin by State (Profit per Capita)
plt.figure(figsize=(12, 6))
colors = sns.color_palette('husl', len(merged_data))  # Create a distinct color for each state
sns.barplot(
    x='State', 
    y='profit_per_capita', 
    data=merged_data.sort_values(by='profit_per_capita', ascending=False), 
    palette=colors  # Use the color palette
)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.title('Profit Per Capita by State', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Profit Per Capita by State', fontsize=12)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()


# Prepare the data for clustering
features = ['profit_per_capita']
X = merged_data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality for better visualization, if necessary
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
merged_data['cluster'] = kmeans.fit_predict(X_pca)

# Visualize the clustering
plt.figure(figsize=(12, 6))
colors = sns.color_palette('husl', len(set(merged_data['cluster'])))
sns.scatterplot(x=merged_data['State'], y=merged_data['profit_per_capita'], hue=merged_data['cluster'], palette=colors, marker='X')
plt.xticks(rotation=90)
plt.title('Clustering of States Based on Profit Per Capita')
plt.xlabel('State')
plt.ylabel('Profit per Capita')
plt.legend(title='Cluster')
plt.show()



# Prepare the data for clustering
features = ['sales_per_capita']
X = merged_data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality for better visualization, if necessary
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
merged_data['cluster'] = kmeans.fit_predict(X_pca)

# Visualize the clustering
plt.figure(figsize=(12, 6))
colors = sns.color_palette('husl', len(set(merged_data['cluster'])))
sns.scatterplot(x=merged_data['State'], y=merged_data['sales_per_capita'], hue=merged_data['cluster'], palette=colors, marker='X')
plt.xticks(rotation=90)
plt.title('Clustering of States Based on Sales Per Capita')
plt.xlabel('State')
plt.ylabel('Sales per Capita')
plt.legend(title='Cluster')
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Loading the datasets
superstore = pd.read_csv('SampleSuperstore.csv')
state_populations = pd.read_csv('us_pop_by_state.csv')

# 2. Data Preprocessing
# Superstore dataset(Grouping by state and calculating total sales and profit)
state_sales_profit = superstore.groupby('State').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

# Merging with the state populations dataset
merged_data = pd.merge(state_sales_profit, state_populations, left_on = 'State', right_on = 'state')

# Calculation of sales-to-population and profit-to-population ratios
merged_data['Sales_to_Population'] = merged_data['Sales'] / merged_data['2020_census']
merged_data['Profit_to_Population'] = merged_data['Profit'] / merged_data['2020_census']

# 3. Define classification thresholds
sales_thresholds = [0.005, 0.01]
profit_thresholds = [0.0025, 0.005]

# Function to classify based on sales
def classify_sales(row):
    if row['Sales_to_Population'] > sales_thresholds[1]:
        return 'High Sales Potential'
    elif row['Sales_to_Population'] > sales_thresholds[0]:
        return 'Moderate Sales Potential'
    else:
        return 'Low Sales Potential'

# Function to classify based on profit
def classify_profit(row):
    if row['Profit_to_Population'] > profit_thresholds[1]:
        return 'High Profit Potential'
    elif row['Profit_to_Population'] > profit_thresholds[0]:
        return 'Moderate Profit Potential'
    else:
        return 'Low Profit Potential'

# Apply the classification functions
merged_data['Sales_Classification'] = merged_data.apply(classify_sales, axis = 1)
merged_data['Profit_Classification'] = merged_data.apply(classify_profit, axis = 1)

# 4. Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Sales Classification Bar Plot
axes[0].bar(merged_data['State'], merged_data['Sales_to_Population'], color = merged_data['Sales_Classification'].map({'High Sales Potential': 'green', 'Moderate Sales Potential': 'yellow', 'Low Sales Potential': 'red'}))
axes[0].set_xticks(range(len(merged_data['State'])))
axes[0].set_xticklabels(merged_data['State'], rotation=90)
axes[0].set_xlabel('State')
axes[0].set_ylabel('Sales-to-Population Ratio')
axes[0].set_title('Sales Potential Classification')

# Profit Classification Bar Plot
axes[1].bar(merged_data['State'], merged_data['Profit_to_Population'], color = merged_data['Profit_Classification'].map({'High Profit Potential': 'green', 'Moderate Profit Potential': 'yellow', 'Low Profit Potential': 'red'}))
axes[1].set_xticks(range(len(merged_data['State'])))
axes[1].set_xticklabels(merged_data['State'], rotation=90)
axes[1].set_xlabel('State')
axes[1].set_ylabel('Profit-to-Population Ratio')
axes[1].set_title('Profit Potential Classification')

# Legends
handles_sales = [plt.Line2D([0], [0], color='green'), 
                 plt.Line2D([0], [0], color='yellow'), 
                 plt.Line2D([0], [0], color='red')]
labels_sales = ['High Sales Potential', 'Moderate Sales Potential', 'Low Sales Potential']
axes[0].legend(handles = handles_sales, labels = labels_sales)

handles_profit = [plt.Line2D([0], [0], color='green'), 
                  plt.Line2D([0], [0], color='yellow'), 
                  plt.Line2D([0], [0], color='red')]
labels_profit = ['High Profit Potential', 'Moderate Profit Potential', 'Low Profit Potential']
axes[1].legend(handles = handles_profit, labels = labels_profit)

plt.tight_layout()
plt.show()


df_store = pd.read_csv('SampleSuperstore.csv')
df_pop = pd.read_csv('us_pop_by_state.csv')

profit = []
count = 0
df_store['Profit'] = df_store['Profit'].astype(float)
for i in range(len(df_store)):
  if df_store['Profit'][i] >= 0:
    profit.append(1)
  else:
    profit.append(0)
    count += 1
print(count)
df_store['Target'] = profit

Low_State_List = []
Med_State_List = []
High_State_List = []
for i in range(len(df_pop)):
  if df_pop['2020_census'][i] < 2.5 * 10**6:
    Low_State_List.append(df_pop["state"][i])
  elif 2.5 * 10**6 < df_pop['2020_census'][i] < 7.5 * 10**6:
    Med_State_List.append(df_pop["state"][i])
  else:
    High_State_List.append(df_pop["state"][i])

Population = []
for i in range(len(df_store)):
  if df_store["State"][i] in Low_State_List:
    Population.append("Low")
  elif df_store["State"][i] in Med_State_List:
    Population.append("Medium")
  elif df_store["State"][i] in High_State_List:
    Population.append("High")
  else: 
    Population.append("Unknown")
    continue

df_store['Population'] = Population

import pandas as pd
import numpy as np
from collections import Counter

# Function to calculate entropy
def entropy(labels):
    """
    Calculate entropy of a list of class labels.
    """
    total = len(labels)
    counts = Counter(labels)
    probabilities = [count / total for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# Function to calculate information gain
def information_gain(parent_labels, left_labels, right_labels):
    """
    Calculate information gain from splitting a parent node into two child nodes.
    """
    total = len(parent_labels)
    parent_entropy = entropy(parent_labels)
    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total
    children_entropy = left_weight * entropy(left_labels) + right_weight * entropy(right_labels)
    return parent_entropy - children_entropy

# Function to calculate IG for all features
def calculate_ig_for_dataframe(df, target_column):
    """
    Calculate information gain for each feature in the DataFrame.
    
    Parameters:
    - df: pandas DataFrame
    - target_column: The name of the column containing the class labels (target).
    
    Returns:
    - A dictionary of IG values for each feature.
    """
    ig_dict = {}
    target = df[target_column]

    for feature in df.columns:
        if feature == target_column:
            continue  # Skip the target column

        # Calculate IG for this feature
        feature_values = df[feature]
        unique_values = feature_values.unique()
        parent_labels = target

        # Create splits for each unique feature value
        weighted_entropy = 0
        for value in unique_values:
            split_labels = target[feature_values == value]
            weight = len(split_labels) / len(parent_labels)
            weighted_entropy += weight * entropy(split_labels)

        ig_dict[feature] = entropy(parent_labels) - weighted_entropy

    return ig_dict

# Example usage
#data = {
   # "Feature1": ["A", "A", "B", "B", "C", "C"],
   # "Feature2": [1, 1, 0, 0, 1, 0],
   # "Target": [1, 1, 0, 0, 1, 0],
#df = pd.DataFrame(data)

ig_results = calculate_ig_for_dataframe(df_store, target_column="Target")
print("Information Gain for each feature:", ig_results)
print(type(ig_results))

ig_results = dict.items(ig_results)

keys = []
values = []
for key, value in ig_results:
    keys.append(key)
    values.append(value)

print(keys)
print(values)
fig, ax = plt.subplots(figsize=(16, 6))
plt.bar(keys, values, color ='maroon')

plt.xlabel("Variables")
plt.ylabel("Information Gain towards Profit")
plt.title("Profit Information Gain for Data Categories")
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("/work/SampleSuperstore.csv")

# Count unique subcategories per state
subcategory_counts = data.groupby("State")["Sub-Category"].nunique()

# Create a ranked DataFrame
ranked_states = subcategory_counts.sort_values(ascending=False).reset_index()
ranked_states.columns = ["State", "Unique Subcategories"]

# Add rank column
ranked_states["Rank"] = ranked_states["Unique Subcategories"].rank(ascending=False, method="min").astype(int)

# Classify based on rank
ranked_states["Classification"] = np.where(
    ranked_states["Rank"] <= 25, "Top 25 (Diverse Markets)",
    np.where(ranked_states["Rank"] > len(ranked_states) - 25, "Bottom 25 (Limited Markets)", "Middle Markets")
)

# Map colors based on classification
colors = ranked_states["Classification"].map({
    "Top 25 (Diverse Markets)": "green",
    "Bottom 25 (Limited Markets)": "red"
})

# Plot the data with spaces between bars
plt.figure(figsize=(20, 10))
plt.bar(ranked_states["State"], ranked_states["Unique Subcategories"], color=colors, edgecolor="black", width=0.8)

# Add labels and title
plt.title("States by Unique Subcategories with Classification", fontsize=16)
plt.xlabel("State", fontsize=12)
plt.ylabel("Unique Subcategories", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="green", edgecolor="black", label="Top 25 (Diverse Markets)"),
    Patch(facecolor="red", edgecolor="black", label="Bottom 25 (Limited Markets)")
]
plt.legend(handles=legend_elements, title="Classification")

# Show the plot
plt.show()




# Merge diversity classification into the main dataset
merged_data = pd.merge(
    merged_data,
    ranked_states[['State', 'Classification']],  # Assuming 'Classification' has diversity info
    on='State',
    how='left'
)

# Combine classifications into a new column
def combine_classifications(row):
    combined = f"{row['Sales_Classification']} & {row['Profit_Classification']}"
    if pd.notna(row['Classification']):
        combined += f" & {row['Classification']}"
    return combined

merged_data['Combined_Classification'] = merged_data.apply(combine_classifications, axis=1)

# Visualization: Combined Classifications
plt.figure(figsize=(18, 6))
combined_class_summary = merged_data['Combined_Classification'].value_counts()
combined_class_summary.plot(kind="bar", color="skyblue", edgecolor="black")

plt.title("Combined Classification of States", fontsize=16)
plt.xlabel("Classification", fontsize=12)
plt.ylabel("Number of States", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Display the DataFrame with combined classifications
display(merged_data)


import matplotlib.pyplot as plt

# Sort the merged data by rank
sorted_data = merged_data.sort_values(by='rank', ascending=True)

# Assign colors based on rank
colors = ['green' if rank <= 10 else 'yellow' if rank <= 25 else 'red' for rank in sorted_data['rank']]

# Plot the ranks of all states
plt.figure(figsize=(18, 8))
plt.bar(sorted_data['State'], sorted_data['rank'], color=colors, edgecolor='black')

# Add labels and title
plt.title("State Ranks", fontsize=16)
plt.xlabel("State", fontsize=12)
plt.ylabel("Rank", fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="green", edgecolor="black", label="Top 10"),
    Patch(facecolor="yellow", edgecolor="black", label="Top 11-25"),
    Patch(facecolor="red", edgecolor="black", label="Above 25")
]
plt.legend(handles=legend_elements, title="Ranking Levels")

# Show the plot
plt.show()
