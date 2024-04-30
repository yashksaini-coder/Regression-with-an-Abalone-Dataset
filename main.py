# Importing the basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_log_error
import warnings
# Disable all warnings
warnings.filterwarnings ('ignore')
import scipy.stats as stats  # noqa: E402
from collections import Counter  # noqa: E402
from skimpy import skim  # noqa: E402

train = pd.read_csv('playground-series-s4e4/train.csv')



train.head()


train.info()


train.isnull().sum()


# ## EDA (Explorartory Data Analysis)


skim(train)


df = train.copy(deep='True')
df = df.drop(columns='id',axis=1)



df.head()


numeric_features = df.select_dtypes(include = ['int', 'float']).columns.to_list()
categoric_features = df.select_dtypes(include = ['object', 'category']).columns.to_list()


numeric_features


categoric_features


# ## Hist Plot


sns.set_style(style="darkgrid")
colors = sns.color_palette(palette='bright', n_colors=len(numeric_features))
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 14))
axs = axs.flat

# Loop through each numeric feature and plot its distribution
for i, num_feat in enumerate(numeric_features):
    sns.histplot(df[num_feat], kde=True, color=colors[i], ax=axs[i], edgecolor="gray")
    axs[i].set_xlabel(num_feat, fontsize=12)
    axs[i].set_ylabel("Density", fontsize=12)
    axs[i].set_title(f"Distribution of {num_feat}", fontsize=14, fontweight='bold', color="black")


for j in range(len(numeric_features), len(axs)):
    fig.delaxes(axs[j])

fig.suptitle("Distribution of numerical variables", fontsize=16, fontweight="bold", color="darkblue")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to accommodate the suptitle
plt.show()



# ## Pair Plot


sns.set(style="white")
sns.pairplot(df, diag_kind="kde", markers="o", plot_kws={"alpha": 0.5})
plt.show()



# ## Box Plot
sns.set(style="whitegrid")
fig, axs = plt.subplots(nrows=len(numeric_features), figsize=(10, 6 * len(numeric_features)))

# Loop through each numerical feature and create a box plot
for i, num_feat in enumerate(numeric_features):
    sns.boxplot(x=num_feat, data=df, ax=axs[i])
    axs[i].set_title(f"Box Plot of {num_feat}", fontsize=14)
    axs[i].set_xlabel(num_feat, fontsize=12)
    axs[i].set_ylabel("Values", fontsize=12)

# plt.tight_layout()
plt.show()




sns.set(style="whitegrid")

for num_feat in numeric_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=num_feat, y='Sex', data=df)
    plt.title(f"Scatter Plot of {num_feat} vs Target Variable(Sex)", fontsize=14)
    plt.xlabel(num_feat, fontsize=12)
    plt.ylabel("Target Variable", fontsize=12)
    plt.show()



# Q-Q Plot
sns.set_style(style="darkgrid")
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))
axs = axs.flat

# Loop through each numerical feature and create a Q-Q plot
for i, num_feat in enumerate(numeric_features):
    # Generate Q-Q plot
    stats.probplot(df[num_feat], dist="norm", plot=axs[i])
    
    # Add reference line
    axs[i].plot(axs[i].get_lines()[1].get_xdata(), axs[i].get_lines()[1].get_ydata(), color='r', linestyle='--')
    axs[i].set_title(f"Q-Q Plot of {num_feat}", fontsize=12, fontweight='bold', color="black")
    axs[i].set_xlabel("Theoretical Quantiles", fontsize=10)
    axs[i].set_ylabel("Ordered Values", fontsize=10)

fig.suptitle("Q-Q Plots", fontsize=14, fontweight="bold", color="darkblue")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()



# ## Pie Plot


plt.style.use("ggplot")
category_data = df[categoric_features[0]]
category_counts = Counter(category_data)

plt.figure(figsize=(8, 6))
plt.pie(x=list(category_counts.values()), 
        labels=list(category_counts.keys()), 
        colors=[color for color in sns.color_palette(palette='bright', n_colors=3)],
        shadow=True, 
        wedgeprops={'edgecolor': 'black'}, 
        textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'black'}, 
        autopct="%.1f%%", 
        startangle=90, 
        explode=[0.1] * len(category_counts)) 

plt.title(categoric_features[0], fontsize=14, fontweight='bold', color='black')
plt.legend(loc="upper right")
plt.axis("equal")
plt.show()



# ## Correlation Matrix
corr_matrix = df[numeric_features].corr(method="spearman")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(corr_matrix, 
            cmap="coolwarm", 
            annot=True, 
            fmt=".2f",  
            annot_kws={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'}, 
            linewidths=1, 
            linecolor='black',
            square=True, 
            mask=mask, 
            ax=ax)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
ax.set_title("Advanced Correlation Matrix", fontsize=14, fontweight="bold", color="black", loc='left', pad=20)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
for i in range(len(corr_matrix)):
    ax.axhline(i, color='black', lw=1)
    ax.axvline(i, color='black', lw=1)

fig.tight_layout()
plt.show()



# ## Feature Processing
test = pd.read_csv('playground-series-s4e4/test.csv')
test.head()


ordinal = OrdinalEncoder(dtype='float')


train[categoric_features] = ordinal.fit_transform(train[categoric_features])
test[categoric_features] = ordinal.transform(test[categoric_features])


# ## Model Training
test = test.drop(columns=['id'], axis=1)
train = train.drop(columns=['id'], axis=1)


y = train.Rings
X = train.drop(['Rings'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=21)
xgb_model = XGBRegressor(n_estimators=1000, early_stopping_rounds=10, learning_rate=0.07, max_depth=7)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


# ## Predictions
pred = xgb_model.predict(X_val)
pred
print("The RMSLE score is: ", mean_squared_log_error(pred, y_val)**0.5)
predictions = xgb_model.predict(test)


df_pred = pd.DataFrame({'Rings': pred.round().astype(int)})
df_vald = pd.DataFrame({'Rings': y_val})
plt.figure(figsize=(8, 4))
sns.countplot(x='Rings', data=df_pred, color='blue', alpha=0.5, label='Predicted Rings')
sns.countplot(x='Rings', data=df_vald, color='red', alpha=0.5, label='Actual Rings')
plt.title("Comparison of the Number of Predicted and Actual Values of Rings")
plt.xlabel('Rings')
plt.ylabel('Count')
plt.legend()
plt.show()


# ## Submission
submission = pd.read_csv('playground-series-s4e4/sample_submission.csv')

submission['Rings'] = predictions
submission.to_csv('submission.csv', index=False)





