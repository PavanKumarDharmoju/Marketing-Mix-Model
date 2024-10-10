import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_marketing_data(file_path):
    # Load the data
    data = pd.read_csv(file_path, delimiter=';')

    # Plot distribution of income
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Income'].dropna(), kde=True, bins=30)
    plt.title('Income Distribution')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.show()

    # Correlation heatmap of marketing spend
    marketing_spend_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    plt.figure(figsize=(10, 6))
    sns.heatmap(data[marketing_spend_columns].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Marketing Spend')
    plt.show()

    # Response rates by education level
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Education', y='Response', data=data)
    plt.title('Response Rate by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Response Rate')
    plt.show()

    # Recency vs Response
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Recency', y='Response', data=data)
    plt.title('Recency vs Response')
    plt.xlabel('Recency')
    plt.ylabel('Response')
    plt.show()

if __name__ == "__main__":
    visualize_marketing_data('marketing_campaign.csv')
