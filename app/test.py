import pandas as pd

df_tree = pd.read_csv('data/processed/df_tree.csv')

# Show the unique values
print("Unique values in 'PreferedOrderCat':")
print(df_tree['PreferedOrderCat'].unique())

# Check exact mapping used by ngroup
group_mapping = dict(zip(df_tree['PreferedOrderCat'].unique(), df_tree.groupby('PreferedOrderCat').ngroup()))
print("\nGroup Mapping (PreferedOrderCat):")
print(group_mapping)
