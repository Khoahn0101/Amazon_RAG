import pandas as pd
import os

def preprocess_data():
    data_dir = r"D:\works\rag-chatbot\data"
    print("Loading data...")
    categories = pd.read_csv(os.path.join(data_dir, "amazon_categories.csv"))
    products = pd.read_csv(os.path.join(data_dir, "amazon_products.csv"))

    print("Initial shapes:", products.shape, categories.shape)
    
    # Remove rows without titles
    products = products[products['title'].notna()]
    
    # Remove rows with price <= 0
    products = products[products['price'] > 0]
    
    # Fill reviews
    products['reviews'] = products['reviews'].replace(0, "No review provided")
    
    # Fill listPrice
    products['listPrice'] = products['listPrice'].where(products['listPrice']!=0, products['price'])
    
    # Drop asin (first column) and some unwanted columns
    products = products.drop(products.columns[0], axis=1)
    products = products.drop(columns=['isBestSeller','boughtInLastMonth'])
    
    # Merge with categories
    merged_data = products.merge(categories, left_on='category_id', right_on='id', how='left')
    merged_data = merged_data.drop(columns=['category_id','id'], axis=1)
    
    print("Sampling data...")
    sampled_data = merged_data.sample(n=100000, random_state=42)
    print(f"Sampled shape: {sampled_data.shape}")
    
    # Save the cleaned data
    output_path = os.path.join(data_dir, "products_cleaned_ready_for_rag.csv")
    sampled_data.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    preprocess_data()
