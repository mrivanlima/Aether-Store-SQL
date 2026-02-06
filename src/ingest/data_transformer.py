import pandas as pd
from loguru import logger

class AmazonDataTransformer:
    """
    Cleans and prepares Amazon Metadata for SQL Server 2025.
    """
    def transform_row(self, row):
        # 1. Combine title and description for better 'Semantic Context'
        combined_text = f"Product: {row.get('title', '')}. Description: {row.get('description', '')}"
        
        # 2. Basic cleaning (Senior touch: handle missing values)
        return {
            "ProductId": row.get('asin'),
            "Title": row.get('title', 'Unknown'),
            "Category": row.get('category', 'Uncategorized'),
            "Price": float(row.get('price', 0.0)) if row.get('price') else 0.0,
            "Description": combined_text[:4000], # SQL NVARCHAR(MAX) is fine, but let's keep vectors concise
            "RawText": combined_text
        }