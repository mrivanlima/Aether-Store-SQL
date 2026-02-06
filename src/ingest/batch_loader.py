import json
import os
import pandas as pd  # We use pandas for efficient CSV reading
import pyodbc
from loguru import logger
from src.ingest.vectorizer import VectorizerFactory 
from src.ingest.data_transformer import AmazonDataTransformer

class AetherBatchLoader:
    def __init__(self):
        self.vectorizer = VectorizerFactory()
        self.transformer = AmazonDataTransformer()
        
        self.conn_str = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USER')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
            "TrustServerCertificate=yes;"
        )

    def load_file(self, file_path: str):
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        # Determine file type
        _, ext = os.path.splitext(file_path)
        
        try:
            if ext.lower() == '.csv':
                logger.info(f"Detected CSV. Processing {file_path} in chunks...")
                # Chunking prevents Memory Overload on large Amazon files
                chunk_size = 1000 
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    # Convert DataFrame chunk to list of dicts
                    batch = chunk.to_dict(orient='records')
                    self._process_batch(batch)
                    
            elif ext.lower() == '.json':
                logger.info(f"Detected JSON. Loading {file_path}...")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self._process_batch(data)
            else:
                logger.error(f"Unsupported file format: {ext}")

        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            raise

    def _process_batch(self, batch_data):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            successful_inserts = 0

            for item in batch_data:
                try:
                    # Amazon CSVs usually have 'title' and 'category' columns
                    # We map them to our schema
                    clean_item = {
                        "ProductId": str(item.get("asin", item.get("ProductId", "UNKNOWN"))), # Fallback if no ID
                        "Title": str(item.get("title", "No Title")),
                        "Category": str(item.get("category", "Uncategorized")),
                        "Price": float(item.get("price", 0.0)),
                        "Description": str(item.get("description", "")),
                        # Construct RawText for embedding
                        "RawText": f"{item.get('title', '')} {item.get('description', '')} {item.get('category', '')}"
                    }
                    
                    # Generate simple ID if missing (common in some CSVs)
                    if clean_item["ProductId"] == "UNKNOWN":
                        # Skip or generate hash? For now, let's skip invalid rows
                        continue

                    # 1. Generate Vector
                    vector = self.vectorizer.generate_embedding(clean_item["RawText"])
                    
                    # 2. Dynamic Column Selection
                    if len(vector) == 1536:
                        target_col = "ProductVector"
                        cast_type = "VECTOR(1536)"
                    elif len(vector) == 384:
                        target_col = "ProductVector_Local"
                        cast_type = "VECTOR(384)"
                    else:
                        continue

                    # 3. Insert
                    sql = f"""
                        INSERT INTO catalog.Products (ProductId, Title, Category, Price, Description, {target_col})
                        VALUES (?, ?, ?, ?, ?, CAST(CAST(? AS NVARCHAR(MAX)) AS {cast_type}))
                    """

                    cursor.execute(sql, (
                        clean_item["ProductId"], 
                        clean_item["Title"], 
                        clean_item["Category"], 
                        clean_item["Price"], 
                        clean_item["Description"], 
                        str(vector) 
                    ))
                    successful_inserts += 1
                    
                except Exception as row_error:
                    if "23000" in str(row_error) or "Violation of PRIMARY KEY" in str(row_error):
                        continue # Ignore duplicates
                    # logger.warning(f"Skipping row: {row_error}") # Commented out to reduce noise
                    continue

            conn.commit()
            # Print progress without spamming logs
            print(f"Batch Processed: {successful_inserts} rows inserted.")

if __name__ == "__main__":
    loader = AetherBatchLoader()
    # Updated to point to the file inside the container
    loader.load_file("data/titles_to_categories.csv")