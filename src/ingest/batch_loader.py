import json
import os
import pyodbc
from loguru import logger
# --- CORRECT IMPORT ---
from src.ingest.vectorizer import ProductVectorizer 
from src.ingest.data_transformer import AmazonDataTransformer

class AetherBatchLoader:
    def __init__(self):
        # --- CORRECT INITIALIZATION ---
        self.vectorizer = ProductVectorizer()
        self.transformer = AmazonDataTransformer()
        
        # Connection String
        self.conn_str = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={os.getenv('SQL_SERVER')};"
            f"DATABASE={os.getenv('SQL_DATABASE')};"
            f"UID={os.getenv('SQL_USER')};"
            f"PWD={os.getenv('SQL_PASSWORD')};"
            "TrustServerCertificate=yes;"
        )

    def load_file(self, file_path: str = "data/sample.json"):
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} records from {file_path}")
            self._process_batch(data)
            
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            raise

    def _process_batch(self, batch_data):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            logger.info("Connected to SQL Server successfully.")
            
            successful_inserts = 0

            for item in batch_data:
                try:
                    clean_item = self.transformer.transform_row(item)
                    
                    # Generate vector
                    vector = self.vectorizer.generate_embedding(clean_item["RawText"])
                    
                    # Insert into SQL Server 2025
                    cursor.execute("""
                        INSERT INTO catalog.Products (ProductId, Title, Category, Price, Description, ProductVector)
                        VALUES (?, ?, ?, ?, ?, CAST(CAST(? AS NVARCHAR(MAX)) AS VECTOR(1536))) 
                    """, (
                        clean_item["ProductId"], 
                        clean_item["Title"], 
                        clean_item["Category"], 
                        clean_item["Price"], 
                        clean_item["Description"], 
                        str(vector) # Goes into the double cast
                    ))
                    successful_inserts += 1
                    
                except Exception as row_error:
                    logger.warning(f"Skipping row {item.get('asin', 'unknown')}: {row_error}")
                    continue

            conn.commit()
            logger.success(f"Pipeline Complete. Successfully inserted {successful_inserts} vectors.")

if __name__ == "__main__":
    loader = AetherBatchLoader()
    loader.load_file()