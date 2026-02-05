-- =============================================
-- Aether-Store-SQL Database Schema
-- SQL Server 2025 with Vector Search Support
-- =============================================

USE master;
GO

-- Create Database
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'AetherStoreDB')
BEGIN
    CREATE DATABASE AetherStoreDB;
END
GO

USE AetherStoreDB;
GO

-- =============================================
-- Products Table with Vector Column
-- =============================================

IF OBJECT_ID('dbo.Products', 'U') IS NOT NULL
    DROP TABLE dbo.Products;
GO

CREATE TABLE dbo.Products
(
    ProductID INT PRIMARY KEY IDENTITY(1,1),
    SKU NVARCHAR(100) NOT NULL UNIQUE,
    ProductName NVARCHAR(500) NOT NULL,
    ProductDescription NVARCHAR(MAX),
    Category NVARCHAR(200),
    Brand NVARCHAR(200),
    Price DECIMAL(18, 2),
    StockQuantity INT DEFAULT 0,
    
    -- Vector embedding for semantic search (1536 dimensions for text-embedding-3-small)
    ProductEmbedding VECTOR(1536) NOT NULL,
    
    -- Metadata
    CreatedAt DATETIME2 DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2 DEFAULT GETUTCDATE(),
    IsActive BIT DEFAULT 1,
    
    INDEX IX_Products_Category NONCLUSTERED (Category),
    INDEX IX_Products_Brand NONCLUSTERED (Brand),
    INDEX IX_Products_IsActive NONCLUSTERED (IsActive)
);
GO

-- Create Vector Index for similarity search
CREATE INDEX IX_Products_VectorSearch 
ON dbo.Products(ProductEmbedding)
USING VECTOR;
GO

-- =============================================
-- Search History Table
-- =============================================

IF OBJECT_ID('dbo.SearchHistory', 'U') IS NOT NULL
    DROP TABLE dbo.SearchHistory;
GO

CREATE TABLE dbo.SearchHistory
(
    SearchID BIGINT PRIMARY KEY IDENTITY(1,1),
    SearchQuery NVARCHAR(1000) NOT NULL,
    QueryEmbedding VECTOR(1536),
    ResultCount INT,
    ExecutionTimeMs INT,
    SearchedAt DATETIME2 DEFAULT GETUTCDATE(),
    
    INDEX IX_SearchHistory_SearchedAt NONCLUSTERED (SearchedAt DESC)
);
GO

-- =============================================
-- Stored Procedure: Vector Similarity Search
-- =============================================

IF OBJECT_ID('dbo.usp_SearchProductsByVector', 'P') IS NOT NULL
    DROP PROCEDURE dbo.usp_SearchProductsByVector;
GO

CREATE PROCEDURE dbo.usp_SearchProductsByVector
    @QueryEmbedding VECTOR(1536),
    @TopK INT = 10,
    @MinScore FLOAT = 0.7
AS
BEGIN
    SET NOCOUNT ON;
    
    SELECT TOP (@TopK)
        p.ProductID,
        p.SKU,
        p.ProductName,
        p.ProductDescription,
        p.Category,
        p.Brand,
        p.Price,
        p.StockQuantity,
        VECTOR_DISTANCE('cosine', p.ProductEmbedding, @QueryEmbedding) AS SimilarityScore
    FROM dbo.Products p
    WHERE p.IsActive = 1
        AND VECTOR_DISTANCE('cosine', p.ProductEmbedding, @QueryEmbedding) >= @MinScore
    ORDER BY SimilarityScore DESC;
END
GO

PRINT 'Schema created successfully with vector search support!';
GO
