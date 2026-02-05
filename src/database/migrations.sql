-- =============================================
-- Aether-Store-SQL Database Migrations
-- Version Control for Schema Changes
-- =============================================

USE AetherStoreDB;
GO

-- Create Migrations Tracking Table
IF OBJECT_ID('dbo.SchemaMigrations', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.SchemaMigrations
    (
        MigrationID INT PRIMARY KEY IDENTITY(1,1),
        VersionNumber NVARCHAR(50) NOT NULL UNIQUE,
        Description NVARCHAR(500),
        AppliedAt DATETIME2 DEFAULT GETUTCDATE(),
        AppliedBy NVARCHAR(100) DEFAULT SYSTEM_USER
    );
    
    INSERT INTO dbo.SchemaMigrations (VersionNumber, Description)
    VALUES ('1.0.0', 'Initial schema with vector search support');
END
GO

PRINT 'Migration tracking initialized';
GO
