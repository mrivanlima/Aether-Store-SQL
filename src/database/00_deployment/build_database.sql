/*
    Aether Store - Master Build Script
    ----------------------------------
    Run this script to rebuild the database from scratch.
*/
USE master;
GO

IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'AetherStore')
BEGIN
    CREATE DATABASE [AetherStore];
    PRINT '>>> Database AetherStore Created';
END
GO

USE AetherStore;
GO

PRINT '>>> Building Schemas...';
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'catalog')
BEGIN
    EXEC('CREATE SCHEMA [catalog]');
END
GO

PRINT '>>> Building Tables...';
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[catalog].[Products]') AND type in (N'U'))
BEGIN
    CREATE TABLE [catalog].[Products](
        [ProductId] [nvarchar](50) NOT NULL,
        [Title] [nvarchar](500) NOT NULL,
        [Category] [nvarchar](100) NULL,
        [Price] [decimal](10, 2) NULL,
        [Description] [nvarchar](max) NULL,
        [ProductVector] [vector](1536) NULL,
        CONSTRAINT [PK_Products] PRIMARY KEY CLUSTERED ([ProductId] ASC)
    );
END
GO

PRINT '>>> Build Complete. AetherStore is ready for vectors.';
