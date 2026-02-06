IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'catalog')
BEGIN
    EXEC('CREATE SCHEMA [catalog]')
END
GO
