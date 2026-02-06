IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[catalog].[Products]') AND type in (N'U'))
BEGIN
    CREATE TABLE [catalog].[Products](
        [ProductId] [nvarchar](50) NOT NULL,
        [Title] [nvarchar](500) NOT NULL,
        [Category] [nvarchar](100) NULL,
        [Price] [decimal](10, 2) NULL,
        [Description] [nvarchar](max) NULL,
        [ProductVector] [vector](1536) NULL,
        CONSTRAINT [PK_Products] PRIMARY KEY CLUSTERED 
        (
            [ProductId] ASC
        )
    )
END
GO
