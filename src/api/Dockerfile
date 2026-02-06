# Use a lightweight Python image
FROM python:3.11-slim

# Step 1: Install basic networking and GPG tools
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Manually download and install the Microsoft GPG key
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg

# Step 3: Add the correct repository (Debian 12 is the current 'slim' base)
RUN echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list

# Step 4: Install the ODBC Driver and SQL Tools
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y \
    msodbcsql18 \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from the root (Docker context is now root)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

CMD ["python", "src/api/app.py"]