# ── Build stage ───────────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["PgsToSrtPlus/PgsToSrtPlus.csproj", "PgsToSrtPlus/"]
RUN dotnet restore "PgsToSrtPlus/PgsToSrtPlus.csproj"
COPY . .
WORKDIR "/src/PgsToSrtPlus"
RUN dotnet publish "./PgsToSrtPlus.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

# Runtime stage
FROM mcr.microsoft.com/dotnet/runtime:10.0 AS final

# Install Python and pip dependencies for PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfontconfig1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        python3 \
        python3-pip \
    && python3 -m pip install --no-cache-dir --break-system-packages \
        paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
    && python3 -m pip install --no-cache-dir --break-system-packages \
        paddleocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy published .NET app (includes paddle_ocr_bridge.py via CopyToOutputDirectory)
COPY --from=build /app/publish .

# Copy fonts
COPY PgsToSrtPlus/fonts/ fonts/

ENTRYPOINT ["dotnet", "PgsToSrtPlus.dll"]
