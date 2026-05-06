# Data Loaders

Utilities for loading data into MongoDB with embeddings from the text-embedding service.

## Scripts

### `embed_and_insert.py`

Generates MongoDB insert commands with real embeddings for templated data.

**What it does:**
1. Scans a templates directory for `.ldr` files organized by category (subdirectories)
2. Converts filenames to human-readable descriptions (e.g., `black_french_beard.ldr` → "Black French Beard")
3. Fetches embeddings from the text-embedding service for each description
4. Generates MongoDB `insertMany` commands with embedded vectors
5. Each category becomes a separate MongoDB collection

**Prerequisites:**
- Python 3.8+
- Text-embedding service running on port 8003
- MongoDB instance ready to receive data

## Installation

```bash
cd infrastructure/data-loaders
pip install -r requirements.txt
```

## Usage

### Basic usage (outputs to stdout)

```bash
python3 embed_and_insert.py ../../lego-service/templates
```

### Save to file

```bash
python3 embed_and_insert.py ../../lego-service/templates \
  -o insert_commands.js
```

### Custom embedding service URL

```bash
python3 embed_and_insert.py ../../lego-service/templates \
  --embedding-url http://localhost:8003/embed \
  -o insert_commands.js
```

### Verbose mode (shows progress)

```bash
python3 embed_and_insert.py ../../lego-service/templates -v
```

### All options together

```bash
python3 embed_and_insert.py ../../lego-service/templates \
  --embedding-url http://embedding-service:8003/embed \
  --output insert_commands.js \
  --verbose
```

## Directory Structure

The script expects a templates directory organized like this:

```
templates/
├── beard/
│   ├── black_french_beard.ldr
│   ├── black_full_beard.ldr
│   └── ...
├── eyes/
│   ├── black_eyes.ldr
│   ├── blue_eyes.ldr
│   └── ...
├── hair/
│   └── ...
└── pants/
    └── ...
```

Each subdirectory becomes a MongoDB collection, and each `.ldr` file becomes a document:

```javascript
{
  "module_name": "black_eyes.ldr",
  "desc": "Black Eyes",
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

## Output Format

The script generates MongoDB shell commands ready to execute:

```javascript
db.beard.insertMany([
  {
    "module_name": "black_french_beard.ldr",
    "desc": "Black French Beard",
    "embedding": [...]
  },
  ...
])

db.eyes.insertMany([
  ...
])
```

## Creating Vector Search Indexes

Run once after data is loaded to enable `$vectorSearch` on each attribute collection:

```powershell
Get-Content create_vector_indexes.js | docker exec -i mongod mongosh -u admin -p admin --authenticationDatabase admin legopersona
```

---

## Inserting into MongoDB

### Option 1: Via MongoDB Shell (mongosh)

```bash
# Run the script and save to file
python3 embed_and_insert.py ../../lego-service/templates -o insert_commands.js

# Execute in MongoDB
mongosh -u admin -p admin --authenticationDatabase admin < insert_commands.js
```

### Option 2: Via Docker Compose (if using mongo-search-setup)

```bash
# Run the script and save to file
python3 embed_and_insert.py ../../lego-service/templates -o insert_commands.js

# Execute in the running MongoDB container
docker exec -i mongod mongosh -u admin -p admin \
  --authenticationDatabase admin < insert_commands.js
```

### Option 3: Stream directly (with bash)

```bash
python3 embed_and_insert.py ../../lego-service/templates | \
  docker exec -i mongod mongosh -u admin -p admin \
    --authenticationDatabase admin
```

## Troubleshooting

### Connection error to embedding service

```
Error: Failed to connect to embedding service at http://localhost:8003/embed
Make sure the embedding service is running on port 8003.
```

**Solution:** Start the text-embedding service first:

```bash
# From text-embedding-service directory
docker build -t text-embedding-service .
docker run -p 8003:8000 text-embedding-service
```

### MongoDB authentication error

```
Error: SASL step 1 failed
```

**Solution:** Verify MongoDB credentials in your `docker-compose.yml` or connection string. Default credentials in mongo-search-setup are:
- User: `admin`
- Password: `admin`

### Timeout errors

If the embedding service is slow, increase the timeout in the script or run locally with a faster machine.

## Development

The script is designed to be:
- **Reusable**: Works with any templates directory structure (not just lego-service)
- **Extensible**: Easy to add support for other file types or embedding strategies
- **Testable**: Separate functions for each operation

## Future Enhancements

- [ ] Support for batch processing with resume capability
- [ ] Direct MongoDB insertion (skip shell script generation)
- [ ] Support for different embedding models/services
- [ ] Data validation and duplicate detection
- [ ] Progress bars and ETA for large datasets
