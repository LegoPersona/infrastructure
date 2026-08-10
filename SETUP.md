# Project Setup

Short step-by-step to get everything running locally.

## 1. Start the database

```bash
cd infrastructure/mongo-search-setup
docker compose up -d
```

This creates the `search-community` network and starts MongoDB + Atlas Search.

## 2. Configure secrets

Open `infrastructure/docker-compose.apps.yml` and fill in any placeholder values (e.g. `HF_TOKEN`) before starting the apps.

## 3. Start the application services

```bash
cd infrastructure
docker compose -f docker-compose.apps.yml up -d --build
```

## 4. Run the data loaders

```bash
cd infrastructure/data-loaders
pip install -r requirements.txt

python3 embed_and_insert.py ../../lego-service/templates -o insert_commands.js

docker exec -i mongod mongosh -u admin -p admin --authenticationDatabase admin legopersona < insert_commands.js

docker exec -i mongod mongosh -u admin -p admin --authenticationDatabase admin legopersona < create_vector_indexes.js
```

Done. The database is seeded and all services are running.
