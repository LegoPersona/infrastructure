# MongoDB Community + Atlas Search — Docker Compose Setup

## Quick start

```bash
docker compose up -d
```

Boots in this order:

1. **keyfile-init** — generates a secure keyfile + password file, fixes file ownership and permissions
2. **mongod** — starts MongoDB with auth + replica-set mode
3. **mongod-rs-init** — initialises `rs0` and creates the `mongot` service user
4. **mongot** — starts the Atlas Search engine, connected to mongod

## Ports

| Service | Port        | Purpose              |
|---------|-------------|----------------------|
| mongod  | 27017       | MongoDB wire protocol|
| mongot  | 8080        | Search HTTP API      |
| mongot  | 9946        | Metrics endpoint     |

## Credentials

| User  | Password | Role              |
|-------|----------|-------------------|
| admin | admin    | Root (MongoDB)    |
| mongot| mongot   | searchCoordinator |

## Useful commands

```bash
# Follow startup logs
docker compose logs -f

# Shell into MongoDB
docker exec -it mongod mongosh -u admin -p admin --authenticationDatabase admin

# Check replica-set status
docker exec -it mongod mongosh -u admin -p admin --authenticationDatabase admin --eval "rs.status()"

# Tear everything down (keeps data volume)
docker compose down

# Tear down AND delete all data
docker compose down -v
```

## File layout

```
.
├── docker-compose.yml
└── config/
    ├── mongod.conf   ← MongoDB config (auth, replication, ports)
    └── mongot.conf   ← Atlas Search config (connection, credentials)
```

## Vector search quick example

```js
// 1. Create collection
db.createCollection("docs")

// 2. Create the vector search index
db.docs.createSearchIndex(
  "vec_idx",
  "vectorSearch",
  { fields: [{ type: "vector", path: "embedding", numDimensions: 4, similarity: "cosine" }] }
)

// 3. Insert data
db.docs.insertMany([
  { name: "Alice",  embedding: [0.9, 0.1, 0.2, 0.1] },
  { name: "Bob",    embedding: [0.8, 0.2, 0.1, 0.3] },
  { name: "Carol",  embedding: [0.1, 0.9, 0.1, 0.2] },
  { name: "Dave",   embedding: [0.2, 0.8, 0.3, 0.1] },
  { name: "Eve",    embedding: [0.3, 0.1, 0.9, 0.2] },
  { name: "Frank",  embedding: [0.1, 0.3, 0.8, 0.1] },
  { name: "Grace",  embedding: [0.7, 0.3, 0.2, 0.4] },
  { name: "Heidi",  embedding: [0.1, 0.1, 0.2, 0.9] },
])

// 4. Query — vector closest to Alice/Bob/Grace cluster
db.docs.aggregate([
  { $vectorSearch: {
      index: "vec_idx",
      path: "embedding",
      queryVector: [0.85, 0.15, 0.15, 0.2],
      numCandidates: 8,
      limit: 3
  }},
  { $project: { name: 1, score: { $meta: "vectorSearchScore" } } }
])
// Expected top 3: Alice, Bob, Grace
```