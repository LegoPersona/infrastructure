// Creates vector search indexes for all attribute collections.
// Run with: Get-Content create_vector_indexes.js | docker exec -i mongod mongosh -u admin -p admin --authenticationDatabase admin legopersona

const COLLECTIONS = ['beard', 'eyebrows', 'eyes', 'glasses', 'hair', 'nose', 'pants', 'shirt'];

for (const name of COLLECTIONS) {
  try {
    db.getCollection(name).createSearchIndex(
      'embedding_index',
      'vectorSearch',
      {
        fields: [{
          type: 'vector',
          path: 'embedding',
          numDimensions: 384,
          similarity: 'cosine',
        }],
      },
    );
    print(`[${name}] index created.`);
  } catch (e) {
    print(`[${name}] ${e.message}`);
  }
}
