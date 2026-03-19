#!/bin/bash
set -e

MONGOSH="mongosh --host localhost:27017 --username admin --password admin --authenticationDatabase admin --quiet"

echo "⏳ Waiting for mongod to accept connections..."
until $MONGOSH --eval "db.adminCommand('ping').ok" > /dev/null 2>&1; do
  echo "   ...not ready yet, retrying in 3s"
  sleep 3
done

echo "✅ mongod is up. Initiating replica set if needed..."
$MONGOSH --eval "
  try {
    const s = rs.status();
    if (s.ok === 1) {
      print('Replica set already initialised, skipping.');
    }
  } catch(e) {
    print('Initiating replica set...');
    rs.initiate({ _id: 'rs0', members: [{ _id: 0, host: 'mongod:27017' }] });
    sleep(5000);
  }

  const adminDb = db.getSiblingDB('admin');
  const existing = adminDb.getUser('mongot');
  if (!existing) {
    adminDb.createUser({
      user: 'mongot',
      pwd: 'mongot',
      roles: [{ role: 'searchCoordinator', db: 'admin' }]
    });
    print('mongot user created.');
  } else {
    print('mongot user already exists.');
  }
"

echo "🎉 RS init complete."