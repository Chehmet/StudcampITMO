Прошу прощения, это моя ошибка. Я полностью убрал все комментарии и пояснения из кода.

Вот стратегия, как быстрее всего набрать максимум баллов, и весь код без единого комментария.

### Стратегия для максимальных баллов при минимуме времени:

Самые "дорогие" части задания — это **Benchmarking (40 баллов)** и **Data Modeling/Storage (35 баллов)**. При этом бенчмаркинг зависит от уже работающих баз данных и запросов.

1.  **Сконцентрируйтесь на PostgreSQL (это даст вам основу для ~50+ баллов).** Это самая простая и понятная часть.
    *   Запустите скрипт `load_data_psql.py`. Если данные загрузились, у вас есть рабочая база.
    *   Напишите 3 SQL-запроса (`q1.sql`, `q2.sql`, `q3.sql`). Они относительно простые.
    *   Как только это готово, вы можете сразу делать бенчмаркинг для PostgreSQL.

2.  **Сделайте Бенчмаркинг для PostgreSQL (40 баллов).**
    *   Запустите каждый из 3-х SQL-запросов по 5 раз, замеряя время. Проще всего это сделать через `psql` с командой `\timing`.
    *   Сделайте скриншоты.
    *   Внесите результаты в таблицу в отчете. Это самая высокооплачиваемая и механическая часть работы.

3.  **Загрузите данные в MongoDB (часть из 35 баллов).**
    *   Запустите скрипт `load_data_mongodb.py`. Он должен отработать без проблем.
    *   Адаптируйте SQL-запросы под MongoDB. Это сложнее, но можно найти аналоги. Запросы `q1.js`, `q2.js`, `q3.js` уже готовы ниже.
    *   Проведите бенчмаркинг для MongoDB.

4.  **Neo4j — если останется время.** Настройка и загрузка графовой базы может занять больше времени. Если время поджимает, лучше иметь идеально сделанные PSQL и Mongo, чем три базы, сделанные наполовину.

Приоритет: **PSQL -> Бенчмаркинг PSQL -> Mongo -> Бенчмаркинг Mongo -> Neo4j -> Бенчмаркинг Neo4j -> Оформление отчета.**

---

### **Код без комментариев**

#### **1. Загрузка данных**

**`/scripts/load_data_psql.py`**
```python
import psycopg2
import os
import pandas as pd
from io import StringIO

DB_NAME = "bigdata_db"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

DATA_DIR = '../data/f01'

def setup_tables(cur):
    cmds = (
        "DROP TABLE IF EXISTS friends, client_first_purchase_date, messages, events, campaigns CASCADE;",
        """
        CREATE TABLE campaigns (
            id BIGINT, campaign_type VARCHAR(255), channel VARCHAR(255), topic VARCHAR(255),
            started_at TIMESTAMP, finished_at TIMESTAMP, total_count INT, ab_test BOOLEAN,
            warmup_mode BOOLEAN, hour_limit INT, subject_length INT, subject_with_personalization BOOLEAN,
            subject_with_deadline BOOLEAN, subject_with_emoji BOOLEAN, subject_with_bonuses BOOLEAN,
            subject_with_discount BOOLEAN, subject_with_saleout BOOLEAN, is_test BOOLEAN, position INT,
            PRIMARY KEY (id, campaign_type)
        );
        """,
        """
        CREATE TABLE events (
            event_time TIMESTAMP, event_type VARCHAR(255), product_id BIGINT, category_id BIGINT,
            category_code VARCHAR(255), brand VARCHAR(255), price FLOAT, user_id BIGINT, user_session UUID
        );
        """,
        """
        CREATE TABLE messages (
            campaign_id BIGINT, message_type VARCHAR(255), channel VARCHAR(255), client_id VARCHAR(255),
            email_provider VARCHAR(255), platform VARCHAR(255), stream VARCHAR(255), date DATE,
            sent_at TIMESTAMP, is_opened BOOLEAN, opened_first_time_at TIMESTAMP, opened_last_time_at TIMESTAMP,
            is_clicked BOOLEAN, clicked_first_time_at TIMESTAMP, clicked_last_time_at TIMESTAMP,
            is_unsubscribed BOOLEAN, unsubscribed_at TIMESTAMP, is_hard_bounced BOOLEAN, hard_bounced_at TIMESTAMP,
            is_soft_bounced BOOLEAN, soft_bounced_at TIMESTAMP, is_complained BOOLEAN, complained_at TIMESTAMP,
            is_blocked BOOLEAN, blocked_at TIMESTAMP, is_purchased BOOLEAN, purchased_at TIMESTAMP
        );
        """,
        """
        CREATE TABLE client_first_purchase_date (
            client_id VARCHAR(255) PRIMARY KEY, first_purchase_date DATE
        );
        """,
        """
        CREATE TABLE friends (
            user_id1 BIGINT, user_id2 BIGINT, PRIMARY KEY (user_id1, user_id2)
        );
        """
    )
    for cmd in cmds:
        cur.execute(cmd)

def psql_copy(cur, file_path, tbl):
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)
        cur.copy_expert(f"COPY {tbl} FROM STDIN WITH CSV", f)

def do_psql_load():
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        cur = conn.cursor()
        setup_tables(cur)
        psql_copy(cur, os.path.join(DATA_DIR, 'campaigns.csv'), 'campaigns')
        psql_copy(cur, os.path.join(DATA_DIR, 'events.csv'), 'events')
        psql_copy(cur, os.path.join(DATA_DIR, 'messages.csv'), 'messages')
        psql_copy(cur, os.path.join(DATA_DIR, 'client_first_purchase_date.csv'), 'client_first_purchase_date')
        psql_copy(cur, os.path.join(DATA_DIR, 'friends.csv'), 'friends')
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':
    do_psql_load()
```

**`/scripts/load_data_mongodb.py`**
```python
import pymongo
import pandas as pd
import os
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "bigdata_db_mongo"
DATA_DIR = '../data/f01'

def do_mongo_insert(client, fname, coll_name):
    db = client[DB_NAME]
    coll = db[coll_name]
    coll.drop()
    file_path = os.path.join(DATA_DIR, fname)
    for chunk in pd.read_csv(file_path, chunksize=10000, low_memory=False):
        coll.insert_many(chunk.to_dict("records"))

def run_mongo_load():
    client = MongoClient(MONGO_URI)
    do_mongo_insert(client, 'campaigns.csv', 'campaigns')
    do_mongo_insert(client, 'events.csv', 'events')
    do_mongo_insert(client, 'messages.csv', 'messages')
    do_mongo_insert(client, 'client_first_purchase_date.csv', 'purchases')
    do_mongo_insert(client, 'friends.csv', 'friends')
    db = client[DB_NAME]
    db.messages.create_index([("is_purchased", 1)])
    db.messages.create_index([("campaign_id", 1)])
    db.events.create_index([("user_id", 1)])
    db.friends.create_index([("user_id1", 1)])
    db.events.create_index([("category_code", "text")])
    client.close()

if __name__ == '__main__':
    run_mongo_load()
```

**`/scripts/load_data_graph.py`**```python
from neo4j import GraphDatabase
import pandas as pd
import os

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password"
DATA_DIR = '../data/f01'

class GraphLoader:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def _exec_query(self, query, params=None):
        with self._driver.session() as session:
            session.run(query, params)

    def load(self):
        self._exec_query("MATCH (n) DETACH DELETE n")
        self._exec_query("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
        self._exec_query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE")
        self._exec_query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Campaign) REQUIRE c.id IS UNIQUE")
        
        events_df = pd.read_csv(os.path.join(DATA_DIR, 'events.csv'), usecols=['user_id']).dropna().drop_duplicates()
        friends_df = pd.read_csv(os.path.join(DATA_DIR, 'friends.csv'))
        all_users = pd.concat([events_df['user_id'], friends_df['user_id1'], friends_df['user_id2']]).unique()
        for user_id in all_users:
             self._exec_query("MERGE (u:User {id: $id})", params={'id': int(user_id)})
        
        prod_df = pd.read_csv(os.path.join(DATA_DIR, 'events.csv'), usecols=['product_id', 'category_code']).dropna().drop_duplicates()
        for _, row in prod_df.iterrows():
            self._exec_query("MERGE (p:Product {id: $id, category_code: $cat_code})", params={'id': int(row['product_id']), 'cat_code': row['category_code']})
        
        camp_df = pd.read_csv(os.path.join(DATA_DIR, 'campaigns.csv'), usecols=['id', 'campaign_type']).dropna().drop_duplicates(subset=['id', 'campaign_type'])
        for _, row in camp_df.iterrows():
            self._exec_query("MERGE (c:Campaign {id: $id, type: $type})", params={'id': int(row['id']), 'type': row['campaign_type']})
        
        for _, row in friends_df.iterrows():
            self._exec_query("MATCH (u1:User {id: $uid1}), (u2:User {id: $uid2}) MERGE (u1)-[:FRIENDS_WITH]->(u2)", params={'uid1': int(row['user_id1']), 'uid2': int(row['user_id2'])})
        
        events_df_rel = pd.read_csv(os.path.join(DATA_DIR, 'events.csv'), usecols=['user_id', 'product_id', 'event_type']).dropna()
        for _, row in events_df_rel.iterrows():
            if row['event_type'] in ['view', 'cart', 'purchase']:
                rel_type = row['event_type'].upper()
                self._exec_query(f"MATCH (u:User {{id: $uid}}), (p:Product {{id: $pid}}) MERGE (u)-[:{rel_type}]->(p)", params={'uid': int(row['user_id']), 'pid': int(row['product_id'])})
        
        msg_df = pd.read_csv(os.path.join(DATA_DIR, 'messages.csv'), usecols=['client_id', 'campaign_id', 'message_type', 'is_purchased']).dropna()
        msg_df['user_id'] = msg_df['client_id'].str.replace('151591562', '').astype(int)
        for _, row in msg_df.iterrows():
            self._exec_query("MATCH (u:User {id: $uid}), (c:Campaign {id: $cid, type: $ctype}) MERGE (u)-[:RECEIVED_MESSAGE {purchased: $purchased}]->(c)", params={'uid': int(row['user_id']), 'cid': int(row['campaign_id']), 'ctype': row['message_type'], 'purchased': bool(row['is_purchased'])})

def run_graph_load():
    loader = GraphLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    loader.load()
    loader.close()

if __name__ == "__main__":
    run_graph_load()
```

---

### **2. Запросы для анализа**

**`/scripts/q1.sql`**
```sql
WITH purchasers AS (
    SELECT DISTINCT client_id FROM messages WHERE is_purchased = TRUE
)
SELECT DISTINCT f.user_id2
FROM friends f
JOIN purchasers p ON f.user_id1 = CAST(REPLACE(p.client_id, '151591562', '') AS BIGINT)
WHERE f.user_id2 NOT IN (
    SELECT CAST(REPLACE(client_id, '151591562', '') AS BIGINT) FROM purchasers
)
LIMIT 100;
```

**`/scripts/q1.js`**
```javascript
db.messages.aggregate([{$match:{is_purchased:true}},{$project:{_id:0,user_id:{$toInt:{$substr:["$client_id",9,-1]}}}},{$lookup:{from:"friends",localField:"user_id",foreignField:"user_id1",as:"friend_relations"}},{$unwind:"$friend_relations"},{$project:{friend_id:"$friend_relations.user_id2"}},{$lookup:{from:"messages",let:{fid:"$friend_id"},pipeline:[{$match:{$expr:{$and:[{$eq:[{$toInt:{$substr:["$client_id",9,-1]}},"$$fid"]},{$eq:["$is_purchased",true]}]}}}] ,as:"friend_purchases"}},{$match:{friend_purchases:{$size:0}}},{$group:{_id:"$friend_id"}},{$project:{friend_to_target:"$_id",_id:0}},{$limit:100}]);
```

**`/scripts/q1.cypher`**
```cypher
MATCH (p:User)-[:RECEIVED_MESSAGE {purchased: true}]->(:Campaign)
MATCH (p)-[:FRIENDS_WITH]->(friend:User)
WHERE NOT (friend)-[:RECEIVED_MESSAGE {purchased: true}]->(:Campaign)
RETURN DISTINCT friend.id
LIMIT 100
```

**`/scripts/q2.sql`**
```sql
WITH user_friends AS (
    SELECT user_id2 AS friend_id FROM friends WHERE user_id1 = 550067820
    UNION
    SELECT user_id1 AS friend_id FROM friends WHERE user_id2 = 550067820
),
friends_activity AS (
    SELECT e.product_id, COUNT(e.product_id) as act_count
    FROM events e JOIN user_friends uf ON e.user_id = uf.friend_id
    WHERE e.event_type IN ('view', 'purchase')
    GROUP BY e.product_id
),
user_activity AS (
    SELECT DISTINCT product_id FROM events WHERE user_id = 550067820
)
SELECT fa.product_id
FROM friends_activity fa LEFT JOIN user_activity ua ON fa.product_id = ua.product_id
WHERE ua.product_id IS NULL
ORDER BY fa.act_count DESC
LIMIT 10;
```

**`/scripts/q2.js`**
```javascript
db.friends.aggregate([{$match:{$or:[{user_id1:550067820},{user_id2:550067820}]}},{$project:{friend_id:{$cond:{if:{$eq:["$user_id1",550067820]},then:"$user_id2",else:"$user_id1"}}}},{$lookup:{from:"events",localField:"friend_id",foreignField:"user_id",as:"friend_events"}},{$unwind:"$friend_events"},{$match:{"friend_events.event_type":{$in:["view","purchase"]}}},{$group:{_id:"$friend_events.product_id",count:{$sum:1}}},{$sort:{count:-1}},{$lookup:{from:"events",let:{pid:"$_id"},pipeline:[{$match:{$expr:{$and:[{$eq:["$user_id",550067820]},{$eq:["$product_id","$$pid"]}]}}}],as:"user_viewed"}},{$match:{user_viewed:{$size:0}}},{$limit:10},{$project:{_id:0,product:"$_id"}}]);
```

**`/scripts/q2.cypher`**
```cypher
MATCH (u:User {id: 550067820})-[:FRIENDS_WITH]-(friend:User)
MATCH (friend)-[r:VIEW|:PURCHASE]->(p:Product)
WHERE NOT (u)-[:VIEW|:PURCHASE]->(p)
RETURN p.id, count(p) AS strength
ORDER BY strength DESC
LIMIT 10
```

**`/scripts/q3.sql`**
```sql
SELECT product_id, category_code, brand, price
FROM events
WHERE category_code ILIKE '%vacuum%'
GROUP BY product_id, category_code, brand, price
LIMIT 20;
```

**`/scripts/q3.js`**
```javascript
db.events.find({ $text: { $search: "vacuum" } }).limit(20);
```

**`/scripts/q3.cypher`**
```cypher
MATCH (p:Product)
WHERE p.category_code CONTAINS 'vacuum'
RETURN p.id, p.category_code
LIMIT 20
```
