from neo4j import GraphDatabase

# Kết nối tới Neo4j
uri = "bolt://localhost:7687"  # Địa chỉ và cổng mặc định
username = "neo4j"             # Tên người dùng
password = "12341234"          # Mật khẩu của bạn

driver = GraphDatabase.driver(uri, auth=(username, password))