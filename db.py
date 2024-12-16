import json
from neo4j import GraphDatabase
from config import *
# Function to reset the database by deleting all nodes and relationships
def reset_database():
    with driver.session() as session:
        session.run("""
            MATCH (n)
            DETACH DELETE n
        """)

# Function to save predictions into Neo4j
def save_to_neo4j_with_img_path(predictions:list):
    with driver.session() as session:
        # Lặp qua từng prediction
        for i in range(len(predictions)):
            # Tạo node Image (tránh trùng lặp với MERGE)
            img_path = predictions[i]['image_path']
            session.run(
                """
                MERGE (Image:Image {title: 'image', url: $img_path})
                """,
                img_path = img_path
            )
            for j in range(len(predictions[i]['predictions'])):
                # Truy xuất dữ liệu từ prediction
                subject_class = predictions[i]['predictions'][j]['subject']['class']
                relation_class = predictions[i]['predictions'][j]['relation']['class']
                object_class = predictions[i]['predictions'][j]['object']['class']

                # Tạo các node và relationship với thuộc tính duy nhất
                session.run(
                    """
                    MATCH (Image:Image {url: $img_path})
                    MERGE (Subject:Subject {name: $subject_class})
                    MERGE (Object:Object {name: $object_class})
                    CREATE (Relation:Relation {name: $relation_class, 
                                            subject_name: $subject_class, 
                                            object_name: $object_class})
                    MERGE (Subject)-[:Belong_to]->(Image)
                    CREATE (Subject)-[:HAS_RELATION]->(Relation)
                    CREATE (Relation)-[:CONNECTED_TO]->(Object)
                    """,
                    subject_class=subject_class,
                    object_class=object_class,
                    relation_class=relation_class,
                    img_path=img_path
                )

def query_images_for_multiple_subject_relation_object(criteria_list):
    all_images = []

    with driver.session() as session:
        for criteria in criteria_list:
            subject_name = criteria.get('subject', {}).get('class', None)
            relation_name = criteria.get('relation', {}).get('class', None)
            object_name = criteria.get('object', {}).get('class', None)

            # Kiểm tra nếu các giá trị subject, relation, object không phải là None
            if subject_name and relation_name and object_name:
                result = session.run("""
                    MATCH (s:Subject {name: $subject_name})-[:HAS_RELATION]->(r:Relation {name: $relation_name})-[:CONNECTED_TO]->(o:Object {name: $object_name})
                    MATCH (i:Image)<-[:Belong_to]-(s)
                    RETURN DISTINCT i.url AS Image
                """, subject_name=subject_name, relation_name=relation_name, object_name=object_name)

                # Thêm kết quả vào danh sách tất cả các ảnh
                all_images.extend([record['Image'] for record in result])

    return all_images



