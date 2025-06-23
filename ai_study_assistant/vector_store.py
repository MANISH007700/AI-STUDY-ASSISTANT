import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from pymilvus.exceptions import ConnectionNotExistException

class StudyMaterialsStore:
    def __init__(
        self,
        collection: str = "study_materials",
        dimension: int = 1536,
        uri: str = None,
        token: Optional[str] = None,
    ):
        self.collection_name = collection
        self.dimension = dimension
        self.workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if uri is None:
            unique_id = str(uuid.uuid4())[:8]
            timestamp = str(int(time.time()))[-6:]
            tmp_dir = os.path.join(self.workspace_root, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            self.uri = os.path.join(tmp_dir, f"sa_{unique_id}_{timestamp}.db")
        else:
            self.uri = uri
        self.token = token
        self.connection_alias = f"sa_{uuid.uuid4().hex[:6]}"
        self.cleanup_old_dbs()
        if "tmp" in self.uri or self.uri.startswith("./"):
            os.makedirs(os.path.dirname(os.path.abspath(self.uri)), exist_ok=True)
        self._connect()
        self._setup_collection()

    def cleanup_old_dbs(self):
        tmp_dir = os.path.join(self.workspace_root, "tmp")
        if os.path.exists(tmp_dir):
            current_time = time.time()
            for db_file in os.listdir(tmp_dir):
                if db_file.endswith(".db"):
                    db_path = os.path.join(tmp_dir, db_file)
                    file_age = current_time - os.path.getmtime(db_path)
                    if file_age > 24 * 3600:
                        try:
                            os.remove(db_path)
                            print(f"Deleted old database: {db_path}")
                        except Exception as e:
                            print(f"Error deleting {db_path}: {e}")

    def _connect(self):
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                try:
                    connections.disconnect(alias=self.connection_alias)
                except:
                    pass
                connections.connect(
                    alias=self.connection_alias,
                    uri=self.uri,
                    token=self.token,
                    timeout=10,
                )
                utility.list_collections(using=self.connection_alias)
                print(f"Connected to Milvus with alias {self.connection_alias}")
                return
            except Exception as e:
                print(f"Connection error (attempt {retry_count+1}/{max_retries}): {e}")
                retry_count += 1
                time.sleep(1)
                if retry_count >= max_retries:
                    if "tmp" in self.uri or self.uri.startswith("./"):
                        unique_id = str(uuid.uuid4())[:8]
                        timestamp = str(int(time.time()))[-6:]
                        self.uri = os.path.join(self.workspace_root, f"tmp/sa_{unique_id}_{timestamp}.db")
                        print(f"Trying with new URI: {self.uri}")
                        os.makedirs(os.path.dirname(os.path.abspath(self.uri)), exist_ok=True)
                        retry_count = 0
                    else:
                        raise ConnectionNotExistException(
                            message=f"Failed to connect to Milvus after {max_retries} attempts: {str(e)}"
                        )

    def _setup_collection(self):
        try:
            if not connections.has_connection(self.connection_alias):
                self._connect()
            if not utility.has_collection(self.collection_name, using=self.connection_alias):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=10000),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                ]
                schema = CollectionSchema(fields=fields)
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using=self.connection_alias,
                )
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024},
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)
                self.collection.load()
            else:
                self.collection = Collection(
                    name=self.collection_name, using=self.connection_alias
                )
                self.collection.load()
        except Exception as e:
            print(f"Setup collection error: {e}")
            raise

    def store_vectors(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict] = None,
    ) -> List[int]:
        try:
            if metadata is None:
                metadata = [{}] * len(texts)
            metadata_json = [json.dumps(m) for m in metadata]
            entities = [texts, metadata_json, embeddings]
            insert_result = self.collection.insert(entities)
            self.collection.flush()
            return insert_result.primary_keys
        except ConnectionNotExistException:
            self._connect()
            self._setup_collection()
            return self.store_vectors(texts, embeddings, metadata)
        except Exception as e:
            print(f"Store vectors error: {e}")
            raise

    def search_vectors(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "metadata"],
            )
            return [
                {
                    "id": hit.id,
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata"),
                    "distance": hit.distance,
                }
                for hit in results[0]
            ]
        except ConnectionNotExistException:
            self._connect()
            self._setup_collection()
            return self.search_vectors(query_embedding, top_k)
        except Exception as e:
            print(f"Search vectors error: {e}")
            raise

    def drop(self):
        try:
            if utility.has_collection(self.collection_name, using=self.connection_alias):
                utility.drop_collection(self.collection_name, using=self.connection_alias)
                self._connect()
                self._setup_collection()
        except Exception as e:
            print(f"Drop collection error: {e}")
            raise