import logging
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from qdrant_client import QdrantClient, models


logger = logging.getLogger(__name__)


class QdrantGraphTraverser:
    """
    Выполняет ограниченный BFS по ребрам, записанным в payload Qdrant.

    Поддерживаются ``graph_edges``/``edges`` с ``target_id`` и простые поля
    ``related_ids``, ``prev_chunk_id`` и ``next_chunk_id``. Идентификатором
    узла может быть Qdrant point id либо ``metadata.node_id/chunk_id``.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        max_depth: int = 1,
        max_nodes: int = 6,
    ):
        self.client = client
        self.collection_name = collection_name
        self.max_depth = max(0, max_depth)
        self.max_nodes = max(0, max_nodes)

    def expand(self, seed_documents: Sequence[Document]) -> List[Document]:
        documents = list(seed_documents)
        if not documents or self.max_depth == 0 or self.max_nodes == 0:
            return documents

        seed_ids = self._seed_ids(documents)
        if not seed_ids:
            return documents

        visited = set(seed_ids)
        frontier = deque((node_id, 0) for node_id in seed_ids)
        expanded: List[Document] = []
        expanded_keys = {self._document_key(doc) for doc in documents}

        try:
            while frontier and len(expanded) < self.max_nodes:
                node_id, depth = frontier.popleft()
                if depth >= self.max_depth:
                    continue

                for point in self._resolve_points(node_id):
                    for target_id, relation in self._extract_edges(point.payload or {}):
                        if target_id in visited:
                            continue
                        visited.add(target_id)
                        target_points = self._resolve_points(target_id)
                        for target in target_points:
                            document = self._point_to_document(
                                target,
                                relation=relation,
                                hop=depth + 1,
                            )
                            if document:
                                key = self._document_key(document)
                                if key not in expanded_keys:
                                    expanded.append(document)
                                    expanded_keys.add(key)
                            resolved_id = self._point_node_id(target)
                            if resolved_id not in visited:
                                visited.add(resolved_id)
                            frontier.append((resolved_id, depth + 1))
                            if len(expanded) >= self.max_nodes:
                                break
                        if len(expanded) >= self.max_nodes:
                            break
                    if len(expanded) >= self.max_nodes:
                        break
        except Exception as exc:
            # Graph context is an optional enrichment and must not break RAG.
            logger.warning("Qdrant graph traversal failed: %s", exc)
            return documents

        return documents + expanded

    @staticmethod
    def _document_key(document: Document) -> str:
        metadata = document.metadata or {}
        return str(
            metadata.get("chunk_id")
            or metadata.get("node_id")
            or hash(document.page_content)
        )

    @staticmethod
    def _seed_ids(documents: Sequence[Document]) -> List[str]:
        result: List[str] = []
        for document in documents:
            metadata = document.metadata or {}
            for key in ("chunk_id", "node_id", "entity_id", "point_id"):
                value = metadata.get(key)
                if value is not None:
                    result.append(str(value))
            for key in ("graph_node_ids", "entity_ids"):
                values = metadata.get(key) or []
                if isinstance(values, (str, int)):
                    values = [values]
                result.extend(str(value) for value in values)
        return list(dict.fromkeys(result))

    @staticmethod
    def _is_point_id(value: Any) -> bool:
        if isinstance(value, int):
            return True
        try:
            uuid.UUID(str(value))
            return True
        except (ValueError, TypeError, AttributeError):
            return str(value).isdigit()

    def _resolve_points(self, node_id: Any) -> List[Any]:
        if self._is_point_id(node_id):
            try:
                point_id = int(node_id) if str(node_id).isdigit() else node_id
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_payload=True,
                    with_vectors=False,
                )
                if points:
                    return list(points)
            except Exception:
                logger.debug("Point id %s was not resolved directly", node_id)

        lookup_fields = (
            "metadata.node_id",
            "metadata.chunk_id",
            "node_id",
            "chunk_id",
        )
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                should=[
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=str(node_id)),
                    )
                    for field in lookup_fields
                ]
            ),
            limit=2,
            with_payload=True,
            with_vectors=False,
        )
        return list(points)

    @staticmethod
    def _extract_edges(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
        metadata = payload.get("metadata") or {}
        containers = (payload, metadata)
        edges: List[Tuple[str, str]] = []

        for container in containers:
            for key in ("graph_edges", "edges", "relations"):
                raw_edges = container.get(key) or []
                if isinstance(raw_edges, dict):
                    raw_edges = [raw_edges]
                for edge in raw_edges:
                    if isinstance(edge, (str, int)):
                        edges.append((str(edge), "related"))
                        continue
                    if not isinstance(edge, dict):
                        continue
                    target = (
                        edge.get("target_id")
                        or edge.get("target")
                        or edge.get("to")
                        or edge.get("node_id")
                    )
                    if target is not None:
                        relation = edge.get("relation") or edge.get("type") or "related"
                        edges.append((str(target), str(relation)))

            for key in ("related_ids", "neighbor_ids"):
                values = container.get(key) or []
                if isinstance(values, (str, int)):
                    values = [values]
                edges.extend((str(value), "related") for value in values)

            for key, relation in (
                ("prev_chunk_id", "previous_chunk"),
                ("next_chunk_id", "next_chunk"),
            ):
                value = container.get(key)
                if value:
                    edges.append((str(value), relation))

        return list(dict.fromkeys(edges))

    @staticmethod
    def _point_node_id(point: Any) -> str:
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}
        return str(
            metadata.get("node_id")
            or metadata.get("chunk_id")
            or payload.get("node_id")
            or payload.get("chunk_id")
            or point.id
        )

    @staticmethod
    def _point_to_document(
        point: Any, relation: str, hop: int
    ) -> Optional[Document]:
        payload = point.payload or {}
        text = (
            payload.get("page_content")
            or payload.get("text")
            or payload.get("content")
            or payload.get("description")
        )
        if not text and payload.get("name"):
            text = f"Узел графа: {payload['name']}"
        if not text:
            return None

        metadata = dict(payload.get("metadata") or {})
        for key in ("node_id", "chunk_id", "node_type", "source"):
            if key in payload and key not in metadata:
                metadata[key] = payload[key]
        metadata.update(
            {
                "graph_relation": relation,
                "graph_hop": hop,
                "retrieval_origin": "graph_traversal",
            }
        )
        return Document(page_content=str(text), metadata=metadata)


class GraphTraversalRetriever(BaseRetriever):
    """Добавляет найденных BFS соседей к результатам исходного retriever."""

    base_retriever: BaseRetriever
    traverser: QdrantGraphTraverser

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> List[Document]:
        documents = self.base_retriever.invoke(query)
        return self.traverser.expand(documents)
