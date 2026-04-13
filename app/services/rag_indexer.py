"""Unified RAG Indexer — batch-optimized vector indexing for all pipelines.

Handles schema, values, and file content across three vector indexes:
    schema_idx  — table & column entries for query planning and entity resolution
    values_idx  — distinct DB values for filter-value grounding
    chunks_idx  — file document chunks for chat-over-file RAG

Design for scale
────────────────
* Batch embedding in configurable chunks (default 256) to avoid OOM on large schemas
* Parallel table scanning via asyncio.gather for value indexing
* Content-hash deduplication: SHA-256 fingerprint stored per entry; skips unchanged content
* Smart file chunking: CSV/Excel keep header context per group of rows;
  DOCX respects heading boundaries; PDF works page-level; text uses
  sentence-boundary sliding window
* Progress events yielded for UI step tracking

Usage
─────
    indexer = RAGIndexer()

    # Re-index schema (incremental by default — skips unchanged tables)
    stats = await indexer.index_schema(graph)

    # Index a file upload
    stats = await indexer.index_file(file_bytes, "report.xlsx", session_id)

    # Full re-index (schema + values) — called by schema_watcher
    stats = await indexer.reindex_all(graph)

    # Drop file chunks when session expires
    await indexer.drop_file_index(session_id)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

from app.config import get_settings
from app.database import get_pool
from app.services.faiss_manager import batch_add_to_index, clear_index
from app.services.embedder import embed_texts, embed_single

logger = logging.getLogger(__name__)

# ── Tuning knobs ──────────────────────────────────────────────────────────────
_EMBED_BATCH      = 256   # Max items per embedding call
_VALUE_CONCURRENCY = 8    # Max parallel table scans when indexing values
_FILE_CHUNK_SIZE  = 1200  # Characters per file chunk
_FILE_CHUNK_OVERLAP = 200 # Overlap between consecutive file chunks
_CSV_ROWS_PER_CHUNK = 30  # Rows grouped together per chunk for CSV/Excel


@dataclass
class IndexStats:
    """Summary of a single indexing run."""
    index_name: str
    added: int = 0
    skipped: int = 0
    errors: int = 0
    batches: int = 0

    def __str__(self) -> str:
        return (
            f"{self.index_name}: added={self.added} "
            f"skipped={self.skipped} errors={self.errors}"
        )


class RAGIndexer:
    """Thread-safe, async RAG indexer.  Instantiate once at startup."""

    # ── Public entry points ───────────────────────────────────────────────────

    async def index_schema(
        self,
        graph,
        force: bool = False,
        expansions: dict | None = None,
    ) -> IndexStats:
        """Embed table + column entries into schema_idx.

        Parameters
        ──────────
        graph      SchemaGraph instance
        force      If True, clear and re-index everything regardless of hashes
        expansions Pre-generated semantic expansions dict (skips LLM call if provided)
        """
        stats = IndexStats("schema_idx")

        if force:
            await clear_index("schema_idx")
            logger.info("schema_idx cleared (force re-index)")

        if expansions is None:
            expansions = await self._load_or_generate_expansions(graph)

        texts, payloads, hashes = self._build_schema_corpus(graph, expansions)
        if not texts:
            logger.info("schema_idx: nothing to index")
            return stats

        # Fingerprint-based incremental update
        if not force:
            texts, payloads, hashes, skipped = await self._filter_unchanged(
                "schema_idx", texts, payloads, hashes
            )
            stats.skipped = skipped

        if not texts:
            logger.info(f"schema_idx: all {stats.skipped} entries unchanged — skipped")
            return stats

        added, errors = await self._batch_embed_store(
            "schema_idx", texts, payloads, mode="schema"
        )
        stats.added = added
        stats.errors = errors
        await self._save_hashes("schema_idx", hashes[:added])
        logger.info(f"index_schema complete: {stats}")
        return stats

    async def index_values(
        self,
        graph,
        tables: list[str] | None = None,
        force: bool = False,
    ) -> IndexStats:
        """Embed distinct DB values into values_idx.

        Parameters
        ──────────
        graph    SchemaGraph
        tables   Restrict to these tables (None = all tables)
        force    Clear and re-index if True
        """
        stats = IndexStats("values_idx")

        if force:
            await clear_index("values_idx")
            logger.info("values_idx cleared (force re-index)")

        target_tables = tables or list(graph.tables.keys())
        texts, payloads = await self._collect_values_parallel(graph, target_tables)

        if not texts:
            logger.info("values_idx: no indexable values found")
            return stats

        if not force:
            hashes = [_sha256(t) for t in texts]
            texts, payloads, hashes, skipped = await self._filter_unchanged(
                "values_idx", texts, payloads, hashes
            )
            stats.skipped = skipped

        if not texts:
            logger.info(f"values_idx: all {stats.skipped} values unchanged — skipped")
            return stats

        added, errors = await self._batch_embed_store(
            "values_idx", texts, payloads, mode="value"
        )
        stats.added = added
        stats.errors = errors
        logger.info(f"index_values complete: {stats}")
        return stats

    async def index_file(
        self,
        file_bytes: bytes,
        file_name: str,
        session_id: str,
    ) -> tuple[IndexStats, list[dict]]:
        """Chunk and embed a file into chunks_idx.

        Accepts raw bytes loaded from PostgreSQL — no filesystem access needed.
        Returns (stats, chunk_records) where chunk_records have:
            {"chunk_index": int, "chunk_text": str, "node_id": int}
        This list is stored in file_chunks table by the caller (file_pipeline).
        """
        import time as _t
        stats = IndexStats("chunks_idx")

        # Parse + chunk from bytes (no temp file)
        logger.info(f"[index_file] parsing '{file_name}' ({len(file_bytes)} bytes)")
        _t0 = _t.time()
        from app.services.file_parser import parse_file_bytes
        content = await parse_file_bytes(file_bytes, file_name)
        logger.info(
            f"[index_file] parsed: {len(content) if content else 0} chars "
            f"in {(_t.time() - _t0) * 1000:.0f}ms"
        )
        if not content or content.startswith("["):
            logger.warning(f"index_file: could not parse '{file_name}': {content[:80]}")
            stats.errors = 1
            return stats, []

        _t0 = _t.time()
        ext = os.path.splitext(file_name)[1].lower()
        chunks = _smart_chunk(content, file_name, ext)
        logger.info(
            f"[index_file] chunked into {len(chunks)} chunks "
            f"in {(_t.time() - _t0) * 1000:.0f}ms"
        )

        if not chunks:
            logger.warning(f"index_file: no chunks generated for '{file_name}'")
            return stats, []

        # NOTE: do NOT refit the encoder here. The encoder was trained at startup
        # on schema + values and calling fit_encoder() would wipe those learned
        # weights and run a synchronous SVD on the event loop — which hung the
        # server on file upload. The schema-trained encoder handles file chunks
        # well enough for RAG retrieval.

        # Embed in batches
        logger.info(f"[index_file] embedding {len(chunks)} chunks")
        _t0 = _t.time()
        chunk_texts = [c[:_FILE_CHUNK_SIZE] for c in chunks]
        embeddings = await self._embed_batched(chunk_texts, mode="passage")
        logger.info(
            f"[index_file] embedded {len(embeddings)} chunks "
            f"in {(_t.time() - _t0) * 1000:.0f}ms"
        )

        # Filter out any failed embeddings and build parallel arrays for batch insert
        good_embeddings: list[list[float]] = []
        good_payloads: list[dict] = []
        good_contents: list[str] = []
        good_chunk_indices: list[int] = []
        good_chunks: list[str] = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            if emb is None:
                stats.errors += 1
                continue
            good_embeddings.append(emb)
            good_payloads.append({
                "session_id": session_id,
                "file_name": file_name,
                "chunk_index": i,
                "chunk_text": chunk[:500],
            })
            good_contents.append(chunk[:_FILE_CHUNK_SIZE])
            good_chunk_indices.append(i)
            good_chunks.append(chunk)

        # Single batched insert — one transaction, numpy cluster assignment
        chunk_records: list[dict] = []
        if good_embeddings:
            logger.info(
                f"[index_file] batch_add_to_index: inserting {len(good_embeddings)} vectors into chunks_idx"
            )
            _t0 = _t.time()
            metadata_ids = await batch_add_to_index(
                "chunks_idx", good_embeddings, good_payloads, contents=good_contents
            )
            logger.info(
                f"[index_file] batch_add_to_index done in {(_t.time() - _t0) * 1000:.0f}ms"
            )
            for idx, node_id, chunk in zip(good_chunk_indices, metadata_ids, good_chunks):
                chunk_records.append({
                    "chunk_index": idx,
                    "chunk_text": chunk,
                    "node_id": node_id,
                })
                stats.added += 1

        logger.info(
            f"index_file '{file_name}': {stats.added} chunks indexed "
            f"(session={session_id[:8]}…)"
        )
        return stats, chunk_records

    async def reindex_all(
        self,
        graph,
        force: bool = False,
    ) -> list[IndexStats]:
        """Full re-index of schema + values. Used by schema_watcher."""
        # 1. Semantic expansions (shared by schema + value enrichment)
        expansions = await self._load_or_generate_expansions(graph)

        # 2. Fit encoder on full corpus before embedding
        schema_texts, _, _ = self._build_schema_corpus(graph, expansions)
        value_texts, _ = await self._collect_values_parallel(graph, list(graph.tables.keys()))
        full_corpus = schema_texts + value_texts
        if full_corpus:
            from app.services.sentence_encoder import fit_encoder
            fit_encoder(full_corpus)
            logger.info(f"reindex_all: encoder fit on {len(full_corpus)} corpus items")

        # 3. Index schema and values concurrently
        schema_stats, value_stats = await asyncio.gather(
            self.index_schema(graph, force=force, expansions=expansions),
            self.index_values(graph, force=force),
        )
        return [schema_stats, value_stats]

    async def drop_file_index(self, session_id: str) -> int:
        """Delete all chunks_idx entries for a session. Returns rows deleted."""
        settings = get_settings()
        pool = get_pool()
        schema = settings.APP_SCHEMA
        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    f"""DELETE FROM {schema}.embedding_metadata
                        WHERE index_name = 'chunks_idx'
                          AND payload->>'session_id' = $1""",
                    str(session_id),
                )
            deleted = int(result.split()[-1]) if result else 0
            logger.info(f"drop_file_index: deleted {deleted} chunks for session {session_id[:8]}…")
            return deleted
        except Exception as e:
            logger.warning(f"drop_file_index failed: {e}")
            return 0

    async def get_index_stats(self) -> dict:
        """Return current row counts for all three indexes."""
        settings = get_settings()
        pool = get_pool()
        schema = settings.APP_SCHEMA
        result = {}
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""SELECT index_name, COUNT(*) AS cnt
                        FROM {schema}.embedding_metadata
                        GROUP BY index_name"""
                )
                for r in rows:
                    result[r["index_name"]] = r["cnt"]
        except Exception as e:
            logger.warning(f"get_index_stats failed: {e}")
        return result

    # ── Schema corpus construction ────────────────────────────────────────────

    def _build_schema_corpus(
        self,
        graph,
        expansions: dict,
    ) -> tuple[list[str], list[dict], list[str]]:
        """Build enriched text corpus for schema_idx.

        Returns (texts, payloads, content_hashes).
        """
        texts: list[str] = []
        payloads: list[dict] = []
        hashes: list[str] = []

        # Reverse synonym map: actual_table → [aliases]
        reverse_syns: dict[str, list[str]] = {}
        for syn, actual in graph.synonyms.items():
            reverse_syns.setdefault(actual, []).append(syn)

        for tname, tmeta in graph.tables.items():
            col_names = ", ".join(tmeta.columns.keys())
            tname_readable = tname.replace("_", " ")
            all_cols = list(tmeta.columns.keys())

            # ── Table-level entry ──────────────────────────────────────────
            parts = [
                f"table {tname} {tname_readable} {tname_readable}: {col_names}"
            ]
            if tmeta.description:
                parts.append(tmeta.description)
            syns = reverse_syns.get(tname, [])
            if syns:
                parts.append(f"also known as: {', '.join(syns)}")
            texpansions = expansions.get(tname, [])
            if texpansions:
                flat = [str(t) for t in texpansions if not isinstance(t, (list, dict))]
                if flat:
                    parts.append(f"related: {', '.join(flat)}")

            text = ". ".join(parts)
            texts.append(text)
            payloads.append({
                "type": "table", "table": tname,
                "description": col_names,
                "columns": all_cols,
            })
            hashes.append(_sha256(text))

            # ── Column-level entries ──────────────────────────────────────
            siblings = [c.replace("_", " ") for c in all_cols]
            for cname, cinfo in tmeta.columns.items():
                cname_readable = cname.replace("_", " ")
                # Repeat table name 3× to boost disambiguation weight
                cparts = [
                    f"column {cname} {cname_readable} in table {tname} {tname_readable}",
                    f"{tname_readable} {tname_readable}",
                    f"type {cinfo['data_type']}",
                ]
                sib_ctx = [c for c in siblings if c != cname_readable][:8]
                if sib_ctx:
                    cparts.append(f"alongside {', '.join(sib_ctx)}")
                if tmeta.description:
                    cparts.append(tmeta.description)
                col_key = f"{tname}.{cname}"
                cexpansions = expansions.get(col_key, [])
                if cexpansions:
                    flat = [str(t) for t in cexpansions if not isinstance(t, (list, dict))]
                    if flat:
                        cparts.append(f"related: {', '.join(flat)}")

                ctext = ". ".join(cparts)
                texts.append(ctext)
                payloads.append({
                    "type": "column", "table": tname, "column": cname,
                    "description": f"{cname} ({cinfo['data_type']})",
                    "data_type": cinfo["data_type"],
                })
                hashes.append(_sha256(ctext))

        return texts, payloads, hashes

    # ── Value collection ──────────────────────────────────────────────────────

    async def _collect_values_parallel(
        self,
        graph,
        tables: list[str],
    ) -> tuple[list[str], list[dict]]:
        """Collect distinct string values from DB in parallel across tables."""
        settings = get_settings()
        db_schema = settings.POSTGRES_SCHEMA
        max_distinct = getattr(settings, "SCHEMA_MAX_DISTINCT_VALUES", 200)

        sem = asyncio.Semaphore(_VALUE_CONCURRENCY)

        async def _scan_table(tname: str) -> tuple[list[str], list[dict]]:
            async with sem:
                t_texts: list[str] = []
                t_payloads: list[dict] = []
                if tname not in graph.tables:
                    return t_texts, t_payloads
                pool = get_pool()
                try:
                    async with pool.acquire() as conn:
                        for cname, cinfo in graph.tables[tname].columns.items():
                            if cinfo["data_type"] not in (
                                "character varying", "text", "varchar", "USER-DEFINED"
                            ):
                                continue
                            try:
                                cnt = await conn.fetchval(
                                    f"SELECT COUNT(DISTINCT {cname}) FROM {db_schema}.{tname}"
                                )
                                if not cnt or cnt > max_distinct:
                                    continue
                                rows = await conn.fetch(
                                    f"SELECT DISTINCT {cname} FROM {db_schema}.{tname} "
                                    f"WHERE {cname} IS NOT NULL ORDER BY {cname} LIMIT $1",
                                    max_distinct,
                                )
                                for row in rows:
                                    val = row[cname]
                                    if val and isinstance(val, str) and val.strip():
                                        t_texts.append(val)
                                        t_payloads.append({
                                            "value": val,
                                            "table": tname,
                                            "column": cname,
                                        })
                            except Exception:
                                continue
                except Exception as e:
                    logger.debug(f"_scan_table({tname}) skipped: {e}")
                return t_texts, t_payloads

        results = await asyncio.gather(*[_scan_table(t) for t in tables])
        all_texts: list[str] = []
        all_payloads: list[dict] = []
        for t_texts, t_payloads in results:
            all_texts.extend(t_texts)
            all_payloads.extend(t_payloads)

        logger.info(
            f"_collect_values_parallel: {len(all_texts)} values "
            f"from {len(tables)} tables"
        )
        return all_texts, all_payloads

    # ── Semantic expansions ───────────────────────────────────────────────────

    async def _load_or_generate_expansions(self, graph) -> dict:
        """Return cached expansions or generate via LLM."""
        settings = get_settings()
        pool = get_pool()
        app_schema = settings.APP_SCHEMA
        try:
            async with pool.acquire() as conn:
                cached = await conn.fetchrow(
                    f"SELECT description FROM {app_schema}.schema_index "
                    f"WHERE table_name = '_semantic_expansions' AND column_name IS NULL"
                )
                if cached and cached["description"]:
                    return json.loads(cached["description"])
        except Exception:
            pass

        # Delegate to schema_seeder's LLM-based generator
        from app.services.schema_seeder import _generate_semantic_expansions
        return await _generate_semantic_expansions(graph)

    # ── Batched embedding + storage ───────────────────────────────────────────

    async def _batch_embed_store(
        self,
        index_name: str,
        texts: list[str],
        payloads: list[dict],
        mode: str = "schema",
    ) -> tuple[int, int]:
        """Embed texts in batches of _EMBED_BATCH and store into PostgreSQL.

        Returns (added_count, error_count).
        """
        added = 0
        errors = 0
        total = len(texts)

        for start in range(0, total, _EMBED_BATCH):
            batch_texts = texts[start: start + _EMBED_BATCH]
            batch_payloads = payloads[start: start + _EMBED_BATCH]
            try:
                embeddings = await embed_texts(batch_texts, mode=mode)
                await batch_add_to_index(
                    index_name, embeddings, batch_payloads, contents=batch_texts
                )
                added += len(batch_texts)
                logger.debug(
                    f"_batch_embed_store({index_name}): "
                    f"batch {start//  _EMBED_BATCH + 1} / {(total + _EMBED_BATCH - 1) // _EMBED_BATCH} "
                    f"({len(batch_texts)} items)"
                )
            except Exception as e:
                logger.error(
                    f"_batch_embed_store({index_name}) batch {start}-{start+_EMBED_BATCH}: {e}"
                )
                errors += len(batch_texts)

        return added, errors

    async def _embed_batched(
        self,
        texts: list[str],
        mode: str = "passage",
    ) -> list[list[float] | None]:
        """Embed in batches; returns None for failed items."""
        results: list[list[float] | None] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start: start + _EMBED_BATCH]
            try:
                embs = await embed_texts(batch, mode=mode)
                results.extend(embs)
            except Exception as e:
                logger.error(f"_embed_batched: batch failed: {e}")
                results.extend([None] * len(batch))
        return results

    # ── Hash-based incremental update ─────────────────────────────────────────

    async def _filter_unchanged(
        self,
        index_name: str,
        texts: list[str],
        payloads: list[dict],
        hashes: list[str],
    ) -> tuple[list[str], list[dict], list[str], int]:
        """Filter out entries whose content hash hasn't changed since last index.

        Returns (new_texts, new_payloads, new_hashes, skipped_count).
        """
        settings = get_settings()
        pool = get_pool()
        app_schema = settings.APP_SCHEMA

        try:
            async with pool.acquire() as conn:
                # Fetch stored hashes for this index
                rows = await conn.fetch(
                    f"""SELECT content_hash FROM {app_schema}.rag_index_checksums
                        WHERE index_name = $1""",
                    index_name,
                )
                known = {r["content_hash"] for r in rows}
        except Exception:
            # Table may not exist yet — treat all as new
            known = set()

        new_texts, new_payloads, new_hashes = [], [], []
        skipped = 0
        for t, p, h in zip(texts, payloads, hashes):
            if h in known:
                skipped += 1
            else:
                new_texts.append(t)
                new_payloads.append(p)
                new_hashes.append(h)

        return new_texts, new_payloads, new_hashes, skipped

    async def _save_hashes(self, index_name: str, hashes: list[str]) -> None:
        """Persist content hashes after successful indexing."""
        if not hashes:
            return
        settings = get_settings()
        pool = get_pool()
        app_schema = settings.APP_SCHEMA
        try:
            async with pool.acquire() as conn:
                await conn.executemany(
                    f"""INSERT INTO {app_schema}.rag_index_checksums
                        (index_name, content_hash)
                        VALUES ($1, $2)
                        ON CONFLICT (index_name, content_hash) DO NOTHING""",
                    [(index_name, h) for h in hashes],
                )
        except Exception as e:
            logger.debug(f"_save_hashes: {e} (table may not exist yet — OK)")

    # ── Encoder helpers ───────────────────────────────────────────────────────

    def _try_refit_encoder(self, new_chunks: list[str]) -> None:
        """Expand encoder vocabulary with new content chunks (best-effort)."""
        try:
            from app.services.sentence_encoder import fit_encoder, get_encoder
            enc = get_encoder()
            if enc.is_fitted:
                fit_encoder([c[:500] for c in new_chunks])
        except Exception as e:
            logger.debug(f"Encoder refit skipped: {e}")


# ── Singleton ─────────────────────────────────────────────────────────────────
_indexer: RAGIndexer | None = None


def get_indexer() -> RAGIndexer:
    """Return the process-wide RAGIndexer singleton."""
    global _indexer
    if _indexer is None:
        _indexer = RAGIndexer()
    return _indexer


# ── File chunking strategies ──────────────────────────────────────────────────

def _smart_chunk(content: str, file_name: str, ext: str) -> list[str]:
    """Choose chunking strategy based on file type for richer retrieval."""
    if ext in (".csv", ".xlsx", ".xls"):
        return _chunk_tabular(content)
    if ext == ".pdf":
        return _chunk_by_pages(content)
    if ext == ".docx":
        return _chunk_by_headings(content)
    # Default: sliding window with sentence boundaries
    return _chunk_sliding(content, _FILE_CHUNK_SIZE, _FILE_CHUNK_OVERLAP)


def _chunk_tabular(content: str) -> list[str]:
    """Chunk CSV/Excel content: group rows, keep header on every chunk."""
    lines = content.split("\n")
    if not lines:
        return []

    # First line is typically the header (CSV) or column names (Excel repr)
    header = lines[0]
    data_lines = [l for l in lines[1:] if l.strip()]
    if not data_lines:
        return [header] if header.strip() else []

    chunks = []
    for i in range(0, len(data_lines), _CSV_ROWS_PER_CHUNK):
        group = data_lines[i: i + _CSV_ROWS_PER_CHUNK]
        chunk = f"{header}\n" + "\n".join(group)
        chunks.append(chunk)
    return chunks


def _chunk_by_pages(content: str) -> list[str]:
    """Chunk PDF-extracted text at page boundaries (form-feed or double newline)."""
    # PyPDF2 uses \x0c (form feed) between pages; fallback to double newlines
    if "\x0c" in content:
        pages = content.split("\x0c")
    else:
        pages = re.split(r"\n{3,}", content)

    chunks = []
    for page in pages:
        page = page.strip()
        if not page:
            continue
        # If page is very long, split it further
        if len(page) > _FILE_CHUNK_SIZE * 2:
            chunks.extend(_chunk_sliding(page, _FILE_CHUNK_SIZE, _FILE_CHUNK_OVERLAP))
        else:
            chunks.append(page)
    return chunks


def _chunk_by_headings(content: str) -> list[str]:
    """Chunk DOCX content at heading boundaries (all-caps lines or short lines)."""
    lines = content.split("\n")
    chunks: list[str] = []
    current: list[str] = []

    def _is_heading(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        # All-caps short line or very short line (likely a heading)
        return (stripped.isupper() and len(stripped) < 60) or len(stripped) < 40

    for line in lines:
        if _is_heading(line) and current:
            text = "\n".join(current).strip()
            if text:
                if len(text) > _FILE_CHUNK_SIZE * 2:
                    chunks.extend(_chunk_sliding(text, _FILE_CHUNK_SIZE, _FILE_CHUNK_OVERLAP))
                else:
                    chunks.append(text)
            current = [line]
        else:
            current.append(line)

    if current:
        text = "\n".join(current).strip()
        if text:
            if len(text) > _FILE_CHUNK_SIZE * 2:
                chunks.extend(_chunk_sliding(text, _FILE_CHUNK_SIZE, _FILE_CHUNK_OVERLAP))
            else:
                chunks.append(text)

    return chunks or _chunk_sliding(content, _FILE_CHUNK_SIZE, _FILE_CHUNK_OVERLAP)


def _chunk_sliding(text: str, size: int, overlap: int) -> list[str]:
    """Sentence-boundary-aware sliding window chunking.

    Tries to break at sentence boundaries ('. ' / '\\n') to avoid mid-sentence cuts.
    """
    if not text:
        return []
    if overlap >= size:
        overlap = size // 2  # guard against bad config — prevents infinite loop
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + size, text_len)
        # Try to extend to nearest sentence boundary
        if end < text_len:
            boundary = text.rfind(". ", start, end + 80)
            if boundary > start + size // 2:
                end = boundary + 1  # include the period
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Stop once we've consumed the tail — don't loop on the trailing overlap window
        if end >= text_len:
            break
        next_start = end - overlap
        # Guarantee forward progress even in pathological cases
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks


# ── Utilities ─────────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ── DB migration helper ───────────────────────────────────────────────────────

async def ensure_checksums_table() -> None:
    """Create rag_index_checksums if it doesn't exist.

    Call once at application startup (e.g. from app/lifespan.py).
    """
    settings = get_settings()
    pool = get_pool()
    app_schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {app_schema}.rag_index_checksums (
                    id           SERIAL PRIMARY KEY,
                    index_name   VARCHAR(64)  NOT NULL,
                    content_hash VARCHAR(64)  NOT NULL,
                    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                    CONSTRAINT rag_checksums_uq UNIQUE (index_name, content_hash)
                )
            """)
        logger.info("rag_index_checksums table ready")
    except Exception as e:
        logger.warning(f"ensure_checksums_table: {e}")
