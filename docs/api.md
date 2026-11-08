# AgentOps AI — API Reference

> Base URL: `http://localhost:8000`

---

## Endpoints

### `POST /chat`
Invoke the AgentOps AI agent with a user message.

#### Request Body
```json
{
  "message": "What is the Starter plan price per year?",
  "session_id": null,
  "use_memory": false
}
```

| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `message` | `string` | ✅ | User query (1–4096 chars) |
| `session_id` | `string` \| `null` | ❌ | Resume an existing session (implies memory) |
| `use_memory` | `boolean` | ❌ | Enable MemorySaver for multi-turn state (default: `false`) |

#### Response `200 OK`
```json
{
  "response": "The Starter plan costs $49/month, so $588 per year.",
  "session_id": "a1b2c3d4-...",
  "latency_ms": 342.1
}
```

#### Error Responses
| Code | Reason |
| ---: | :--- |
| `422` | Prompt injection detected, PII guardrail violation, or empty message |
| `500` | Internal agent execution error |

#### Examples

**Stateless single query:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the Starter plan cost?"}'
```

**Start a stateful session:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the Starter plan cost?", "use_memory": true}'
# → returns a session_id

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "And the Professional plan?", "session_id": "<returned-session-id>"}'
```

---

### `POST /ingest`
Re-ingest Markdown documents from the knowledge base directory into ChromaDB.

#### Request Body
```json
{
  "knowledge_dir": null
}
```

| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `knowledge_dir` | `string` \| `null` | ❌ | Custom path to knowledge docs. Defaults to `data/knowledge/` |

#### Response `200 OK`
```json
{
  "status": "success",
  "chunks_indexed": 17,
  "message": "Successfully indexed 17 chunk(s) into the ChromaDB collection."
}
```

#### Error Responses
| Code | Reason |
| ---: | :--- |
| `404` | Specified `knowledge_dir` does not exist |
| `500` | Ingestion pipeline error |

#### Example
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

### `GET /health`
Check service health and configuration status.

#### Response `200 OK`
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "memory_enabled": true,
  "knowledge_dir_exists": true,
  "chroma_db_exists": true,
  "uptime_ms": 12345.67
}
```

#### Example
```bash
curl http://localhost:8000/health
```

---

## Interactive Docs

The FastAPI server auto-generates interactive Swagger UI and ReDoc documentation:

| UI | URL |
| :--- | :--- |
| **Swagger UI** | http://localhost:8000/docs |
| **ReDoc** | http://localhost:8000/redoc |
| **OpenAPI JSON** | http://localhost:8000/openapi.json |

---

## Running the Server

```bash
# Start with auto-reload (development)
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload

# Or via Python
python -m app.api.server
```
