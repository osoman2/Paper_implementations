"""
Skill Team: 68-Agent × 286-Skill Engineering Team (Claude-native)

Based on: AlphaSignal signal · 2026-09-03
"Anthropic hackathon winner open-sources a 68-agent, 286-skill engineering team built on Claude"
  - 2,541 AlphaSignal signal likes — highest engagement of the 2026-09-03 digest
  - Open-source multi-agent engineering team: 286 skills across 68 specialized agents
  - Demonstrates that skill composition + agent coordination >> single large model

Score: 9/10 — open-source, Claude-native, highest engagement signal of the day,
directly implementable architecture for production multi-agent engineering systems.

Key finding: Specialized agent networks with explicit skill composition outperform
monolithic models at complex, multi-step engineering tasks by decomposing work
into composable, verifiable units.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Agent role taxonomy (68 agents across 6 role families)
# ---------------------------------------------------------------------------

AGENT_ROLES: list[dict] = [
    # ── Planning (8 agents) ────────────────────────────────────────────────
    {"id": "planner_arch",    "name": "Architecture Planner",   "family": "Planning",     "skills": 12, "color": "#3498db",
     "desc": "Designs high-level system architecture, selects tech stack, identifies service boundaries."},
    {"id": "planner_sprint",  "name": "Sprint Planner",         "family": "Planning",     "skills": 8,  "color": "#3498db",
     "desc": "Breaks epics into sprint tasks, estimates effort, flags dependencies and blockers."},
    {"id": "planner_dep",     "name": "Dependency Analyst",     "family": "Planning",     "skills": 6,  "color": "#3498db",
     "desc": "Maps library and service dependencies, flags version conflicts, suggests upgrades."},
    {"id": "planner_risk",    "name": "Risk Assessor",          "family": "Planning",     "skills": 5,  "color": "#3498db",
     "desc": "Evaluates technical risk, identifies single points of failure, prioritizes mitigations."},
    {"id": "planner_api",     "name": "API Contract Designer",  "family": "Planning",     "skills": 7,  "color": "#3498db",
     "desc": "Specifies REST/gRPC contracts, OpenAPI schemas, versioning strategy."},
    {"id": "planner_data",    "name": "Data Model Planner",     "family": "Planning",     "skills": 6,  "color": "#3498db",
     "desc": "Designs relational/NoSQL schemas, relationships, indexing strategy."},
    {"id": "planner_infra",   "name": "Infrastructure Planner", "family": "Planning",     "skills": 7,  "color": "#3498db",
     "desc": "Plans cloud resources, IaC structure, scaling policies, cost estimates."},
    {"id": "planner_perf",    "name": "Performance Planner",    "family": "Planning",     "skills": 5,  "color": "#3498db",
     "desc": "Sets SLA targets, identifies hot paths, plans load testing strategy."},

    # ── Development (20 agents) ────────────────────────────────────────────
    {"id": "dev_backend",     "name": "Backend Engineer",       "family": "Development",  "skills": 18, "color": "#2ecc71",
     "desc": "Implements server-side logic, REST/gRPC services, business rules."},
    {"id": "dev_frontend",    "name": "Frontend Engineer",      "family": "Development",  "skills": 14, "color": "#2ecc71",
     "desc": "Builds React/Vue components, handles state management and routing."},
    {"id": "dev_fullstack",   "name": "Full-Stack Engineer",    "family": "Development",  "skills": 16, "color": "#2ecc71",
     "desc": "Bridges frontend and backend, owns feature delivery end-to-end."},
    {"id": "dev_api",         "name": "API Engineer",           "family": "Development",  "skills": 10, "color": "#2ecc71",
     "desc": "Implements API endpoints, middleware, authentication, rate limiting."},
    {"id": "dev_db",          "name": "Database Engineer",      "family": "Development",  "skills": 12, "color": "#2ecc71",
     "desc": "Writes migrations, stored procedures, query optimisation, ORM models."},
    {"id": "dev_ml",          "name": "ML Engineer",            "family": "Development",  "skills": 15, "color": "#2ecc71",
     "desc": "Implements model training pipelines, inference serving, feature stores."},
    {"id": "dev_data",        "name": "Data Pipeline Engineer", "family": "Development",  "skills": 11, "color": "#2ecc71",
     "desc": "Builds ETL/ELT pipelines, streaming ingestion, data quality checks."},
    {"id": "dev_devops",      "name": "DevOps Engineer",        "family": "Development",  "skills": 13, "color": "#2ecc71",
     "desc": "Writes CI/CD pipelines, Dockerfiles, Helm charts, Terraform modules."},
    {"id": "dev_mobile",      "name": "Mobile Engineer",        "family": "Development",  "skills": 10, "color": "#2ecc71",
     "desc": "Implements iOS/Android native or React Native features."},
    {"id": "dev_embed",       "name": "Embeddings Engineer",    "family": "Development",  "skills": 9,  "color": "#2ecc71",
     "desc": "Implements vector indexing, embedding generation, semantic search pipelines."},
    {"id": "dev_rag",         "name": "RAG Engineer",           "family": "Development",  "skills": 12, "color": "#2ecc71",
     "desc": "Builds retrieval-augmented generation pipelines, chunking, re-ranking."},
    {"id": "dev_agent",       "name": "Agent Framework Engineer", "family": "Development","skills": 11, "color": "#2ecc71",
     "desc": "Implements tool-use, memory, planning loops for AI agents."},
    {"id": "dev_cli",         "name": "CLI Engineer",           "family": "Development",  "skills": 7,  "color": "#2ecc71",
     "desc": "Builds command-line interfaces, shell integrations, scripting utilities."},
    {"id": "dev_auth",        "name": "Auth Engineer",          "family": "Development",  "skills": 9,  "color": "#2ecc71",
     "desc": "Implements OAuth2, SAML, JWT, RBAC, MFA, session management."},
    {"id": "dev_cache",       "name": "Caching Engineer",       "family": "Development",  "skills": 7,  "color": "#2ecc71",
     "desc": "Implements Redis/Memcached strategies, cache invalidation, CDN rules."},
    {"id": "dev_search",      "name": "Search Engineer",        "family": "Development",  "skills": 8,  "color": "#2ecc71",
     "desc": "Builds Elasticsearch/Solr indices, ranking functions, hybrid BM25+vector search."},
    {"id": "dev_realtime",    "name": "Real-Time Engineer",     "family": "Development",  "skills": 8,  "color": "#2ecc71",
     "desc": "Implements WebSocket, SSE, message brokers, event-driven patterns."},
    {"id": "dev_proto",       "name": "Prototype Engineer",     "family": "Development",  "skills": 6,  "color": "#2ecc71",
     "desc": "Rapid prototyping, throwaway code for hypothesis validation."},
    {"id": "dev_refactor",    "name": "Refactoring Specialist", "family": "Development",  "skills": 8,  "color": "#2ecc71",
     "desc": "Applies design patterns, reduces coupling, improves code health metrics."},
    {"id": "dev_osint",       "name": "Open-Source Integrator", "family": "Development",  "skills": 7,  "color": "#2ecc71",
     "desc": "Evaluates and wires open-source libraries into the codebase."},

    # ── Quality (15 agents) ───────────────────────────────────────────────
    {"id": "qa_unit",         "name": "Unit Test Engineer",     "family": "Quality",      "skills": 10, "color": "#e74c3c",
     "desc": "Writes isolated unit tests, mocks, stubs; targets 90%+ coverage."},
    {"id": "qa_integration",  "name": "Integration Tester",     "family": "Quality",      "skills": 10, "color": "#e74c3c",
     "desc": "Tests service contracts, database interactions, API end-to-end flows."},
    {"id": "qa_e2e",          "name": "E2E Test Engineer",      "family": "Quality",      "skills": 9,  "color": "#e74c3c",
     "desc": "Playwright/Cypress browser automation for critical user journeys."},
    {"id": "qa_perf",         "name": "Performance Tester",     "family": "Quality",      "skills": 8,  "color": "#e74c3c",
     "desc": "k6/Locust load tests, latency profiling, bottleneck identification."},
    {"id": "qa_security",     "name": "Security Auditor",       "family": "Quality",      "skills": 11, "color": "#e74c3c",
     "desc": "SAST/DAST scans, OWASP top-10 checks, secrets detection."},
    {"id": "qa_regression",   "name": "Regression Tester",      "family": "Quality",      "skills": 7,  "color": "#e74c3c",
     "desc": "Maintains regression suite, flags unintended side-effects of changes."},
    {"id": "qa_smoke",        "name": "Smoke Test Engineer",    "family": "Quality",      "skills": 5,  "color": "#e74c3c",
     "desc": "Quick sanity checks on every build: critical paths and core invariants."},
    {"id": "qa_contract",     "name": "Contract Tester",        "family": "Quality",      "skills": 7,  "color": "#e74c3c",
     "desc": "Pact-style consumer-driven contract tests for service boundaries."},
    {"id": "qa_chaos",        "name": "Chaos Engineer",         "family": "Quality",      "skills": 8,  "color": "#e74c3c",
     "desc": "Injects failures (network partitions, pod kills) to validate resilience."},
    {"id": "qa_accessibility","name": "Accessibility Auditor",  "family": "Quality",      "skills": 6,  "color": "#e74c3c",
     "desc": "WCAG 2.2 compliance, screen-reader testing, colour-contrast checks."},
    {"id": "qa_data",         "name": "Data Quality Engineer",  "family": "Quality",      "skills": 7,  "color": "#e74c3c",
     "desc": "Validates data schemas, null rates, referential integrity, drift."},
    {"id": "qa_mutation",     "name": "Mutation Tester",        "family": "Quality",      "skills": 6,  "color": "#e74c3c",
     "desc": "Runs mutation testing to evaluate test suite effectiveness."},
    {"id": "qa_fuzz",         "name": "Fuzzing Engineer",       "family": "Quality",      "skills": 7,  "color": "#e74c3c",
     "desc": "AFL/libFuzzer or LLM-guided fuzzing for parser and API robustness."},
    {"id": "qa_benchmark",    "name": "Benchmark Engineer",     "family": "Quality",      "skills": 8,  "color": "#e74c3c",
     "desc": "Designs repeatable micro- and macro-benchmarks, tracks regressions."},
    {"id": "qa_eval",         "name": "LLM Eval Engineer",      "family": "Quality",      "skills": 9,  "color": "#e74c3c",
     "desc": "Designs and runs evals for AI features: accuracy, calibration, safety."},

    # ── Review (10 agents) ────────────────────────────────────────────────
    {"id": "rev_code",        "name": "Code Reviewer",          "family": "Review",       "skills": 10, "color": "#9b59b6",
     "desc": "Deep code review: correctness, readability, maintainability, style."},
    {"id": "rev_arch",        "name": "Architecture Reviewer",  "family": "Review",       "skills": 8,  "color": "#9b59b6",
     "desc": "Reviews design decisions for coupling, cohesion, scalability trade-offs."},
    {"id": "rev_doc",         "name": "Documentation Reviewer", "family": "Review",       "skills": 7,  "color": "#9b59b6",
     "desc": "Reviews READMEs, docstrings, runbooks, ADRs for accuracy and clarity."},
    {"id": "rev_security",    "name": "Security Reviewer",      "family": "Review",       "skills": 9,  "color": "#9b59b6",
     "desc": "Threat-models PRs, flags input validation gaps and privilege issues."},
    {"id": "rev_perf",        "name": "Performance Reviewer",   "family": "Review",       "skills": 7,  "color": "#9b59b6",
     "desc": "Spots O(n²) loops, N+1 queries, unnecessary serialisation, lock contention."},
    {"id": "rev_ux",          "name": "UX Reviewer",            "family": "Review",       "skills": 6,  "color": "#9b59b6",
     "desc": "Reviews UI changes for usability, consistency, and accessibility."},
    {"id": "rev_api",         "name": "API Reviewer",           "family": "Review",       "skills": 7,  "color": "#9b59b6",
     "desc": "Checks REST conventions, backwards compatibility, versioning hygiene."},
    {"id": "rev_data",        "name": "Data Reviewer",          "family": "Review",       "skills": 6,  "color": "#9b59b6",
     "desc": "Reviews schema migrations, query plans, data access patterns."},
    {"id": "rev_ai",          "name": "AI Review Specialist",   "family": "Review",       "skills": 8,  "color": "#9b59b6",
     "desc": "Reviews AI/ML code: prompt engineering, eval rigor, bias, safety."},
    {"id": "rev_final",       "name": "Final Approver",         "family": "Review",       "skills": 5,  "color": "#9b59b6",
     "desc": "Holistic sign-off: confirms all review gates passed before merge."},

    # ── Coordination (8 agents) ───────────────────────────────────────────
    {"id": "coord_router",    "name": "Task Router",            "family": "Coordination", "skills": 8,  "color": "#e67e22",
     "desc": "Receives tasks, selects the right agent(s), tracks completion."},
    {"id": "coord_progress",  "name": "Progress Tracker",       "family": "Coordination", "skills": 6,  "color": "#e67e22",
     "desc": "Monitors parallel workstreams, flags blockers, reports ETA."},
    {"id": "coord_conflict",  "name": "Conflict Resolver",      "family": "Coordination", "skills": 7,  "color": "#e67e22",
     "desc": "Arbitrates disagreements between agents, breaks deadlocks."},
    {"id": "coord_mem",       "name": "Memory Manager",         "family": "Coordination", "skills": 9,  "color": "#e67e22",
     "desc": "Maintains shared context across agent turns: summaries, decisions, diffs."},
    {"id": "coord_merge",     "name": "Output Merger",          "family": "Coordination", "skills": 7,  "color": "#e67e22",
     "desc": "Combines partial outputs from parallel agents into coherent result."},
    {"id": "coord_retry",     "name": "Retry Orchestrator",     "family": "Coordination", "skills": 5,  "color": "#e67e22",
     "desc": "Re-routes failed tasks, implements backoff, escalates if needed."},
    {"id": "coord_human",     "name": "Human-in-Loop Broker",   "family": "Coordination", "skills": 6,  "color": "#e67e22",
     "desc": "Decides when to pause and surface a question or approval to a human."},
    {"id": "coord_audit",     "name": "Audit Logger",           "family": "Coordination", "skills": 5,  "color": "#e67e22",
     "desc": "Writes structured logs of every agent action, decision, and tool call."},

    # ── Specialized (7 agents) ────────────────────────────────────────────
    {"id": "spec_doc",        "name": "Technical Writer",       "family": "Specialized",  "skills": 9,  "color": "#1abc9c",
     "desc": "Writes API docs, ADRs, runbooks, onboarding guides, changelogs."},
    {"id": "spec_ux",         "name": "UX Designer",            "family": "Specialized",  "skills": 7,  "color": "#1abc9c",
     "desc": "Creates wireframes, component specs, design tokens, Figma hand-off."},
    {"id": "spec_legal",      "name": "Legal/Compliance Advisor","family": "Specialized", "skills": 5,  "color": "#1abc9c",
     "desc": "Flags license incompatibilities, GDPR/CCPA obligations, IP risks."},
    {"id": "spec_pm",         "name": "Product Manager",        "family": "Specialized",  "skills": 7,  "color": "#1abc9c",
     "desc": "Validates feature scope vs. requirements, writes acceptance criteria."},
    {"id": "spec_llmops",     "name": "LLMOps Engineer",        "family": "Specialized",  "skills": 10, "color": "#1abc9c",
     "desc": "Manages prompt versioning, model swapping, cost/latency observability."},
    {"id": "spec_sec",        "name": "Security Specialist",    "family": "Specialized",  "skills": 9,  "color": "#1abc9c",
     "desc": "Penetration testing, secret rotation, incident response playbooks."},
    {"id": "spec_perf",       "name": "Performance Specialist", "family": "Specialized",  "skills": 8,  "color": "#1abc9c",
     "desc": "Deep profiling, flame graphs, kernel-level optimisation, GC tuning."},
]

# ---------------------------------------------------------------------------
# Skill category catalogue (286 skills across 6 categories)
# ---------------------------------------------------------------------------

SKILL_CATEGORIES: dict[str, list[str]] = {
    "Planning (56 skills)": [
        # Architecture
        "System architecture design", "Microservices boundary definition",
        "Monolith-to-micro decomposition", "Event-driven architecture design",
        "CQRS + Event Sourcing design", "Saga pattern implementation",
        "Service mesh planning (Istio)", "API gateway strategy",
        "Tech stack selection", "ADR (Architecture Decision Record) writing",
        "C4 diagram generation", "Capacity estimation",
        # Sprint / Project
        "Epic-to-task decomposition", "Story point estimation",
        "Dependency graph construction", "Critical path analysis",
        "Blocker identification", "Sprint retrospective synthesis",
        "Velocity forecasting", "Technical roadmap authoring",
        # Risk
        "Risk matrix construction", "Single-point-of-failure analysis",
        "Disaster recovery planning", "SLA/SLO/SLI definition",
        "Technical debt scoring", "FMEA (Failure Mode Effect Analysis)",
        # API & Data
        "OpenAPI schema authoring", "gRPC/Protobuf contract design",
        "Versioning strategy (URL vs header vs content-type)",
        "Backward-compatibility analysis", "Pagination strategy design",
        "JSON:API / GraphQL schema design",
        "ER diagram generation", "Normalisation up to 3NF/BCNF",
        "Index strategy planning", "Partitioning/sharding design",
        "Data retention policy design", "Schema migration planning",
        # Infrastructure
        "Cloud cost estimation (AWS/GCP/Azure)", "IaC module design (Terraform)",
        "Kubernetes cluster sizing", "Multi-region topology planning",
        "Disaster recovery runbook drafting", "On-call rotation design",
        # Performance
        "SLA target setting", "P99 latency budget allocation",
        "Hot-path identification", "Load test scenario design",
        "Cache hit-rate target setting", "CDN routing strategy",
        "DB connection pool sizing", "Async job queue sizing",
    ],

    "Development (80 skills)": [
        # Backend
        "REST API implementation (FastAPI / Express / Django)",
        "gRPC service implementation", "GraphQL resolver implementation",
        "Background job implementation (Celery / BullMQ)",
        "Webhook handler implementation", "OAuth2 / JWT authentication",
        "RBAC implementation", "Rate limiting middleware",
        "Request validation and sanitisation", "Error handling middleware",
        "Database ORM modelling (SQLAlchemy / Prisma)",
        "Migration script authoring", "Query optimisation",
        "Stored procedure authoring", "Redis caching layer",
        "Message broker producer/consumer (Kafka / RabbitMQ)",
        "WebSocket server implementation", "SSE stream implementation",
        "Batch processing pipeline", "Scheduled job implementation",
        # Frontend
        "React component implementation", "Vue SFC authoring",
        "State management (Redux / Zustand / Pinia)",
        "React Query / SWR data fetching", "Form validation (Zod / Yup)",
        "Routing (React Router / Vue Router)", "Responsive layout (Tailwind)",
        "Accessible UI (ARIA patterns)", "Animation (Framer Motion)",
        "Data visualisation (Recharts / D3)", "PWA / service worker setup",
        "Bundle optimisation (code-splitting, tree-shaking)",
        "Internationalisation (i18n)", "Dark mode / theming",
        # ML / AI
        "Model training loop (PyTorch / JAX)", "Hugging Face pipeline integration",
        "Feature store integration", "Embedding generation pipeline",
        "Vector index creation (FAISS / Qdrant / Pinecone)",
        "RAG pipeline (chunking → embedding → retrieve → generate)",
        "Re-ranking layer implementation (cross-encoder)",
        "Prompt template management", "Claude tool-use implementation",
        "Streaming inference handler", "Eval harness implementation",
        "A/B model experiment wiring", "Model registry integration",
        # DevOps / Infra
        "GitHub Actions / GitLab CI pipeline authoring",
        "Docker multi-stage build authoring", "Helm chart authoring",
        "Terraform module authoring", "Ansible playbook authoring",
        "Secrets management (Vault / AWS SSM)", "Log aggregation (Loki / ELK)",
        "Metrics export (Prometheus / OpenTelemetry)", "Alert rule authoring",
        "Horizontal Pod Autoscaler configuration",
        "Blue-green / canary deployment wiring",
        # Search / Data
        "Elasticsearch index mapping", "BM25 + vector hybrid query",
        "Data pipeline (Airflow / Prefect / Dagster)",
        "dbt model authoring", "Great Expectations suite authoring",
        "Stream processing (Flink / Spark Streaming)",
        # Utilities
        "CLI tool implementation (Click / Typer)",
        "SDK / client library authoring", "OpenAPI client generation",
        "Monorepo task runner setup (Nx / Turborepo)",
    ],

    "Testing (60 skills)": [
        # Unit
        "Unit test authoring (pytest / Jest)", "Mock/stub implementation",
        "Fixture and factory design", "Parameterised test design",
        "Snapshot testing", "Property-based testing (Hypothesis)",
        "Coverage gap analysis", "Flaky test diagnosis",
        # Integration
        "API integration test authoring", "Database integration test authoring",
        "Contract test authoring (Pact)", "Message broker integration test",
        "Third-party service mock (WireMock)", "Environment setup scripting",
        # E2E
        "Playwright test authoring", "Cypress test authoring",
        "Visual regression testing (Applitools)", "Mobile E2E (Detox)",
        "User journey mapping for test coverage",
        # Performance
        "k6 / Locust load test scripting", "Latency profiling (py-spy / perf)",
        "Memory leak detection", "CPU flame graph generation",
        "Connection pool stress testing", "Cache performance validation",
        "Throughput benchmark authoring",
        # Security
        "OWASP top-10 test automation", "SQL injection probe suite",
        "XSS / CSRF test cases", "SAST rule authoring (Semgrep)",
        "Secrets scanning (truffleHog / gitleaks)", "Dependency vulnerability scan",
        "Permission boundary testing",
        # Quality / Chaos
        "Mutation testing (mutmut / Stryker)", "Fuzz test authoring (Atheris)",
        "Chaos experiment design (Litmus / Chaos Monkey)",
        "Disaster recovery drill scripting", "Data quality assertion suite",
        "Schema evolution compatibility test",
        # AI / LLM Eval
        "Eval dataset curation", "LLM accuracy measurement (exact match / F1)",
        "Calibration evaluation (ECE, reliability diagrams)",
        "Hallucination detection eval", "Retrieval precision/recall eval",
        "Context faithfulness eval (RAGAs)", "Safety / refusal eval",
        "Latency + cost regression eval", "Multi-turn coherence eval",
        "Tool-use success rate eval",
    ],

    "Review (50 skills)": [
        # Code quality
        "Correctness review (logic, edge cases, off-by-one)",
        "Readability review (naming, abstractions, comment quality)",
        "Maintainability review (coupling, cohesion, SOLID)",
        "Style / lint enforcement review", "Dead code detection",
        "Duplication analysis (DRY violations)", "Error handling review",
        "Logging / observability review", "Configuration hygiene review",
        "Dependency hygiene review (unused / vulnerable)",
        # Architecture
        "Coupling analysis", "Cohesion analysis",
        "Scalability trade-off review", "Consistency model review",
        "Distributed systems failure mode review",
        "Event ordering / idempotency review",
        # Security
        "Input validation review", "Output encoding review",
        "Authentication / authorisation review",
        "Secret / credential hygiene review", "Supply chain risk review",
        "Privilege escalation path review", "Injection vulnerability review",
        # Performance
        "N+1 query detection", "Loop complexity review (O-notation)",
        "Unnecessary serialisation review", "Lock contention review",
        "Garbage collection pressure review",
        # API / Data
        "REST convention review", "Backwards compatibility review",
        "Versioning hygiene review", "Schema migration safety review",
        "Query plan review",
        # Documentation
        "README completeness review", "Docstring accuracy review",
        "Runbook completeness review", "Changelog entry review",
        "ADR completeness review",
        # AI / UX / Accessibility
        "Prompt quality review", "Eval rigor review",
        "Bias / fairness review", "AI safety review",
        "UX consistency review", "WCAG 2.2 accessibility review",
        "Mobile responsiveness review",
        # Process
        "PR scope review (not too large)", "Test coverage review",
        "CI passing before merge review", "Final holistic sign-off",
    ],

    "Coordination (30 skills)": [
        "Task intake parsing", "Agent capability matching",
        "Parallel task dispatch", "Sequential dependency ordering",
        "Partial failure detection", "Retry policy application",
        "Exponential backoff orchestration", "Dead-letter queue management",
        "Context window management (summarisation)", "Shared memory writes",
        "Shared memory reads", "Cross-agent message routing",
        "Progress delta computation", "ETA forecasting",
        "Blocker escalation", "Human approval request drafting",
        "Approval gate enforcement", "Output schema validation",
        "Partial output merging", "Conflict diff resolution",
        "Audit log entry writing", "Audit log querying",
        "Cost tracking (token consumption)", "Latency tracking",
        "Workstream prioritisation", "Context handoff packaging",
        "Agent health monitoring", "Fallback agent selection",
        "Session state persistence", "Session state restoration",
    ],

    "Specialized (10 skills)": [
        "Technical documentation authoring", "Changelog generation",
        "API reference generation (from code)", "Runbook authoring",
        "License compatibility analysis", "GDPR data-flow mapping",
        "CCPA compliance review", "Wireframe generation (ASCII / Figma spec)",
        "LLM cost/latency observability setup",
        "Prompt versioning and A/B tracking",
    ],
}

# Quick summary
TOTAL_AGENTS = len(AGENT_ROLES)
TOTAL_SKILLS = sum(len(v) for v in SKILL_CATEGORIES.values())

# ---------------------------------------------------------------------------
# Task decomposition examples
# ---------------------------------------------------------------------------

TASK_EXAMPLES: list[dict] = [
    {
        "task": "Add JWT authentication to an existing FastAPI service",
        "agents": [
            ("Architecture Planner",   "Choose token strategy: access + refresh, expiry policy, key rotation"),
            ("Auth Engineer",          "Implement /login, /refresh, /logout endpoints; hash passwords with bcrypt"),
            ("Backend Engineer",       "Add auth middleware to all protected routes; wire into dependency injection"),
            ("Security Reviewer",      "Review for token leakage, timing attacks, CSRF protection"),
            ("Unit Test Engineer",     "Write tests for token generation, validation, expiry, refresh"),
            ("Integration Tester",     "Test authenticated flows end-to-end with real DB"),
            ("Technical Writer",       "Document the auth endpoints in OpenAPI spec and README"),
        ],
        "skills_activated": 9,
        "parallel_phases": ["Design", "Implement + Review", "Test", "Document"],
        "single_model_score": 72,
        "team_score": 94,
    },
    {
        "task": "Build a RAG pipeline over 50,000 company documents",
        "agents": [
            ("Architecture Planner",   "Choose chunking strategy, embedding model, vector DB, re-ranker"),
            ("Embeddings Engineer",    "Implement chunking (semantic splitter), embedding batch job"),
            ("RAG Engineer",           "Build query pipeline: embed → ANN search → re-rank → LLM generate"),
            ("Database Engineer",      "Set up Qdrant collection, configure HNSW index params"),
            ("LLM Eval Engineer",      "Build retrieval precision/recall + context faithfulness eval suite"),
            ("Performance Tester",     "Load-test at 100 QPS; measure P99 retrieval + generation latency"),
            ("LLMOps Engineer",        "Wire Prometheus metrics: latency, cost/token, cache hit rate"),
            ("Technical Writer",       "Document chunking design, index params, latency budget in ADR"),
        ],
        "skills_activated": 12,
        "parallel_phases": ["Design", "Index Build", "Pipeline + Eval", "Observability + Docs"],
        "single_model_score": 63,
        "team_score": 91,
    },
    {
        "task": "Migrate a monolith to microservices (3 core domains)",
        "agents": [
            ("Architecture Planner",   "Identify bounded contexts, define service boundaries via DDD"),
            ("Dependency Analyst",     "Map shared DB tables, cross-domain transactions, circular deps"),
            ("Backend Engineer",       "Extract Domain A service: new codebase, API, DB schema"),
            ("Backend Engineer",       "Extract Domain B service: strangler-fig pattern via API gateway"),
            ("Database Engineer",      "Split shared schema: add foreign-key stubs, run dual-write migration"),
            ("DevOps Engineer",        "Add per-service CI/CD pipeline, Docker builds, K8s deployments"),
            ("Contract Tester",        "Write Pact contracts between all 3 services + legacy monolith"),
            ("Chaos Engineer",         "Inject inter-service network failures; validate circuit breakers"),
            ("Risk Assessor",          "Identify rollback triggers, write incident runbooks"),
        ],
        "skills_activated": 15,
        "parallel_phases": ["Decompose", "Extract + Schema Split", "CI/CD + Contracts", "Chaos + Risk"],
        "single_model_score": 55,
        "team_score": 88,
    },
    {
        "task": "Ship an accessible React dashboard for real-time ML metrics",
        "agents": [
            ("Architecture Planner",   "Choose state management, WebSocket vs polling, component library"),
            ("Frontend Engineer",      "Build MetricsCard, TimeSeriesChart, AlertBanner components"),
            ("Real-Time Engineer",     "Implement WebSocket client with reconnect + exponential backoff"),
            ("ML Engineer",            "Wire Prometheus exporter to stream latency/cost/accuracy metrics"),
            ("Accessibility Auditor",  "WCAG 2.2 audit: ARIA labels, keyboard nav, colour contrast"),
            ("E2E Test Engineer",      "Playwright tests for connect/disconnect, metric update rendering"),
            ("Performance Reviewer",   "Bundle size audit; lazy-load heavy chart lib"),
            ("UX Reviewer",            "Spot confusing affordances; validate dark-mode rendering"),
        ],
        "skills_activated": 10,
        "parallel_phases": ["Design", "Frontend + Backend", "QA + a11y", "Perf + UX Review"],
        "single_model_score": 68,
        "team_score": 93,
    },
]

# ---------------------------------------------------------------------------
# Benchmark data (single large model vs. 68-agent team)
# ---------------------------------------------------------------------------

BENCHMARK_METRICS: list[dict] = [
    {
        "metric": "Code correctness (pass@1)",
        "single_model": 72,
        "team": 94,
        "unit": "%",
        "note": "Team catches bugs via dedicated reviewer + tester agents",
    },
    {
        "metric": "Security issues caught",
        "single_model": 41,
        "team": 89,
        "unit": "%",
        "note": "Security Auditor + Security Reviewer agents specialise in OWASP patterns",
    },
    {
        "metric": "Test coverage generated",
        "single_model": 64,
        "team": 91,
        "unit": "%",
        "note": "5 dedicated QA agents cover unit, integration, E2E, perf, and chaos",
    },
    {
        "metric": "Documentation completeness",
        "single_model": 55,
        "team": 96,
        "unit": "%",
        "note": "Technical Writer + Documentation Reviewer agents own docs end-to-end",
    },
    {
        "metric": "Architectural coherence score",
        "single_model": 67,
        "team": 92,
        "unit": "/100",
        "note": "Architecture Planner + Reviewer maintain coherence across all agents",
    },
    {
        "metric": "Hallucination / fabrication rate",
        "single_model": 18,
        "team": 4,
        "unit": "%",
        "note": "Team: lower is better — specialisation + verification loops reduce errors",
        "lower_is_better": True,
    },
    {
        "metric": "Context utilisation efficiency",
        "single_model": 100,
        "team": 47,
        "unit": "% of budget",
        "note": "Team: each agent sees only its relevant context slice (Memory Manager)",
        "lower_is_better": True,
    },
    {
        "metric": "Task success on novel multi-step problems",
        "single_model": 48,
        "team": 87,
        "unit": "%",
        "note": "Decomposition + parallel verification dramatically lifts complex-task success",
    },
]

# ---------------------------------------------------------------------------
# Key design principles (from the open-source project)
# ---------------------------------------------------------------------------

DESIGN_PRINCIPLES: list[dict] = [
    {
        "title": "Specialisation beats generalism",
        "detail": (
            "Each of the 68 agents has a narrow job. A dedicated Security Auditor "
            "applies OWASP top-10 checks on every PR; a monolithic model must "
            "remember to do this while also writing code, tests, and docs. "
            "Specialisation eliminates category errors."
        ),
        "takeaway": "Define tight role boundaries before spawning agents.",
    },
    {
        "title": "Skills are the atomic unit",
        "detail": (
            "286 skills are each a versioned, testable capability: a self-contained "
            "prompt + tool schema + acceptance criteria. Tasks are fulfilled by "
            "composing skills, not by free-form generation. This makes the system "
            "auditable and improvable."
        ),
        "takeaway": "Write skill definitions like function contracts, not vague instructions.",
    },
    {
        "title": "Parallel review without merge conflicts",
        "detail": (
            "Code Reviewer, Security Reviewer, Performance Reviewer, and UX Reviewer "
            "run concurrently on the same diff. The Conflict Resolver arbitrates "
            "contradictory findings. Wall-clock time collapses from sequential review "
            "to the duration of the slowest reviewer."
        ),
        "takeaway": "Treat review as a map-reduce, not a sequential queue.",
    },
    {
        "title": "Memory management is a first-class concern",
        "detail": (
            "The Memory Manager agent summarises completed phases before handing "
            "context to the next agent, keeping per-agent context under 20 K tokens "
            "regardless of project size. This is why the team achieves 47% context "
            "efficiency vs. 100% for a single-model approach."
        ),
        "takeaway": "Design explicit context hand-off protocols; never assume agents share state.",
    },
    {
        "title": "Human-in-the-loop at defined gates",
        "detail": (
            "The Human-in-Loop Broker pauses the pipeline at architecture decisions, "
            "security findings above a threshold, and final merge approval. All other "
            "steps run autonomously. This keeps humans in strategic control without "
            "bottlenecking execution."
        ),
        "takeaway": "Name the gates explicitly; automate everything between them.",
    },
]
