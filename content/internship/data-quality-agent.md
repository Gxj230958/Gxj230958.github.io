---
title: "PB-Scale Data Quality Inspection Agent (Ray-based)"
weight: 1
summary: "Built a Ray-powered data quality inspection agent for PB-scale datasets, supporting natural-language validation rules, automatic tool generation and parallel distributed checking. Deployed internally for data governance."
---

I designed and implemented an end-to-end general data quality inspection agent built on Ray. Users can describe validation constraints (format, size, schema, content checks) in natural language. The system parses intents, matches or generates inspection tools on demand, runs checks asynchronously, and outputs structured audit reports with an approval workflow. Tasks are partitioned and executed in parallel to handle PB-scale corpora including POV, OGame and Pandas. On a 110-core node, the framework processes multiple large datasets concurrently for improved throughput. It has passed full end-to-end tests and serves as a standardized internal component for dataset governance.
