#!/usr/bin/env python3
"""
graph.py — Repository Knowledge Graph Generator

Builds a dependency/relationship graph from a GitHub repo by:
1. Fetching source file contents via the GitHub API
2. Parsing import/require statements across languages
3. Building a structured graph of file relationships
4. Outputting Mermaid diagrams, JSON graphs, and LLM context docs
"""

import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ──────────────────────────────────────────────
# Import Parsers (multi-language)
# ──────────────────────────────────────────────
IMPORT_PATTERNS = {
    ".py": [
        (r'^import\s+([\w.]+)', "module"),
        (r'^from\s+([\w.]+)\s+import', "module"),
    ],
    ".js": [
        (r'(?:import|export)\s+.*?from\s+[\'"]([^"\']+)[\'"]', "path"),
        (r'require\s*\(\s*[\'"]([^"\']+)[\'"]\s*\)', "path"),
    ],
    ".ts": [
        (r'(?:import|export)\s+.*?from\s+[\'"]([^"\']+)[\'"]', "path"),
        (r'require\s*\(\s*[\'"]([^"\']+)[\'"]\s*\)', "path"),
    ],
    ".tsx": [
        (r'(?:import|export)\s+.*?from\s+[\'"]([^"\']+)[\'"]', "path"),
    ],
    ".jsx": [
        (r'(?:import|export)\s+.*?from\s+[\'"]([^"\']+)[\'"]', "path"),
    ],
    ".go": [
        (r'"([^"]+)"', "path"),
    ],
    ".rs": [
        (r'(?:use|mod)\s+([\w:]+)', "module"),
        (r'extern\s+crate\s+(\w+)', "module"),
    ],
    ".java": [
        (r'^import\s+([\w.]+)', "module"),
    ],
    ".rb": [
        (r"require\s+['\"]([^'\"]+)['\"]", "path"),
        (r"require_relative\s+['\"]([^'\"]+)['\"]", "path"),
    ],
    ".c": [
        (r'#include\s*[<"]([^>"]+)[>"]', "path"),
    ],
    ".h": [
        (r'#include\s*[<"]([^>"]+)[>"]', "path"),
    ],
    ".cpp": [
        (r'#include\s*[<"]([^>"]+)[>"]', "path"),
    ],
    ".swift": [
        (r'^import\s+(\w+)', "module"),
    ],
}

# File extensions to fetch content for
SOURCE_EXTENSIONS = set(IMPORT_PATTERNS.keys())

# Files to skip (binary, vendor, generated)
SKIP_PATTERNS = [
    "node_modules/", "vendor/", "dist/", "build/", ".min.",
    "__pycache__/", ".pyc", "venv/", ".venv/",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".woff", ".ttf", ".eot", ".mp3", ".mp4",
    ".zip", ".tar", ".gz", ".jar", ".exe",
    "package-lock.json", "yarn.lock", "Pipfile.lock",
    "go.sum", "Cargo.lock",
]

# Config/entry point files to always fetch
PRIORITY_FILES = [
    "package.json", "pyproject.toml", "setup.py", "setup.cfg",
    "Cargo.toml", "go.mod", "Gemfile", "composer.json",
    "pom.xml", "build.gradle", "Makefile", "Dockerfile",
    "docker-compose.yml", "docker-compose.yaml",
    ".github/workflows/", "tsconfig.json", "vite.config",
    "webpack.config", "next.config", "tailwind.config",
]


def should_skip(path):
    """Check if a file should be skipped."""
    return any(skip in path for skip in SKIP_PATTERNS)


def is_source_file(path):
    """Check if a file is a parseable source file."""
    return any(path.endswith(ext) for ext in SOURCE_EXTENSIONS)


def is_priority_file(path):
    """Check if a file is a config/entry point we always want."""
    return any(p in path for p in PRIORITY_FILES)


# ──────────────────────────────────────────────
# Content Fetcher
# ──────────────────────────────────────────────
def fetch_file_contents(session, owner, repo, tree, max_files=250):
    """Fetch source file contents in parallel, respecting limits."""
    base = f"/repos/{owner}/{repo}/contents"

    # Filter files to fetch
    files_to_fetch = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if should_skip(path):
            continue
        size = item.get("size", 0)
        if size > 100_000:  # Skip files > 100KB
            continue
        if is_source_file(path) or is_priority_file(path):
            files_to_fetch.append(path)

    # Limit total files
    if len(files_to_fetch) > max_files:
        # Prioritize: priority files first, then source files sorted by path depth
        priority = [f for f in files_to_fetch if is_priority_file(f)]
        source = [f for f in files_to_fetch if not is_priority_file(f)]
        source.sort(key=lambda p: p.count("/"))  # shallower files first
        files_to_fetch = priority + source[:max_files - len(priority)]

    print(f"📡  Fetching {len(files_to_fetch)} source files for graph analysis…")

    file_contents = {}

    def fetch_one(path):
        from features import api_get_retry
        data = api_get_retry(session, f"{base}/{path}")
        if data and data.get("content"):
            try:
                content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                return path, content
            except Exception:
                pass
        return path, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_one, path) for path in files_to_fetch]
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 20 == 0 or done == len(futures):
                print(f"    {done}/{len(futures)} files fetched…")
            try:
                path, content = future.result()
                if content is not None:
                    file_contents[path] = content
            except Exception:
                pass

    return file_contents


# ──────────────────────────────────────────────
# Import Parser
# ──────────────────────────────────────────────
def parse_imports(path, content):
    """Parse import statements from a source file."""
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    patterns = IMPORT_PATTERNS.get(ext, [])
    if not patterns:
        return []

    imports = []
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        for pattern, import_type in patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                imports.append({"raw": match, "type": import_type})
    return imports


def parse_definitions(path, content):
    """Extract class/function definitions from source files."""
    defs = []
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""

    if ext == ".py":
        for match in re.finditer(r'^class\s+(\w+)', content, re.MULTILINE):
            defs.append({"type": "class", "name": match.group(1)})
        for match in re.finditer(r'^def\s+(\w+)', content, re.MULTILINE):
            defs.append({"type": "function", "name": match.group(1)})
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        for match in re.finditer(r'(?:export\s+)?(?:default\s+)?class\s+(\w+)', content):
            defs.append({"type": "class", "name": match.group(1)})
        for match in re.finditer(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)', content):
            defs.append({"type": "function", "name": match.group(1)})
        for match in re.finditer(r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(', content):
            defs.append({"type": "function", "name": match.group(1)})
    elif ext == ".go":
        for match in re.finditer(r'^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)', content, re.MULTILINE):
            defs.append({"type": "function", "name": match.group(1)})
        for match in re.finditer(r'^type\s+(\w+)\s+struct', content, re.MULTILINE):
            defs.append({"type": "struct", "name": match.group(1)})
    elif ext == ".rs":
        for match in re.finditer(r'^pub\s+fn\s+(\w+)|^fn\s+(\w+)', content, re.MULTILINE):
            name = match.group(1) or match.group(2)
            defs.append({"type": "function", "name": name})
        for match in re.finditer(r'^(?:pub\s+)?struct\s+(\w+)', content, re.MULTILINE):
            defs.append({"type": "struct", "name": match.group(1)})
    elif ext == ".java":
        for match in re.finditer(r'(?:public|private|protected)?\s*class\s+(\w+)', content):
            defs.append({"type": "class", "name": match.group(1)})

    return defs


# ──────────────────────────────────────────────
# Graph Builder
# ──────────────────────────────────────────────
def resolve_import(importing_file, raw_import, import_type, all_paths):
    """Try to resolve an import to an actual file path in the repo."""
    if import_type == "module":
        # Python-style: convert dots to slashes
        candidates = [
            raw_import.replace(".", "/") + ".py",
            raw_import.replace(".", "/") + "/__init__.py",
            "src/" + raw_import.replace(".", "/") + ".py",
            "src/" + raw_import.replace(".", "/") + "/__init__.py",
            "lib/" + raw_import.replace(".", "/") + ".py",
        ]
    else:
        # Path-style: relative resolution
        dir_of_file = "/".join(importing_file.split("/")[:-1])
        clean = raw_import
        if clean.startswith("./"):
            clean = clean[2:]
        elif clean.startswith("../"):
            parts = dir_of_file.split("/")
            while clean.startswith("../"):
                clean = clean[3:]
                if parts:
                    parts.pop()
            dir_of_file = "/".join(parts)

        if clean.startswith("@") or clean.startswith("~"):
            return None  # aliased imports, can't resolve

        base = f"{dir_of_file}/{clean}" if dir_of_file else clean
        candidates = [
            base,
            base + ".py", base + ".js", base + ".ts", base + ".tsx",
            base + ".jsx", base + ".go", base + ".rs",
            base + "/index.js", base + "/index.ts", base + "/index.tsx",
            base + "/mod.rs", base + "/__init__.py",
        ]

    for candidate in candidates:
        normalized = candidate.lstrip("/")
        if normalized in all_paths:
            return normalized
    return None


def build_graph(file_contents, tree):
    """Build the full knowledge graph from parsed files."""
    all_paths = {item["path"] for item in tree if item.get("type") == "blob"}
    dir_paths = {item["path"] for item in tree if item.get("type") == "tree"}

    nodes = {}  # path -> node data
    edges = []  # list of {from, to, type}

    # Build nodes from ALL tree items
    for item in tree:
        path = item.get("path", "")
        if item.get("type") == "tree":
            nodes[path] = {
                "id": path,
                "type": "directory",
                "name": path.split("/")[-1],
            }
        else:
            ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
            node = {
                "id": path,
                "type": "file",
                "name": path.split("/")[-1],
                "extension": ext,
                "size": item.get("size", 0),
            }
            if path in file_contents:
                content = file_contents[path]
                node["definitions"] = parse_definitions(path, content)
                node["imports_raw"] = parse_imports(path, content)
                node["lines"] = content.count("\n") + 1
            nodes[path] = node

    # Build directory containment edges
    for path in all_paths:
        parts = path.split("/")
        if len(parts) > 1:
            parent_dir = "/".join(parts[:-1])
            if parent_dir in dir_paths:
                edges.append({
                    "from": parent_dir,
                    "to": path,
                    "type": "contains",
                })

    # Build import/dependency edges
    for path, content in file_contents.items():
        imports = parse_imports(path, content)
        for imp in imports:
            resolved = resolve_import(path, imp["raw"], imp["type"], all_paths)
            if resolved and resolved != path:
                edges.append({
                    "from": path,
                    "to": resolved,
                    "type": "imports",
                    "raw": imp["raw"],
                })

    # Detect entry points
    entry_indicators = [
        "main.py", "app.py", "__main__.py", "index.js", "index.ts",
        "main.go", "main.rs", "Main.java", "server.py", "server.js",
        "cli.py", "manage.py", "wsgi.py", "asgi.py",
    ]
    for path in all_paths:
        name = path.split("/")[-1]
        if name in entry_indicators and path in nodes:
            nodes[path]["is_entry_point"] = True

    return {"nodes": nodes, "edges": edges}


# ──────────────────────────────────────────────
# Output: Mermaid Diagram
# ──────────────────────────────────────────────
def build_mermaid_diagram(graph, max_nodes=60):
    """Generate a Mermaid flowchart from the graph."""
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Filter to only import edges and their connected nodes
    import_edges = [e for e in edges if e["type"] == "imports"]
    if not import_edges:
        return ""

    # Get nodes involved in imports
    import_nodes = set()
    for e in import_edges:
        import_nodes.add(e["from"])
        import_nodes.add(e["to"])

    # Limit nodes
    if len(import_nodes) > max_nodes:
        # Keep only the most connected nodes
        connection_count = defaultdict(int)
        for e in import_edges:
            connection_count[e["from"]] += 1
            connection_count[e["to"]] += 1
        import_nodes = set(
            sorted(connection_count, key=connection_count.get, reverse=True)[:max_nodes]
        )
        import_edges = [e for e in import_edges
                        if e["from"] in import_nodes and e["to"] in import_nodes]

    lines = ["\n## 🕸️ Dependency Graph\n"]
    lines.append("```mermaid")
    lines.append("graph LR")

    # Create sanitized node IDs
    def node_id(path):
        return path.replace("/", "_").replace(".", "_").replace("-", "_")

    # Group by directory for subgraphs
    dirs = defaultdict(list)
    for path in import_nodes:
        parts = path.split("/")
        dir_name = "/".join(parts[:-1]) if len(parts) > 1 else "root"
        dirs[dir_name].append(path)

    # Add subgraphs
    for dir_name, dir_files in sorted(dirs.items()):
        if len(dir_files) > 1:
            safe_dir = dir_name.replace("/", "_").replace(".", "_").replace("-", "_")
            lines.append(f"    subgraph {safe_dir}[\"{dir_name}/\"]")
            for path in dir_files:
                name = path.split("/")[-1]
                nid = node_id(path)
                node = nodes.get(path, {})
                if node.get("is_entry_point"):
                    lines.append(f'        {nid}[["⚡ {name}"]]')
                elif node.get("definitions"):
                    defs = node["definitions"]
                    classes = [d["name"] for d in defs if d["type"] == "class"][:2]
                    label = f"{name}"
                    if classes:
                        label += f"\\n({', '.join(classes)})"
                    lines.append(f'        {nid}["{label}"]')
                else:
                    lines.append(f'        {nid}["{name}"]')
            lines.append("    end")
        else:
            for path in dir_files:
                name = path.split("/")[-1]
                nid = node_id(path)
                lines.append(f'    {nid}["{name}"]')

    # Add edges
    for e in import_edges:
        f_id = node_id(e["from"])
        t_id = node_id(e["to"])
        lines.append(f"    {f_id} --> {t_id}")

    lines.append("```\n")

    # Stats
    lines.append(f"**Nodes:** {len(import_nodes)} files | "
                  f"**Edges:** {len(import_edges)} dependencies\n")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Output: JSON Graph
# ──────────────────────────────────────────────
def export_graph_json(graph, output_dir, owner, repo):
    """Export the graph as a structured JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{owner}_{repo}_graph.json")

    # Build clean export
    export = {
        "repository": f"{owner}/{repo}",
        "generated_at": None,
        "nodes": [],
        "edges": [],
        "stats": {},
    }

    from datetime import datetime, timezone
    export["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Nodes
    for path, node in graph["nodes"].items():
        clean_node = {
            "id": node["id"],
            "type": node["type"],
            "name": node["name"],
        }
        if node["type"] == "file":
            clean_node["extension"] = node.get("extension", "")
            clean_node["size"] = node.get("size", 0)
            clean_node["lines"] = node.get("lines", 0)
            if node.get("definitions"):
                clean_node["definitions"] = node["definitions"]
            if node.get("is_entry_point"):
                clean_node["is_entry_point"] = True
            if node.get("imports_raw"):
                clean_node["imports"] = [i["raw"] for i in node["imports_raw"]]
        export["nodes"].append(clean_node)

    # Edges
    for edge in graph["edges"]:
        export["edges"].append({
            "from": edge["from"],
            "to": edge["to"],
            "type": edge["type"],
        })

    # Stats
    file_nodes = [n for n in export["nodes"] if n["type"] == "file"]
    import_edges = [e for e in export["edges"] if e["type"] == "imports"]
    export["stats"] = {
        "total_files": len(file_nodes),
        "total_directories": len([n for n in export["nodes"] if n["type"] == "directory"]),
        "total_import_edges": len(import_edges),
        "total_containment_edges": len([e for e in export["edges"] if e["type"] == "contains"]),
        "entry_points": [n["id"] for n in export["nodes"] if n.get("is_entry_point")],
        "languages": list(set(n.get("extension", "") for n in file_nodes if n.get("extension"))),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    return filepath


# ──────────────────────────────────────────────
# Output: LLM Context Document
# ──────────────────────────────────────────────
def build_llm_context(graph, owner, repo, data=None):
    """Build an LLM-optimized context document for the repository."""
    nodes = graph["nodes"]
    edges = graph["edges"]

    lines = []
    lines.append(f"# Repository Context: {owner}/{repo}")
    lines.append(f"# Use this document to understand the full structure and relationships in this codebase.\n")

    # 1. High-level overview
    repo_data = data.get("repo", {}) if data else {}
    desc = repo_data.get("description", "No description")
    langs = data.get("languages", {}) if data else {}
    top_langs = sorted(langs.items(), key=lambda x: -x[1])[:5] if langs else []

    lines.append("## OVERVIEW")
    lines.append(f"Repository: {owner}/{repo}")
    lines.append(f"Description: {desc}")
    if top_langs:
        lines.append(f"Languages: {', '.join(f'{l[0]} ({l[1]:,}B)' for l in top_langs)}")
    lines.append("")

    # 2. Architecture
    file_nodes = {p: n for p, n in nodes.items() if n["type"] == "file"}
    dir_nodes = {p: n for p, n in nodes.items() if n["type"] == "directory"}

    lines.append("## DIRECTORY STRUCTURE")
    # Top-level dirs with file counts
    top_dirs = {}
    for path in file_nodes:
        top = path.split("/")[0]
        top_dirs[top] = top_dirs.get(top, 0) + 1
    for d, count in sorted(top_dirs.items(), key=lambda x: -x[1])[:20]:
        if d in dir_nodes:
            lines.append(f"  {d}/ ({count} files)")
        else:
            lines.append(f"  {d}")
    lines.append("")

    # 3. Entry points
    entry_points = [p for p, n in nodes.items() if n.get("is_entry_point")]
    if entry_points:
        lines.append("## ENTRY POINTS")
        for ep in entry_points:
            node = nodes[ep]
            defs = node.get("definitions", [])
            def_str = ", ".join(d["name"] for d in defs[:5]) if defs else "no definitions parsed"
            lines.append(f"  ⚡ {ep} → defines: {def_str}")
        lines.append("")

    # 4. Module definitions
    lines.append("## KEY MODULES & DEFINITIONS")
    # Sort by number of definitions (most important first)
    defined_files = [(p, n) for p, n in file_nodes.items()
                     if n.get("definitions")]
    defined_files.sort(key=lambda x: -len(x[1]["definitions"]))

    for path, node in defined_files[:50]:
        defs = node["definitions"]
        classes = [d for d in defs if d["type"] in ("class", "struct")]
        funcs = [d for d in defs if d["type"] == "function"]
        parts = []
        if classes:
            parts.append(f"classes=[{', '.join(d['name'] for d in classes[:5])}]")
        if funcs:
            func_names = [d["name"] for d in funcs if not d["name"].startswith("_")][:8]
            if func_names:
                parts.append(f"functions=[{', '.join(func_names)}]")
        if parts:
            lines.append(f"  {path}: {'; '.join(parts)}")
    lines.append("")

    # 5. Dependency map
    import_edges = [e for e in edges if e["type"] == "imports"]
    if import_edges:
        lines.append("## DEPENDENCY MAP (file → imports)")
        # Group by source file
        deps_by_file = defaultdict(list)
        for e in import_edges:
            deps_by_file[e["from"]].append(e["to"])

        for src in sorted(deps_by_file, key=lambda x: -len(deps_by_file[x])):
            targets = deps_by_file[src]
            lines.append(f"  {src}")
            for t in targets:
                lines.append(f"    → {t}")
        lines.append("")

    # 6. Reverse dependencies (what depends on this file?)
    if import_edges:
        lines.append("## REVERSE DEPENDENCIES (file ← depended on by)")
        rev_deps = defaultdict(list)
        for e in import_edges:
            rev_deps[e["to"]].append(e["from"])

        # Sort by most depended-on
        for target in sorted(rev_deps, key=lambda x: -len(rev_deps[x]))[:30]:
            dependents = rev_deps[target]
            lines.append(f"  {target} ← used by {len(dependents)} files: "
                         f"{', '.join(dependents[:5])}"
                         f"{'…' if len(dependents) > 5 else ''}")
        lines.append("")

    # 7. Config files
    config_files = [p for p in file_nodes if is_priority_file(p)]
    if config_files:
        lines.append("## CONFIGURATION FILES")
        for cf in sorted(config_files):
            lines.append(f"  📋 {cf}")
        lines.append("")

    # 8. Stats
    lines.append("## STATS")
    lines.append(f"  Total files: {len(file_nodes)}")
    lines.append(f"  Total directories: {len(dir_nodes)}")
    lines.append(f"  Files with parsed imports: {len([n for n in file_nodes.values() if n.get('imports_raw')])}")
    lines.append(f"  Import relationships: {len(import_edges)}")
    lines.append(f"  Entry points: {len(entry_points)}")
    total_defs = sum(len(n.get("definitions", [])) for n in file_nodes.values())
    lines.append(f"  Total definitions found: {total_defs}")
    lines.append("")

    return "\n".join(lines)


def export_llm_context(graph, output_dir, owner, repo, data=None):
    """Save the LLM context document to a file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{owner}_{repo}_context.txt")
    content = build_llm_context(graph, owner, repo, data)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath
