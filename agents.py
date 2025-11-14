# agents.py
# ──────────────────────────────────────────────────────────
# Build CrewAI Agent objects from a YAML config file.
# The file path defaults to  “config.agents.yaml”.
# Each YAML agent block may look like:
#
# agents:
#   - id: web_researcher
#     role: Web‑Research Analyst
#     backstory: |
#       Former investment analyst specialising in …
#     goal: |
#       Find credible web sources about potential investors …
#     model: gpt-3.5-turbo
#     temperature: 0.2
#     allow_delegation: false
#     tools: [ddg_search, website_search]
#
# Supported tools keys:
#   ddg_search      → DuckDuckGo instant‑answer search
#   website_search  → crewai WebsiteSearchTool (HTML scraper)
#   wikipedia       → WikipediaSearchTool
# Add more in `_tool_from_key()` if you need them.
# ──────────────────────────────────────────────────────────

from __future__ import annotations
import os, yaml, importlib
from pathlib import Path
from textwrap import dedent

from crewai          import Agent
from langchain_openai import ChatOpenAI
from crewai_tools    import WebsiteSearchTool, ScrapeWebsiteTool

# ── optional tools (import lazily to avoid hard deps)
def _safe_import(module:str, name:str):
    try:
        return getattr(importlib.import_module(module), name)
    except Exception:
        return None


class AgentFactory:
    """Load agents from a YAML file and return instantiated Agent objects."""

    def __init__(self, yaml_path: str | Path = "config.agents.yaml"):
        self.yaml_path = Path(yaml_path)

        if not self.yaml_path.exists():
            raise FileNotFoundError(
                f"[AgentFactory] YAML file not found: {self.yaml_path}"
            )

        with open(self.yaml_path, "r", encoding="utf-8") as fh:
            self.raw_cfg = yaml.safe_load(fh) or {}

        if not isinstance(self.raw_cfg.get("agents"), list):
            raise ValueError("YAML must have top‑level key `agents:` as a list")

    # --------------------------------------------------
    # public helpers
    # --------------------------------------------------
    def build_all(self) -> list[Agent]:
        """Instantiate and return every agent defined in YAML."""
        return [self._build_agent(block) for block in self.raw_cfg["agents"]]

    def build_by_id(self, agent_id: str) -> Agent:
        """Retrieve a single Agent by its YAML `id` field."""
        for block in self.raw_cfg["agents"]:
            if block.get("id") == agent_id:
                return self._build_agent(block)
        raise KeyError(f"No agent with id='{agent_id}' found in {self.yaml_path}")

    # --------------------------------------------------
    # internal
    # --------------------------------------------------
    def _build_agent(self, cfg: dict) -> Agent:
        llm = ChatOpenAI(
            model_name  = cfg.get("model", "gpt-3.5-turbo"),
            temperature = cfg.get("temperature", 0.3),
            max_tokens  = cfg.get("max_tokens", 512),
        )

        tools = [WebsiteSearchTool(), ScrapeWebsiteTool()]
        # for key in cfg.get("tools", []):
        #     t = self._tool_from_key(key)
        #     if t:
        #         tools.append(t)

        return Agent(
            role             = cfg.get("role", "CrewAI Agent"),
            backstory        = dedent(cfg.get("backstory", "")).strip(),
            goal             = dedent(cfg.get("goal", "")).strip(),
            tools            = tools,
            allow_delegation = bool(cfg.get("allow_delegation", False)),
            verbose          = bool(cfg.get("verbose", True)),
            llm              = llm,
        )

    # --------------------------------------------------
    # map YAML tool keys → real Tool objects
    # --------------------------------------------------
    def _tool_from_key(self, key:str):
        k = key.lower().strip
        if k == "websitesearchtool":
            return WebsiteSearchTool()
        if k == "scrapewebsitetool":
            return ScrapeWebsiteTool()
        print(f"[AgentFactory] unknown tool key: {key} (ignored)")
        return None