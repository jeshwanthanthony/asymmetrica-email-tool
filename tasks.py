# tasks.py
# ──────────────────────────────────────────────────────────
# Turn YAML task definitions (config.task.yaml) into
# CrewAI Task objects, resolving the agent IDs declared there.
# ──────────────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
from textwrap import dedent
from pandas.core.frame import deprecate_nonkeyword_arguments
import yaml
import importlib

import logging

from crewai import Task, agent
from agents import AgentFactory

# tasks.py  – compatibility shim for CrewAI ≤ 0.21
try:
    from crewai import OutputFormat            # ≥0.22
except ImportError:
    from enum import Enum

    class OutputFormat(str, Enum):
        RAW   = "raw"
        JSON  = "json"
        JSONL = "jsonl"

def _lazy_import(module: str, attr: str):
    try:
        return getattr(importlib.import_module(module), attr)
    except Exception:
        return None


class TaskFactory:
    """
    Build Task objects from a YAML config file.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the task‑configuration file (default: "config.task.yaml").
    runtime_vars : dict | None
        Values (e.g. search query, max_urls) to .format() into
        description / expected_output placeholders.
    agent_factory : AgentFactory | None
        Re‑use an existing AgentFactory or create a fresh one.
    """

    def __init__(self,
         yaml_path: str | Path = "config.tasks.yaml",
         agent_factory: AgentFactory | None = None):
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"[TaskFactory] YAML file not found: {self.yaml_path}")

        with open(self.yaml_path, "r", encoding="utf-8") as fh:
            self.raw_cfg = yaml.safe_load(fh) or {}

        if not isinstance(self.raw_cfg.get("tasks"), list):
            raise ValueError("YAML must have top‑level key `tasks:` as a list")

        # allow task YAML to reference agents by id
        self.agent_factory = agent_factory or AgentFactory()

    # -------------------------------------------------- #
    #  Public helpers
    # -------------------------------------------------- #
    def build_all(self, ctx: dict | None = None) -> list[Task]:
        """Instantiate all tasks in their listed order."""
        ctx = ctx or {}
        return [self._build_task(block, ctx) for block in self.raw_cfg["tasks"]]

    def build_by_id(self, task_id: str, ctx: dict | None = None) -> Task:
        """Fetch a single task by the YAML `id`."""
        ctx = ctx or {}
        for block in self.raw_cfg["tasks"]:
            if block.get("id") == task_id:
                return self._build_task(block, ctx)
        raise KeyError(f"No task with id='{task_id}' in {self.yaml_path}")

    # -------------------------------------------------- #
    #  Internal
    # -------------------------------------------------- #
    def _build_task(self, cfg: dict, ctx: dict) -> Task:
        descr = dedent(cfg.get("description", ""))
        exp_tpl = dedent(cfg.get("expected_output", ""))
        
        # first format the description
        try:
            descr = descr.format(**ctx)
        except KeyError as e:
            missing = e.args[0]
            print(f"[DEBUG] description template:\n{descr}")
            print(f"[DEBUG] context keys: {list(ctx.keys())}")
            raise KeyError(f"Missing context key in description: {missing}")
        
        # now try the expected_output, with extra debug if it fails
        try:
            exp = exp_tpl.format(**ctx)
        except KeyError as e:
            missing = e.args[0]
            print("❌ [DEBUG] Failed to format expected_output template:")
            print(exp_tpl)
            print("❌ [DEBUG] Available context keys:")
            print(list(ctx.keys()))
            raise KeyError(f"Missing context key for expected_output formatting: {missing}")

        agent_id = cfg.get("agent")
        if not agent_id:
            raise ValueError(f"[TaskFactory] Task block is missing `agent:` field:\n{cfg}")
        agent    = self.agent_factory.build_by_id(agent_id)

        depends = cfg.get("depends_on", [])
        tools   = self._tools_from_yaml(cfg.get("tools", []))
        
        fmt_key = cfg.get("output_format","raw").lower()
        fmt     = OutputFormat.JSON   if fmt_key=="json"  else \
                  OutputFormat.JSONL  if fmt_key=="jsonl" else \
                  OutputFormat.RAW

        return Task(
            name            = cfg["id"],
            description     = descr,
            expected_output = exp,
            output_format   = fmt,
            agent           = agent,
            async_execution = bool(cfg.get("async_execution", False)),
            # You can pass more Task‑level params here if needed
        )

    def _tools_from_yaml(self, keys: list[str]):
        """Return a list[Tool] for the YAML *tools:* keys (may be empty)."""
        if not keys:
            return None            # CrewAI Task accepts None or [] for tools
    
        out = []
        for key in keys:
            k = key.lower().strip()
    
            if k == "ddg_search":
                DDG = _lazy_import("langchain.tools", "DuckDuckGoSearchRun")
                if DDG: out.append(DDG())
    
            elif k == "website_search":
                WS = _lazy_import("crewai_tools", "WebsiteSearchTool")
                if WS: out.append(WS())
    
            elif k == "email_search":
                # our custom wrapper defined in email_search.py
                EmailSearch = _lazy_import("email_search", "EmailSearchTool")
                if EmailSearch: out.append(EmailSearch())
    
            else:
                print(f"[TaskFactory] Unknown tool key '{key}' – ignored")
    
        return out or None
