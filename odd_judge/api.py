"""OddJudge facade: load config, evaluate observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config.loader import (
    load_observation,
    load_profile_bundle,
    load_switches,
    parse_profile_bundle,
)
from .core.aggregate import aggregate
from .core.evaluate import evaluate_rule
from .core.types import (
    DefinitionMode,
    EvalContext,
    JudgmentReport,
    MissingPolicy,
    Observation,
    Rule,
    RuleResult,
    ScenarioState,
)


class SwitchPolicy:
    """Decide which rules are active based on switches config."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.global_enabled: bool = cfg.get("global_enabled", True)
        self.by_element: dict[str, bool] = dict(cfg.get("by_element", {}))
        self.by_rule_type: dict[str, bool] = dict(cfg.get("by_rule_type", {}))
        self.by_tag: dict[str, bool] = dict(cfg.get("by_tag", {}))
        self.boundary_warnings_enabled: bool = cfg.get("boundary_warnings_enabled", True)

    def is_active(self, rule: Rule) -> tuple[bool, str | None]:
        """Return (active, reason_if_skipped)."""
        if not self.global_enabled:
            return False, "global_enabled=false"
        if rule.element_id in self.by_element and not self.by_element[rule.element_id]:
            return False, f"by_element[{rule.element_id}]=false"
        if rule.rule_type in self.by_rule_type and not self.by_rule_type[rule.rule_type]:
            return False, f"by_rule_type[{rule.rule_type}]=false"
        for tag in rule.tags:
            if tag in self.by_tag and not self.by_tag[tag]:
                return False, f"by_tag[{tag}]=false"
        return True, None


class OddJudge:
    """Main API: load profiles/switches, evaluate observations."""

    def __init__(
        self,
        profile_id: str,
        definition_mode: DefinitionMode,
        rules: list[Rule],
        switches: dict[str, Any] | None = None,
        missing_policy: MissingPolicy | None = None,
    ):
        self.profile_id = profile_id
        self.definition_mode = definition_mode
        self.rules = rules
        self.switch_policy = SwitchPolicy(switches)
        self.switches_config = switches or {}
        self.missing_policy = missing_policy

    @classmethod
    def from_files(
        cls,
        profile_path: str | Path,
        switches_path: str | Path | None = None,
        extends_paths: list[str | Path] | None = None,
        override_paths: list[str | Path] | None = None,
    ) -> "OddJudge":
        bundle = load_profile_bundle(
            profile_path, extends_paths=extends_paths, override_paths=override_paths
        )
        switches = load_switches(switches_path)
        return cls(
            bundle.profile_id,
            bundle.definition_mode,
            bundle.rules,
            switches,
            bundle.missing_policy,
        )

    @classmethod
    def from_dicts(
        cls,
        profile_data: dict[str, Any],
        switches: dict[str, Any] | None = None,
    ) -> "OddJudge":
        bundle = parse_profile_bundle(profile_data)
        return cls(
            bundle.profile_id,
            bundle.definition_mode,
            bundle.rules,
            switches,
            bundle.missing_policy,
        )

    def evaluate(self, obs: Observation | dict[str, Any]) -> JudgmentReport:
        if isinstance(obs, dict):
            obs = Observation.from_dict(obs)

        ctx = EvalContext(definition_mode=self.definition_mode, missing_policy=self.missing_policy)
        results: list[RuleResult] = []
        skipped: list[dict[str, Any]] = []
        undefined_handled: list[dict[str, Any]] = []

        for rule in self.rules:
            active, skip_reason = self.switch_policy.is_active(rule)
            if not active:
                skipped.append({"element_id": rule.element_id, "reason": skip_reason})
                continue

            result = evaluate_rule(rule, obs, ctx)
            results.append(result)

            if result.state == ScenarioState.UNDEFINED:
                entry = {
                    "element_id": rule.element_id,
                    "fallback": result.reason_code or "undefined",
                    "mode": self.definition_mode.value,
                }
                if result.policy_source:
                    entry["policy"] = result.policy_source
                undefined_handled.append(entry)

        return aggregate(
            profile_id=self.profile_id,
            results=results,
            skipped=skipped,
            undefined_handled=undefined_handled,
            active_switches=list(self.switches_config.keys()),
            timestamp=obs.timestamp,
            definition_mode=self.definition_mode.value,
            boundary_warnings_enabled=self.switch_policy.boundary_warnings_enabled,
        )

    def evaluate_path(self, obs_path: str | Path) -> JudgmentReport:
        data = load_observation(obs_path)
        return self.evaluate(data)

    def evaluate_stream(self, obs_path: str | Path) -> list[JudgmentReport]:
        from .perception.json_stream import load_observation_stream

        observations = load_observation_stream(obs_path)
        return [self.evaluate(o) for o in observations]

    def list_rules(self, tag: str | None = None) -> list[Rule]:
        if tag is None:
            return list(self.rules)
        return [r for r in self.rules if tag in r.tags]
