# odd_judge 分歧评审报告（Phase 0）

> 生成时间：2026-11-06。对比对象：
> **J** = `~/projects/ODD_Judge`（独立仓），**S** = `~/projects/odd_perception_suite`（suite 内嵌拷贝）。
> 原始全量 diff 见文末附录。**请在 D 决议表逐行填 ✓（采纳建议）/ ✗（备注改判）**。

## 时间线（重要）

| 侧 | 最近修改 | 说明 |
|---|---|---|
| J 侧包文件 | **2026-11-06 11:56–11:57** | 最近刚做过一轮**加固**：NaN 守卫、严格 bool、统一 margin 语义、强 schema 校验、CLI exit code、mock fixture 扩字段 |
| S 侧包文件 | 2026-11-05 12:27–12:35 | 最近做了 **MissingPolicy 体系**：bundle API、missing_policy 校验、warning_codes 校验 |
| S 侧未提交 | 约 440 行（19 modified + 10 untracked） | 含新测试 3 个、新 profile 2 个、新 docs |

**结论：分歧是双向的，两边都比对方新，禁止整目录单向覆盖，必须按项合并。**

---

## A. 包文件（8 个有内容差异）

### A1. `cli.py` — J 领先（exit code 体系）｜建议：**取 J**
- J：`evaluate` 返回 0/1/3（INSIDE/BOUNDARY/OUTSIDE），4 = 输入/配置错误（stderr 报错）；`try/except (OddJudgeError, ValueError, OSError)` 包裹。
- S：算出 `worst` 却 **`return 0` 恒成功（死代码/缺陷）**；无错误捕获，异常直接 traceback。
- 附带涟漪：S `tests/test_judge_e2e.py:91` 断言 BOUNDARY → `returncode==0`，采纳 J 后须改为 `== 1`（J 侧 `test_e2e.py` 已断言 1/3）。

### A2. `core/margin.py` — J 领先（语义重构）｜建议：**取 J**
- J：统一三态判定式（violated → OUTSIDE；`_reaches_band` → INSIDE；否则 BOUNDARY）、`EDGE_REL_TOL=1e-12`（带宽恰落在浮点边上时稳定判定）、`math.isfinite` 守卫、完整文档字符串。
- S：旧的逐 if 分支实现，无 NaN 守卫，`==`/`!=` 边界语义较弱（`<= band` 闭区间 vs J 的闭带边+fp 容差）。

### A3. `core/evaluate.py` — **双向冲突，需合并**｜建议：**S 的 missing_policy + J 的其余全部**
| 片段 | 归属 | 说明 |
|---|---|---|
| `missing_policy` 解析与三分支（allowed/not_allowed/undefined，带 `policy_source`/`reason_code`） | **S** | MissingPolicy 体系核心 |
| `undefined_as == UNDEFINED` → UNDEFINED 态 | **J** | S 无此分支（旧版直接落到 mode 回退） |
| `NaN/±inf → _handle_missing` | **J** | S 缺失（NaN 会流进 classify） |
| 未知枚举值 → 按缺失处理（走 definition_mode） | **J** | S 直接按 allowed/not_allowed 硬判，语义更激进 |
| 严格 bool 解析（字符串 "false" → 缺失；0/1 合法） | **J** | S 是 `bool(val)`，`"false"` 会错判为 True |
| bool 失配置信度：`confidence_observation_key` + 缺失/低置信 → 缺失回退 | **J** | S 用 `spec.values[0]` 当置信度 key（设计不同，且 suite 的 types 根本没有该字段） |
| 合并顺序（推荐）：`requirement=undefined` → 元素 `undefined_as`（含 UNDEFINED）→ `missing_policy` → `definition_mode` | 合并 | J 的 undefined_as 分支在 S 代码中前置，天然兼容，保留两段即可 |

### A4. `config/schema.py` — **双向冲突，需合并**｜建议：**J 的强校验骨架 + S 的两块新校验**
- **J 领先（S 完全没有）**：`_validate_composite`（applies_when 结构/op 白名单）、`_validate_enum`（allowed/forbidden 互斥、集合交集检查）、`_validate_confidence`（confidence_threshold∈[0,1]、必须配 `confidence_observation_key`、仅限 qualitative_bool）、`_validate_numeric`（有限数、range lo<hi、equality 必配 threshold）、`undefined_as`/`expect` 类型校验、`boundary_margin.value ≥ 0`。
- **S 领先（J 完全没有）**：`warning_codes` key/格式校验 + `check_warning_code_duplicates`（跨 profile 与 registry 元素查重）、`missing_policy` 块校验（`VALID_MISSING_POLICY_VALUES`）。
- ⚠️ 注意：J 的 `_validate_confidence` 要求 `confidence_observation_key` 必填 —— 合并后 S 侧现存 profile/元素若有 confidence_threshold 而无 key 会校验失败（见 D3）。

### A5. `api.py` + `config/loader.py` — **双向，主体取 S + 恢复 J 字段**｜建议：**按下表合并**
- **取 S**：`MissingPolicy`、`ProfileBundle`/`parse_profile_bundle`/`load_profile_bundle`、`_apply_overrides` 重构、`_merge_missing_policy`（extends/override 后来者覆盖）、跨源 warning_codes 查重、`load_profile_with_extends` 兼容包装。
- **取 J**：`_parse_rule` 里的 `confidence_observation_key=el.get(...)`（S 已删；A3/A4 合并后必须有）、`load_switches` 默认值中删除 `boundary_severity` 死键（S 删了，J 还注入 —— 见 C2，两侧均无消费者，建议随 S 删）。

### A6. `core/types.py` — 纯互补｜建议：**取并集**
- 取 S：`MissingPolicy` 类、`RuleResult.policy_source`（含 to_dict）、`EvalContext.missing_policy`。
- 取 J：`Rule.confidence_observation_key` 字段（S 缺失）。

### A7. `perception/mock.py` — J 领先｜建议：**取 J**
- J 三个场景各加 5 个观测字段：`legal_l3_area`、`has_no_marking`、`has_ramp`、`accident_scene`、`traffic_control`（供 ODC 扩展/城市 override 类规则用）；S 还是 10-28 的旧 fixture。

### A8. `__init__.py` / `report/` 等 — 无差异
（`diff -rq` 确认其余文件一致；`__pycache__` 忽略。）

---

## B. 测试对照

### B1. 共有测试文件（4 对，双向互补，**取并集合并**）

| S 文件 | J 文件 | diff 行 | J 独有 | S 独有 | 建议 |
|---|---|---|---|---|---|
| `tests/test_judge_loader.py` | `tests/test_loader.py` | 291 | 强 schema 14 例、`merge_profiles` 2 例、`confidence_observation_key` 解析 | warning_codes 校验 5 例 + **doc-drift 1 例**（引用 `scripts/gen_warning_codes_doc.py`，依赖 suite 资产） | 主体并集入 J；doc-drift 例**抽回 suite** 存为 `tests/test_warning_codes_doc.py` |
| `tests/test_judge_evaluate.py` | `tests/test_evaluate.py` | 212 | NaN/inf/严格 bool/置信度/未知枚举/`undefined_as==UNDEFINED` 共 15 例 | （无独有函数名） | 取 J 全量为准 |
| `tests/test_judge_margin.py` | `tests/test_margin.py` | 106 | 带边语义 10 例（`EDGE_REL_TOL`、`isfinite`） | （无独有函数名） | 取 J 全量为准 |
| `tests/test_judge_e2e.py` | `tests/test_e2e.py` | 148 | exit code 0/1/3 5 例（含 mock 判决一致性、effective_switches） | 用 suite 路径 `examples/observations/`；**exit 断言 `==0` 与 J 冲突** | **e2e 留 suite**（集成）但 exit 断言改 1/3 对齐 A1；J 自留其版 |

`tests/test_judge_composite_aggregate.py` ↔ `tests/test_composite_aggregate.py`：**diff = 0**，纯重复，suite 侧删除。

### B2. S 新增测试（未提交）

| 文件 | 用例 | 建议 |
|---|---|---|
| `tests/test_missing_policy.py`（12 例） | MissingPolicy 解析/优先级/合并/`test_switches_config_has_no_dead_keys` | **移入 J**（含死键断言，正好覆盖 A5/C2 的删除） |
| `tests/test_city_override.py`（4 例） | override 提升 intersection 禁令等 | **移入 J**（需复制 `profiles/overrides/city_noa_cn.json` 进 J） |
| `tests/test_ext_profile.py`（9 例） | extends：灯光/摩擦/密度/限速规则 | **移入 J**（需复制 `profiles/odc_ext_cn.json` 进 J） |
| `tests/test_oem_coverage.py` | 读 `docs/oem_practice.md`（suite 资产） | **留 suite** |

留 suite 的其余：`test_pipeline_mock`、`test_cli_demo`、`test_mapping_*`、`test_observation_builder`、`test_smoke_attr`、`test_judge_e2e`。

---

## C. profiles / switches / docs / observations

### C1. `profiles/byd_tjp.json`（diff 20 行）｜建议：**取 S**
仅 2 处 warning_codes 更名：`PER-CONF-NEAR` → `PER-CAM-CONF-NEAR` / `PER-LIDAR-CONF-NEAR`（S 侧 warning_codes 文档工作的配套，消除同码复用）。

### C2. `switches/default_switches.json`（diff 17 行）｜建议：**取 S**
- S 新增分组 `surface`、`traffic`、`extension`（新 profile 元素依赖）。
- S 删除 `boundary_severity`、`on_switch_mismatch` —— **grep 确认两侧代码均无消费者（死键）**，且 S 测试 `test_switches_config_has_no_dead_keys` 明确断言不存在。合并时 J 的 `loader.py` 默认值注入（A5）同步删除。

### C3. `docs/semantics.md`（diff 127 行）｜建议：**合并改写**
- J 版（11-06）：`undefined_as=undefined`、非法观测（NaN/非 bool/未知枚举/低置信）视同缺失清单、统一 margin 判定式、指向 `docs/odd_rules.md`。
- S 版（11-05）：`missing_policy` 覆盖层优先级表、`policy_source` 字段、指向 suite `architecture_contract.md`。
- 合并目标（迁入 J 时改写）：优先级链 = `requirement=undefined` → `undefined_as` → `missing_policy(by_element→by_tag→default)` → `definition_mode`，两侧清单全保留。

### C4. S 新增文件（未提交）
- **复制进 J**：`profiles/odc_ext_cn.json`、`profiles/overrides/city_noa_cn.json`（B2 移入测试依赖；是复制，S 保留副本供自己 CLI 用）。
- **留 S**：`docs/architecture_contract.md`、`docs/oem_practice.md`、`docs/warning_codes.md` 改动、`docs/adapters.md`、`docs/mapping.md`、`plans/`、`examples/sidecar_highway.json` 改动、`odd_suite/**` 全部改动（第一方胶水，与 J 无关）。

### C5. `observations/`
`ODD_Judge/observations/` 与 `suite/examples/observations/` **逐字节一致**（diff -rq 无输出），无冲突。

### C6. ODD_Judge 独有资产
`docs/odd_rules.md`、`.pytest_cache/`（不入库，加 .gitignore 即可）。

---

## D. 决议表（逐行确认）

| # | 项 | 建议 | 你批准(✓/✗) | 备注 |
|---|---|---|---|---|
| D1 | `cli.py` exit code 0/1/3/4 + 错误处理 | 取 **J**；S e2e 断言 `==0` 改 `==1` | | |
| D2 | `margin.py` 统一语义 + `EDGE_REL_TOL` + isfinite | 取 **J** | | |
| D3 | `evaluate.py`：missing_policy 取 **S**，NaN/未知枚举/严格 bool/置信度(`confidence_observation_key`)/undefined_as-UNDEFINED 取 **J** | 合并（顺序：undefined_as → missing_policy → mode） | | J 的 confidence 校验要求 profile 必须带 `confidence_observation_key`，验收时跑全量 profile 校验 |
| D4 | `schema.py`：J 强校验骨架 + S 的 warning_codes/missing_policy 校验 | 合并 | | |
| D5 | `api.py`/`loader.py`：S 的 bundle/missing_policy/查重 + 恢复 J 的 `confidence_observation_key` 解析、删 `boundary_severity` 默认 | 合并 | | |
| D6 | `types.py`：并集（S: MissingPolicy/policy_source；J: confidence_observation_key） | 并集 | | |
| D7 | `mock.py` fixture 扩 5 字段 | 取 **J** | | |
| D8 | 共有测试 4 对取并集入 J；doc-drift 例抽回 suite；composite 重复删 | 按 B1 | | |
| D9 | S 新测试 `test_missing_policy`/`test_city_override`/`test_ext_profile` 移入 J（复制 2 个 profile）；`test_oem_coverage` 留 S | 按 B2 | | |
| D10 | `byd_tjp.json` warning_codes 更名 | 取 **S** | | |
| D11 | `switches`：新增三组 + 删 2 个死键（含 J loader 默认值同步删） | 取 **S** | | |
| D12 | `semantics.md` 合并改写（优先级链含 missing_policy） | 合并 | | |
| D13 | S 未提交的 `odd_suite/**`、docs、examples 改动 | 全部留在 S（基线提交），与 J 无关 | | |

**评审方式**：直接在上表批注，或口头告诉我"全按建议"/"Dx 改为取另一侧"。
确认后进入 Phase 1（同步进 ODD_Judge + `git init`）。

---
---

# 附录：全量原始 diff

（A 包文件 8 项 → B 测试 4 对 → C 配置/文档；`<` = ODD_Judge(J)，`>` = suite(S)）

---

## 附录 A: odd_judge/api.py

```diff
--- ODD_Judge/odd_judge/api.py	2026-11-06 11:57:39.640094521 +0800
+++ odd_perception_suite/odd_judge/api.py	2026-11-05 12:30:52.830580449 +0800
@@ -7,9 +7,9 @@
 
 from .config.loader import (
     load_observation,
-    load_profile_with_extends,
+    load_profile_bundle,
     load_switches,
-    parse_profile,
+    parse_profile_bundle,
 )
 from .core.aggregate import aggregate
 from .core.evaluate import evaluate_rule
@@ -17,6 +17,7 @@
     DefinitionMode,
     EvalContext,
     JudgmentReport,
+    MissingPolicy,
     Observation,
     Rule,
     RuleResult,
@@ -48,24 +49,6 @@
                 return False, f"by_tag[{tag}]=false"
         return True, None
 
-    def effective_switches(self) -> list[str]:
-        """Switches set to a non-default value (i.e. those altering the judgment)."""
-        out: list[str] = []
-        if not self.global_enabled:
-            out.append("global_enabled=false")
-        for key in sorted(self.by_element):
-            if self.by_element[key] is False:
-                out.append(f"by_element[{key}]=false")
-        for key in sorted(self.by_rule_type):
-            if self.by_rule_type[key] is False:
-                out.append(f"by_rule_type[{key}]=false")
-        for key in sorted(self.by_tag):
-            if self.by_tag[key] is False:
-                out.append(f"by_tag[{key}]=false")
-        if not self.boundary_warnings_enabled:
-            out.append("boundary_warnings_enabled=false")
-        return out
-
 
 class OddJudge:
     """Main API: load profiles/switches, evaluate observations."""
@@ -76,12 +59,14 @@
         definition_mode: DefinitionMode,
         rules: list[Rule],
         switches: dict[str, Any] | None = None,
+        missing_policy: MissingPolicy | None = None,
     ):
         self.profile_id = profile_id
         self.definition_mode = definition_mode
         self.rules = rules
         self.switch_policy = SwitchPolicy(switches)
         self.switches_config = switches or {}
+        self.missing_policy = missing_policy
 
     @classmethod
     def from_files(
@@ -91,11 +76,17 @@
         extends_paths: list[str | Path] | None = None,
         override_paths: list[str | Path] | None = None,
     ) -> "OddJudge":
-        pid, mode, rules = load_profile_with_extends(
+        bundle = load_profile_bundle(
             profile_path, extends_paths=extends_paths, override_paths=override_paths
         )
         switches = load_switches(switches_path)
-        return cls(pid, mode, rules, switches)
+        return cls(
+            bundle.profile_id,
+            bundle.definition_mode,
+            bundle.rules,
+            switches,
+            bundle.missing_policy,
+        )
 
     @classmethod
     def from_dicts(
@@ -103,14 +94,20 @@
         profile_data: dict[str, Any],
         switches: dict[str, Any] | None = None,
     ) -> "OddJudge":
-        pid, mode, rules = parse_profile(profile_data)
-        return cls(pid, mode, rules, switches)
+        bundle = parse_profile_bundle(profile_data)
+        return cls(
+            bundle.profile_id,
+            bundle.definition_mode,
+            bundle.rules,
+            switches,
+            bundle.missing_policy,
+        )
 
     def evaluate(self, obs: Observation | dict[str, Any]) -> JudgmentReport:
         if isinstance(obs, dict):
             obs = Observation.from_dict(obs)
 
-        ctx = EvalContext(definition_mode=self.definition_mode)
+        ctx = EvalContext(definition_mode=self.definition_mode, missing_policy=self.missing_policy)
         results: list[RuleResult] = []
         skipped: list[dict[str, Any]] = []
         undefined_handled: list[dict[str, Any]] = []
@@ -125,20 +122,21 @@
             results.append(result)
 
             if result.state == ScenarioState.UNDEFINED:
-                undefined_handled.append(
-                    {
-                        "element_id": rule.element_id,
-                        "fallback": result.reason_code or "undefined",
-                        "mode": self.definition_mode.value,
-                    }
-                )
+                entry = {
+                    "element_id": rule.element_id,
+                    "fallback": result.reason_code or "undefined",
+                    "mode": self.definition_mode.value,
+                }
+                if result.policy_source:
+                    entry["policy"] = result.policy_source
+                undefined_handled.append(entry)
 
         return aggregate(
             profile_id=self.profile_id,
             results=results,
             skipped=skipped,
             undefined_handled=undefined_handled,
-            active_switches=self.switch_policy.effective_switches(),
+            active_switches=list(self.switches_config.keys()),
             timestamp=obs.timestamp,
             definition_mode=self.definition_mode.value,
             boundary_warnings_enabled=self.switch_policy.boundary_warnings_enabled,
```

---

## 附录 A: odd_judge/cli.py

```diff
--- ODD_Judge/odd_judge/cli.py	2026-11-06 11:57:56.348555030 +0800
+++ odd_perception_suite/odd_judge/cli.py	2026-10-28 14:24:18.962986914 +0800
@@ -1,11 +1,4 @@
-"""CLI: evaluate | validate | list-rules.
-
-Exit codes for `evaluate` (2 is reserved by argparse usage errors):
-    0  INSIDE / UNDEFINED / SKIPPED  → recommendation `continue`
-    1  BOUNDARY                      → recommendation `warn_driver`
-    3  OUTSIDE                       → recommendation `mrm_or_refuse`
-    4  input / config / runtime error (message on stderr)
-"""
+"""CLI: evaluate | validate | list-rules."""
 
 from __future__ import annotations
 
@@ -17,22 +10,8 @@
 from .api import OddJudge
 from .config.loader import load_profile_with_extends, load_switches
 from .config.schema import validate_profile_dict, validate_switches_dict
-from .core.errors import OddJudgeError
 from .report.render import render
 
-EXIT_OK = 0
-EXIT_BOUNDARY = 1
-EXIT_OUTSIDE = 3
-EXIT_ERROR = 4
-
-_STATE_EXIT = {
-    "INSIDE": EXIT_OK,
-    "UNDEFINED": EXIT_OK,
-    "SKIPPED": EXIT_OK,
-    "BOUNDARY": EXIT_BOUNDARY,
-    "OUTSIDE": EXIT_OUTSIDE,
-}
-
 
 def _resolve_profile_path(name_or_path: str, base: Path) -> Path:
     p = Path(name_or_path)
@@ -48,55 +27,41 @@
     raise FileNotFoundError(f"Profile not found: {name_or_path}")
 
 
-def _fail(msg: str) -> int:
-    print(f"ERROR: {msg}", file=sys.stderr)
-    return EXIT_ERROR
-
-
 def cmd_evaluate(args: argparse.Namespace) -> int:
-    if not args.obs and not args.scenario:
-        return _fail("需要 --obs <path> 或 --scenario <name> 之一")
-
     base = Path(args.project_root) if args.project_root else Path.cwd()
-    try:
-        profile_path = _resolve_profile_path(args.profile, base)
+    profile_path = _resolve_profile_path(args.profile, base)
 
-        extends = []
-        if args.extends:
-            for e in args.extends:
-                extends.append(_resolve_profile_path(e, base))
-        overrides = []
-        if args.override:
-            for o in args.override:
-                overrides.append(Path(o) if Path(o).exists() else base / "profiles" / "overrides" / o)
-
-        switches_path = args.switches
-        if switches_path and not Path(switches_path).exists():
-            switches_path = base / args.switches
-
-        judge = OddJudge.from_files(
-            profile_path,
-            switches_path=switches_path,
-            extends_paths=extends or None,
-            override_paths=overrides or None,
-        )
+    extends = []
+    if args.extends:
+        for e in args.extends:
+            extends.append(_resolve_profile_path(e, base))
+    overrides = []
+    if args.override:
+        for o in args.override:
+            overrides.append(Path(o) if Path(o).exists() else base / "profiles" / "overrides" / o)
+
+    switches_path = args.switches
+    if switches_path and not Path(switches_path).exists():
+        switches_path = base / args.switches
+
+    judge = OddJudge.from_files(
+        profile_path,
+        switches_path=switches_path,
+        extends_paths=extends or None,
+        override_paths=overrides or None,
+    )
 
-        if args.scenario:
-            from .perception.mock import MockAdapter
+    if args.scenario:
+        from .perception.mock import MockAdapter
 
-            adapter = MockAdapter(args.scenario)
-            reports = [judge.evaluate(adapter.get_observation())]
+        adapter = MockAdapter(args.scenario)
+        reports = [judge.evaluate(adapter.get_observation())]
+    else:
+        obs_path = Path(args.obs) if Path(args.obs).exists() else base / args.obs
+        if args.stream:
+            reports = judge.evaluate_stream(obs_path)
         else:
-            obs_path = Path(args.obs) if Path(args.obs).exists() else base / args.obs
-            if args.stream:
-                reports = judge.evaluate_stream(obs_path)
-            else:
-                reports = [judge.evaluate_path(obs_path)]
-    except (OddJudgeError, ValueError, OSError) as e:
-        return _fail(str(e))
-
-    if not reports:
-        return _fail("观测流为空，无可判决帧")
+            reports = [judge.evaluate_path(obs_path)]
 
     fmt = args.format
     outputs = [render(r, fmt=fmt) for r in reports]
@@ -108,22 +73,18 @@
     else:
         print(text)
 
-    # Exit code reflects the worst state across the (possibly streamed) run
-    worst = max(_STATE_EXIT.get(r.state.value, EXIT_OK) for r in reports)
-    return worst
+    # Exit code reflects worst state
+    worst = max((r.state.value for r in reports), key=lambda s: {"OUTSIDE": 3, "BOUNDARY": 2, "INSIDE": 1, "UNDEFINED": 0, "SKIPPED": 0}.get(s, 0))
+    return 0
 
 
 def cmd_validate(args: argparse.Namespace) -> int:
-    try:
-        path = Path(args.profile)
-        if not path.exists():
-            base = Path(args.project_root) if args.project_root else Path.cwd()
-            path = _resolve_profile_path(args.profile, base)
-        with open(path, "r", encoding="utf-8") as f:
-            data = json.load(f)
-    except (OddJudgeError, ValueError, OSError) as e:
-        return _fail(str(e))
-
+    path = Path(args.profile)
+    if not path.exists():
+        base = Path(args.project_root) if args.project_root else Path.cwd()
+        path = _resolve_profile_path(args.profile, base)
+    with open(path, "r", encoding="utf-8") as f:
+        data = json.load(f)
     errors = validate_profile_dict(data)
     if errors:
         print(f"INVALID: {path}")
@@ -131,17 +92,13 @@
             print(f"  - {e}")
         return 1
     print(f"VALID: {path}  ({len(data.get('elements', []))} elements)")
-    return EXIT_OK
+    return 0
 
 
 def cmd_list_rules(args: argparse.Namespace) -> int:
     base = Path(args.project_root) if args.project_root else Path.cwd()
-    try:
-        profile_path = _resolve_profile_path(args.profile, base)
-        pid, mode, rules = load_profile_with_extends(profile_path)
-    except (OddJudgeError, ValueError, OSError) as e:
-        return _fail(str(e))
-
+    profile_path = _resolve_profile_path(args.profile, base)
+    pid, mode, rules = load_profile_with_extends(profile_path)
     if args.tag:
         rules = [r for r in rules if args.tag in r.tags]
     print(f"profile={pid}  mode={mode.value}  rules={len(rules)}")
@@ -149,7 +106,7 @@
         tags = ",".join(r.tags) if r.tags else "-"
         codes = r.warning_codes.get("outside") or r.warning_codes.get("boundary") or ""
         print(f"  {r.element_id:<40} {r.rule_type:<20} req={r.requirement.value:<12} tags={tags:<16} code={codes}")
-    return EXIT_OK
+    return 0
 
 
 def _add_common(p: argparse.ArgumentParser) -> None:
```

---

## 附录 A: odd_judge/config/loader.py

```diff
--- ODD_Judge/odd_judge/config/loader.py	2026-11-06 11:56:29.543012251 +0800
+++ odd_perception_suite/odd_judge/config/loader.py	2026-11-05 12:35:05.099154327 +0800
@@ -3,6 +3,7 @@
 from __future__ import annotations
 
 import json
+from dataclasses import dataclass
 from pathlib import Path
 from typing import Any
 
@@ -11,12 +12,17 @@
 from ..core.types import (
     BoundaryMargin,
     DefinitionMode,
+    MissingPolicy,
     Requirement,
     Rule,
     ThresholdSpec,
 )
 from .registry import get_extra_elements
-from .schema import validate_profile_dict, validate_switches_dict
+from .schema import (
+    check_warning_code_duplicates,
+    validate_profile_dict,
+    validate_switches_dict,
+)
 
 
 def _load_json(path: str | Path) -> dict[str, Any]:
@@ -77,11 +83,34 @@
         road_rule_match=el.get("road_rule_match"),
         confidence_threshold=el.get("confidence_threshold"),
         confidence_band=el.get("confidence_band"),
-        confidence_observation_key=el.get("confidence_observation_key"),
     )
 
 
-def parse_profile(data: dict[str, Any]) -> tuple[str, DefinitionMode, list[Rule]]:
+def parse_missing_policy(data: dict[str, Any]) -> MissingPolicy | None:
+    """Build a ``MissingPolicy`` from a profile dict, or ``None`` when absent."""
+    raw = data.get("missing_policy")
+    if raw is None:
+        return None
+    return MissingPolicy(
+        by_element=dict(raw.get("by_element") or {}),
+        by_tag=dict(raw.get("by_tag") or {}),
+        default=str(raw.get("default", "mode")),
+    )
+
+
+@dataclass(frozen=True)
+class ProfileBundle:
+    """A parsed profile plus its missing-observation policy."""
+
+    profile_id: str
+    definition_mode: DefinitionMode
+    rules: list[Rule]
+    missing_policy: MissingPolicy | None = None
+
+
+def _parse_profile_core(
+    data: dict[str, Any],
+) -> tuple[str, DefinitionMode, list[Rule], MissingPolicy | None]:
     errors = validate_profile_dict(data)
     if errors:
         raise ConfigError("Profile validation failed:\n  " + "\n  ".join(errors))
@@ -90,53 +119,103 @@
     mode = DefinitionMode(data.get("definition_mode", "default"))
     rules = [_parse_rule(el) for el in data["elements"]]
 
-    # Append registry-registered extra elements
-    for el in get_extra_elements().values():
+    # Append registry-registered extra elements (cross-source duplicate check)
+    extras = get_extra_elements()
+    dup_errors = check_warning_code_duplicates(list(data["elements"]) + list(extras.values()))
+    if dup_errors:
+        raise ConfigError("Profile validation failed:\n  " + "\n  ".join(dup_errors))
+    for el in extras.values():
         rules.append(_parse_rule(el))
 
+    return profile_id, mode, rules, parse_missing_policy(data)
+
+
+def parse_profile(data: dict[str, Any]) -> tuple[str, DefinitionMode, list[Rule]]:
+    profile_id, mode, rules, _ = _parse_profile_core(data)
     return profile_id, mode, rules
 
 
+def parse_profile_bundle(data: dict[str, Any]) -> ProfileBundle:
+    profile_id, mode, rules, policy = _parse_profile_core(data)
+    return ProfileBundle(profile_id, mode, rules, policy)
+
+
 def load_profile(path: str | Path) -> tuple[str, DefinitionMode, list[Rule]]:
     data = _load_json(path)
     return parse_profile(data)
 
 
+def _merge_missing_policy(*policies: dict[str, Any] | None) -> dict[str, Any] | None:
+    """Overlay missing_policy blocks; later sources win per sub-key.
+
+    Elements merge with ``base_priority`` (base wins), but policy refines safety
+    semantics per source, so the later (more specific) source wins.
+    """
+    merged: dict[str, Any] = {}
+    seen = False
+    for pol in policies:
+        if not pol:
+            continue
+        seen = True
+        for group in ("by_element", "by_tag"):
+            sub = pol.get(group)
+            if isinstance(sub, dict):
+                merged.setdefault(group, {}).update(sub)
+        if "default" in pol:
+            merged["default"] = pol["default"]
+    return merged if seen else None
+
+
+def _apply_overrides(
+    base: dict[str, Any],
+    override_paths: list[str | Path] | None,
+) -> dict[str, Any]:
+    """Same element_id replaces; new element_id appends; policy overlays later-wins."""
+    for op in (override_paths or []):
+        od = _load_json(op)
+        by_id = {el["element_id"]: el for el in base.get("elements", []) if "element_id" in el}
+        for el in od.get("elements", []):
+            eid = el.get("element_id")
+            if eid and eid in by_id:
+                by_id[eid] = el
+            elif eid:
+                by_id[eid] = el
+        base["elements"] = list(by_id.values())
+        base["missing_policy"] = _merge_missing_policy(
+            base.get("missing_policy"), od.get("missing_policy")
+        )
+    return base
+
+
 def merge_profiles(
     base: dict[str, Any],
     *others: dict[str, Any],
     base_priority: bool = True,
 ) -> dict[str, Any]:
-    """Merge profile dicts. On element_id collision, base wins if base_priority.
-
-    With base_priority=True the first occurrence (base) is kept; otherwise
-    later sources overwrite, so non-base profiles win over the base.
-    """
+    """Merge profile dicts. On element_id collision, base wins if base_priority."""
     merged_elements: dict[str, dict[str, Any]] = {}
     order: list[str] = []
 
-    if base_priority:
-        # base first: first write wins
-        for src in (base, *others):
-            for el in src.get("elements", []):
-                eid = el.get("element_id")
-                if not eid or eid in merged_elements:
-                    continue
+    sources = [base, *others] if base_priority else [*others, base]
+    for src in sources:
+        for el in src.get("elements", []):
+            eid = el.get("element_id")
+            if not eid:
+                continue
+            if eid not in merged_elements:
                 order.append(eid)
+            # later sources do not overwrite earlier if base_priority
+            # with base_priority=True, base is first so it wins
+            if eid not in merged_elements:
                 merged_elements[eid] = el
-    else:
-        # base first, others overwrite: last write wins
-        for src in (base, *others):
-            for el in src.get("elements", []):
-                eid = el.get("element_id")
-                if not eid:
-                    continue
-                if eid not in merged_elements:
-                    order.append(eid)
+            elif not base_priority:
                 merged_elements[eid] = el
 
     result = dict(base)
     result["elements"] = [merged_elements[eid] for eid in order]
+    result["missing_policy"] = _merge_missing_policy(
+        *[src.get("missing_policy") for src in sources]
+    )
     return result
 
 
@@ -145,24 +224,21 @@
     extends_paths: list[str | Path] | None = None,
     override_paths: list[str | Path] | None = None,
 ) -> tuple[str, DefinitionMode, list[Rule]]:
+    bundle = load_profile_bundle(path, extends_paths, override_paths)
+    return bundle.profile_id, bundle.definition_mode, bundle.rules
+
+
+def load_profile_bundle(
+    path: str | Path,
+    extends_paths: list[str | Path] | None = None,
+    override_paths: list[str | Path] | None = None,
+) -> ProfileBundle:
     base = _load_json(path)
     others = [_load_json(p) for p in (extends_paths or [])]
     if others:
         base = merge_profiles(base, *others, base_priority=True)
-
-    # Overrides: same element_id replaces; new element_id appends
-    for op in (override_paths or []):
-        od = _load_json(op)
-        by_id = {el["element_id"]: el for el in base.get("elements", []) if "element_id" in el}
-        for el in od.get("elements", []):
-            eid = el.get("element_id")
-            if eid and eid in by_id:
-                by_id[eid] = el
-            elif eid:
-                by_id[eid] = el
-        base["elements"] = list(by_id.values())
-
-    return parse_profile(base)
+    base = _apply_overrides(base, override_paths)
+    return parse_profile_bundle(base)
 
 
 def load_switches(path: str | Path | None = None) -> dict[str, Any]:
@@ -173,7 +249,6 @@
             "by_rule_type": {},
             "by_tag": {},
             "boundary_warnings_enabled": True,
-            "boundary_severity": "warning",
         }
     data = _load_json(path)
     errors = validate_switches_dict(data)
@@ -185,7 +260,6 @@
         "by_rule_type": {},
         "by_tag": {},
         "boundary_warnings_enabled": True,
-        "boundary_severity": "warning",
     }
     defaults.update(data)
     return defaults
```

---

## 附录 A: odd_judge/config/schema.py

```diff
--- ODD_Judge/odd_judge/config/schema.py	2026-11-06 11:57:03.829063178 +0800
+++ odd_perception_suite/odd_judge/config/schema.py	2026-11-05 12:28:42.809628325 +0800
@@ -2,140 +2,19 @@
 
 from __future__ import annotations
 
-import math
+import re
 from typing import Any
 
+from ..core.errors import ConfigError
+
 VALID_REQUIREMENTS = {"allowed", "not_allowed", "undefined"}
 VALID_RULE_TYPES = {"numeric_range", "qualitative_enum", "qualitative_bool"}
-# numeric_range ops (classify_numeric contract)
-NUMERIC_OPS = {">=", "<=", ">", "<", "between", "==", "!="}
-# applies_when / composite condition leaf ops (evaluate._eval_leaf contract)
-CONDITION_OPS = {">=", "<=", ">", "<", "in", "not_in", "==", "!=", "bool"}
+VALID_OPS = {">=", "<=", ">", "<", "between", "in", "not_in", "==", "!=", "bool"}
 VALID_MARGIN_MODES = {"absolute", "relative"}
 VALID_DEFINITION_MODES = {"permissive", "restrictive", "default"}
-VALID_UNDEFINED_AS = {"allowed", "not_allowed", "undefined"}
-
-
-def _is_number(v: Any) -> bool:
-    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
-
-
-def _validate_composite(cond: Any, prefix: str, errors: list[str]) -> None:
-    """Validate an applies_when tree; mirrors composite.parse_composite."""
-    if not isinstance(cond, dict):
-        errors.append(f"{prefix}: condition must be an object")
-        return
-    if "all_of" in cond:
-        children = cond["all_of"]
-        if not isinstance(children, list):
-            errors.append(f"{prefix}.all_of: must be a list")
-            return
-        for i, c in enumerate(children):
-            _validate_composite(c, f"{prefix}.all_of[{i}]", errors)
-    elif "any_of" in cond:
-        children = cond["any_of"]
-        if not isinstance(children, list):
-            errors.append(f"{prefix}.any_of: must be a list")
-            return
-        for i, c in enumerate(children):
-            _validate_composite(c, f"{prefix}.any_of[{i}]", errors)
-    elif "not" in cond:
-        _validate_composite(cond["not"], f"{prefix}.not", errors)
-    elif "key" in cond and "op" in cond:
-        op = cond["op"]
-        if op not in CONDITION_OPS:
-            errors.append(f"{prefix}: invalid condition op {op!r}")
-        if op in ("in", "not_in") and not isinstance(cond.get("value"), list):
-            errors.append(f"{prefix}: op {op} requires value to be a list")
-    else:
-        errors.append(f"{prefix}: unrecognized composite structure: {sorted(cond.keys())}")
-
-
-def _validate_enum(el: dict[str, Any], prefix: str, eid: Any, req: str, errors: list[str]) -> None:
-    allowed_raw = el.get("allowed_values")
-    boundary_raw = el.get("boundary_values")
-    forbidden_raw = el.get("forbidden_values")
-
-    if req == "allowed" and not allowed_raw:
-        errors.append(f"{prefix} ({eid}): enum with requirement=allowed needs allowed_values")
-    if req == "not_allowed":
-        if allowed_raw:
-            errors.append(
-                f"{prefix} ({eid}): enum with requirement=not_allowed must not define "
-                "allowed_values (use forbidden_values)"
-            )
-        if not forbidden_raw:
-            errors.append(f"{prefix} ({eid}): enum with requirement=not_allowed needs forbidden_values")
-
-    sets: dict[str, set[Any]] = {}
-    for name, raw in (
-        ("allowed_values", allowed_raw),
-        ("boundary_values", boundary_raw),
-        ("forbidden_values", forbidden_raw),
-    ):
-        if not raw:
-            continue
-        if not isinstance(raw, list):
-            errors.append(f"{prefix} ({eid}): {name} must be a list")
-            continue
-        try:
-            sets[name] = set(raw)
-        except TypeError:
-            errors.append(f"{prefix} ({eid}): {name} entries must be scalar values")
-            return
-
-    pairs = [
-        ("allowed_values", "boundary_values"),
-        ("allowed_values", "forbidden_values"),
-        ("boundary_values", "forbidden_values"),
-    ]
-    for a, b in pairs:
-        if a in sets and b in sets and (sets[a] & sets[b]):
-            overlap = ", ".join(sorted(map(str, sets[a] & sets[b])))
-            errors.append(f"{prefix} ({eid}): {a} and {b} overlap on: {overlap}")
-
-
-def _validate_confidence(el: dict[str, Any], prefix: str, eid: Any, rt: str | None, errors: list[str]) -> None:
-    threshold = el.get("confidence_threshold")
-    band = el.get("confidence_band")
-    key = el.get("confidence_observation_key")
-    if threshold is None and band is None and key is None:
-        return
-
-    if rt != "qualitative_bool":
-        errors.append(f"{prefix} ({eid}): confidence_* fields are only valid on qualitative_bool rules")
-        return
-    if threshold is None:
-        errors.append(f"{prefix} ({eid}): confidence_band/confidence_observation_key requires confidence_threshold")
-        return
-    if not _is_number(threshold) or not (0.0 <= float(threshold) <= 1.0):
-        errors.append(f"{prefix} ({eid}): confidence_threshold must be a number in [0, 1]")
-    if not key:
-        errors.append(f"{prefix} ({eid}): confidence_threshold requires confidence_observation_key")
-    if band is not None and (not _is_number(band) or float(band) < 0.0):
-        errors.append(f"{prefix} ({eid}): confidence_band must be a finite number >= 0")
-
-
-def _validate_numeric(el: dict[str, Any], prefix: str, eid: Any, errors: list[str]) -> None:
-    op = el.get("op")
-    if op not in NUMERIC_OPS:
-        errors.append(f"{prefix} ({eid}): invalid numeric op {op!r} (allowed: {sorted(NUMERIC_OPS)})")
-        return
-    if op in (">=", "<=", ">", "<", "==", "!="):
-        if el.get("threshold") is None:
-            errors.append(f"{prefix} ({eid}): op {op} requires threshold")
-        elif not _is_number(el.get("threshold")):
-            errors.append(f"{prefix} ({eid}): threshold must be a finite number")
-    if op == "between":
-        rng = el.get("range")
-        if rng is None:
-            errors.append(f"{prefix} ({eid}): op between requires range")
-        elif not isinstance(rng, (list, tuple)) or len(rng) != 2:
-            errors.append(f"{prefix} ({eid}): range must be [lo, hi]")
-        elif not all(_is_number(x) for x in rng):
-            errors.append(f"{prefix} ({eid}): range bounds must be finite numbers")
-        elif float(rng[0]) >= float(rng[1]):
-            errors.append(f"{prefix} ({eid}): range requires lo < hi")
+VALID_MISSING_POLICY_VALUES = {"allowed", "not_allowed", "undefined", "mode"}
+VALID_WARNING_CODE_KEYS = {"outside", "boundary"}
+WARNING_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]*(-[A-Z0-9]+)+$")
 
 
 def validate_profile_dict(data: dict[str, Any]) -> list[str]:
@@ -154,6 +33,8 @@
     if mode not in VALID_DEFINITION_MODES:
         errors.append(f"invalid definition_mode: {mode!r}")
 
+    errors.extend(_check_missing_policy(data.get("missing_policy")))
+
     elements = data["elements"]
     if not isinstance(elements, list):
         errors.append("elements must be a list")
@@ -177,29 +58,27 @@
         if req not in VALID_REQUIREMENTS:
             errors.append(f"{prefix} ({eid}): invalid requirement {req!r}")
 
-        ua = el.get("undefined_as")
-        if ua is not None and ua not in VALID_UNDEFINED_AS:
-            errors.append(f"{prefix} ({eid}): invalid undefined_as {ua!r}")
-
-        if "expect" in el and el.get("expect") is not None and not isinstance(el.get("expect"), bool):
-            errors.append(f"{prefix} ({eid}): expect must be a boolean")
-
         rt = el.get("rule_type")
         if rt is not None and rt not in VALID_RULE_TYPES:
-            # registered custom types pass through; strict checks below only
-            # apply to the built-ins
-            pass
+            # allow registered custom types — only warn-level, not error
+            # but validate the known ones strictly
+            if rt not in ("numeric_range", "qualitative_enum", "qualitative_bool"):
+                pass  # custom type, skip strict check
 
         if not el.get("observation_key") and req != "undefined":
             errors.append(f"{prefix} ({eid}): missing observation_key")
 
         if rt == "numeric_range":
-            _validate_numeric(el, prefix, eid, errors)
-
-        if rt == "qualitative_enum" and req != "undefined":
-            _validate_enum(el, prefix, eid, req, errors)
+            op = el.get("op")
+            if op not in VALID_OPS:
+                errors.append(f"{prefix} ({eid}): invalid op {op!r}")
+            if op in (">=", "<=", ">", "<") and el.get("threshold") is None:
+                errors.append(f"{prefix} ({eid}): op {op} requires threshold")
+            if op == "between" and el.get("range") is None:
+                errors.append(f"{prefix} ({eid}): op between requires range")
 
-        _validate_confidence(el, prefix, eid, rt, errors)
+        if rt == "qualitative_enum" and req != "undefined" and not el.get("allowed_values") and not el.get("forbidden_values"):
+            errors.append(f"{prefix} ({eid}): enum rule needs allowed_values or forbidden_values")
 
         bm = el.get("boundary_margin")
         if bm is not None:
@@ -208,13 +87,81 @@
             else:
                 if bm.get("mode") not in VALID_MARGIN_MODES:
                     errors.append(f"{prefix} ({eid}): invalid boundary_margin.mode")
-                bmv = bm.get("value", 0.0)
-                if not _is_number(bmv) or float(bmv) < 0.0:
-                    errors.append(f"{prefix} ({eid}): boundary_margin.value must be a finite number >= 0")
 
-        if el.get("applies_when") is not None:
-            _validate_composite(el["applies_when"], f"{prefix} ({eid}).applies_when", errors)
+        wc = el.get("warning_codes")
+        if wc is not None:
+            if not isinstance(wc, dict):
+                errors.append(f"{prefix} ({eid}): warning_codes must be object")
+            else:
+                for key, val in wc.items():
+                    if key not in VALID_WARNING_CODE_KEYS:
+                        errors.append(
+                            f"{prefix} ({eid}): invalid warning_codes key {key!r} "
+                            f"(allowed: outside, boundary)"
+                        )
+                    if not isinstance(val, str) or not WARNING_CODE_RE.match(val):
+                        errors.append(f"{prefix} ({eid}): invalid warning code format: {val!r}")
+
+    errors.extend(check_warning_code_duplicates(elements))
+    return errors
 
+
+def _check_missing_policy(mp: object) -> list[str]:
+    """Validate the optional top-level ``missing_policy`` block."""
+    if mp is None:
+        return []
+    if not isinstance(mp, dict):
+        return ["missing_policy must be an object"]
+    errors: list[str] = []
+    for key in mp:
+        if key not in ("by_element", "by_tag", "default"):
+            errors.append(f"missing_policy: unknown key {key!r}")
+    for group in ("by_element", "by_tag"):
+        sub = mp.get(group)
+        if sub is None:
+            continue
+        if not isinstance(sub, dict):
+            errors.append(f"missing_policy.{group} must be an object")
+            continue
+        for name, value in sub.items():
+            if value not in VALID_MISSING_POLICY_VALUES:
+                errors.append(
+                    f"missing_policy.{group}[{name}]: invalid value {value!r} "
+                    f"(allowed: {sorted(VALID_MISSING_POLICY_VALUES)})"
+                )
+    default = mp.get("default")
+    if default is not None and default not in VALID_MISSING_POLICY_VALUES:
+        errors.append(
+            f"missing_policy.default: invalid value {default!r} "
+            f"(allowed: {sorted(VALID_MISSING_POLICY_VALUES)})"
+        )
+    return errors
+
+
+def check_warning_code_duplicates(elements: list[Any]) -> list[str]:
+    """Return errors when the same warning code is used by more than one element.
+
+    `boundary` and `outside` codes are counted uniformly; a code may appear
+    at most once across the whole profile (duplicate codes make report
+    consumers unable to attribute a warning to its source element).
+    """
+    errors: list[str] = []
+    code_owners: dict[str, str] = {}
+    for i, el in enumerate(elements):
+        if not isinstance(el, dict):
+            continue
+        eid = el.get("element_id") or f"elements[{i}]"
+        wc = el.get("warning_codes")
+        if not isinstance(wc, dict):
+            continue
+        for val in wc.values():
+            if not isinstance(val, str):
+                continue
+            owner = code_owners.get(val)
+            if owner is None:
+                code_owners[val] = eid
+            elif owner != eid:
+                errors.append(f"duplicate warning code {val!r}: used by {owner} and {eid}")
     return errors
 
 
```

---

## 附录 A: odd_judge/core/evaluate.py

```diff
--- ODD_Judge/odd_judge/core/evaluate.py	2026-11-06 11:56:23.018804513 +0800
+++ odd_perception_suite/odd_judge/core/evaluate.py	2026-11-05 12:27:57.525707459 +0800
@@ -2,7 +2,6 @@
 
 from __future__ import annotations
 
-import math
 from typing import Any
 
 from .margin import classify_numeric
@@ -90,40 +89,70 @@
     mode = ctx.definition_mode
     if rule.undefined_as is not None:
         # Element-level override for missing observation
-        if rule.undefined_as == Requirement.UNDEFINED:
+        if rule.undefined_as == Requirement.ALLOWED:
             return RuleResult(
                 element_id=rule.element_id,
-                state=ScenarioState.UNDEFINED,
+                state=ScenarioState.INSIDE,
                 observed=None,
-                message_zh=f"{rule.label_zh or rule.element_id}: 观测缺失按不定义处理",
+                message_zh=f"{rule.label_zh or rule.element_id}: 观测缺失按允许处理",
                 label_zh=rule.label_zh,
                 requirement=rule.requirement.value,
                 rule_type=rule.rule_type,
                 tags=rule.tags,
-                reason_code="UNDEFINED-AS-UNDEFINED",
             )
-        if rule.undefined_as == Requirement.ALLOWED:
+        if rule.undefined_as == Requirement.NOT_ALLOWED:
+            return RuleResult(
+                element_id=rule.element_id,
+                state=ScenarioState.OUTSIDE,
+                observed=None,
+                message_zh=f"{rule.label_zh or rule.element_id}: 观测缺失按不允许处理",
+                label_zh=rule.label_zh,
+                requirement=rule.requirement.value,
+                rule_type=rule.rule_type,
+                tags=rule.tags,
+                reason_code="UNDEFINED-AS-NOT-ALLOWED",
+            )
+
+    if ctx.missing_policy is not None:
+        value, source = ctx.missing_policy.resolve(rule)
+        if value == "allowed":
             return RuleResult(
                 element_id=rule.element_id,
                 state=ScenarioState.INSIDE,
                 observed=None,
-                message_zh=f"{rule.label_zh or rule.element_id}: 观测缺失按允许处理",
+                message_zh=f"{rule.label_zh or rule.element_id}: 观测缺失(missing_policy {source}→域内)",
                 label_zh=rule.label_zh,
                 requirement=rule.requirement.value,
                 rule_type=rule.rule_type,
                 tags=rule.tags,
+                reason_code="MISSING-POLICY-ALLOWED",
+                policy_source=source,
             )
-        if rule.undefined_as == Requirement.NOT_ALLOWED:
+        if value == "not_allowed":
             return RuleResult(
                 element_id=rule.element_id,
                 state=ScenarioState.OUTSIDE,
                 observed=None,
-                message_zh=f"{rule.label_zh or rule.element_id}: 观测缺失按不允许处理",
+                message_zh=f"{rule.label_zh or rule.element_id}: 观测缺失(missing_policy {source}→域外)",
                 label_zh=rule.label_zh,
                 requirement=rule.requirement.value,
                 rule_type=rule.rule_type,
                 tags=rule.tags,
-                reason_code="UNDEFINED-AS-NOT-ALLOWED",
+                reason_code="MISSING-POLICY-NOT-ALLOWED",
+                policy_source=source,
+            )
+        if value == "undefined":
+            return RuleResult(
+                element_id=rule.element_id,
+                state=ScenarioState.UNDEFINED,
+                observed=None,
+                message_zh=f"{rule.label_zh or rule.element_id}: 观测缺失(missing_policy {source}→不定义)",
+                label_zh=rule.label_zh,
+                requirement=rule.requirement.value,
+                rule_type=rule.rule_type,
+                tags=rule.tags,
+                reason_code="MISSING-POLICY-UNDEFINED",
+                policy_source=source,
             )
 
     if mode == DefinitionMode.PERMISSIVE:
@@ -204,9 +233,6 @@
         fval = float(val)
     except (TypeError, ValueError):
         return _handle_missing(rule, ctx)
-    if not math.isfinite(fval):
-        # NaN / ±inf are invalid sensor data, same fallback as a missing value
-        return _handle_missing(rule, ctx)
 
     raw_state, remaining, band = classify_numeric(fval, rule.spec)
     state = _apply_requirement(raw_state, rule.requirement)
@@ -250,19 +276,18 @@
         elif val in forbidden:
             raw = ScenarioState.OUTSIDE
         else:
-            # unknown enum value: treat as missing observation so the outcome
-            # follows definition_mode uniformly for every requirement
-            return _handle_missing(rule, ctx)
+            # unknown enum value
+            if rule.requirement == Requirement.ALLOWED:
+                raw = ScenarioState.OUTSIDE
+            else:
+                raw = ScenarioState.INSIDE
     elif forbidden:
-        # Reached only when allowed_values is empty; schema restricts such
-        # rules to requirement=not_allowed, so raw here lives in the
-        # "forbidden present" frame and _apply_requirement flips it.
         if val in forbidden:
-            raw = ScenarioState.INSIDE  # forbidden present → flips to OUTSIDE
+            raw = ScenarioState.INSIDE  # forbidden present → will flip for NOT_ALLOWED
         elif val in boundary:
             raw = ScenarioState.BOUNDARY
         else:
-            raw = ScenarioState.OUTSIDE  # forbidden absent → flips to INSIDE
+            raw = ScenarioState.OUTSIDE  # forbidden absent
     else:
         raw = ScenarioState.INSIDE
 
@@ -294,46 +319,33 @@
     if isinstance(val, ValueMissing) or val is VALUE_MISSING:
         return _handle_missing(rule, ctx)
 
-    # Only real booleans (and exact 0/1) are valid for a bool rule; strings
-    # like "false" must not be truth-evaluated into a wrong verdict.
-    if isinstance(val, bool):
-        actual = val
-    elif isinstance(val, (int, float)) and val in (0, 1):
-        actual = bool(val)
-    else:
-        return _handle_missing(rule, ctx)
-
     expect = rule.expect if rule.expect is not None else True
+    # Confidence-based boundary (optional)
+    conf_key = rule.spec.values[0] if (rule.spec and rule.spec.values) else None
     confidence = 1.0
     raw: ScenarioState
 
+    actual = bool(val)
     if actual == expect:
         raw = ScenarioState.INSIDE
-    elif rule.confidence_threshold is None:
-        raw = ScenarioState.OUTSIDE
     else:
-        # Confidence is consulted only on a mismatch; schema guarantees the
-        # key when a threshold is configured.
-        ckey = rule.confidence_observation_key
-        cval = obs.get(ckey) if ckey else VALUE_MISSING
-        if isinstance(cval, ValueMissing) or cval is VALUE_MISSING:
-            return _handle_missing(rule, ctx)
-        try:
-            confidence = float(cval)
-        except (TypeError, ValueError):
-            return _handle_missing(rule, ctx)
-        if not math.isfinite(confidence):
-            return _handle_missing(rule, ctx)
-
-        thr = float(rule.confidence_threshold)
-        band = float(rule.confidence_band or 0.0)
-        if confidence >= thr:
-            raw = ScenarioState.OUTSIDE  # reliable mismatch
-        elif confidence >= thr - band:
-            raw = ScenarioState.BOUNDARY  # inside the critical band
+        # Check confidence band for boundary
+        if rule.confidence_threshold is not None and conf_key:
+            cval = obs.get(conf_key)
+            if not isinstance(cval, ValueMissing) and cval is not VALUE_MISSING:
+                try:
+                    confidence = float(cval)
+                except (TypeError, ValueError):
+                    confidence = 1.0
+                band = rule.confidence_band or 0.0
+                if confidence >= rule.confidence_threshold - band:
+                    raw = ScenarioState.BOUNDARY
+                else:
+                    raw = ScenarioState.OUTSIDE
+            else:
+                raw = ScenarioState.OUTSIDE
         else:
-            # Too unreliable to assert anything → same fallback as missing
-            return _handle_missing(rule, ctx)
+            raw = ScenarioState.OUTSIDE
 
     state = _apply_requirement(raw, rule.requirement)
 
```

---

## 附录 A: odd_judge/core/margin.py

```diff
--- ODD_Judge/odd_judge/core/margin.py	2026-11-06 11:55:45.028540146 +0800
+++ odd_perception_suite/odd_judge/core/margin.py	2026-10-28 14:24:01.864723951 +0800
@@ -2,23 +2,10 @@
 
 from __future__ import annotations
 
-import math
 from typing import Optional
 
 from .types import BoundaryMargin, ScenarioState, ThresholdSpec
 
-# Relative tolerance applied when a safety margin lands exactly on the band
-# width, so a decimal value that is "one band away" classifies the same way
-# regardless of binary rounding (e.g. 2.5+0.2 vs the literal 2.7).
-EDGE_REL_TOL = 1e-12
-
-
-def _reaches_band(margin_value: float, band: float) -> bool:
-    """True when the safety margin is at least one full band (INSIDE edge)."""
-    if margin_value >= band:
-        return True
-    return band > 0.0 and math.isclose(margin_value, band, rel_tol=EDGE_REL_TOL)
-
 
 def classify_numeric(
     value: float,
@@ -26,77 +13,81 @@
 ) -> tuple[ScenarioState, Optional[float], Optional[float]]:
     """Classify a numeric observation against a threshold spec.
 
-    Unified margin principle for relational/range ops (>=, >, <=, <, between):
-
-        OUTSIDE   the violating set is hit (margin < 0, or == 0 for strict ops)
-        BOUNDARY  0 <= margin < band        (margin == 0 stays BOUNDARY)
-        INSIDE    margin >= band            (band edge belongs to INSIDE)
-
-    For equality ops the band is an inclusive tolerance zone around the
-    satisfying point (`==`) / forbidden point (`!=`): the outer band edge is
-    BOUNDARY, beyond it flips (== → OUTSIDE, != → INSIDE).
-
-    Returns (state, remaining_margin, margin_band); remaining_margin is the
-    signed safety margin (positive = inside, negative = outside).
+    Returns (state, remaining_margin, margin_band).
+    remaining_margin is signed distance to the nearest threshold
+    (positive = inside, negative = outside).
     """
-    if not math.isfinite(value):
-        raise ValueError(f"classify_numeric requires a finite value, got {value!r}")
-
     op = spec.op
-    margin_cfg = spec.margin or BoundaryMargin(mode="absolute", value=0.0)
+    margin = spec.margin or BoundaryMargin(mode="absolute", value=0.0)
 
     if op in (">=", ">") and spec.threshold is not None:
         thr = float(spec.threshold)
-        band = margin_cfg.band_for_threshold(thr)
-        m = value - thr  # exact sign: subtraction of two doubles never lies
-        violated = m < 0 if op == ">=" else m <= 0
-        remaining = m
+        band = margin.band_for_threshold(thr)
+        remaining = value - thr
+        if op == ">=":
+            if value >= thr + band:
+                return ScenarioState.INSIDE, remaining, band
+            if value >= thr:
+                return ScenarioState.BOUNDARY, remaining, band
+            return ScenarioState.OUTSIDE, remaining, band
+        # op == ">"
+        if value > thr + band:
+            return ScenarioState.INSIDE, remaining, band
+        if value > thr:
+            return ScenarioState.BOUNDARY, remaining, band
+        return ScenarioState.OUTSIDE, remaining, band
 
-    elif op in ("<=", "<") and spec.threshold is not None:
+    if op in ("<=", "<") and spec.threshold is not None:
         thr = float(spec.threshold)
-        band = margin_cfg.band_for_threshold(thr)
-        m = thr - value
-        violated = m < 0 if op == "<=" else m <= 0
-        remaining = m
+        band = margin.band_for_threshold(thr)
+        remaining = thr - value
+        if op == "<=":
+            if value <= thr - band:
+                return ScenarioState.INSIDE, remaining, band
+            if value <= thr:
+                return ScenarioState.BOUNDARY, remaining, band
+            return ScenarioState.OUTSIDE, remaining, band
+        # op == "<"
+        if value < thr - band:
+            return ScenarioState.INSIDE, remaining, band
+        if value < thr:
+            return ScenarioState.BOUNDARY, remaining, band
+        return ScenarioState.OUTSIDE, remaining, band
 
-    elif op == "between" and spec.range is not None:
+    if op == "between" and spec.range is not None:
         lo, hi = float(spec.range[0]), float(spec.range[1])
-        band = margin_cfg.band_for_range(lo, hi)
-        remaining = min(value - lo, hi - value)
-        violated = remaining < 0
+        band = margin.band_for_range(lo, hi)
+        if value < lo:
+            remaining = value - lo
+            return ScenarioState.OUTSIDE, remaining, band
+        if value > hi:
+            remaining = hi - value
+            return ScenarioState.OUTSIDE, remaining, band
+        dist_lo = value - lo
+        dist_hi = hi - value
+        remaining = min(dist_lo, dist_hi)
+        if remaining <= band:
+            return ScenarioState.BOUNDARY, remaining, band
+        return ScenarioState.INSIDE, remaining, band
 
-    elif op == "==" and spec.threshold is not None:
+    if op == "==" and spec.threshold is not None:
         thr = float(spec.threshold)
-        band = margin_cfg.band_for_threshold(thr)
-        d = abs(value - thr)
-        if d == 0.0:
+        band = margin.band_for_threshold(thr) if margin.value else 0.0
+        remaining = thr - value
+        if value == thr:
             return ScenarioState.INSIDE, 0.0, band
-        # tolerance zone closed at the band edge (fp-edge absorbed)
-        if band > 0.0 and (d < band or math.isclose(d, band, rel_tol=EDGE_REL_TOL)):
-            return ScenarioState.BOUNDARY, thr - value, band
-        return ScenarioState.OUTSIDE, thr - value, band
+        if band > 0 and abs(value - thr) <= band:
+            return ScenarioState.BOUNDARY, remaining, band
+        return ScenarioState.OUTSIDE, remaining, band
 
-    elif op == "!=" and spec.threshold is not None:
+    if op == "!=" and spec.threshold is not None:
         thr = float(spec.threshold)
-        band = margin_cfg.band_for_threshold(thr)
-        d = abs(value - thr)
-        if d == 0.0:
-            return ScenarioState.OUTSIDE, 0.0, band
-        # forbidden point hit only when exactly equal; near-miss is BOUNDARY
-        # while the band lasts, INSIDE (with fp tolerance) at the band edge.
-        if band > 0.0 and (d < band or math.isclose(d, band, rel_tol=EDGE_REL_TOL)):
-            return ScenarioState.BOUNDARY, d, band
-        return ScenarioState.INSIDE, d, band
+        band = margin.band_for_threshold(thr) if margin.value else 0.0
+        if value != thr:
+            return ScenarioState.INSIDE, abs(value - thr), band
+        return ScenarioState.OUTSIDE, 0.0, band
 
-    else:
-        raise ValueError(f"Unsupported numeric op for classify_numeric: {op!r}")
-
-    # Relational / between: unified three-way split on the safety margin.
-    if violated:
-        return ScenarioState.OUTSIDE, remaining, band
-    if _reaches_band(remaining, band):
-        return ScenarioState.INSIDE, remaining, band
-    return ScenarioState.BOUNDARY, remaining, band
+    raise ValueError(f"Unsupported numeric op for classify_numeric: {op!r}")
 
 
 def relative_band(threshold: float, fraction: float) -> float:
```

---

## 附录 A: odd_judge/core/types.py

```diff
--- ODD_Judge/odd_judge/core/types.py	2026-11-06 11:56:24.821862184 +0800
+++ odd_perception_suite/odd_judge/core/types.py	2026-11-05 12:27:40.730186327 +0800
@@ -115,7 +115,6 @@
     road_rule_match: Optional[str] = None
     confidence_threshold: Optional[float] = None
     confidence_band: Optional[float] = None
-    confidence_observation_key: Optional[str] = None
 
 
 @dataclass(frozen=True)
@@ -126,6 +125,7 @@
     remaining_margin: Optional[float] = None
     margin_band: Optional[float] = None
     reason_code: Optional[str] = None
+    policy_source: Optional[str] = None
     confidence: float = 1.0
     message_zh: str = ""
     label_zh: str = ""
@@ -141,6 +141,7 @@
             "remaining_margin": self.remaining_margin,
             "margin_band": self.margin_band,
             "reason_code": self.reason_code,
+            "policy_source": self.policy_source,
             "confidence": self.confidence,
             "message_zh": self.message_zh,
             "label_zh": self.label_zh,
@@ -216,10 +217,42 @@
         return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
 
 
+@dataclass(frozen=True)
+class MissingPolicy:
+    """Fallback for missing observations, resolved per element / tag / default.
+
+    Value ``"mode"`` means: fall through to the profile ``definition_mode``.
+    Resolution order (highest first): ``by_element`` -> first matching entry in
+    ``by_tag`` over ``rule.tags`` -> ``default``. Element-level ``undefined_as``
+    and ``requirement == undefined`` are checked by the caller before this.
+    """
+
+    by_element: dict[str, str] = field(default_factory=dict)
+    by_tag: dict[str, str] = field(default_factory=dict)
+    default: str = "mode"
+
+    def resolve(self, rule: "Rule") -> tuple[Optional[str], Optional[str]]:
+        """Return ``(value, source)`` for ``rule``; ``value=None`` when nothing matches.
+
+        ``source`` names the matched entry (``by_element[id]`` / ``by_tag[tag]`` /
+        ``default``) so reports can say which policy fired.
+        """
+        value = self.by_element.get(rule.element_id)
+        if value is not None:
+            return value, f"by_element[{rule.element_id}]"
+        for tag in rule.tags:
+            if tag in self.by_tag:
+                return self.by_tag[tag], f"by_tag[{tag}]"
+        if self.default and self.default != "mode":
+            return self.default, "default"
+        return None, None
+
+
 @dataclass
 class EvalContext:
     definition_mode: DefinitionMode = DefinitionMode.DEFAULT
     strict_undefined: bool = False
+    missing_policy: Optional[MissingPolicy] = None
 
 
 class RuleEvaluator(Protocol):
```

---

## 附录 A: odd_judge/perception/mock.py

```diff
--- ODD_Judge/odd_judge/perception/mock.py	2026-11-06 11:57:31.833875026 +0800
+++ odd_perception_suite/odd_judge/perception/mock.py	2026-10-28 14:18:16.512815260 +0800
@@ -26,11 +26,6 @@
                 "has_toll_station": False,
                 "has_railway_crossing": False,
                 "lane_count": 3,
-                "legal_l3_area": True,
-                "has_no_marking": False,
-                "has_ramp": False,
-                "accident_scene": False,
-                "traffic_control": False,
             },
             "vehicle": {
                 "speed_kmh": 72.0,
@@ -88,11 +83,6 @@
                 "has_toll_station": False,
                 "has_railway_crossing": False,
                 "lane_count": 2,
-                "legal_l3_area": True,
-                "has_no_marking": False,
-                "has_ramp": False,
-                "accident_scene": False,
-                "traffic_control": False,
             },
             "vehicle": {
                 "speed_kmh": 78.0,
@@ -150,11 +140,6 @@
                 "has_toll_station": False,
                 "has_railway_crossing": False,
                 "lane_count": 3,
-                "legal_l3_area": True,
-                "has_no_marking": False,
-                "has_ramp": False,
-                "accident_scene": False,
-                "traffic_control": False,
             },
             "vehicle": {
                 "speed_kmh": 60.0,
```

---

# 附录 B: 测试对照

## 附录 B: J/tests/test_loader.py ↔ S/tests/test_judge_loader.py

```diff
--- ODD_Judge/tests/test_loader.py	2026-11-06 11:59:04.341312795 +0800
+++ odd_perception_suite/tests/test_judge_loader.py	2026-10-29 10:47:45.360694820 +0800
@@ -74,215 +74,88 @@
     assert len(rules) == 1
 
 
-# ---------------------------------------------------------------------------
-# Schema: configuration errors must be caught at validate time
-# ---------------------------------------------------------------------------
-
-
-def _profile(elements):
-    return {"profile_id": "probe", "elements": elements}
-
-
-def _num(**overrides):
-    el = {
-        "element_id": "n1",
-        "requirement": "allowed",
-        "rule_type": "numeric_range",
-        "observation_key": "a.b",
-        "op": ">=",
-        "threshold": 1.0,
-    }
-    el.update(overrides)
-    return el
-
-
-def test_schema_rejects_condition_ops_on_numeric_range():
-    el = _num(op="in")
-    errors = validate_profile_dict(_profile([el]))
-    assert any("invalid numeric op" in e for e in errors)
-
-
-def test_schema_requires_threshold_for_equality_ops():
-    el = _num(op="==")
-    del el["threshold"]
-    errors = validate_profile_dict(_profile([el]))
-    assert any("requires threshold" in e for e in errors)
-
-
-def test_schema_requires_valid_range_for_between():
-    errors = validate_profile_dict(_profile([_num(op="between", range=[9.0, -9.0])]))
-    assert any("lo < hi" in e for e in errors)
-    errors = validate_profile_dict(_profile([_num(op="between", range=None)]))
-    assert any("requires range" in e for e in errors)
-
-
-def test_schema_rejects_negative_boundary_margin():
-    errors = validate_profile_dict(
-        _profile([_num(boundary_margin={"mode": "absolute", "value": -1.0})])
-    )
-    assert any("boundary_margin.value" in e for e in errors)
-
-
-def test_schema_rejects_bad_undefined_as():
-    errors = validate_profile_dict(_profile([_num(undefined_as="maybe")]))
-    assert any("invalid undefined_as" in e for e in errors)
-
-
-def test_schema_rejects_non_bool_expect():
-    el = {
-        "element_id": "b1",
-        "requirement": "allowed",
-        "rule_type": "qualitative_bool",
-        "observation_key": "a.b",
-        "expect": "yes",
-    }
-    errors = validate_profile_dict(_profile([el]))
-    assert any("expect must be a boolean" in e for e in errors)
-
-
-def test_schema_enum_allowed_requires_allowed_values():
-    el = {
-        "element_id": "e1",
-        "requirement": "allowed",
-        "rule_type": "qualitative_enum",
-        "observation_key": "a.b",
-        "forbidden_values": ["x"],
-    }
-    errors = validate_profile_dict(_profile([el]))
-    assert any("needs allowed_values" in e for e in errors)
-
-
-def test_schema_enum_not_allowed_must_not_use_allowed_values():
-    el = {
-        "element_id": "e2",
-        "requirement": "not_allowed",
-        "rule_type": "qualitative_enum",
-        "observation_key": "a.b",
-        "allowed_values": ["x"],
-        "forbidden_values": ["y"],
-    }
-    errors = validate_profile_dict(_profile([el]))
-    assert any("must not define" in e for e in errors)
-
-
-def test_schema_rejects_overlapping_enum_sets():
-    el = {
-        "element_id": "e3",
-        "requirement": "allowed",
-        "rule_type": "qualitative_enum",
-        "observation_key": "a.b",
-        "allowed_values": ["x", "y"],
-        "boundary_values": ["y"],
-    }
-    errors = validate_profile_dict(_profile([el]))
-    assert any("overlap" in e for e in errors)
-
-
-def test_schema_confidence_requires_key_and_bool_rule():
-    # threshold without key
-    errors = validate_profile_dict(
-        _profile([
-            {
-                "element_id": "c1",
-                "requirement": "allowed",
-                "rule_type": "qualitative_bool",
-                "observation_key": "a.b",
-                "confidence_threshold": 0.7,
-            }
-        ])
-    )
-    assert any("requires confidence_observation_key" in e for e in errors)
-
-    # confidence fields on a numeric rule → dead config, rejected
-    errors = validate_profile_dict(
-        _profile([
-            _num(confidence_threshold=0.7, confidence_observation_key="a.c")
-        ])
-    )
-    assert any("only valid on qualitative_bool" in e for e in errors)
-
-    # threshold out of range
-    errors = validate_profile_dict(
-        _profile([
+def _mini_profile(warning_codes_by_id: dict) -> dict:
+    """Build a minimal valid profile with one element per given warning_codes."""
+    return {
+        "profile_id": "mini",
+        "elements": [
             {
-                "element_id": "c2",
+                "element_id": eid,
                 "requirement": "allowed",
                 "rule_type": "qualitative_bool",
                 "observation_key": "a.b",
-                "confidence_threshold": 1.5,
-                "confidence_observation_key": "a.c",
+                "expect": True,
+                "warning_codes": wc,
             }
-        ])
-    )
-    assert any("confidence_threshold" in e and "[0, 1]" in e for e in errors)
-
-
-def test_schema_rejects_bad_applies_when_op():
-    el = _num(applies_when={"key": "s.type", "op": "like", "value": "x"})
-    errors = validate_profile_dict(_profile([el]))
-    assert any("invalid condition op" in e for e in errors)
-
-
-def test_schema_rejects_in_without_list_value():
-    el = _num(applies_when={"key": "s.type", "op": "in", "value": "expressway"})
-    errors = validate_profile_dict(_profile([el]))
-    assert any("requires value to be a list" in e for e in errors)
-
-
-def test_schema_accepts_valid_confidence_rule():
-    el = {
-        "element_id": "c3",
-        "requirement": "allowed",
-        "rule_type": "qualitative_bool",
-        "observation_key": "a.b",
-        "expect": True,
-        "confidence_threshold": 0.7,
-        "confidence_band": 0.1,
-        "confidence_observation_key": "a.c",
+            for eid, wc in warning_codes_by_id.items()
+        ],
     }
-    assert validate_profile_dict(_profile([el])) == []
-
-
-# ---------------------------------------------------------------------------
-# merge_profiles priority
-# ---------------------------------------------------------------------------
-
 
-def test_merge_profiles_base_priority_true_keeps_base():
-    from odd_judge.config.loader import merge_profiles
-
-    base = {"profile_id": "b", "elements": [{"element_id": "E", "v": 1}]}
-    other = {"profile_id": "o", "elements": [{"element_id": "E", "v": 2}]}
-    merged = merge_profiles(base, other, base_priority=True)
-    assert merged["elements"][0]["v"] == 1
 
+def test_warning_codes_invalid_key():
+    data = _mini_profile({"x1": {"warn": "X-001"}})
+    errors = validate_profile_dict(data)
+    assert any("invalid warning_codes key" in e for e in errors)
+
+
+def test_warning_codes_invalid_format():
+    data = _mini_profile({"x1": {"outside": "lower case code"}})
+    errors = validate_profile_dict(data)
+    assert any("invalid warning code format" in e for e in errors)
+
+
+def test_warning_codes_duplicate_across_elements():
+    data = _mini_profile(
+        {
+            "x1": {"outside": "X-001", "boundary": "X-NEAR"},
+            "x2": {"boundary": "X-NEAR"},
+        }
+    )
+    errors = validate_profile_dict(data)
+    assert any("duplicate warning code" in e and "X-NEAR" in e for e in errors)
 
-def test_merge_profiles_base_priority_false_keeps_other():
-    from odd_judge.config.loader import merge_profiles
 
-    base = {"profile_id": "b", "elements": [{"element_id": "E", "v": 1}]}
-    other = {"profile_id": "o", "elements": [{"element_id": "E", "v": 2}]}
-    merged = merge_profiles(base, other, base_priority=False)
-    assert merged["elements"][0]["v"] == 2
+def test_warning_codes_valid():
+    data = _mini_profile(
+        {
+            "x1": {"outside": "X-001", "boundary": "X-NEAR"},
+            "x2": {"outside": "X-002"},
+        }
+    )
+    assert validate_profile_dict(data) == []
 
 
-def test_loader_parses_confidence_observation_key():
-    data = {
-        "profile_id": "mini",
-        "elements": [
-            {
-                "element_id": "x1",
-                "requirement": "allowed",
-                "rule_type": "qualitative_bool",
-                "observation_key": "a.b",
-                "expect": True,
-                "confidence_threshold": 0.7,
-                "confidence_band": 0.1,
-                "confidence_observation_key": "a.conf",
-            }
-        ],
-    }
-    _, _, rules = parse_profile(data)
-    assert rules[0].confidence_observation_key == "a.conf"
-    assert rules[0].confidence_threshold == 0.7
-    assert rules[0].confidence_band == 0.1
+def test_warning_codes_duplicate_via_registry_element():
+    """Plugin-registered elements must not collide with profile codes."""
+    from odd_judge.config import registry
+
+    data = _mini_profile({"x1": {"outside": "X-001"}})
+    registry.register_element(
+        {
+            "element_id": "plug.1",
+            "requirement": "allowed",
+            "rule_type": "qualitative_bool",
+            "observation_key": "a.b",
+            "expect": True,
+            "warning_codes": {"boundary": "X-001"},
+        }
+    )
+    try:
+        with pytest.raises(ConfigError, match="duplicate warning code"):
+            parse_profile(data)
+    finally:
+        registry.clear()
+
+
+def test_warning_codes_doc_up_to_date():
+    """docs/warning_codes.md code tables must match profiles/ (no drift)."""
+    import subprocess
+    import sys
+
+    script = ROOT / "scripts" / "gen_warning_codes_doc.py"
+    res = subprocess.run(
+        [sys.executable, str(script), "--check"],
+        capture_output=True,
+        text=True,
+    )
+    assert res.returncode == 0, res.stdout + res.stderr
```

## 附录 B: J/tests/test_evaluate.py ↔ S/tests/test_judge_evaluate.py

```diff
--- ODD_Judge/tests/test_evaluate.py	2026-11-06 11:58:49.715949058 +0800
+++ odd_perception_suite/tests/test_judge_evaluate.py	2026-10-28 18:17:11.300631600 +0800
@@ -196,209 +196,3 @@
     )
     r = evaluate_rule(rule, make_obs(x={}), EvalContext(DefinitionMode.DEFAULT))
     assert r.state == ScenarioState.UNDEFINED
-
-
-# ---------------------------------------------------------------------------
-# Invalid observation data → same fallback as a missing value
-# ---------------------------------------------------------------------------
-
-
-def _numeric_rule(observation_key="v.x"):
-    return Rule(
-        element_id="t.num",
-        path=("num",),
-        requirement=Requirement.ALLOWED,
-        rule_type="numeric_range",
-        observation_key=observation_key,
-        spec=ThresholdSpec(op="between", range=(0.0, 80.0), margin=BoundaryMargin("absolute", 3.0)),
-        warning_codes={"outside": "N"},
-    )
-
-
-def test_numeric_nan_falls_back_by_definition_mode():
-    rule = _numeric_rule()
-    strict = evaluate_rule(rule, make_obs(v={"x": float("nan")}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert strict.state == ScenarioState.OUTSIDE
-    assert strict.reason_code == "UNDEFINED-STRICT"
-    loose = evaluate_rule(rule, make_obs(v={"x": float("nan")}), EvalContext(DefinitionMode.PERMISSIVE))
-    assert loose.state == ScenarioState.INSIDE
-
-
-def test_numeric_infinity_is_invalid_not_inside():
-    rule = _numeric_rule()
-    r = evaluate_rule(rule, make_obs(v={"x": float("inf")}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.OUTSIDE
-    assert r.reason_code == "UNDEFINED-STRICT"
-
-
-def test_numeric_non_numeric_string_is_missing():
-    rule = _numeric_rule()
-    r = evaluate_rule(rule, make_obs(v={"x": "N/A"}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.OUTSIDE
-    assert r.reason_code == "UNDEFINED-STRICT"
-
-
-# ---------------------------------------------------------------------------
-# Unknown enum value → treated as missing, consistently for both config shapes
-# ---------------------------------------------------------------------------
-
-
-def _enum_rule(requirement, allowed=None, forbidden=None):
-    return Rule(
-        element_id="t.enum",
-        path=("enum",),
-        requirement=requirement,
-        rule_type="qualitative_enum",
-        observation_key="s.type",
-        allowed_values=allowed,
-        forbidden_values=forbidden,
-        warning_codes={"outside": "E"},
-    )
-
-
-def test_unknown_enum_is_missing_for_allowed_config():
-    rule = _enum_rule(Requirement.ALLOWED, allowed=("a",), forbidden=("b",))
-    r = evaluate_rule(rule, make_obs(s={"type": "zzz"}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.OUTSIDE
-    assert r.reason_code == "UNDEFINED-STRICT"
-    r2 = evaluate_rule(rule, make_obs(s={"type": "zzz"}), EvalContext(DefinitionMode.PERMISSIVE))
-    assert r2.state == ScenarioState.INSIDE
-
-
-def test_forbidden_only_config_anything_not_forbidden_is_compliant():
-    """Forbidden-list semantics: a value outside the list is compliant, not
-    unknown — the two ambiguous config shapes are rejected by schema instead."""
-    rule = _enum_rule(Requirement.NOT_ALLOWED, forbidden=("b",))
-    r = evaluate_rule(rule, make_obs(s={"type": "zzz"}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.INSIDE
-    # ...while the allowed-list shape treats unlisted values as missing
-    rule2 = _enum_rule(Requirement.ALLOWED, allowed=("a",), forbidden=("b",))
-    r2 = evaluate_rule(rule2, make_obs(s={"type": "zzz"}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r2.state == ScenarioState.OUTSIDE
-    assert r2.reason_code == "UNDEFINED-STRICT"
-
-
-def test_known_forbidden_value_still_flips_for_not_allowed():
-    rule = _enum_rule(Requirement.NOT_ALLOWED, forbidden=("b",))
-    r = evaluate_rule(rule, make_obs(s={"type": "b"}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.OUTSIDE
-    assert r.reason_code == "E"  # configured warning code, not the missing fallback
-    r2 = evaluate_rule(rule, make_obs(s={"type": "ok"}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r2.state == ScenarioState.INSIDE
-
-
-# ---------------------------------------------------------------------------
-# Bool rules: strict type check + live confidence handling
-# ---------------------------------------------------------------------------
-
-
-def test_bool_string_value_is_missing_not_truthy():
-    rule = Rule(
-        element_id="t.flag",
-        path=("flag",),
-        requirement=Requirement.ALLOWED,
-        rule_type="qualitative_bool",
-        observation_key="a.flag",
-        expect=False,
-    )
-    # "false" must not be truth-evaluated as True → falls back as missing
-    r = evaluate_rule(rule, make_obs(a={"flag": "false"}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.OUTSIDE
-    assert r.reason_code == "UNDEFINED-STRICT"
-    # numeric 0/1 are accepted as booleans
-    assert evaluate_rule(rule, make_obs(a={"flag": 0}), EvalContext(DefinitionMode.RESTRICTIVE)).state == ScenarioState.INSIDE
-    assert evaluate_rule(rule, make_obs(a={"flag": 1}), EvalContext(DefinitionMode.RESTRICTIVE)).state == ScenarioState.OUTSIDE
-
-
-def _confidence_rule():
-    return Rule(
-        element_id="t.conf",
-        path=("conf",),
-        requirement=Requirement.ALLOWED,
-        rule_type="qualitative_bool",
-        observation_key="a.flag",
-        expect=True,
-        confidence_threshold=0.7,
-        confidence_band=0.1,
-        confidence_observation_key="a.score",
-        warning_codes={"boundary": "CB", "outside": "CO"},
-    )
-
-
-def test_bool_confidence_match_is_inside_without_consulting_confidence():
-    r = evaluate_rule(_confidence_rule(), make_obs(a={"flag": True, "score": 0.2}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.INSIDE
-
-
-def test_bool_confidence_reliable_mismatch_is_outside():
-    r = evaluate_rule(_confidence_rule(), make_obs(a={"flag": False, "score": 0.9}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.OUTSIDE
-    assert r.confidence == 0.9
-    assert r.reason_code == "CO"
-
-
-def test_bool_confidence_critical_band_is_boundary():
-    r = evaluate_rule(_confidence_rule(), make_obs(a={"flag": False, "score": 0.65}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.BOUNDARY
-    assert r.reason_code == "CB"
-    assert r.confidence == 0.65
-
-
-def test_bool_confidence_below_band_falls_back_as_missing():
-    r = evaluate_rule(_confidence_rule(), make_obs(a={"flag": False, "score": 0.5}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.OUTSIDE
-    assert r.reason_code == "UNDEFINED-STRICT"
-
-
-def test_bool_confidence_missing_key_falls_back_as_missing():
-    r = evaluate_rule(_confidence_rule(), make_obs(a={"flag": False}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.OUTSIDE
-    assert r.reason_code == "UNDEFINED-STRICT"
-
-
-def test_bool_without_confidence_config_is_binary():
-    rule = Rule(
-        element_id="t.plain",
-        path=("p",),
-        requirement=Requirement.ALLOWED,
-        rule_type="qualitative_bool",
-        observation_key="a.flag",
-        expect=True,
-    )
-    r = evaluate_rule(rule, make_obs(a={"flag": False}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.OUTSIDE
-    assert r.reason_code is None
-
-
-# ---------------------------------------------------------------------------
-# undefined_as element-level override
-# ---------------------------------------------------------------------------
-
-
-def test_undefined_as_undefined_returns_undefined():
-    rule = Rule(
-        element_id="t.u",
-        path=("u",),
-        requirement=Requirement.ALLOWED,
-        rule_type="qualitative_bool",
-        observation_key="a.gone",
-        expect=True,
-        undefined_as=Requirement.UNDEFINED,
-    )
-    r = evaluate_rule(rule, make_obs(a={}), EvalContext(DefinitionMode.PERMISSIVE))
-    assert r.state == ScenarioState.UNDEFINED
-    assert r.reason_code == "UNDEFINED-AS-UNDEFINED"
-
-
-def test_undefined_as_allowed_overrides_restrictive():
-    rule = Rule(
-        element_id="t.u2",
-        path=("u",),
-        requirement=Requirement.ALLOWED,
-        rule_type="qualitative_bool",
-        observation_key="a.gone",
-        expect=True,
-        undefined_as=Requirement.ALLOWED,
-    )
-    r = evaluate_rule(rule, make_obs(a={}), EvalContext(DefinitionMode.RESTRICTIVE))
-    assert r.state == ScenarioState.INSIDE
```

## 附录 B: J/tests/test_margin.py ↔ S/tests/test_judge_margin.py

```diff
--- ODD_Judge/tests/test_margin.py	2026-11-06 11:58:25.120320283 +0800
+++ odd_perception_suite/tests/test_judge_margin.py	2026-10-28 18:17:11.305631532 +0800
@@ -64,103 +64,3 @@
     state, rem, _ = classify_numeric(2.0, spec)
     assert state == ScenarioState.OUTSIDE
     assert rem == -0.5
-
-
-# ---------------------------------------------------------------------------
-# Band-edge unification: a value exactly one band away belongs to INSIDE
-# for relational/range ops (>= > <= < between), stable under fp rounding.
-# ---------------------------------------------------------------------------
-
-
-def test_ge_band_edge_is_inside():
-    spec = ThresholdSpec(op=">=", threshold=0.6, margin=BoundaryMargin("absolute", 0.05))
-    assert classify_numeric(0.65, spec)[0] == ScenarioState.INSIDE  # JSON literal
-    assert classify_numeric(0.6 + 0.05, spec)[0] == ScenarioState.INSIDE  # expression
-    assert classify_numeric(0.64999999, spec)[0] == ScenarioState.BOUNDARY
-    assert classify_numeric(0.6, spec)[0] == ScenarioState.BOUNDARY  # threshold itself
-
-
-def test_le_band_edge_is_inside():
-    spec = ThresholdSpec(op="<=", threshold=14.9, margin=BoundaryMargin("absolute", 1.5))
-    assert classify_numeric(13.4, spec)[0] == ScenarioState.INSIDE  # thr - band
-    assert classify_numeric(13.400001, spec)[0] == ScenarioState.BOUNDARY
-    assert classify_numeric(14.9, spec)[0] == ScenarioState.BOUNDARY
-
-
-def test_strict_gt_edge_semantics():
-    spec = ThresholdSpec(op=">", threshold=200.0, margin=BoundaryMargin("absolute", 20.0))
-    assert classify_numeric(200.0, spec)[0] == ScenarioState.OUTSIDE  # strict: threshold violates
-    assert classify_numeric(220.0, spec)[0] == ScenarioState.INSIDE  # band edge → INSIDE
-    assert classify_numeric(210.0, spec)[0] == ScenarioState.BOUNDARY
-
-
-def test_between_band_edges_are_inside_stable():
-    """Both ends: exact band distance → INSIDE regardless of fp direction."""
-    lane = ThresholdSpec(op="between", range=(2.5, 5.5), margin=BoundaryMargin("absolute", 0.2))
-    assert classify_numeric(2.7, lane)[0] == ScenarioState.INSIDE
-    assert classify_numeric(5.3, lane)[0] == ScenarioState.INSIDE
-    cross = ThresholdSpec(op="between", range=(-2.0, 2.0), margin=BoundaryMargin("absolute", 0.2))
-    assert classify_numeric(1.8, cross)[0] == ScenarioState.INSIDE
-    assert classify_numeric(-1.8, cross)[0] == ScenarioState.INSIDE
-    grade = ThresholdSpec(op="between", range=(-9.0, 9.0), margin=BoundaryMargin("absolute", 0.9))
-    assert classify_numeric(8.1, grade)[0] == ScenarioState.INSIDE
-    assert classify_numeric(-8.1, grade)[0] == ScenarioState.INSIDE
-
-
-def test_between_range_endpoints_are_boundary():
-    speed = ThresholdSpec(op="between", range=(0.0, 80.0), margin=BoundaryMargin("absolute", 3.0))
-    assert classify_numeric(0.0, speed)[0] == ScenarioState.BOUNDARY
-    assert classify_numeric(80.0, speed)[0] == ScenarioState.BOUNDARY
-    assert classify_numeric(3.0, speed)[0] == ScenarioState.INSIDE  # band edge
-    assert classify_numeric(77.0, speed)[0] == ScenarioState.INSIDE  # band edge
-    assert classify_numeric(2.9, speed)[0] == ScenarioState.BOUNDARY
-    assert classify_numeric(80.1, speed)[0] == ScenarioState.OUTSIDE
-
-
-def test_zero_band_has_no_boundary_zone():
-    spec = ThresholdSpec(op="<=", threshold=0.0, margin=BoundaryMargin("absolute", 0.0))
-    assert classify_numeric(0.0, spec)[0] == ScenarioState.INSIDE
-    assert classify_numeric(1.0, spec)[0] == ScenarioState.OUTSIDE
-    ge = ThresholdSpec(op=">=", threshold=100.0, margin=BoundaryMargin("absolute", 0.0))
-    assert classify_numeric(100.0, ge)[0] == ScenarioState.INSIDE
-    assert classify_numeric(99.9, ge)[0] == ScenarioState.OUTSIDE
-
-
-# ---------------------------------------------------------------------------
-# == / != : the band is an inclusive tolerance zone
-# ---------------------------------------------------------------------------
-
-
-def test_neq_produces_boundary_near_forbidden_value():
-    spec = ThresholdSpec(op="!=", threshold=10.0, margin=BoundaryMargin("absolute", 1.0))
-    assert classify_numeric(10.0, spec)[0] == ScenarioState.OUTSIDE
-    assert classify_numeric(10.5, spec)[0] == ScenarioState.BOUNDARY
-    assert classify_numeric(9.5, spec)[0] == ScenarioState.BOUNDARY
-    assert classify_numeric(11.0, spec)[0] == ScenarioState.BOUNDARY  # band edge, closed
-    assert classify_numeric(11.000001, spec)[0] == ScenarioState.INSIDE
-    assert classify_numeric(12.0, spec)[0] == ScenarioState.INSIDE
-
-
-def test_neq_zero_band_is_binary():
-    spec = ThresholdSpec(op="!=", threshold=10.0, margin=BoundaryMargin("absolute", 0.0))
-    assert classify_numeric(10.0, spec)[0] == ScenarioState.OUTSIDE
-    assert classify_numeric(10.5, spec)[0] == ScenarioState.INSIDE
-
-
-def test_eq_tolerance_zone_closed_at_edge():
-    spec = ThresholdSpec(op="==", threshold=10.0, margin=BoundaryMargin("absolute", 1.0))
-    assert classify_numeric(10.0, spec)[0] == ScenarioState.INSIDE
-    assert classify_numeric(10.5, spec)[0] == ScenarioState.BOUNDARY
-    assert classify_numeric(11.0, spec)[0] == ScenarioState.BOUNDARY  # edge closed
-    assert classify_numeric(11.000001, spec)[0] == ScenarioState.OUTSIDE
-
-
-def test_non_finite_value_raises():
-    spec = ThresholdSpec(op="between", range=(0.0, 80.0), margin=BoundaryMargin("absolute", 3.0))
-    for bad in (float("nan"), float("inf"), float("-inf")):
-        try:
-            classify_numeric(bad, spec)
-        except ValueError:
-            pass
-        else:
-            raise AssertionError(f"expected ValueError for {bad!r}")
```

## 附录 B: J/tests/test_e2e.py ↔ S/tests/test_judge_e2e.py

```diff
--- ODD_Judge/tests/test_e2e.py	2026-11-06 11:58:05.949814170 +0800
+++ odd_perception_suite/tests/test_judge_e2e.py	2026-10-28 18:17:24.175458117 +0800
@@ -20,7 +20,7 @@
 
 
 def test_json_file_adapter():
-    obs = load_observation_file(ROOT / "observations" / "sample_highway_inside.json")
+    obs = load_observation_file(ROOT / "examples" / "observations" / "sample_highway_inside.json")
     assert obs.get("scenery.road_type") == "expressway_main"
 
 
@@ -29,7 +29,7 @@
         ROOT / "profiles" / "byd_tjp.json",
         switches_path=ROOT / "switches" / "default_switches.json",
     )
-    report = judge.evaluate_path(ROOT / "observations" / "sample_highway_inside.json")
+    report = judge.evaluate_path(ROOT / "examples" / "observations" / "sample_highway_inside.json")
     assert report.state.value == "INSIDE"
     assert report.recommendation == "continue"
 
@@ -39,7 +39,7 @@
         ROOT / "profiles" / "byd_tjp.json",
         switches_path=ROOT / "switches" / "default_switches.json",
     )
-    report = judge.evaluate_path(ROOT / "observations" / "sample_curve_boundary.json")
+    report = judge.evaluate_path(ROOT / "examples" / "observations" / "sample_curve_boundary.json")
     assert report.state.value == "BOUNDARY"
     assert report.recommendation == "warn_driver"
     assert len(report.boundary_reasons) > 0
@@ -50,7 +50,7 @@
         ROOT / "profiles" / "byd_tjp.json",
         switches_path=ROOT / "switches" / "default_switches.json",
     )
-    report = judge.evaluate_path(ROOT / "observations" / "sample_rain_outside.json")
+    report = judge.evaluate_path(ROOT / "examples" / "observations" / "sample_rain_outside.json")
     assert report.state.value == "OUTSIDE"
     assert report.recommendation == "mrm_or_refuse"
     assert len(report.outside_reasons) > 0
@@ -69,7 +69,7 @@
         switches_path=None,
     )
     judge.switch_policy.by_element["ODC.3.1.2.rain_intensity"] = False
-    report = judge.evaluate_path(ROOT / "observations" / "sample_rain_outside.json")
+    report = judge.evaluate_path(ROOT / "examples" / "observations" / "sample_rain_outside.json")
     skipped_ids = {s["element_id"] for s in report.skipped}
     assert "ODC.3.1.2.rain_intensity" in skipped_ids
 
@@ -80,7 +80,7 @@
             sys.executable, "-m", "odd_judge", "evaluate",
             "--project-root", str(ROOT),
             "--profile", "byd_tjp",
-            "--obs", str(ROOT / "observations" / "sample_curve_boundary.json"),
+            "--obs", str(ROOT / "examples" / "observations" / "sample_curve_boundary.json"),
             "--switches", str(ROOT / "switches" / "default_switches.json"),
             "--format", "text",
         ],
@@ -88,7 +88,7 @@
         text=True,
         cwd=str(ROOT),
     )
-    assert result.returncode == 1, result.stderr  # BOUNDARY → warn exit code
+    assert result.returncode == 0, result.stderr
     assert "BOUNDARY" in result.stdout
 
 
@@ -101,82 +101,3 @@
     )
     assert result.returncode == 0, result.stderr
     assert "VALID" in result.stdout
-
-
-def _run_cli(*argv: str):
-    return subprocess.run(
-        [sys.executable, "-m", "odd_judge", *argv],
-        capture_output=True,
-        text=True,
-        cwd=str(ROOT),
-    )
-
-
-def test_cli_exit_codes_reflect_worst_state():
-    """0 = INSIDE, 1 = BOUNDARY, 3 = OUTSIDE."""
-    common = ["evaluate", "--project-root", str(ROOT), "--profile", "byd_tjp", "--format", "text"]
-
-    inside = _run_cli(*common, "--obs", str(ROOT / "observations" / "sample_highway_inside.json"))
-    assert inside.returncode == 0, inside.stderr + inside.stdout
-
-    boundary = _run_cli(*common, "--obs", str(ROOT / "observations" / "sample_curve_boundary.json"))
-    assert boundary.returncode == 1, boundary.stderr + boundary.stdout
-
-    outside = _run_cli(*common, "--obs", str(ROOT / "observations" / "sample_rain_outside.json"))
-    assert outside.returncode == 3, outside.stderr + outside.stdout
-
-
-def test_cli_missing_input_is_clean_error():
-    result = _run_cli("evaluate", "--project-root", str(ROOT), "--profile", "byd_tjp")
-    assert result.returncode == 4
-    assert "ERROR:" in result.stderr
-    assert "Traceback" not in result.stderr
-
-
-def test_cli_unknown_obs_file_is_clean_error():
-    result = _run_cli(
-        "evaluate",
-        "--project-root",
-        str(ROOT),
-        "--profile",
-        "byd_tjp",
-        "--obs",
-        str(ROOT / "observations" / "does_not_exist.json"),
-    )
-    assert result.returncode == 4
-    assert "ERROR:" in result.stderr
-    assert "Traceback" not in result.stderr
-
-
-def test_mock_scenarios_match_json_verdicts():
-    """Mock fixtures must carry the same keys as the JSON samples → same verdict."""
-    judge = OddJudge.from_files(
-        ROOT / "profiles" / "byd_tjp.json",
-        switches_path=ROOT / "switches" / "default_switches.json",
-    )
-    pairs = [
-        ("highway_inside", "sample_highway_inside.json", "INSIDE"),
-        ("curve_boundary", "sample_curve_boundary.json", "BOUNDARY"),
-        ("rain_outside", "sample_rain_outside.json", "OUTSIDE"),
-    ]
-    for scenario, sample, expected in pairs:
-        mock_state = judge.evaluate(MockAdapter(scenario).get_observation()).state.value
-        json_state = judge.evaluate_path(ROOT / "observations" / sample).state.value
-        assert mock_state == expected, f"mock {scenario}: {mock_state}"
-        assert json_state == expected, f"json {sample}: {json_state}"
-
-
-def test_effective_switches_in_report():
-    """active_switches lists only switches that actually alter the judgment."""
-    judge = OddJudge.from_files(
-        ROOT / "profiles" / "byd_tjp.json",
-        switches_path=ROOT / "switches" / "default_switches.json",
-    )
-    report = judge.evaluate_path(ROOT / "observations" / "sample_highway_inside.json")
-    assert report.active_switches == []
-
-    judge.switch_policy.by_element["ODC.3.1.2.rain_intensity"] = False
-    judge.switch_policy.boundary_warnings_enabled = False
-    report = judge.evaluate_path(ROOT / "observations" / "sample_rain_outside.json")
-    assert "by_element[ODC.3.1.2.rain_intensity]=false" in report.active_switches
-    assert "boundary_warnings_enabled=false" in report.active_switches
```

---

# 附录 C: 配置/文档

## 附录 C: profiles/byd_tjp.json

```diff
--- ODD_Judge/profiles/byd_tjp.json	2026-10-28 14:24:01.889724340 +0800
+++ odd_perception_suite/profiles/byd_tjp.json	2026-10-29 10:13:46.959797922 +0800
@@ -489,7 +489,7 @@
       "threshold": 0.6,
       "boundary_margin": {"mode": "absolute", "value": 0.05},
       "tags": ["sensor", "perception"],
-      "warning_codes": {"boundary": "PER-CONF-NEAR", "outside": "PER-CONF-001"}
+      "warning_codes": {"boundary": "PER-CAM-CONF-NEAR", "outside": "PER-CONF-001"}
     },
     {
       "element_id": "ODC.7.1.4.lidar_confidence",
@@ -502,7 +502,7 @@
       "threshold": 0.6,
       "boundary_margin": {"mode": "absolute", "value": 0.05},
       "tags": ["sensor", "perception"],
-      "warning_codes": {"boundary": "PER-CONF-NEAR", "outside": "PER-CONF-002"}
+      "warning_codes": {"boundary": "PER-LIDAR-CONF-NEAR", "outside": "PER-CONF-002"}
     }
   ]
 }
```

## 附录 C: switches/default_switches.json

```diff
--- ODD_Judge/switches/default_switches.json	2026-10-28 14:20:56.771451221 +0800
+++ odd_perception_suite/switches/default_switches.json	2026-11-05 12:31:12.296996415 +0800
@@ -22,9 +22,10 @@
     "odc": true,
     "system": true,
     "sensor": true,
-    "perception": true
+    "perception": true,
+    "surface": true,
+    "traffic": true,
+    "extension": true
   },
-  "boundary_warnings_enabled": true,
-  "boundary_severity": "warning",
-  "on_switch_mismatch": "skip"
+  "boundary_warnings_enabled": true
 }
```

## 附录 C: docs/semantics.md

```diff
--- ODD_Judge/docs/semantics.md	2026-11-06 12:01:35.761728118 +0800
+++ odd_perception_suite/docs/semantics.md	2026-11-05 12:34:10.463272063 +0800
@@ -1,8 +1,5 @@
 # ODD 判决语义
 
-> 逐条规则与具体边界值见 **`docs/odd_rules.md`**（profile = `byd_tjp`，42 条，边界值经实测校验）。
-> 本文描述通用语义，与当前实现一致（2026-11-06 基准）；逐条规则与精确边界值见 `docs/odd_rules.md`。
-
 ## 1. 三态分类（ISO 34503 Clause 6）
 
 | 状态 | 含义 | 系统期望行为 |
@@ -11,8 +8,6 @@
 | **BOUNDARY** | 接近限制值 / Conditional 切换区 | 降级、限速、预警、准备 MRM |
 | **OUTSIDE** | 不满足 ODD | 拒绝激活 / transition demand / MRM |
 
-另有两个不参与聚合的辅助态：`UNDEFINED`（不定义/按 default 回退）、`SKIPPED`（开关关闭或 `applies_when` 不成立）。
-
 ## 2. 元素要求映射
 
 ### allowed（允许）
@@ -41,56 +36,40 @@
 
 ## 3. 定义模式（DefinitionMode）— 缺失观测回退
 
-优先级：`requirement=undefined` > 元素级 `undefined_as` > profile 级 `definition_mode`。
-
 | 模式 | 行为 | 用途 |
 |------|------|------|
 | **permissive** | 视为 INSIDE，跳过 | 演示 / 宽松原型 |
 | **restrictive** | 视为 OUTSIDE（`UNDEFINED-STRICT`） | **BYD TJP 默认**，安全优先 |
 | **default** | UNDEFINED，不入聚合 | ODC 骨架探索 |
 
-`undefined_as` 三个取值均有效：`allowed` → INSIDE、`not_allowed` → OUTSIDE、`undefined` → UNDEFINED（`UNDEFINED-AS-UNDEFINED`），逐元素覆盖 profile 模式。
-
-**视同缺失的非法观测**（同样走上表回退）：数值规则的非数字字符串与 `NaN`/`±inf`；布尔规则的非布尔类型（字符串 `"false"` 等，数值 `0`/`1` 合法）；枚举规则（allowed 形态）未命中任何集合的未知值；失配时的低置信数据（`c < threshold − band`）。
+以上是最后一级回退。更细的覆盖层（高 → 低）：元素 `requirement=undefined` →
+元素级 `undefined_as` → profile 顶层 `missing_policy.by_element` → `by_tag` →
+`default` → 本节的 `definition_mode`。`missing_policy` 取值
+`allowed | not_allowed | undefined | mode`，命中来源记入
+`RuleResult.policy_source` 与 `report.undefined_handled[].policy`。
+完整优先级表、写法与「缺信息时哪些判据生效」的说明见
+[`architecture_contract.md`](architecture_contract.md) 第 4 节。
 
 ## 4. 定量边界（BoundaryMargin）
 
 ```
 absolute: band = value
 relative: band = |threshold| * value   (between: band = (hi-lo) * value)
-未配置:   band = 0（两态，无 BOUNDARY）
 ```
 
-判定式（`odd_judge/core/margin.py`，统一安全裕度原理）：
-
+判定（以 `>=` 为例）：
 ```
-margin = 到违反集的距离（域内为正；margin == 0 恰在阈值/区间端点上）
-
-序关系/区间类（>= > <= < between）:
-    margin < 0（严格算子 == 0 也违反） → OUTSIDE
-    0 ≤ margin < band                 → BOUNDARY
-    margin ≥ band                     → INSIDE（带边缘归 INSIDE）
-
-比较类（== / !=）:
-    band 是闭合容差带：|v−thr| ∈ (0, band] → BOUNDARY；带边缘归 BOUNDARY
-    == : v == thr → INSIDE，|v−thr| > band → OUTSIDE
-    != : v == thr → OUTSIDE，|v−thr| > band → INSIDE（band=0 时为两态）
+value >= thr + band   → INSIDE
+thr <= value < thr+band → BOUNDARY   (remaining_margin = value - thr)
+value < thr            → OUTSIDE    (remaining_margin < 0)
 ```
 
-- **端点稳定**：`margin` 与 `band` 比较时启用 `EDGE_REL_TOL = 1e-12`（`math.isclose`），十进制字面量端点（`220`、`13.4`、`2.7`、`1.8`、`0.65`…）无论二进制舍入方向都稳定归 INSIDE。
-- `between` 的边界带**两端都生效**（如速度 `[0,80]` band 3 ⇒ `[0,3)` 与 `(77,80]` 为 BOUNDARY，含 0 km/h 停车；`3` 与 `77` 归 INSIDE）。
-- `NaN` / `±inf`：`classify_numeric` 直接抛 `ValueError`；评估层视同缺失，按第 3 节回退（不再可能误判 INSIDE）。
-
-`remaining_margin`：到最近阈值的**有符号距离**（正=域内，0=恰在阈值，负=越界）。
+`remaining_margin`：到最近阈值的**有符号距离**（正=域内，负=越界）。
 
 ## 5. 定性边界
 
-- **枚举**：判定优先级 `allowed_values` > `boundary_values` > `forbidden_values`；**未命中任何集合的未知值视同缺失**（按第 3 节回退）。配置形态由 schema 约束：`allowed` 要求必须给 `allowed_values`；`not_allowed` 要求必须给 `forbidden_values` 且不得给 `allowed_values`；三集合两两不重叠——两种形态下语义唯一，无分叉。
-- **布尔**：默认严格两态（INSIDE / OUTSIDE）。可选置信度三段判别（仅失配时启用，`confidence_threshold` + `confidence_band` + `confidence_observation_key` 三字段成组、仅限 `qualitative_bool`）：
-  - `c ≥ threshold` → OUTSIDE（可信失配）；
-  - `threshold − band ≤ c < threshold` → BOUNDARY（临界带）；
-  - `c < threshold − band` / 置信度缺失 → 视同缺失回退。
-- 观测值须为 JSON 布尔或数值 `0`/`1`；字符串等其他类型视同缺失（不再 `bool()` 真值化）。
+- **枚举**：`boundary_values` 列表显式标注边界取值
+- **布尔**：二值无边界；可选配 `confidence_threshold` + `confidence_band`，置信度落入带内 → BOUNDARY
 
 ## 6. 组合条件（Conditional / AllOf / AnyOf / Not）
 
@@ -101,27 +80,24 @@
 ]}
 ```
 
-- `applies_when` 为假时规则跳过（`CONDITIONAL-SKIP`），不参与判决，**不走缺失回退**。
-- 叶子引用的键缺失 → 该叶子为**假**；因此 `not` 内部叶子缺失时整体为**真**（`not` 分支把缺失当"满足"）。
-- 求值顺序：开关 → `applies_when` → `requirement=undefined` → 缺失回退 → 规则类型评估。
+`applies_when` 为假时规则跳过（`CONDITIONAL-SKIP`），不参与判决。
 
 ## 7. 聚合与建议
 
 ```
-参与态: INSIDE / BOUNDARY / OUTSIDE（UNDEFINED、SKIPPED 剔除）
-优先级: OUTSIDE > BOUNDARY > INSIDE；无参与规则 → UNDEFINED
-severity: OUTSIDE→critical, BOUNDARY→warning, 其余→info
-recommendation: INSIDE→continue, BOUNDARY→warn_driver, OUTSIDE→mrm_or_refuse, UNDEFINED→continue
+优先级: OUTSIDE > BOUNDARY > INSIDE > UNDEFINED
+severity: OUTSIDE→critical, BOUNDARY→warning, INSIDE→info
+recommendation: INSIDE→continue, BOUNDARY→warn_driver, OUTSIDE→mrm_or_refuse
 ```
 
+逐规则告警码（`reason_code`）的命名规范、完整码表与引擎码见 [`warning_codes.md`](warning_codes.md)。
+
 ## 8. 开关（Switches）
 
 - `global_enabled=false`：全部跳过
 - `by_element[id]=false`：按元素关闭
 - `by_rule_type[type]=false`：按规则类型关闭
-- `by_tag[tag]=false`：按标签关闭（规则任一标签命中即跳过）
-- `boundary_warnings_enabled=false`：`boundary_reasons` 清空，**整体状态由 BOUNDARY 降为 INSIDE**（建议随之变 `continue`），但 `results[]` 中各规则仍保留 BOUNDARY 原值
+- `by_tag[tag]=false`：按标签关闭
+- `boundary_warnings_enabled=false`：BOUNDARY 不输出预警，整体可降为 INSIDE
 
 关闭的规则记入 `skipped[]`，不参与判决。
-报告字段 `active_switches` 只列出**实际生效（非默认）**的开关项（全默认时为空列表）。
-
```
