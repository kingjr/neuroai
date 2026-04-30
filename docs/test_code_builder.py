# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.

"""Tests for the interactive code-builder page (docs/neuralset/code_builder.rst).

Four layers:
  L0  YAML schema + cls resolution + JS-in-sync.            Always runs.
  L1  AST-parse the rendered script per axis option.        Always runs.
  L2  Per-axis pydantic instantiation.                      Always runs.
  L3  End-to-end exec of the rendered script per option.    @pytest.mark.slow

L1/L2/L3 all parametrize on the same per-axis-pinned set: one combo per
axis option, with the other axes at their defaults. That set already
exercises every conditional branch in the renderer; a full Cartesian
sweep would multiply redundant leaves on top.

L1/L2/L3 share a Python re-implementation of the renderer that mirrors
``docs/_static/code-builder.js`` line-for-line. Keep them in sync.
(See PR plan to retire this mirror in favour of a precomputed bundle.)
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).parent
DATA_YAML = DOCS / "_data" / "code-builder-data.yaml"
DATA_JS = DOCS / "_static" / "code-builder-data.js"
RENDERER_JS = DOCS / "_static" / "code-builder.js"

sys.path.insert(0, str(DOCS / "_data"))
import build_data  # type: ignore  # noqa: E402

# ── Shared helpers ──────────────────────────────────────────────────────────


def _load_yaml() -> dict:
    return build_data.load(DATA_YAML)


DATA = _load_yaml()
AXES = DATA["axes"]
STUDIES = DATA.get("studies", {}) or {}
DEFAULT_STUDY = DATA["default_study"]
AXIS_ORDER = ["neuro", "stim", "task", "model", "style", "compute"]


def _resolve_study(sel: dict) -> dict:
    """Mirror of `study()` in code-builder.js — pick a real public dataset
    when (neuro, stim) maps to one, otherwise the bundled synthetic study."""
    return STUDIES.get(f"{sel['neuro']}-{sel['stim']}", DEFAULT_STUDY)


def _resolve_model(sel: dict) -> dict:
    """Mirror of `model()` in code-builder.js."""
    return AXES["model"]["options"][sel["model"]]


def _axis_pinned() -> list[dict]:
    """One combo per axis option, with the other axes at their defaults.

    The all-default combo would be emitted once per axis (whenever ``key``
    equals that axis's default); de-dupe by keying on the tuple of values
    in canonical order.
    """
    base = {a: AXES[a]["default"] for a in AXIS_ORDER}
    seen: set[tuple] = set()
    out: list[dict] = []
    for axis in AXIS_ORDER:
        for key in AXES[axis]["options"]:
            sel = {**base, axis: key}
            sig = tuple(sel[a] for a in AXIS_ORDER)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(sel)
    return out


def _id(sel: dict) -> str:
    return "-".join(sel[a] for a in AXIS_ORDER)


# ── Pure-Python renderer (mirrors docs/_static/code-builder.js) ─────────────


def _quote_pip(p: str) -> str:
    """Quote `extras` brackets so zsh/bash don't expand them as globs."""
    return f"'{p}'" if "[" in p else p


def _build_install(neuro: dict, stim: dict, stu: dict) -> str:
    """Render the bash install block with framework + dataset deps split.

    Mirror of `buildInstall()` in `docs/_static/code-builder.js`. The output
    is a multi-line string with two `pip install` calls (the dataset section
    is omitted entirely for FakeMulti).
    """

    def _uniq(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in items:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    extras = _uniq(list(neuro.get("pip") or []) + list(stim.get("pip") or []))
    fw_pkgs = _uniq(
        list(neuro.get("pip_packages") or []) + list(stim.get("pip_packages") or [])
    )
    ns_token = f"'neuralset[{','.join(extras)}]'" if extras else "neuralset"
    fw_line = "pip install " + " ".join([ns_token] + [_quote_pip(p) for p in fw_pkgs])

    lines = ["# NeuralSet + extractor dependencies", fw_line]
    for p in (neuro.get("post_install"), stim.get("post_install")):
        if p:
            lines.append(p)

    ds_pkgs = list(stu.get("pip_packages") or [])
    if ds_pkgs:
        lines.append("")
        lines.append(f"# Dataset: {stu['name']}")
        lines.append("pip install " + " ".join(_quote_pip(p) for p in ds_pkgs))
    return "\n".join(lines)


_EVENTS_LINE = "events = study.run()  # simple pd.DataFrame"


def _direction_label(task: dict) -> str:
    return (
        "Decoding (brain -> stim)"
        if task.get("direction") == "decoding"
        else "Encoding (stim -> brain)"
    )


def _load_demo_lines(indent: int = 0) -> list[str]:
    pad = " " * indent
    return [
        pad + "loader = DataLoader(dset, batch_size=8, collate_fn=dset.collate_fn)",
        pad + "batch = next(iter(loader))",
        pad + 'print(batch.data["neuro"].shape)',
        pad + 'print(batch.data["stim"].shape)',
    ]


def _load_imports(task: dict) -> list[str]:
    # Torch ML branch already imports DataLoader through `_ml_imports`, so
    # only emit it here when no ML branch is active (keeps the mirror in
    # lock-step with `loadImports` in code-builder.js).
    if task.get("needs_ml"):
        return []
    return ["from torch.utils.data import DataLoader"]


def _infra_to_py(infra_literal: str) -> str:
    """Render the infra literal as a Python dict expression.

    Anything that holds the ``"$CACHE"`` placeholder is rewritten to
    interpolate the runtime ``CACHE`` ``Path`` (Pydantic coerces Path -> str
    so we don't wrap in ``str(...)``):
      ``"$CACHE"``       → ``CACHE``
      ``"$CACHE/foo"``   → ``CACHE / "foo"``
    """
    import json

    parts: list[str] = []
    for k, v in json.loads(infra_literal).items():
        if isinstance(v, str) and v == "$CACHE":
            rhs = "CACHE"
        elif isinstance(v, str) and v.startswith("$CACHE/"):
            rhs = f'CACHE / "{v[len("$CACHE/") :]}"'
        elif v is None:
            rhs = "None"
        else:
            rhs = json.dumps(v)
        parts.append(f'"{k}": {rhs}')
    return "{" + ", ".join(parts) + "}"


def _multiline_call(prefix: str, kwargs: list[str], trailing: str | None) -> str:
    items = list(kwargs or [])
    if trailing:
        items.append(trailing)
    if not items:
        return prefix + "()"
    body = ",\n    ".join(items)
    return f"{prefix}(\n    {body},\n)"


def _ml_exprs(neuro: dict, stim: dict, task: dict) -> dict:
    """Common machinery: assemble the X / y expressions used by both flavours."""
    is_class = bool(stim.get("is_classification"))
    is_dec = task.get("direction") == "decoding"
    t_lines: list[str] = []
    if "x_expr" in neuro:
        x_neuro = neuro["x_expr"]
    else:
        cmt = f"  # {neuro['t_comment']}" if neuro.get("t_comment") else ""
        t_lines.append(f"t = {neuro['t']}{cmt}")
        x_neuro = 'batch.data["neuro"][:, :, t]'
    stim_flat = 'batch.data["stim"].reshape(len(batch), -1)'
    stim_labels = 'batch.data["stim"].argmax(-1)'
    stim_one_hot = 'batch.data["stim"].float()'
    y_stim = stim_labels if is_class else stim_flat
    x_stim = stim_one_hot if is_class else stim_flat
    return dict(
        is_class=is_class,
        is_dec=is_dec,
        t_lines=t_lines,
        x_neuro=x_neuro,
        y_stim=y_stim,
        x_stim=x_stim,
    )


def _ml_imports(stim: dict, task: dict, model: dict) -> list[str]:
    if not task.get("needs_ml"):
        return []
    if model.get("kind") == "torch":
        return [
            "import torch.nn.functional as F",
            "from torch import optim, nn",
            "from torch.utils.data import DataLoader",
        ]
    is_class = bool(stim.get("is_classification"))
    is_dec = task.get("direction") == "decoding"
    imps: list[str] = []
    if model.get("kind") == "ridge":
        cls = "RidgeClassifier" if (is_dec and is_class) else "Ridge"
        imps.append(f"from sklearn.linear_model import {cls}")
        imps.append("from sklearn.model_selection import cross_val_score")
        return imps
    # sgd: streaming via partial_fit on a DataLoader, mirroring the torch
    # path but with sklearn estimators.
    imps.append("from torch.utils.data import DataLoader")
    if is_dec and is_class:
        imps.append("from sklearn.linear_model import SGDClassifier")
        imps.append("from sklearn.metrics import accuracy_score")
    else:
        imps.append("from sklearn.linear_model import SGDRegressor")
        imps.append("from sklearn.multioutput import MultiOutputRegressor")
        imps.append("from sklearn.metrics import mean_squared_error")
    return imps


def _ml_label(stim: dict, task: dict, model: dict) -> str:
    if not task.get("needs_ml"):
        return "score"
    kind = model.get("kind")
    # Streaming branches both report a per-batch training loss.
    if kind in ("torch", "sgd"):
        return "final loss"
    is_class = bool(stim.get("is_classification"))
    is_dec = task.get("direction") == "decoding"
    return "balanced accuracy" if (is_dec and is_class) else "score"


def _ml_metric_expr(model: dict) -> str:
    return "scores.mean()" if model.get("kind") == "ridge" else "loss"


def _ml_compute_lines(
    neuro: dict, stim: dict, task: dict, model: dict, indent: int
) -> list[str]:
    if not task.get("needs_ml"):
        return []
    e = _ml_exprs(neuro, stim, task)
    X = e["x_neuro"] if e["is_dec"] else e["x_stim"]
    y = e["y_stim"] if e["is_dec"] else e["x_neuro"]
    pad = " " * indent

    if model.get("kind") == "torch":
        if e["is_dec"] and e["is_class"]:
            loss_fn = "F.cross_entropy"
            out_dim_expr = 'batch.data["stim"].shape[-1]'
        else:
            loss_fn = "F.mse_loss"
            out_dim_expr = f"{y}.shape[-1]"
        lines: list[str] = []
        lines.append(
            pad + "loader = DataLoader(dset, batch_size=32, "
            "collate_fn=dset.collate_fn, shuffle=True)"
        )
        for l in e["t_lines"]:
            lines.append(pad + l)
        lines.append(pad + "batch = next(iter(loader))")
        lines.append(f"{pad}model = nn.LazyLinear({out_dim_expr})")
        lines.append(
            pad + "opt   = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)"
        )
        lines.append(pad + "for batch in loader:")
        lines.append(f"{pad}    X = {X}")
        lines.append(f"{pad}    y = {y}")
        lines.append(pad + "    opt.zero_grad()")
        lines.append(f"{pad}    loss = {loss_fn}(model(X), y)")
        lines.append(pad + "    loss.backward()")
        lines.append(pad + "    opt.step()")
        return lines

    if model.get("kind") == "sgd":
        # Streaming online learning via `partial_fit` — same DataLoader
        # pattern as the torch branch, just with an sklearn estimator.
        lines = []
        lines.append(
            pad + "loader = DataLoader(dset, batch_size=32, "
            "collate_fn=dset.collate_fn, shuffle=True)"
        )
        for l in e["t_lines"]:
            lines.append(pad + l)
        if e["is_dec"] and e["is_class"]:
            lines.append(
                pad + 'classes = range(next(iter(loader)).data["stim"].shape[-1])'
            )
            lines.append(pad + "model = SGDClassifier()")
            lines.append(pad + "for batch in loader:")
            lines.append(f"{pad}    X = {X}")
            lines.append(f"{pad}    y = {y}")
            lines.append(pad + "    model.partial_fit(X, y, classes=classes)")
            lines.append(pad + "    loss = 1.0 - accuracy_score(y, model.predict(X))")
        else:
            lines.append(pad + "model = MultiOutputRegressor(SGDRegressor())")
            lines.append(pad + "for batch in loader:")
            lines.append(f"{pad}    X = {X}")
            lines.append(f"{pad}    y = {y}")
            lines.append(pad + "    model.partial_fit(X, y)")
            lines.append(pad + "    loss = mean_squared_error(y, model.predict(X))")
        return lines

    # ridge — classic full-RAM cross-validated score.
    est = "RidgeClassifier()" if (e["is_dec"] and e["is_class"]) else "Ridge()"
    lines = [f"{pad}batch = dset.load_all()"]
    for l in e["t_lines"]:
        lines.append(pad + l)
    lines.append(f"{pad}scores = cross_val_score(")
    lines.append(f"{pad}    estimator={est},")
    lines.append(f"{pad}    X={X},")
    lines.append(f"{pad}    y={y},")
    lines.append(f"{pad})")
    return lines


def _build_kwargs_script(
    neuro: dict, stim: dict, comp: dict, task: dict, stu: dict, model: dict
) -> str:
    win = neuro["window"]
    s_trailing = None if stim.get("accepts_infra") is False else "infra=infra"
    c_infra = _infra_to_py(comp["infra_literal"])

    lines: list[str] = [
        "from pathlib import Path",
        "import neuralset as ns",
    ]
    lines.extend(_ml_imports(stim, task, model))
    lines.extend(_load_imports(task))
    lines.append("")
    lines.append('CACHE = Path.home() / "neuroai_data" / ".cache"')
    lines.append('STUDIES = Path.home() / "neuroai_data" / ".studies"')
    lines.append("STUDIES.mkdir(parents=True, exist_ok=True)")
    lines.append("infra = " + c_infra)
    lines.append("")
    lines.append(f"# 1. {stu['comment']}")
    # Parent dir only — Study.download() resolves the study-name subfolder
    # for unfrozen instances. `infra={"folder": CACHE}` enables caching of
    # `study.run()` events between calls. (YAML mode keeps the explicit
    # subfolder because the Experiment freezes the Study.)
    lines.append("study = ns.Study(")
    lines.append(f'    name="{stu["name"]}",')
    lines.append("    path=STUDIES,")
    lines.append('    infra={"folder": CACHE},')
    lines.append(")")
    lines.append("")
    lines.append("# 2. Define extractors")
    # Real-data studies may pin extractor kwargs (e.g. allow_maxshield=True
    # for Bel's MaxShield-recorded MEG). Append them to the per-modality
    # kwargs so the rendered call carries them.
    n_kwargs = list(neuro["kwargs"]) + list(stu.get("neuro_kwargs") or [])
    s_kwargs = list(stim["kwargs"]) + list(stu.get("stim_kwargs") or [])
    lines.append(
        _multiline_call(
            f"neuro_ext = ns.extractors.{neuro['cls']}", n_kwargs, "infra=infra"
        )
    )
    lines.append(
        _multiline_call(f"stim_ext  = ns.extractors.{stim['cls']}", s_kwargs, s_trailing)
    )
    lines.append("")
    lines.append(f'# 3. Segment around each "{stim["event_type"]}" event')
    lines.append("segmenter = ns.Segmenter(")
    lines.append(f"    start={win['start']}, duration={win['duration']},")
    lines.append(f"    trigger_query='type==\"{stim['event_type']}\"',")
    lines.append("    extractors=dict(neuro=neuro_ext, stim=stim_ext),")
    lines.append("    drop_incomplete=True,")
    lines.append(")")
    lines.append("")
    # All instances defined — now run the pipeline.
    lines.append("# 4. Run the study and apply the segmenter")
    lines.append("study.download()")
    lines.append(_EVENTS_LINE)
    lines.append("dset = segmenter.apply(events)")
    lines.append("dset.prepare()")

    ml = _ml_compute_lines(neuro, stim, task, model, 0)
    if ml:
        lines.append("")
        lines.append(f"# 5. {_direction_label(task)}")
        lines.extend(ml)
        lines.append(
            'print(f"'
            + _ml_label(stim, task, model)
            + " = {"
            + _ml_metric_expr(model)
            + ':.3f}")'
        )
    else:
        lines.append("")
        lines.append("# 5. Inspect one batch of 8 segments")
        lines.extend(_load_demo_lines(0))
    return "\n".join(lines)


def _py_value_to_yaml(v: str) -> str:
    v = v.strip()
    if v.startswith("(") and v.endswith(")"):
        inner = v[1:-1].rstrip(",").strip()
        items = []
        for s in inner.split(","):
            t = s.strip()
            if (t.startswith("'") and t.endswith("'")) or (
                t.startswith('"') and t.endswith('"')
            ):
                t = t[1:-1]
            items.append(t)
        return "[" + ", ".join(items) + "]"
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        if "\\" not in v:
            return v[1:-1]
    return v


def _kwargs_list_to_yaml(kw_list: list[str], indent: int) -> str:
    pad = " " * indent
    out: list[str] = []
    for kv in kw_list or []:
        eq = kv.index("=")
        k = kv[:eq].strip()
        v = _py_value_to_yaml(kv[eq + 1 :])
        out.append(f"{pad}{k}: {v}")
    return "\n".join(out)


def _infra_to_yaml(infra_literal: str, indent: int) -> str:
    """Render the infra literal as YAML key/value lines at the given indent."""
    import json

    pad = " " * indent
    return "\n".join(
        f"{pad}{k}: {'null' if v is None else v}"
        for k, v in json.loads(infra_literal).items()
    )


def _build_yaml_script(
    neuro: dict, stim: dict, comp: dict, task: dict, stu: dict, model: dict
) -> str:
    win = neuro["window"]
    needs_ml = bool(task.get("needs_ml"))

    n_kw = (
        list(neuro.get("kwargs") or [])
        + list(neuro.get("yaml_extra") or [])
        + list(stu.get("neuro_kwargs") or [])
    )
    s_kw = (
        list(stim.get("kwargs") or [])
        + list(stim.get("yaml_extra") or [])
        + list(stu.get("stim_kwargs") or [])
    )

    stim_block = [
        "    stim:",
        f"      name: {stim['cls']}",
        _kwargs_list_to_yaml(s_kw, 6),
    ]
    if stim.get("accepts_infra") is not False:
        stim_block += [
            "      infra:",
            _infra_to_yaml(comp["infra_literal"], 8),
        ]

    # `infra: {folder: $CACHE}` enables caching of `study.run()` events
    # between calls. Same `$CACHE` placeholder as the extractor MapInfras.
    yaml_sections = [
        f"# {stu['comment']}",
        "study:",
        f"  name: {stu['name']}",
        f"  path: $STUDIES/{stu['name']}",
        "  infra:",
        "    folder: $CACHE",
        "segmenter:",
        f"  start: {win['start']}",
        f"  duration: {win['duration']}",
        f"  trigger_query: 'type==\"{stim['event_type']}\"'",
        "  drop_incomplete: true",
        "  extractors:",
        "    neuro:",
        f"      name: {neuro['cls']}",
        _kwargs_list_to_yaml(n_kw, 6),
        "      infra:",
        _infra_to_yaml(comp["infra_literal"], 8),
        *stim_block,
    ]
    if needs_ml:
        yaml_sections += [
            "infra:",
            _infra_to_yaml(comp["infra_literal"], 2),
        ]
    yaml_body = "\n".join(yaml_sections)

    imports: list[str] = [
        "from pathlib import Path",
        "import yaml, pydantic",
        "import neuralset as ns",
    ]
    if needs_ml:
        imports.append("import exca")
    imports.extend(_ml_imports(stim, task, model))
    imports.extend(_load_imports(task))

    py_lines: list[str] = imports[:]
    py_lines.append("")
    py_lines.append("config = '''")
    py_lines.append(yaml_body)
    py_lines.append("'''")
    py_lines.append("")
    py_lines.append('CACHE = Path.home() / "neuroai_data" / ".cache"')
    py_lines.append('STUDIES = Path.home() / "neuroai_data" / ".studies"')
    py_lines.append("STUDIES.mkdir(parents=True, exist_ok=True)")
    # `str.replace` requires str args, so str() is unavoidable here
    # (unlike the kwargs-mode infra dict where Pydantic coerces Path).
    py_lines.append("cfg = yaml.safe_load(")
    py_lines.append(
        '    config.replace("$CACHE", str(CACHE)).replace("$STUDIES", str(STUDIES))'
    )
    py_lines.append(")")
    py_lines.append("")
    py_lines.append("class Experiment(pydantic.BaseModel):")
    py_lines.append(
        "    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)"
    )
    py_lines.append("    study: ns.Study")
    py_lines.append("    segmenter: ns.Segmenter")
    if needs_ml:
        py_lines.append("    infra: exca.TaskInfra = exca.TaskInfra()")
        py_lines.append("")
        py_lines.append("    @infra.apply")
        py_lines.append("    def score(self) -> float:")
        py_lines.append("        self.study.download()")
        py_lines.append(
            "        " + _EVENTS_LINE.replace("study.run()", "self.study.run()")
        )
        py_lines.append("        dset = self.segmenter.apply(events)")
        py_lines.append("        dset.prepare()")
        py_lines.extend(_ml_compute_lines(neuro, stim, task, model, 8))
        py_lines.append("        return float(" + _ml_metric_expr(model) + ")")
    py_lines.append("")
    py_lines.append("exp = Experiment(**cfg)")
    if needs_ml:
        py_lines.append("")
        py_lines.append(f"# {_direction_label(task)}")
        py_lines.append("score = exp.score()")
        py_lines.append('print(f"' + _ml_label(stim, task, model) + ' = {score:.3f}")')
    else:
        py_lines.append("exp.study.download()")
        py_lines.append(_EVENTS_LINE.replace("study.run()", "exp.study.run()"))
        py_lines.append("dset = exp.segmenter.apply(events)")
        py_lines.append("dset.prepare()")
        py_lines.append("")
        py_lines.append("# Inspect one batch of 8 segments")
        py_lines.extend(_load_demo_lines(0))
    return "\n".join(py_lines)


def _render(sel: dict) -> str:
    neuro = AXES["neuro"]["options"][sel["neuro"]]
    stim = AXES["stim"]["options"][sel["stim"]]
    task = AXES["task"]["options"][sel["task"]]
    comp = AXES["compute"]["options"][sel["compute"]]
    stu = _resolve_study(sel)
    model = _resolve_model(sel)
    if sel["style"] == "yaml":
        return _build_yaml_script(neuro, stim, comp, task, stu, model)
    return _build_kwargs_script(neuro, stim, comp, task, stu, model)


# ── L0: YAML schema ─────────────────────────────────────────────────────────


def test_l0_yaml_axes_match_expected_set() -> None:
    assert set(AXES) == set(AXIS_ORDER), AXES.keys()
    for axis in AXIS_ORDER:
        meta = AXES[axis]
        assert "label" in meta
        assert "default" in meta and meta["default"] in meta["options"]
        assert meta["options"], f"{axis} has no options"


def test_l0_neuro_options_have_required_fields() -> None:
    for key, opt in AXES["neuro"]["options"].items():
        for f in ("cls", "label", "kwargs", "window", "event_type"):
            assert f in opt, f"neuro.{key} missing {f}"
        assert isinstance(opt["kwargs"], list), f"neuro.{key}.kwargs must be a list"
        # Each option must declare either a `t` (with optional `t_comment`)
        # or a raw `x_expr` for the ML one-liner.
        assert ("t" in opt) ^ ("x_expr" in opt), (
            f"neuro.{key} must define exactly one of `t` or `x_expr`"
        )


def test_l0_stim_options_have_required_fields() -> None:
    for key, opt in AXES["stim"]["options"].items():
        for f in ("cls", "label", "kwargs", "event_type"):
            assert f in opt, f"stim.{key} missing {f}"
        assert isinstance(opt["kwargs"], list), f"stim.{key}.kwargs must be a list"


def test_l0_extractor_classes_resolve() -> None:
    import neuralset.extractors as nex

    for axis in ("neuro", "stim"):
        for key, opt in AXES[axis]["options"].items():
            cls = getattr(nex, opt["cls"], None)
            assert cls is not None, (
                f"{axis}.{key}.cls={opt['cls']!r} not on neuralset.extractors"
            )


def test_l0_generated_js_is_in_sync() -> None:
    """The committed .js must match what the generator would produce now."""
    expected = build_data.render(_load_yaml())
    actual = DATA_JS.read_text()
    assert expected == actual, (
        "docs/_static/code-builder-data.js is stale. "
        "Regenerate with: python docs/_data/build_data.py"
    )


# ── L1: AST sweep over per-axis-pinned combos ──────────────────────────────
# (Switched from the full Cartesian sweep — the per-axis-pinned set already
# covers every conditional branch in the renderer; the Cartesian sweep was
# adding ~850 redundant tests for ~zero coverage gain.)


@pytest.mark.parametrize("sel", _axis_pinned(), ids=_id)
def test_l1_rendered_script_parses(sel: dict) -> None:
    script = _render(sel)
    try:
        ast.parse(script)
    except SyntaxError as e:
        pytest.fail(
            f"SyntaxError in rendered script for {_id(sel)}:\n{e}\n--- script ---\n{script}"
        )


# ── L2: Per-axis instantiation against real neuralset ───────────────────────


@pytest.mark.parametrize("sel", _axis_pinned(), ids=_id)
def test_l2_pydantic_instantiates(sel: dict, tmp_path: Path) -> None:
    """Instantiate extractors without running them (no Study, no Segmenter)."""
    if sel["compute"] == "slurm":
        pytest.skip("slurm path is generated-only; not instantiated here")

    import neuralset.extractors as nex

    n = AXES["neuro"]["options"][sel["neuro"]]
    s = AXES["stim"]["options"][sel["stim"]]

    g: dict = {"nex": nex, "infra": {"folder": str(tmp_path), "cluster": None}}
    n_kw = ", ".join(n["kwargs"])
    s_kw = ", ".join(s["kwargs"])
    s_infra = "" if s.get("accepts_infra") is False else ", infra=infra"
    src = (
        f"neuro_ext = nex.{n['cls']}({n_kw}, infra=infra)\n"
        f"stim_ext  = nex.{s['cls']}({s_kw}{s_infra})\n"
    )
    try:
        exec(src, g)
    except Exception as e:
        pytest.fail(
            f"L2 extractor instantiation failed for {_id(sel)}:\n{e}\n--- src ---\n{src}"
        )


# ── L3: End-to-end (slow, opt-in) ───────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("sel", _axis_pinned(), ids=_id)
def test_l3_pipeline_runs_end_to_end(sel: dict, tmp_path: Path) -> None:
    """Exec the rendered script in a subprocess (clean Study.catalog())."""
    if sel["compute"] == "slurm":
        pytest.skip("slurm path is generated-only; not exec'd here")

    for axis in ("neuro", "stim"):
        opt = AXES[axis]["options"][sel[axis]]
        if opt.get("l3_skip"):
            pytest.skip(f"{axis}.{sel[axis]}: {opt['l3_skip']}")

    # Real public datasets (Allen2022Massive, Bel2026Petit, etc.) require
    # large network downloads, are subject to upstream availability, and
    # often need credentials — keep L3 to FakeMulti combos so it stays
    # fast and self-contained.
    stu = _resolve_study(sel)
    if stu["name"] != DEFAULT_STUDY["name"]:
        pytest.skip(f"{_id(sel)}: uses real public dataset {stu['name']!r}")

    script = _render(sel)
    pre = f"from pathlib import Path as _P\n_TMP = _P({str(tmp_path)!r})\n"
    script = script.replace(
        'CACHE = Path.home() / "neuroai_data" / ".cache"',
        'CACHE = _TMP / ".cache"',
    ).replace(
        'STUDIES = Path.home() / "neuroai_data" / ".studies"',
        'STUDIES = _TMP / ".studies"',
    )
    script_path = tmp_path / "script.py"
    script_path.write_text(pre + script)

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"L3 exec failed for {_id(sel)}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr[-3000:]}\n"
            f"--- script ({script_path}) ---\n{script}"
        )
