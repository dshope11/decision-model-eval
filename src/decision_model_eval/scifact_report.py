"""The SciFact control run's report: accuracy, calibration, working points, agreement.

``report(run_dir, data_dir)`` reads every log of one complete run (``scifact.attempts``
refuses a log of another configuration) and writes ``results.json``, ``summary.md``,
``items.jsonl`` (per-pair scores) and the figures into ``<run_dir>/report/``.

Each item has ``REPEATS`` answers per backend, which give two scores per option: the
**stated mean** (the mean stated probability) and the **vote fraction** (the share of
repeats that answered it). An item's answer is its stated mean's top option.

- **Accuracy, calibration, agreement and Jev's confidence check use every item**:
  nothing in them is tuned.
- **Working points use a seeded 50/50 split, stratified by label.** An answered item
  scores +1 if right and -c if wrong; an escalated item scores 0; the score is the top
  option's probability. Three rows per c, all evaluated on the reporting half:
  ``frozen`` (threshold selected on the selection half), ``oracle`` (selected on the
  reporting half itself, a biased upper bound shown only for the size of the bias) and
  ``stated`` (act when the top probability is at least c / (1 + c), the break-even point
  if the probabilities are honest; nothing tuned).
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from decision_model_eval import jsonl, scifact
from decision_model_eval.calibration import (
    Reliability,
    brier,
    classwise_ece,
    reliability,
    top_label,
    top_label_ece,
)
from decision_model_eval.scifact import Pair, call_key
from decision_model_eval.stats import clopper_pearson
from decision_model_eval.thresholds import ThresholdReport, report_at, scan_thresholds

COSTS = (1, 3, 9)
SCORES = ("mean", "vote")
# Probabilities are stated to about two decimals, and a stated mean of five 0.9s can land
# a rounding error below 0.9; this keeps ">= c / (1 + c)" meaning what it says.
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Answers:
    """One backend's answers to one question, item by item in release order."""

    point: str
    backend: str
    options: list[str]
    pairs: list[Pair]
    y: np.ndarray  # (items,) index of the true option
    stated: np.ndarray  # (items, repeats, options) stated probabilities
    chosen: np.ndarray  # (items, repeats) index of the option answered
    confidence: np.ndarray | None  # (items, repeats), where the backend reports one

    def scores(self, kind: str) -> np.ndarray:
        """(items, options): the stated mean, or the vote fraction."""
        if kind == "mean":
            return np.asarray(self.stated.mean(axis=1))
        counts = [np.bincount(row, minlength=len(self.options)) for row in self.chosen]
        return np.asarray(np.array(counts) / self.chosen.shape[1])


def truth(point: str, pair: Pair) -> str:
    return pair.label if point == "scifact.verdict" else pair.supports_label


def split(pairs: Sequence[Pair], seed: int) -> np.ndarray:
    """True for the selection half: a seeded half of each label (rounded down)."""
    rng = random.Random(seed)
    selection = np.zeros(len(pairs), dtype=bool)
    for label in scifact.VERDICT.options:
        members = [i for i, pair in enumerate(pairs) if pair.label == label]
        rng.shuffle(members)
        selection[members[: len(members) // 2]] = True
    return selection


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = clopper_pearson(np.array([k]), np.array([n]))
    return {"k": k, "n": n, "value": k / n, "lo": float(lo[0]), "hi": float(hi[0])}


def _point(row: ThresholdReport, n: int, threshold: float | None = None) -> dict[str, Any]:
    answered = int(round(row.tp + row.fp))
    nominal = row.threshold if threshold is None else threshold
    return {
        "threshold": nominal if np.isfinite(nominal) else None,  # None: escalate every item
        "utility_per_item": row.value / n, "coverage": answered / n, "answered": answered,
        "wrong": int(round(row.fp)), "accuracy_answered": row.tp / answered if answered else None,
    }


def working_points(score: np.ndarray, correct: np.ndarray,
                   selection: np.ndarray) -> dict[str, dict[str, Any]]:
    """Frozen, oracle and stated rows per c, for one score of the top option."""
    y = correct.astype(int)
    w = np.ones(len(score))
    sel, rep = selection, ~selection
    n = int(rep.sum())
    rows = {}
    for c in COSTS:
        def objective(tp: float, fp: float, pos: float, neg: float, c: float = c) -> float:
            return tp - c * fp
        scan = scan_thresholds(score[sel], y[sel], w[sel], score[rep], y[rep], w[rep],
                               objective)
        stated = report_at(score[rep], y[rep], w[rep], c / (1 + c) - TOLERANCE, objective)
        rows[f"c={c}"] = {"frozen": _point(scan.frozen, n), "oracle": _point(scan.oracle, n),
                          "stated": _point(stated, n, threshold=c / (1 + c))}
    return rows


def _curve(curve: Reliability) -> dict[str, list[float]]:
    return {"stated": curve.stated.tolist(), "observed": curve.observed.tolist(),
            "lo": curve.lo.tolist(), "hi": curve.hi.tolist(), "n": curve.n.tolist()}


def _curves(p: np.ndarray, y: np.ndarray, options: list[str]) -> dict[str, Any]:
    curves = {"top": _curve(reliability(*top_label(p, y)))}
    for k, option in enumerate(options):
        curves[option] = _curve(reliability(p[:, k], y == k))
    return curves


def collect(run_dir: Path, data_dir: str | Path
            ) -> tuple[dict[str, Any], list[dict[str, Any]], list[Answers]]:
    """The run header, every attempt, and the answers per question and backend.

    Refuses a run with a planned call still unanswered.
    """
    logs = scifact.run_logs(run_dir)
    if not logs:
        raise ValueError(f"{run_dir}: no logs")
    header: dict[str, Any] = jsonl.read_records(logs[0])[0][1]["run"]
    backends = list(header["points"]["scifact.verdict"]["backends"])
    config = scifact.run_config(backends=backends, seed=header["seed"], limit=header["limit"])
    attempts = scifact.attempts(run_dir, config)
    decisions: dict[str, dict[str, Any]] = {}
    for record in attempts:
        if record["error"] is None:
            key = call_key(record["decision_point"], record["backend"], record["item"],
                           record["repeat_index"])
            if key in decisions:
                raise ValueError(f"{run_dir}: {key} answered twice")
            decisions[key] = record
    pairs = scifact.load_pairs(data_dir)
    planned = scifact.plan(pairs, backends=backends, seed=header["seed"],
                           limit=header["limit"])
    missing = [call for call in planned if call.key not in decisions]
    if missing:
        raise ValueError(f"{run_dir}: {len(missing)} of {len(planned)} planned calls "
                         "unanswered; run it again to finish")

    answers = []
    for point, question in scifact.POINTS.items():
        for backend in header["points"][point]["backends"]:
            items = list(dict.fromkeys(c.pair for c in planned
                                       if c.point == point and c.backend == backend))
            records = [[decisions[call_key(point, backend, pair.item, r)]
                        for r in range(scifact.REPEATS)] for pair in items]
            options = question.options
            confidence = [[r["confidence"] for r in row] for row in records]
            answers.append(Answers(
                point=point, backend=backend, options=options, pairs=items,
                y=np.array([options.index(truth(point, pair)) for pair in items]),
                stated=np.array([[[r["output"]["probabilities"][o] for o in options]
                                  for r in row] for row in records]),
                chosen=np.array([[options.index(r["output"]["answer"]) for r in row]
                                 for row in records]),
                confidence=None if any(c is None for row in confidence for c in row)
                else np.array(confidence)))
    return header, attempts, answers


def _cost(attempts: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    mine = [r for r in attempts if r["backend"] == backend]
    ok = [r for r in mine if r["error"] is None]
    usd = sum(r["cost_usd"] or 0.0 for r in mine)
    thinking = [r["raw_response"].get("usage", {}).get("output_tokens_details", {})
                .get("thinking_tokens") for r in ok]
    known = [t for t in thinking if t is not None]
    return {
        "attempts": len(mine), "failed_attempts": len(mine) - len(ok), "usd": usd,
        "usd_per_answer": usd / len(ok), "latency_ms_median": statistics.median(
            r["latency_ms"] for r in ok),
        "tokens_in_median": statistics.median(r["tokens_in"] for r in ok),
        "thinking_tokens": {"median": statistics.median(known),
                            "share_zero": sum(t == 0 for t in known) / len(known)}
        if known else None,
    }


def analyze(header: dict[str, Any], attempts: list[dict[str, Any]],
            answers: list[Answers]) -> dict[str, Any]:
    verdicts = {a.backend: a for a in answers if a.point == "scifact.verdict"}
    pairs = next(iter(verdicts.values())).pairs
    selection = split(pairs, header["seed"])
    results: dict[str, Any] = {
        "run": {key: header[key] for key in ("dataset", "split", "release_sha256", "seed",
                                             "repeats", "limit", "points", "sampling")},
        "split": {"seed": header["seed"],
                  "selection": dict(Counter(p.label for p, s in zip(pairs, selection,
                                                                    strict=True) if s)),
                  "reporting": dict(Counter(p.label for p, s in zip(pairs, selection,
                                                                    strict=True) if not s))},
        "points": {},
    }
    counts = Counter(pair.label for pair in pairs)
    majority = max(scifact.VERDICT.options, key=lambda option: counts[option])
    results["majority_baseline"] = {"answer": majority,
                                    "accuracy": _rate(counts[majority], len(pairs))}

    for a in answers:
        mean = a.scores("mean")
        answer = mean.argmax(axis=1)
        entry: dict[str, Any] = {
            "model": header["points"][a.point]["backends"][a.backend],
            "items": len(a.pairs),
            "accuracy": _rate(int((answer == a.y).sum()), len(a.pairs)),
            "accuracy_single_call": _rate(int((a.chosen == a.y[:, None]).sum()),
                                          a.chosen.size),
            "answer_changed_across_repeats": float(np.mean(
                [len(set(row.tolist())) > 1 for row in a.chosen])),
            "sd_true_option_across_repeats": float(np.mean(
                a.stated[np.arange(len(a.y)), :, a.y].std(axis=1))),
            "calibration": {}, "reliability": {}, "cost": _cost(
                [r for r in attempts if r["decision_point"] == a.point], a.backend),
        }
        for kind in SCORES:
            p = a.scores(kind)
            entry["calibration"][kind] = {"top_label_ece": top_label_ece(p, a.y),
                                          "classwise_ece": classwise_ece(p, a.y),
                                          "brier": brier(p, a.y)}
            entry["reliability"][kind] = _curves(p, a.y, a.options)
        if a.point == "scifact.verdict":
            entry["working_points"] = {}
            for kind in SCORES:
                top, correct = top_label(a.scores(kind), a.y)
                entry["working_points"][kind] = working_points(top, correct, selection)
        if a.confidence is not None:
            k = len(a.options)
            rescaled = (k * a.stated.max(axis=2) - 1) / (k - 1)
            residual = np.abs(a.confidence - rescaled)
            entry["confidence_vs_rescaled_top"] = {
                "formula": "(K * pmax - 1) / (K - 1)", "K": k, "calls": int(residual.size),
                "max_abs_residual": float(residual.max()),
                "mean_abs_residual": float(residual.mean())}
        results["points"].setdefault(a.point, {})[a.backend] = entry

    if len(verdicts) == 2:
        first, second = verdicts.values()
        options = first.options
        x = first.scores("mean").argmax(axis=1)
        y = second.scores("mean").argmax(axis=1)
        matrix = np.zeros((len(options), len(options)), dtype=int)
        np.add.at(matrix, (x, y), 1)
        results["agreement"] = {"rows": first.backend, "columns": second.backend,
                                "options": options, "matrix": matrix.tolist(),
                                "agreement": float(np.mean(x == y))}
    return results


def _pct(rate: dict[str, Any]) -> str:
    return (f"{100 * rate['value']:.1f}% (+{100 * (rate['hi'] - rate['value']):.1f} / "
            f"-{100 * (rate['value'] - rate['lo']):.1f})")


def _row_cells(row: dict[str, Any]) -> str:
    threshold = "all" if row["threshold"] is None else f"{row['threshold']:.3f}"
    accuracy = ("-" if row["accuracy_answered"] is None
                else f"{100 * row['accuracy_answered']:.1f}%")
    return (f"{threshold} | {100 * row['coverage']:.0f}% | {accuracy} | "
            f"{row['utility_per_item']:+.3f}")


def summary(results: dict[str, Any], name: str) -> str:
    """The results as Markdown tables, generated so they never drift from results.json."""
    verdict = results["points"]["scifact.verdict"]
    run = results["run"]
    lines = [
        f"# SciFact control run: {name}", "",
        f"{run['split']} split, {next(iter(verdict.values()))['items']} (claim, abstract) "
        f"pairs, {run['repeats']} repeats per backend; release sha256 "
        f"{run['release_sha256'][:12]}. Rates are given with 1 sigma Clopper-Pearson "
        "intervals.", "",
        "## Accuracy and calibration (3-option verdict, every pair, stated mean)", "",
        "| Backend | Accuracy | Single-call accuracy | Top-label ECE | Classwise ECE "
        "| Brier | Cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in verdict.values():
        cal = entry["calibration"]["mean"]
        lines.append(f"| {entry['model']} | {_pct(entry['accuracy'])} | "
                     f"{_pct(entry['accuracy_single_call'])} | {cal['top_label_ece']:.3f} | "
                     f"{cal['classwise_ece']:.3f} | {cal['brier']:.3f} | "
                     f"${entry['cost']['usd']:.3f} |")
    base = results["majority_baseline"]
    lines += [f"| majority class ({base['answer']}) | {_pct(base['accuracy'])} | | | | | |",
              "", "## Working points (reporting half, top stated-mean probability)", "",
              f"Split: selection {results['split']['selection']}, "
              f"reporting {results['split']['reporting']}. An answer scores +1 if right "
              "and -c if wrong; an escalated item 0. Each cell is threshold | coverage | "
              "accuracy of the answered | utility per item. The oracle row is a biased "
              "upper bound.", "",
              "| Backend | c | Row | Threshold | Coverage | Accuracy answered | Utility |",
              "|---|---|---|---|---|---|---|"]
    for backend, entry in verdict.items():
        for c, rows in entry["working_points"]["mean"].items():
            for row in ("frozen", "stated", "oracle"):
                lines.append(f"| {backend} | {c.removeprefix('c=')} | {row} | "
                             f"{_row_cells(rows[row])} |")
    if "agreement" in results:
        agree = results["agreement"]
        lines += ["", f"## Agreement ({agree['rows']} rows, {agree['columns']} columns)", "",
                  "| | " + " | ".join(agree["options"]) + " |",
                  "|---" * (len(agree["options"]) + 1) + "|"]
        for option, row in zip(agree["options"], agree["matrix"], strict=True):
            lines.append(f"| {option} | " + " | ".join(str(v) for v in row) + " |")
        lines += ["", f"Same answer on {100 * agree['agreement']:.1f}% of pairs."]
    checks = [(point, backend, entry["confidence_vs_rescaled_top"])
              for point, entries in results["points"].items()
              for backend, entry in entries.items() if "confidence_vs_rescaled_top" in entry]
    if checks:
        lines += ["", "## Jev's confidence vs (K * pmax - 1) / (K - 1)", "",
                  "| Question | K | Calls | Max abs residual | Mean abs residual |",
                  "|---|---|---|---|---|"]
        for point, _, check in checks:
            lines.append(f"| {point} | {check['K']} | {check['calls']} | "
                         f"{check['max_abs_residual']:.4f} | {check['mean_abs_residual']:.4f} |")
    return "\n".join(lines) + "\n"


def items(answers: list[Answers]) -> list[dict[str, Any]]:
    """One row per question and pair: the truth and each backend's scores."""
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for a in answers:
        mean, vote = a.scores("mean"), a.scores("vote")
        for i, pair in enumerate(a.pairs):
            row = rows.setdefault((a.point, pair.claim_id, pair.doc_id), {
                "point": a.point, **pair.item, "label": truth(a.point, pair),
                "claim": pair.claim})
            row[a.backend] = {"answer": a.options[int(mean[i].argmax())],
                              "mean": dict(zip(a.options, mean[i].tolist(), strict=True)),
                              "vote": dict(zip(a.options, vote[i].tolist(), strict=True))}
    return list(rows.values())


def plot(results: dict[str, Any], answers: list[Answers], out: Path) -> list[Path]:
    """Reliability diagrams of the verdict, one figure per score kind, and the
    distribution of the top stated-mean probability, right and wrong answers stacked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    verdicts = [a for a in answers if a.point == "scifact.verdict"]
    fig, axes = plt.subplots(1, len(verdicts), squeeze=False,
                             figsize=(4.5 * len(verdicts), 3.4))
    for ax, a in zip(axes[0], verdicts, strict=True):
        top, correct = top_label(a.scores("mean"), a.y)
        ax.hist([top[correct], top[~correct]], bins=np.linspace(1 / 3, 1, 21),
                stacked=True, label=["right", "wrong"])
        ax.set(title=a.backend, xlabel="top stated-mean probability", ylabel="pairs")
        ax.legend()
    fig.tight_layout()
    path = out / "top_probability.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    verdict = results["points"]["scifact.verdict"]
    for kind in SCORES:
        panels = ["top", *scifact.VERDICT.options]
        fig, axes = plt.subplots(len(verdict), len(panels), squeeze=False,
                                 figsize=(3.2 * len(panels), 3.2 * len(verdict)))
        for row, (backend, entry) in enumerate(verdict.items()):
            for col, panel in enumerate(panels):
                curve = entry["reliability"][kind][panel]
                ax = axes[row][col]
                ax.plot([0, 1], [0, 1], color="0.7", linewidth=1)
                stated, observed = np.array(curve["stated"]), np.array(curve["observed"])
                ax.errorbar(stated, observed, fmt="o", capsize=2,
                            yerr=[observed - np.array(curve["lo"]),
                                  np.array(curve["hi"]) - observed])
                ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03),
                       title=f"{backend}: {'top option' if panel == 'top' else panel}",
                       xlabel=f"stated {kind}", ylabel="observed frequency")
        fig.tight_layout()
        path = out / f"reliability_{kind}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def report(run_dir: str | Path, data_dir: str | Path
           ) -> tuple[dict[str, Any], str, list[Path]]:
    """Analyze a complete run; write results.json, summary.md, items.jsonl and the
    figures to report/."""
    run_dir = Path(run_dir)
    header, attempts, answers = collect(run_dir, data_dir)
    results = analyze(header, attempts, answers)
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    text = summary(results, run_dir.name)
    (out / "summary.md").write_text(text)
    (out / "items.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items(answers)))
    written = [out / name for name in ("results.json", "summary.md", "items.jsonl")]
    return results, text, written + plot(results, answers, out)
