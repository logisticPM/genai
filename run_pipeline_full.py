"""Two-phase Pipeline experiment on Misviz real dataset (aligned with B's 271 samples).

Phase 1: Batch OCR all images (subprocess per batch, solves memory leak)
Phase 2: Multi-threaded VLM calls (8 threads, reads pre-computed OCR)

Usage:
    python run_pipeline_full.py                    # Full run (both phases)
    python run_pipeline_full.py --phase 1          # OCR only
    python run_pipeline_full.py --phase 2          # VLM only (requires Phase 1 done)
    python run_pipeline_full.py --workers 4        # Fewer threads
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

PYTHON = sys.executable
OUT_DIR = Path("data/eval_results/pipeline_full")
OCR_CACHE_DIR = OUT_DIR / "ocr_cache"
BATCH_SIZE = 5  # charts per OCR subprocess


def select_samples_aligned_with_b() -> list[dict]:
    """Use B's exact 271 sample IDs from his results file."""
    import json as _json
    from data_tools.misviz.loader import MisvizLoader

    b_results_path = Path("C:/Users/chntw/Documents/7180/PCBZ_FinChartAudit/results/claude_vision_only.json")
    if not b_results_path.exists():
        raise FileNotFoundError(f"B's results not found at {b_results_path}. Clone B's repo first.")

    b_data = _json.loads(b_results_path.read_text(encoding="utf-8"))
    b_items = b_data["results"]
    print(f"B's sample count: {len(b_items)}")

    # B loads from HuggingFace datasets, which may have different ordering.
    # We need to match by content (ground truth labels) since IDs are positional.
    # B's results contain gt_misleaders for each sample — use those as ground truth.
    loader = MisvizLoader()
    real_data = loader.load_real()

    # Build a lookup: frozenset(misleaders) + chart_type -> list of local indices
    from collections import defaultdict as _defaultdict
    content_to_local = _defaultdict(list)
    for i, d in enumerate(real_data):
        key = (frozenset(d.get("misleader", [])), tuple(sorted(d.get("chart_type", []))))
        content_to_local[key].append(i)

    # Match B's samples to local indices
    samples = []
    matched = 0
    used_local_indices = set()

    for b_item in b_items:
        b_gt = b_item["gt_misleaders"]
        b_chart_type = b_item.get("chart_type", [])
        key = (frozenset(b_gt), tuple(sorted(b_chart_type)))

        candidates = [idx for idx in content_to_local.get(key, []) if idx not in used_local_indices]
        if candidates:
            idx = candidates[0]
            used_local_indices.add(idx)
            instance = loader.get_real_instance(idx)
            if Path(instance.image_path).exists():
                samples.append({
                    "idx": idx,
                    "instance_id": str(b_item["id"]),  # Use B's ID for alignment
                    "image_path": instance.image_path,
                    "ground_truth": instance.misleader,
                    "b_id": b_item["id"],
                })
                matched += 1

    print(f"Matched {matched}/{len(b_items)} of B's samples to local data")
    return samples


# ── Phase 1: Batch OCR ──────────────────────────────────────────────────────

OCR_WORKER_SCRIPT = OUT_DIR / "_ocr_worker.py"

def write_ocr_worker():
    """Write a worker script that OCRs a batch of images."""
    OCR_WORKER_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    # Embed the project root so the worker can find modules regardless of where it's run from
    project_root = str(Path(__file__).parent.resolve()).replace("\\", "\\\\")
    worker_code = '''
"""Worker: OCR a batch of images. Input: JSON list of {id, image_path} on stdin. Output: JSON to stdout."""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
sys.path.insert(0, r"__PROJECT_ROOT__")

# Read batch from stdin
batch = json.loads(sys.stdin.read())

from finchartaudit.tools.traditional_ocr import TraditionalOCRTool
from finchartaudit.tools.rule_check import RuleEngine

ocr = TraditionalOCRTool()
engine = RuleEngine()
results = {}

import re

def extract_numbers(result):
    numbers = []
    for b in result.get("text_blocks", result.get("texts", [])):
        text = b.get("text", b) if isinstance(b, dict) else str(b)
        for match in re.findall(r"-?\\d+\\.?\\d*", text):
            try:
                numbers.append(float(match))
            except ValueError:
                pass
    return sorted(set(numbers))

def format_ocr(result):
    blocks = result.get("text_blocks", [])
    lines = []
    for b in blocks[:20]:
        text = b.get("text", "")
        conf = b.get("confidence", 0)
        if conf > 0.5 and text.strip():
            lines.append(text.strip())
    return "\\n".join(lines) if lines else "No confident text detected."

for item in batch:
    image_path = item["image_path"]
    iid = item["id"]
    try:
        full_result = ocr.run(image_path, "full", "bbox")
        y_result = ocr.run(image_path, "y_axis", "text")
        right_result = ocr.run(image_path, "right_axis", "text")
        x_result = ocr.run(image_path, "x_axis", "text")

        ocr_text = format_ocr(full_result)
        ocr_axis = format_ocr(y_result)
        axis_values = extract_numbers(y_result)
        right_axis_values = extract_numbers(right_result)
        x_axis_values = extract_numbers(x_result)

        # Run rule checks
        rule_results = []
        if axis_values:
            for check_type in ["truncated_axis", "broken_scale"]:
                try:
                    r = engine.run_check(check_type, {"axis_values": axis_values, "chart_type": "bar"})
                    rule_results.append(f"{check_type}: {r['explanation']}")
                except Exception:
                    pass
            try:
                r = engine.run_check("inverted_axis", {"axis_values": axis_values})
                if r["is_inverted"]:
                    rule_results.append(f"inverted_axis: {r['explanation']}")
            except Exception:
                pass
            try:
                r = engine.run_check("inappropriate_axis_range", {"axis_values": axis_values})
                if r["is_inappropriate"]:
                    rule_results.append(f"inappropriate_axis_range: {r['explanation']}")
            except Exception:
                pass

        if axis_values and right_axis_values:
            try:
                r = engine.run_check("dual_axis", {
                    "left_axis_values": axis_values, "right_axis_values": right_axis_values})
                if r["has_dual_axis"]:
                    rule_results.append(f"dual_axis: {r['explanation']}")
            except Exception:
                pass

        if len(x_axis_values) >= 3:
            try:
                r = engine.run_check("inconsistent_binning", {"bin_edges": x_axis_values})
                if r["is_inconsistent"]:
                    rule_results.append(f"inconsistent_binning: {r['explanation']}")
            except Exception:
                pass

        results[iid] = {
            "ocr_text": ocr_text,
            "ocr_axis": ocr_axis,
            "axis_values": axis_values,
            "right_axis_values": right_axis_values,
            "x_axis_values": x_axis_values,
            "rule_results": rule_results,
        }
    except Exception as e:
        results[iid] = {"error": str(e)}

print(json.dumps(results))
'''.replace("__PROJECT_ROOT__", project_root)
    OCR_WORKER_SCRIPT.write_text(worker_code, encoding="utf-8")


def run_phase1(samples: list[dict]):
    """Batch OCR all images with subprocess isolation."""
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    write_ocr_worker()

    # Check what's already cached
    done = set()
    for f in OCR_CACHE_DIR.glob("*.json"):
        try:
            cached = json.loads(f.read_text(encoding="utf-8"))
            done.update(cached.keys())
        except Exception:
            pass

    remaining = [s for s in samples if s["instance_id"] not in done]
    print(f"Phase 1: OCR {len(remaining)} images ({len(done)} cached)")

    if not remaining:
        print("All OCR already cached, skipping Phase 1")
        return

    # Process in batches
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} images)...", end="", flush=True)

        batch_input = [{"id": s["instance_id"], "image_path": s["image_path"]} for s in batch]
        start = time.time()

        try:
            result = subprocess.run(
                [PYTHON, "-X", "utf8", str(OCR_WORKER_SCRIPT)],
                input=json.dumps(batch_input),
                capture_output=True, text=True, timeout=900,
                env={**os.environ, "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
            )
            if result.returncode != 0:
                print(f" ERROR: {result.stderr[-200:]}")
                continue

            batch_results = json.loads(result.stdout)
            cache_file = OCR_CACHE_DIR / f"batch_{batch_num:04d}.json"
            cache_file.write_text(json.dumps(batch_results, indent=2), encoding="utf-8")
            elapsed = time.time() - start
            errors = sum(1 for v in batch_results.values() if "error" in v)
            print(f" done ({elapsed:.0f}s, {errors} errors)")

        except subprocess.TimeoutExpired:
            print(" TIMEOUT")
        except Exception as e:
            print(f" ERROR: {e}")

    # Cleanup worker
    if OCR_WORKER_SCRIPT.exists():
        OCR_WORKER_SCRIPT.unlink()


def load_ocr_cache() -> dict:
    """Load all cached OCR results into a single dict."""
    cache = {}
    for f in sorted(OCR_CACHE_DIR.glob("*.json")):
        try:
            batch = json.loads(f.read_text(encoding="utf-8"))
            cache.update(batch)
        except Exception:
            pass
    return cache


# ── Phase 2: Multi-threaded VLM ─────────────────────────────────────────────

def run_phase2(samples: list[dict], workers: int = 8):
    """Parallel VLM calls using pre-computed OCR + rules."""
    from finchartaudit.config import get_config
    from finchartaudit.agents.t2_pipeline import PIPELINE_SYSTEM_PROMPT, PIPELINE_PROMPT
    from finchartaudit.prompts.t2_visual import MISLEADER_DEFINITIONS, COMPLETENESS_CHECKS
    from data_tools.misviz.evaluator import MisvizEvaluator

    get_config.cache_clear()
    config = get_config()
    print(f"Phase 2: VLM calls with {workers} threads")
    print(f"Model: {config.vlm_model}")

    ocr_cache = load_ocr_cache()
    print(f"OCR cache loaded: {len(ocr_cache)} entries")

    missing_ocr = [s for s in samples if s["instance_id"] not in ocr_cache]
    if missing_ocr:
        print(f"WARNING: {len(missing_ocr)} samples missing OCR, run Phase 1 first")

    # Check existing results
    results_file = OUT_DIR / "raw_results.json"
    if results_file.exists():
        existing = json.loads(results_file.read_text(encoding="utf-8"))
        done_ids = {r["instance_id"] for r in existing if "error" not in r}
        print(f"Resuming: {len(done_ids)} already done")
    else:
        existing = []
        done_ids = set()

    remaining = [s for s in samples if s["instance_id"] not in done_ids]
    print(f"Remaining: {len(remaining)} samples")

    if not remaining:
        print("All done!")
        return

    # Build prompts
    misleader_list = "\n".join(f"- {k}: {v}" for k, v in MISLEADER_DEFINITIONS.items())
    completeness_list = "\n".join(f"- {k}: {v}" for k, v in COMPLETENESS_CHECKS.items())

    # Use OpenAI client directly for thread safety (like B does)
    from openai import OpenAI
    import base64
    from PIL import Image
    from io import BytesIO

    client = OpenAI(api_key=config.openrouter_api_key, base_url=config.openrouter_base_url)

    results = list(existing)
    lock = threading.Lock()
    completed = len(existing)
    errors = sum(1 for r in existing if "error" in r)

    def process_one(sample):
        nonlocal completed, errors
        iid = sample["instance_id"]
        ocr_data = ocr_cache.get(iid, {})

        if "error" in ocr_data:
            ocr_text = f"OCR failed: {ocr_data['error']}"
            ocr_axis = ""
            rule_results_str = "No rule checks (OCR failed)."
        else:
            ocr_text = ocr_data.get("ocr_text", "No OCR results.")
            ocr_axis = ocr_data.get("ocr_axis", "No axis values.")
            rule_results = ocr_data.get("rule_results", [])
            rule_results_str = "\n".join(rule_results) if rule_results else "No rule checks applicable."

        prompt = PIPELINE_PROMPT.format(
            chart_id=f"eval_{iid}",
            page=1,
            ocr_text=ocr_text,
            ocr_axis=ocr_axis,
            rule_results=rule_results_str,
            misleader_list=misleader_list,
            completeness_list=completeness_list,
        )

        # Encode image
        img = Image.open(sample["image_path"])
        if img.mode in ("CMYK", "RGBA", "P"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()

        try:
            start = time.time()
            response = client.chat.completions.create(
                model=config.vlm_model,
                messages=[
                    {"role": "system", "content": PIPELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ]},
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            elapsed = time.time() - start
            raw_text = response.choices[0].message.content or ""

            # Parse response (reuse pipeline logic)
            predicted = _parse_predicted(raw_text)

            result = {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": predicted,
                "elapsed_s": round(elapsed, 1),
                "rule_evidence": ocr_data.get("rule_results", []),
            }
        except Exception as e:
            errors += 1
            result = {
                "instance_id": iid,
                "ground_truth": sample["ground_truth"],
                "predicted": [],
                "error": str(e),
            }

        with lock:
            completed += 1
            results.append(result)
            status = result.get("predicted", "ERR")
            if "error" in result:
                print(f"[{completed}/{len(samples)}] id={iid} ERROR: {result['error'][:60]}")
            else:
                print(f"[{completed}/{len(samples)}] id={iid} gt={sample['ground_truth']} -> {status} ({result.get('elapsed_s', '?')}s)")

            # Save every 20
            if completed % 20 == 0:
                _save(results)

        return result

    # Run with thread pool
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one, s) for s in remaining]
        for future in as_completed(futures):
            try:
                future.result(timeout=120)
            except Exception as e:
                print(f"Thread error: {e}")

    _save(results)

    # Compute metrics
    evaluator = MisvizEvaluator()
    for r in results:
        if "error" not in r:
            evaluator.add_prediction(
                instance_id=r["instance_id"],
                ground_truth=r["ground_truth"],
                predicted=r["predicted"],
                condition="pipeline",
                model="claude_haiku",
            )

    evaluator.print_summary()
    metrics = evaluator.compute_metrics()
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nDone: {completed} charts, {errors} errors")


def _parse_predicted(text: str) -> list[str]:
    """Extract predicted misleader types from VLM JSON response."""
    data = _extract_json(text)
    if not data:
        return []
    predicted = []
    for name, assessment in data.get("misleaders", {}).items():
        if isinstance(assessment, dict) and assessment.get("present"):
            confidence = float(assessment.get("confidence", 0))
            if confidence >= 0.3:
                predicted.append(name)
    return list(set(predicted))


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    for marker in ["```json", "```"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _save(results):
    (OUT_DIR / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline experiment (two-phase)")
    parser.add_argument("--phase", type=int, default=0, help="1=OCR only, 2=VLM only, 0=both")
    parser.add_argument("--workers", type=int, default=8, help="VLM thread count")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Selecting samples (aligned with B's stratified sampling)...")
    samples = select_samples_aligned_with_b()
    print(f"Selected {len(samples)} samples")

    if args.phase in (0, 1):
        print(f"\n{'='*60}")
        print("PHASE 1: Batch OCR")
        print(f"{'='*60}")
        t1 = time.time()
        run_phase1(samples)
        print(f"Phase 1 completed in {time.time() - t1:.0f}s")

    if args.phase in (0, 2):
        print(f"\n{'='*60}")
        print("PHASE 2: Multi-threaded VLM")
        print(f"{'='*60}")
        t2 = time.time()
        run_phase2(samples, workers=args.workers)
        print(f"Phase 2 completed in {time.time() - t2:.0f}s")


if __name__ == "__main__":
    main()
