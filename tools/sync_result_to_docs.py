import json
import pickle
import re
import shutil
from pathlib import Path
import nbformat

BASE = Path(__file__).resolve().parents[1]

PROJECTS = {
    "project_defect": {
        "readme": BASE / "project_defect" / "README.md",
        "models_dir": BASE / "project_defect" / "models",
        "card_marker": "제조 공정 품질 이상 감지",
        "prefer": ["Random Forest", "XGBoost", "Logistic"]
    },
    "project_failure": {
        "readme": BASE / "project_failure" / "README.md",
        "models_dir": BASE / "project_failure" / "models",
        "card_marker": "설비 이상 감지",
        "prefer": ["GRU", "CNN", "LSTM"]
    }
}

def load_results_from_models(models_dir: Path):
    results = {}
    # 1) summary json
    summary = models_dir / "results_summary.json"
    if summary.exists():
        try:
            arr = json.loads(summary.read_text(encoding="utf-8"))
            for r in arr:
                name = (r.get("model") or r.get("model_name") or r.get("name") or r.get("Model"))
                if name:
                    results[str(name)] = r
            return results
        except Exception:
            pass
    # 2) any results_*.json
    for p in models_dir.glob("results_*.json"):
        try:
            arr = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(arr, list):
                for r in arr:
                    name = r.get("model") or r.get("model_name")
                    if name:
                        results[str(name)] = r
            elif isinstance(arr, dict):
                name = arr.get("model") or arr.get("model_name")
                if name:
                    results[str(name)] = arr
        except Exception:
            continue
    # 3) pickle files
    for p in models_dir.glob("*.pkl"):
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                name = data.get("model") or data.get("model_name") or data.get("Model") or p.stem
                results[str(name)] = data
        except Exception:
            continue
    return results

def pct(v):
    try:
        fv = float(v)
        # already 0..1
        if 0 <= fv <= 1:
            return f"{fv*100:.2f}%"
        # already percent (0-100)
        if fv > 1 and fv <= 100:
            return f"{fv:.2f}%"
        return str(v)
    except Exception:
        return str(v) if v is not None else "N/A"

def update_readme(readme_path: Path, models_order, results):
    if not readme_path.exists():
        print("  README missing:", readme_path)
        return
    bak = readme_path.with_suffix(".md.bak")
    shutil.copy2(readme_path, bak)
    txt = readme_path.read_text(encoding="utf-8")

    # 찾아넣기: '## 주요 성과' 뒤에 첫 번째 헤더(## ... ) 전까지 영역을 대체(없으면 파일 끝까지)
    m_start = re.search(r"^##\s+주요 성과\s*$", txt, flags=re.MULTILINE)
    if not m_start:
        print("  README: '## 주요 성과' section not found, skipping")
        return
    start_idx = m_start.end()
    # find next top-level header '## ' after start_idx
    m_next = re.search(r"^##\s+", txt[start_idx:], flags=re.MULTILINE)
    end_idx = start_idx + m_next.start() if m_next else len(txt)

    # build table
    header = "| 지표 | " + " | ".join(models_order) + " |\n"
    header += "|------" + "|------" * len(models_order) + "|\n"
    rows = []
    for metric_key, label in [("accuracy","Accuracy"), ("precision","Precision"), ("recall","Recall")]:
        row = f"| **{label}** "
        for mn in models_order:
            r = results.get(mn, {})
            # support different key casings
            val = r.get(metric_key) if isinstance(r, dict) else None
            if val is None:
                val = r.get(label) or r.get(label.lower()) if isinstance(r, dict) else None
            row += "| " + (pct(val) if val is not None else "N/A") + " "
        row += "|\n"
        rows.append(row)
    table = header + "".join(rows)

    new_txt = txt[:start_idx] + "\n\n" + table + "\n" + txt[end_idx:]
    readme_path.write_text(new_txt, encoding="utf-8")
    print("  README updated:", readme_path)

def update_notebook_conclusion(nb_path: Path, results, models_order):
    if not nb_path.exists():
        print("  Notebook not found:", nb_path)
        return
    bak = nb_path.with_suffix(".ipynb.bak")
    shutil.copy2(nb_path, bak)
    nb = nbformat.read(nb_path, as_version=4)
    replaced = False
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.strip().startswith("# 최종 결론"):
            # build inserted summary block
            # pick representative model by prefer order
            rep = None
            for mn in models_order:
                if mn in results:
                    rep = results[mn]; break
            if rep is None and results:
                rep = next(iter(results.values()))
            summary_lines = []
            summary_lines.append("\n\n---\n")
            summary_lines.append("### 자동 갱신 - 성능 요약\n\n")
            if rep is None:
                summary_lines.append("- 결과 파일이 없습니다.\n")
            else:
                name = rep.get("model") or rep.get("model_name") or "선택 모델"
                summary_lines.append(f"- 선택된 모델: **{name}**\n")
                summary_lines.append(f"- Accuracy: {pct(rep.get('accuracy'))}\n")
                summary_lines.append(f"- Precision: {pct(rep.get('precision'))}\n")
                summary_lines.append(f"- Recall: {pct(rep.get('recall'))}\n")
                notes = rep.get("notes") or rep.get("note") or ""
                if notes:
                    summary_lines.append(f"- 메모: {notes}\n")
            # append summary to existing cell (do not delete existing content)
            cell.source = cell.source.rstrip() + "".join(summary_lines)
            replaced = True
            break
    if replaced:
        nbformat.write(nb, nb_path)
        print("  Notebook conclusion updated:", nb_path)
    else:
        print("  No '# 최종 결론' markdown cell found in", nb_path)

def update_home_and_pages(card_marker, results, prefer):
    # helper to produce badge string
    def badge_text(prefer_list):
        rep = None
        for p in prefer_list:
            if p in results:
                rep = results[p]; break
        if rep is None and results:
            rep = next(iter(results.values()))
        if rep is None:
            return "N/A Accuracy · Recall N/A"
        return f"{pct(rep.get('accuracy'))} Accuracy · Recall {pct(rep.get('recall'))}"

    # Home.py
    home = BASE / "app" / "Home.py"
    if home.exists():
        s = home.read_text(encoding="utf-8")
        bak = home.with_suffix(".py.bak")
        shutil.copy2(home, bak)
        # replace inner text of performance-badge for matching card header robustly
        pattern = (
            rf"(<div\s+class=[\"']card-header[\"']\s*>{re.escape(card_marker)}</div>[\s\S]*?"
            rf"<div\s+class=[\"']performance-badge[\"']\s*>)([\s\S]*?)(</div>)"
        )
        new_s, n = re.subn(pattern, lambda m: m.group(1) + badge_text(prefer) + m.group(3), s, flags=re.MULTILINE)
        if n:
            home.write_text(new_s, encoding="utf-8")
            print("  app/Home.py updated")
        else:
            print("  app/Home.py: pattern not found or unchanged")
    # pages
    pages_dir = BASE / "app" / "pages"
    if pages_dir.exists():
        for p in pages_dir.glob("*.py"):
            txt = p.read_text(encoding="utf-8")
            if card_marker in txt:
                bak = p.with_suffix(".py.bak")
                shutil.copy2(p, bak)
                new_txt, n = re.subn(pattern, lambda m: m.group(1) + badge_text(prefer) + m.group(3), txt, flags=re.MULTILINE)
                if n:
                    p.write_text(new_txt, encoding="utf-8")
                    print("  app/pages updated:", p.name)

def main():
    for proj, cfg in PROJECTS.items():
        print(f"\n=== Syncing project: {proj} ===")
        models_dir = cfg["models_dir"]
        results = load_results_from_models(models_dir)
        if not results:
            print("  No results found in", models_dir)
            continue
        # normalize keys
        normalized = {}
        for k, v in results.items():
            kk = str(k)
            if "rf" in kk.lower() or "random" in kk.lower():
                nk = "Random Forest"
            elif "xgb" in kk.lower() or "xgboost" in kk.lower():
                nk = "XGBoost"
            elif "logistic" in kk.lower():
                nk = "Logistic"
            elif "lstm" in kk.lower():
                nk = "LSTM"
            elif "gru" in kk.lower():
                nk = "GRU"
            elif "cnn" in kk.lower():
                nk = "CNN"
            else:
                nk = kk
            normalized[nk] = v
        update_readme(cfg["readme"], cfg["prefer"], normalized)
        nb = BASE / proj / "researching" / "06_model_comparison.ipynb"
        update_notebook_conclusion(nb, normalized, cfg["prefer"])
        update_home_and_pages(cfg["card_marker"], normalized, cfg["prefer"])
    print("\nSync complete.")

if __name__ == "__main__":
    main()