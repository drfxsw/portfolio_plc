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

    # 1. 프로젝트 요약 테이블의 결과 열 업데이트
    def update_project_summary():
        nonlocal txt
        def safe_replace(pattern, value, text):
            """정규표현식 그룹 참조 오류를 방지하는 안전한 치환 함수"""
            import re
            def replacer(match):
                return match.group(1) + value + match.group(2)
            return re.sub(pattern, replacer, text)
        
        for model_name, result in results.items():
            if "Random Forest" in model_name or "rf" in model_name.lower():
                recall_val = pct(result.get('recall') or result.get('test_recall'))
                pattern = r'(\| 04_random_forest.*?\| Recall )[^|]+(\|)'
                txt = safe_replace(pattern, recall_val, txt)
            elif "XGBoost" in model_name or "xgb" in model_name.lower():
                recall_val = pct(result.get('recall') or result.get('test_recall'))
                pattern = r'(\| 05_xgboost.*?\| Recall )[^|]+(\|)'
                txt = safe_replace(pattern, recall_val, txt)
            elif "Logistic" in model_name or "logistic" in model_name.lower():
                recall_val = pct(result.get('recall') or result.get('test_recall'))
                pattern = r'(\| 03_logistic.*?\| Recall )[^|]+(\|)'
                txt = safe_replace(pattern, recall_val, txt)

    # 2. 주요 성과 테이블만 업데이트 (핵심 성과는 보존)
    m_start = re.search(r"^##\s+주요 성과\s*$", txt, flags=re.MULTILINE)
    if not m_start:
        print("  README: '## 주요 성과' section not found, skipping")
        return
    
    start_idx = m_start.end()
    
    # 기존 테이블만 찾아서 교체 (핵심 성과나 다른 내용은 보존)
    table_pattern = r"\n\n\|[^#]*?(?=\n\n###|\n##|\Z)"
    m_table = re.search(table_pattern, txt[start_idx:], flags=re.DOTALL)
    
    if m_table:
        table_start = start_idx + m_table.start()
        table_end = start_idx + m_table.end()
        
        # build new table
        header = "| 지표 | " + " | ".join(models_order) + " |\n"
        header += "|------" + "|------" * len(models_order) + "|\n"
        rows = []
        for metric_key, label in [("accuracy","Accuracy"), ("precision","Precision"), ("recall","Recall")]:
            row = f"| **{label}** "
            for mn in models_order:
                r = results.get(mn, {})
                val = r.get(metric_key) or r.get(f'test_{metric_key}') if isinstance(r, dict) else None
                if val is None:
                    val = r.get(label) or r.get(label.lower()) if isinstance(r, dict) else None
                row += "| " + (pct(val) if val is not None else "N/A") + " "
            row += "|\n"
            rows.append(row)
        new_table = "\n\n" + header + "".join(rows)
        
        txt = txt[:table_start] + new_table + txt[table_end:]
    else:
        print("  README: Performance table not found, skipping table update")

    # 프로젝트 요약도 업데이트
    update_project_summary()
    
    readme_path.write_text(txt, encoding="utf-8")
    print("  README updated:", readme_path)

def update_notebook_conclusion(nb_path: Path, results, models_order):
    if not nb_path.exists():
        print("  Notebook not found:", nb_path)
        return
    bak = nb_path.with_suffix(".ipynb.bak")
    shutil.copy2(nb_path, bak)
    nb = nbformat.read(nb_path, as_version=4)
    updated = False
    
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            original_content = cell.source
            updated_content = original_content
            
            # 각 모델의 성능 지표를 실제 값으로 업데이트
            for model_name, result in results.items():
                if isinstance(result, dict):
                    accuracy = result.get('accuracy') or result.get('test_accuracy')
                    precision = result.get('precision') or result.get('test_precision') 
                    recall = result.get('recall') or result.get('test_recall')
                    
                    def safe_replace_nb(pattern, value, text):
                        """노트북용 안전한 치환 함수"""
                        def replacer(match):
                            return match.group(1) + value
                        return re.sub(pattern, replacer, text, flags=re.IGNORECASE)
                    
                    if accuracy:
                        # 다양한 패턴으로 정확도 업데이트
                        patterns = [
                            rf"({model_name}.*?정확도[:\s]*)[0-9]+\.[0-9]+%",
                            rf"({model_name}.*?Accuracy[:\s]*)[0-9]+\.[0-9]+%",
                            rf"(Test Accuracy[:\s]*)[0-9]+\.[0-9]+%"
                        ]
                        acc_val = pct(accuracy)
                        for pattern in patterns:
                            updated_content = safe_replace_nb(pattern, acc_val, updated_content)
                    
                    if precision:
                        patterns = [
                            rf"({model_name}.*?정밀도[:\s]*)[0-9]+\.[0-9]+%",
                            rf"({model_name}.*?Precision[:\s]*)[0-9]+\.[0-9]+%",
                            rf"(Test Precision[:\s]*)[0-9]+\.[0-9]+%"
                        ]
                        prec_val = pct(precision)
                        for pattern in patterns:
                            updated_content = safe_replace_nb(pattern, prec_val, updated_content)
                    
                    if recall:
                        patterns = [
                            rf"({model_name}.*?재현율[:\s]*)[0-9]+\.[0-9]+%",
                            rf"({model_name}.*?Recall[:\s]*)[0-9]+\.[0-9]+%", 
                            rf"(Test Recall[:\s]*)[0-9]+\.[0-9]+%"
                        ]
                        recall_val = pct(recall)
                        for pattern in patterns:
                            updated_content = safe_replace_nb(pattern, recall_val, updated_content)
            
            # 내용이 변경되었으면 업데이트
            if updated_content != original_content:
                cell.source = updated_content
                updated = True
    
    if updated:
        nbformat.write(nb, nb_path)
        print("  Notebook metrics updated:", nb_path)
    else:
        print("  No metrics found to update in", nb_path)

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
        
        # README.md 업데이트
        update_readme(cfg["readme"], cfg["prefer"], normalized)
        
        # 06_model_comparison.ipynb 노트북 업데이트
        nb = BASE / proj / "researching" / "06_model_comparison.ipynb"
        update_notebook_conclusion(nb, normalized, cfg["prefer"])
        
        print(f"  {proj}: README.md and 06_model_comparison.ipynb updated")
    
    print("\nSync complete - README and notebooks only.")
    print("Note: Home.py and pages/ now use dynamic loading from pkl files.")

if __name__ == "__main__":
    main()