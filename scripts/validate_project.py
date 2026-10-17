"""Automated project validation and QA test runner for Vision Sentinel."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, desc: str) -> bool:
    print(f"\n[RUNNING] {desc}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[PASS] {desc}")
        return True
    else:
        print(f"[FAIL] {desc}\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}")
        return False


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    python_bin = sys.executable

    print("=" * 65)
    print("  VISION SENTINEL — COMPREHENSIVE PROJECT VALIDATION")
    print("=" * 65)

    checks = [
        (
            [python_bin, str(project_root / "scripts" / "check_environment.py")],
            "Environment & OpenCV Diagnostic Check",
        ),
        ([python_bin, "-m", "ruff", "check", "."], "Ruff Code Formatting & Lint Check"),
        ([python_bin, "-m", "pytest", "-v", "--tb=short"], "Pytest Unit & Integration Test Suite"),
        (
            [python_bin, str(project_root / "benchmarks" / "run_benchmark.py"), "--runs", "1"],
            "Performance Benchmark Run",
        ),
        (
            [python_bin, str(project_root / "scripts" / "generate_evaluation_report.py")],
            "Evaluation Report Generation",
        ),
    ]

    results = []
    for cmd, desc in checks:
        ok = run_command(cmd, desc)
        results.append((desc, ok))

    print("\n" + "=" * 65)
    print("  FINAL VALIDATION VERDICT MATRIX")
    print("=" * 65)
    all_passed = True
    for desc, ok in results:
        status_text = "PASS" if ok else "FAIL"
        print(f"  {desc.ljust(48)}: [{status_text}]")
        if not ok:
            all_passed = False

    print("=" * 65)
    print(f"  OVERALL RESULT: {'ALL VERIFICATIONS PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 65)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
