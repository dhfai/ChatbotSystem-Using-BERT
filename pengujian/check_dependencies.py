#!/usr/bin/env python3
"""
Dependency Checker - Memvalidasi semua dependencies yang diperlukan untuk evaluasi
"""

import sys
import subprocess
import importlib
from packaging import version

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    current_version = sys.version_info
    required_version = (3, 8)

    if current_version >= required_version:
        print(f"✅ Python {current_version.major}.{current_version.minor}.{current_version.micro} (minimum 3.8 required)")
        return True
    else:
        print(f"❌ Python {current_version.major}.{current_version.minor} found, but 3.8+ required")
        return False

def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally check version"""
    if import_name is None:
        import_name = package_name

    try:
        # Try to import the package
        module = importlib.import_module(import_name)

        # Check version if specified
        if min_version:
            try:
                if hasattr(module, '__version__'):
                    current_version = module.__version__
                elif package_name == 'torch' and hasattr(module, '__version__'):
                    current_version = module.__version__
                else:
                    # Try to get version from pip
                    result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if line.startswith('Version:'):
                                current_version = line.split(':', 1)[1].strip()
                                break
                    else:
                        current_version = "unknown"

                if current_version != "unknown":
                    if version.parse(current_version) >= version.parse(min_version):
                        print(f"✅ {package_name} {current_version} (minimum {min_version})")
                        return True
                    else:
                        print(f"❌ {package_name} {current_version} found, but {min_version}+ required")
                        return False
                else:
                    print(f"⚠️  {package_name} installed but version unknown")
                    return True
            except Exception as e:
                print(f"⚠️  {package_name} installed but version check failed: {e}")
                return True
        else:
            print(f"✅ {package_name} installed")
            return True

    except ImportError:
        print(f"❌ {package_name} not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking {package_name}: {e}")
        return False

def check_all_dependencies():
    """Check all required dependencies"""
    print("="*60)
    print("DEPENDENCY VALIDATION FOR CHATBOT EVALUATION SYSTEM")
    print("="*60)

    # Check Python version first
    python_ok = check_python_version()

    print("\n📦 Checking Core Dependencies...")

    # Core dependencies
    core_deps = [
        ("torch", "torch", "1.9.0"),
        ("transformers", "transformers", "4.20.0"),
        ("faiss-cpu", "faiss", None),
        ("pandas", "pandas", "1.3.0"),
        ("numpy", "numpy", "1.21.0"),
    ]

    core_ok = True
    for pkg_name, import_name, min_ver in core_deps:
        if not check_package(pkg_name, import_name, min_ver):
            core_ok = False

    print("\n📊 Checking Evaluation Dependencies...")

    # Evaluation dependencies
    eval_deps = [
        ("scikit-learn", "sklearn", "1.0.0"),
        ("matplotlib", "matplotlib", "3.3.0"),
        ("seaborn", "seaborn", "0.11.0"),
        ("nltk", "nltk", "3.6.0"),
    ]

    eval_ok = True
    for pkg_name, import_name, min_ver in eval_deps:
        if not check_package(pkg_name, import_name, min_ver):
            eval_ok = False

    print("\n📝 Checking NLP Metrics Dependencies...")

    # NLP metrics dependencies
    nlp_deps = [
        ("bert-score", "bert_score", None),
        ("rouge-score", "rouge_score", None),
    ]

    nlp_ok = True
    for pkg_name, import_name, min_ver in nlp_deps:
        if not check_package(pkg_name, import_name, min_ver):
            nlp_ok = False

    print("\n🔧 Checking Optional Dependencies...")

    # Optional dependencies
    optional_deps = [
        ("jupyter", "jupyter", None),
        ("ipython", "IPython", None),
    ]

    optional_count = 0
    for pkg_name, import_name, min_ver in optional_deps:
        if check_package(pkg_name, import_name, min_ver):
            optional_count += 1

    # Summary
    print("\n" + "="*60)
    print("DEPENDENCY CHECK SUMMARY")
    print("="*60)

    total_checks = 4  # python, core, eval, nlp
    passed_checks = sum([python_ok, core_ok, eval_ok, nlp_ok])

    print(f"Python Version: {'✅ OK' if python_ok else '❌ FAIL'}")
    print(f"Core Dependencies: {'✅ OK' if core_ok else '❌ FAIL'}")
    print(f"Evaluation Dependencies: {'✅ OK' if eval_ok else '❌ FAIL'}")
    print(f"NLP Metrics Dependencies: {'✅ OK' if nlp_ok else '❌ FAIL'}")
    print(f"Optional Dependencies: {optional_count}/{len(optional_deps)} available")

    overall_status = passed_checks == total_checks

    print(f"\nOverall Status: {'🎉 ALL DEPENDENCIES OK' if overall_status else '⚠️ SOME DEPENDENCIES MISSING'}")

    # Installation instructions if needed
    if not overall_status:
        print("\n" + "="*60)
        print("INSTALLATION INSTRUCTIONS")
        print("="*60)

        print("\n📋 To install missing dependencies, run:")
        print("```bash")

        if not core_ok:
            print("# Core dependencies")
            print("pip install torch transformers faiss-cpu pandas numpy")

        if not eval_ok:
            print("# Evaluation dependencies")
            print("pip install scikit-learn matplotlib seaborn nltk")

        if not nlp_ok:
            print("# NLP metrics dependencies")
            print("pip install bert-score rouge-score")

        print("```")

        print("\n📋 Or install all at once:")
        print("```bash")
        print("pip install -r requirements.txt")
        print("pip install bert-score rouge-score scikit-learn matplotlib seaborn nltk")
        print("```")

        print("\n⚠️ Note: Some packages may require additional setup:")
        print("- NLTK may need data downloads: python -c \"import nltk; nltk.download('punkt')\"")
        print("- BERT-Score downloads models on first use")
        print("- FAISS may need different builds for different systems")

    else:
        print("\n🎉 All dependencies are satisfied! You can run the evaluation suite.")
        print("\nNext steps:")
        print("1. Run complete evaluation: python run_all_evaluations.py")
        print("2. Run individual tests: python test1_standalone.py")
        print("3. View results: python view_evaluation_results.py")

    return overall_status

if __name__ == "__main__":
    success = check_all_dependencies()
    sys.exit(0 if success else 1)
