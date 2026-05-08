"""
Verification script to ensure all dependencies and data files are ready
before launching the Streamlit app
"""

import os
import sys
import importlib


def check_module(module_name, package_name=None):
    """Check if a module is installed"""
    if package_name is None:
        package_name = module_name

    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {package_name:<20} {version}")
        return True
    except ImportError:
        print(f"✗ {package_name:<20} NOT INSTALLED")
        return False


def check_file(path, description):
    """Check if a file exists"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        size_mb = size / (1024 * 1024)
        print(f"✓ {description:<40} {size_mb:.1f} MB")
        return True
    else:
        print(f"✗ {description:<40} NOT FOUND")
        return False


def check_directory(path, description):
    """Check if a directory exists"""
    if os.path.isdir(path):
        file_count = len(
            [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        )
        print(f"✓ {description:<40} ({file_count} files)")
        return True
    else:
        print(f"✗ {description:<40} NOT FOUND")
        return False


def main():
    """Run all verification checks"""

    print("=" * 70)
    print("🏥 Hospital Readmission Predictor - Deployment Verification")
    print("=" * 70)
    print()

    # Change to project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    all_ok = True

    # ====================================================================
    # 1. Check Python Packages
    # ====================================================================
    print("📦 Checking Python Dependencies...")
    print("-" * 70)

    required_packages = [
        ("streamlit", "Streamlit"),
        ("shap", "SHAP"),
        ("pandas", "pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
    ]

    for module_name, package_name in required_packages:
        if not check_module(module_name, package_name):
            all_ok = False

    print()

    # ====================================================================
    # 2. Check Data Files
    # ====================================================================
    print("📊 Checking Data Files...")
    print("-" * 70)

    data_files = [
        ("data/preprocessed/X_train.csv", "Training data (X_train)"),
        ("data/preprocessed/X_test.csv", "Test data (X_test)"),
        ("data/preprocessed/y_train.csv", "Training labels (y_train)"),
        ("data/preprocessed/y_test.csv", "Test labels (y_test)"),
        ("results/fairness_metrics_by_group.csv", "Fairness metrics"),
    ]

    for file_path, description in data_files:
        if not check_file(file_path, description):
            all_ok = False

    print()

    # ====================================================================
    # 3. Check App Files
    # ====================================================================
    print("🎨 Checking App Files...")
    print("-" * 70)

    app_files = [
        ("app/app.py", "Main Streamlit application"),
        ("app/utils.py", "Utility functions"),
        ("app/README.md", "App documentation"),
    ]

    for file_path, description in app_files:
        if not check_file(file_path, description):
            all_ok = False

    print()

    # ====================================================================
    # 4. Check Project Structure
    # ====================================================================
    print("📁 Checking Project Structure...")
    print("-" * 70)

    directories = [
        ("notebooks", "Notebooks directory"),
        ("data/preprocessed", "Preprocessed data directory"),
        ("results", "Results directory"),
        ("app", "App directory"),
        ("src", "Source code directory"),
    ]

    for dir_path, description in directories:
        if not check_directory(dir_path, description):
            all_ok = False

    print()

    # ====================================================================
    # Summary
    # ====================================================================
    print("=" * 70)

    if all_ok:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("You are ready to launch the Streamlit app:")
        print()
        print("  streamlit run app/app.py")
        print()
        print("Then open your browser to: http://localhost:8501")
        return 0
    else:
        print("✗ SOME CHECKS FAILED!")
        print()
        print("Please fix the issues above before running the app.")
        print()
        print("Common fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Run preprocessing notebooks: 01_EDA, 02_preprocessing, 03_modeling")
        print("3. Run fairness audit: 05_fairness.ipynb")
        return 1


if __name__ == "__main__":
    sys.exit(main())
