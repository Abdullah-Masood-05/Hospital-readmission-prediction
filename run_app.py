#!/usr/bin/env python3
"""
Quick-start script for Streamlit app
Run this to launch the Hospital Readmission Prediction app
"""

import os
import sys
import subprocess


def main():
    """Launch the Streamlit application"""

    # Change to app directory
    app_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("🏥 Hospital Readmission Risk Predictor")
    print("Phase 7: Deployment & Clinical Decision Support")
    print("=" * 70)
    print()

    # Check if data exists
    data_path = os.path.join(app_dir, "..", "data", "preprocessed")
    if not os.path.exists(data_path):
        print("❌ Error: Preprocessed data not found at:", data_path)
        print("Please run notebooks 01-03 first to generate the data.")
        sys.exit(1)

    print("✓ Data verified")
    print("✓ Launching Streamlit app...")
    print()
    print("Opening browser to: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print()
    print("=" * 70)
    print()

    # Launch Streamlit
    app_file = os.path.join(app_dir, "app.py")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app_file,
                "--client.showErrorDetails=true",
            ],
            cwd=app_dir,
        )
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user.")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Run from the project root directory")
        print("3. Check that data files exist")
        sys.exit(1)


if __name__ == "__main__":
    main()
