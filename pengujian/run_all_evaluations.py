#!/usr/bin/env python3
"""
Master Test Runner - Menjalankan semua test evaluasi chatbot secara berurutan
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")

    try:
        start_time = time.time()
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {description} completed successfully in {duration:.2f}s")
            if result.stdout:
                print("\nOutput highlights:")
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:  # Show last 10 lines
                    if any(keyword in line.lower() for keyword in ['test', 'score', 'accuracy', 'completed', '✅', '❌']):
                        print(f"  {line}")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    """Run all evaluation tests"""
    print("="*80)
    print("CHATBOT SYSTEM COMPREHENSIVE EVALUATION SUITE".center(80))
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    # Create output directory
    output_dir = "pengujian/output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    total_start_time = time.time()
    tests_passed = 0
    tests_total = 3

    # Test 1: Two Questions Evaluation
    print("\n🔍 STARTING TEST 1: Two Questions Evaluation")
    if run_command("python test1_standalone.py", "Test 1: Two Questions Evaluation"):
        tests_passed += 1

    # Wait between tests
    time.sleep(3)

    # Test 2: Session Interactions Evaluation
    print("\n📊 STARTING TEST 2: Session Interactions Evaluation")
    if run_command("python test2_standalone.py", "Test 2: Session Interactions Evaluation"):
        tests_passed += 1

    # Wait between tests
    time.sleep(3)

    # Test 3: BERT Model Accuracy Evaluation
    print("\n🤖 STARTING TEST 3: BERT Model Accuracy Evaluation")
    if run_command("python test3_standalone.py", "Test 3: BERT Model Accuracy Evaluation"):
        tests_passed += 1

    # Generate final summary
    total_duration = time.time() - total_start_time

    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY".center(80))
    print("="*80)

    print(f"Total Execution Time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tests Passed: {tests_passed}/{tests_total}")

    # Show success rate
    success_rate = (tests_passed / tests_total) * 100
    if success_rate == 100:
        status = "🎉 ALL TESTS PASSED"
        color = "SUCCESS"
    elif success_rate >= 66:
        status = "⚡ MOSTLY SUCCESSFUL"
        color = "WARNING"
    else:
        status = "❌ NEEDS ATTENTION"
        color = "ERROR"

    print(f"Success Rate: {success_rate:.1f}% - {status}")

    # List output files
    print(f"\n📁 Generated Output Files:")
    if os.path.exists(output_dir):
        files = sorted(os.listdir(output_dir))
        if files:
            for file in files:
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  📄 {file} ({file_size:,} bytes)")
        else:
            print("  No output files found")
    else:
        print("  Output directory not found")

    # Recommendations
    print(f"\n💡 Recommendations:")
    if tests_passed == tests_total:
        print("  ✅ Excellent! All evaluation tests passed successfully.")
        print("  ✅ Check the output files for detailed analysis.")
        print("  ✅ Review the visualizations for insights.")
    else:
        print("  ⚠️  Some tests failed. Check the error messages above.")
        print("  ⚠️  Ensure all dependencies are installed correctly.")
        print("  ⚠️  Verify the chatbot system is working properly.")

    print("\n" + "="*80)
    print("EVALUATION SUITE COMPLETED")
    print("="*80)

    return tests_passed == tests_total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
