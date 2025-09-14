#!/usr/bin/env python3
"""
Clean Evaluation Outputs - Membersihkan hasil evaluasi lama
"""

import os
import shutil
import glob
from datetime import datetime

def clean_output_directory():
    """Clean all output files from previous evaluations"""

    print("="*60)
    print("CLEANING EVALUATION OUTPUT DIRECTORY")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    output_dir = "pengujian/output"

    # Check if output directory exists
    if not os.path.exists(output_dir):
        print(f"📁 Output directory does not exist: {output_dir}")
        print("✅ Nothing to clean")
        return True

    print(f"📁 Checking output directory: {os.path.abspath(output_dir)}")

    # Get all files in output directory
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    png_files = glob.glob(os.path.join(output_dir, "*.png"))
    log_files = glob.glob(os.path.join(output_dir, "*.log"))
    txt_files = glob.glob(os.path.join(output_dir, "*.txt"))

    all_files = json_files + png_files + log_files + txt_files

    if not all_files:
        print("📂 Output directory is already empty")
        print("✅ Nothing to clean")
        return True

    print(f"\n📊 Files found:")
    print(f"  📄 JSON files: {len(json_files)}")
    print(f"  🖼️  PNG files: {len(png_files)}")
    print(f"  📋 Log files: {len(log_files)}")
    print(f"  📝 TXT files: {len(txt_files)}")
    print(f"  📁 Total files: {len(all_files)}")

    # Calculate total size
    total_size = 0
    for file_path in all_files:
        try:
            total_size += os.path.getsize(file_path)
        except:
            pass

    print(f"💾 Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

    # List files to be deleted
    print(f"\n🗑️  Files to be deleted:")
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        try:
            file_size = os.path.getsize(file_path)
            print(f"  🗙 {file_name} ({file_size:,} bytes)")
        except:
            print(f"  🗙 {file_name} (size unknown)")

    # Delete files
    print(f"\n🧹 Cleaning up...")

    deleted_count = 0
    errors = []

    for file_path in all_files:
        try:
            os.remove(file_path)
            deleted_count += 1
            print(f"  ✅ Deleted: {os.path.basename(file_path)}")
        except Exception as e:
            errors.append(f"Failed to delete {os.path.basename(file_path)}: {e}")
            print(f"  ❌ Failed: {os.path.basename(file_path)} - {e}")

    # Summary
    print(f"\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)

    print(f"Files deleted: {deleted_count}/{len(all_files)}")
    print(f"Space freed: ~{total_size/1024/1024:.2f} MB")

    if errors:
        print(f"Errors encountered: {len(errors)}")
        for error in errors:
            print(f"  ⚠️  {error}")
        return False
    else:
        print("✅ All files cleaned successfully")

        # Verify directory is empty
        remaining_files = glob.glob(os.path.join(output_dir, "*"))
        if remaining_files:
            print(f"⚠️  Warning: {len(remaining_files)} files remain in directory")
        else:
            print("✅ Output directory is now clean")

        return True

def backup_outputs():
    """Create backup of current outputs before cleaning"""

    output_dir = "pengujian/output"

    if not os.path.exists(output_dir):
        return True

    # Check if there are files to backup
    files = glob.glob(os.path.join(output_dir, "*"))
    if not files:
        return True

    # Create backup directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f"pengujian/backup_output_{timestamp}"

    try:
        print(f"\n💾 Creating backup: {backup_dir}")
        shutil.copytree(output_dir, backup_dir)
        print(f"✅ Backup created successfully")
        print(f"📁 Backup location: {os.path.abspath(backup_dir)}")
        return True
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return False

def main():
    """Main function with user interaction"""

    print("🧹 EVALUATION OUTPUT CLEANER")
    print("="*60)

    # Check if we should create backup first
    output_dir = "pengujian/output"
    if os.path.exists(output_dir):
        files = glob.glob(os.path.join(output_dir, "*"))
        if files:
            print(f"\n⚠️  Found {len(files)} files in output directory")

            # In automated mode, create backup automatically
            if '--auto' in sys.argv or '--force' in sys.argv:
                print("🔄 Auto mode: Creating backup before cleanup...")
                backup_outputs()
            else:
                response = input("Would you like to create a backup before cleaning? (y/N): ")
                if response.lower() in ['y', 'yes']:
                    backup_outputs()

    # Perform cleanup
    success = clean_output_directory()

    if success:
        print(f"\n🎉 Cleanup completed successfully!")
        print("✅ Ready for fresh evaluation runs")
        print("\nNext steps:")
        print("  python run_all_evaluations.py    # Run full evaluation suite")
        print("  python test1_standalone.py       # Run individual tests")
    else:
        print(f"\n⚠️  Cleanup completed with some errors")
        print("   Check the error messages above")

    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
