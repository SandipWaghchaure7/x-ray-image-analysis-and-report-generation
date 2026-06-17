import os
import zipfile
import shutil
from pathlib import Path

def extract_and_setup_dataset(zip_path='archive.zip', extract_to='dataset'):
    """
    Extracts the Kaggle chest X-ray dataset and sets it up properly
    
    Args:
        zip_path: Path to the downloaded zip file
        extract_to: Directory name for extracted dataset
    """
    
    print("=" * 70)
    print("CHEST X-RAY DATASET EXTRACTION & SETUP")
    print("=" * 70)
    
    # Check if zip file exists
    if not os.path.exists(zip_path):
        print(f"\n❌ Error: Could not find '{zip_path}'")
        print("\nPlease ensure the zip file is in the same directory as this script.")
        print("Expected file names:")
        print("  - chest-xray-pneumonia.zip")
        print("  - archive.zip (sometimes Kaggle names it this)")
        return False
    
    print(f"\n✅ Found zip file: {zip_path}")
    print(f"📦 File size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
    
    # Create extraction directory
    if os.path.exists(extract_to):
        print(f"\n⚠️  Directory '{extract_to}' already exists!")
        response = input("Do you want to delete it and re-extract? (yes/no): ").lower()
        if response == 'yes':
            shutil.rmtree(extract_to)
            print(f"✅ Removed existing directory")
        else:
            print("❌ Extraction cancelled")
            return False
    
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract zip file
    print(f"\n📂 Extracting to '{extract_to}/'...")
    print("⏳ This may take 2-5 minutes...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total number of files
            total_files = len(zip_ref.namelist())
            print(f"📄 Total files to extract: {total_files}")
            
            # Extract with progress
            for i, file in enumerate(zip_ref.namelist(), 1):
                zip_ref.extract(file, extract_to)
                if i % 500 == 0:
                    print(f"   Extracted {i}/{total_files} files...")
            
            print(f"✅ Extracted all {total_files} files!")
    
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return False
    
    # Find the actual dataset directory
    print("\n🔍 Locating dataset directories...")
    
    # The Kaggle dataset usually extracts to: chest_xray/
    possible_paths = [
        os.path.join(extract_to, 'chest_xray'),
        os.path.join(extract_to, 'chest-xray'),
        os.path.join(extract_to, 'chest_xray_pneumonia'),
        extract_to
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'train')):
            dataset_path = path
            break
    
    if not dataset_path:
        print("❌ Could not find train/ directory in extracted files")
        print(f"📁 Contents of '{extract_to}':")
        for item in os.listdir(extract_to):
            print(f"   - {item}")
        return False
    
    print(f"✅ Found dataset at: {dataset_path}")
    
    # Move to chest_xray/ for consistency
    if dataset_path != 'chest_xray':
        if os.path.exists('chest_xray'):
            shutil.rmtree('chest_xray')
        shutil.move(dataset_path, 'chest_xray')
        print("✅ Moved dataset to 'chest_xray/' directory")
    
    # Verify structure and count files
    print("\n📊 Dataset Structure & Statistics:")
    print("=" * 70)
    
    base_path = 'chest_xray'
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            print(f"❌ Missing '{split}' directory!")
            continue
        
        print(f"\n📁 {split.upper()}/")
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path) 
                        if f.endswith(('.jpeg', '.jpg', '.png'))]
                count = len(files)
                total_images += count
                print(f"   ├── {class_name:12s}: {count:4d} images")
            else:
                print(f"   ├── {class_name:12s}: ❌ Missing!")
    
    print(f"\n{'='*70}")
    print(f"📊 TOTAL IMAGES: {total_images}")
    print(f"{'='*70}")
    
    # Show expected structure
    print("\n✅ Dataset is ready for training!")
    print("\n📂 Final Structure:")
    print("""
    chest_xray/
    ├── train/
    │   ├── NORMAL/       (1,341 images)
    │   └── PNEUMONIA/    (3,875 images)
    ├── val/
    │   ├── NORMAL/       (8 images)
    │   └── PNEUMONIA/    (8 images)
    └── test/
        ├── NORMAL/       (234 images)
        └── PNEUMONIA/    (390 images)
    """)
    
    # Check for common issues
    print("\n🔍 Checking for potential issues...")
    
    issues_found = False
    
    # Check validation set size
    val_normal = len([f for f in os.listdir('chest_xray/val/NORMAL') 
                     if f.endswith(('.jpeg', '.jpg', '.png'))])
    val_pneumonia = len([f for f in os.listdir('chest_xray/val/PNEUMONIA') 
                        if f.endswith(('.jpeg', '.jpg', '.png'))])
    
    if val_normal + val_pneumonia < 50:
        print("⚠️  WARNING: Validation set is very small (only 16 images)")
        print("   Consider creating a better split using the code in DATASET_SETUP.md")
        issues_found = True
    
    # Check class imbalance
    train_normal = len([f for f in os.listdir('chest_xray/train/NORMAL') 
                       if f.endswith(('.jpeg', '.jpg', '.png'))])
    train_pneumonia = len([f for f in os.listdir('chest_xray/train/PNEUMONIA') 
                          if f.endswith(('.jpeg', '.jpg', '.png'))])
    
    ratio = train_pneumonia / train_normal if train_normal > 0 else 0
    if ratio > 2.5:
        print(f"⚠️  WARNING: Class imbalance detected (ratio: {ratio:.2f}:1)")
        print("   Don't worry - the training scripts handle this with class weights!")
        issues_found = True
    
    if not issues_found:
        print("✅ No issues detected!")
    
    # Cleanup temporary extraction folder if different
    if extract_to != 'chest_xray' and os.path.exists(extract_to):
        try:
            shutil.rmtree(extract_to)
            print(f"\n🧹 Cleaned up temporary directory '{extract_to}'")
        except:
            pass
    
    print("\n" + "=" * 70)
    print("🎉 SETUP COMPLETE!")
    print("=" * 70)
    print("\n📝 Next Steps:")
    print("1. Run training script:")
    print("   python train_transfer_learning.py")
    print("\n2. Or for custom CNN:")
    print("   python train_model.py")
    print("\n3. Wait for training to complete (20-30 minutes)")
    print("4. Your model will be saved as 'x-ray-classification.h5'")
    print("5. Use it in your Flask app!")
    
    return True


if __name__ == "__main__":
    # Try common zip file names
    zip_names = [
        'chest-xray-pneumonia.zip',
        'archive.zip',
        'chest_xray.zip'
    ]
    
    zip_file = None
    for name in zip_names:
        if os.path.exists(name):
            zip_file = name
            break
    
    if zip_file:
        extract_and_setup_dataset(zip_file)
    else:
        print("❌ Could not find zip file!")
        print("\nPlease ensure one of these files is in the current directory:")
        for name in zip_names:
            print(f"  - {name}")
        print("\nCurrent directory contents:")
        for item in os.listdir('.'):
            if item.endswith('.zip'):
                print(f"  📦 {item}")