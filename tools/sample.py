import shutil
from pathlib import Path

# 소스와 타겟 경로
src_base = Path('D:/workplace/portfolio_plc/project_vision/processed_data/test/images')
dst_base = Path('D:/workplace/portfolio_plc/app/assets/pcb_samples')

classes = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

print("=== PCB 샘플 이미지 복사 ===\n")

for cls in classes:
    src_dir = src_base / cls
    dst_dir = dst_base / cls
    
    # 이미지 찾기
    images = list(src_dir.glob('*.jpg'))
    
    print(f"[{cls}]")
    print(f"  발견: {len(images)}장")
    
    # 5장 복사
    for i, img in enumerate(images[:5], 1):
        dst_file = dst_dir / f'sample_{i:02d}.jpg'
        shutil.copy(img, dst_file)
        print(f"  ✓ {img.name} → {dst_file.name}")
    
    print()
