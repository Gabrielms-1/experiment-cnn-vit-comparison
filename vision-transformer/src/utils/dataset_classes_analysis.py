import glob
import os
from collections import Counter
from typing import Dict

def count_classes_in_split(split_path: str) -> Dict[str, int]:
    class_counts = Counter()
    
    for file in glob.glob(os.path.join(split_path, "*/*.jpg"), recursive=True):
        class_name = os.path.basename(os.path.dirname(file))
        class_counts[class_name] += 1
        
    return dict(class_counts)

def analyze_dataset_balance(dataset_path: str) -> Dict[str, Dict[str, int]]:
    splits = ['train', 'test', 'valid']
    dataset_stats = {}
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            dataset_stats[split] = count_classes_in_split(split_path)
    
    return dataset_stats

def print_dataset_analysis(stats: Dict[str, Dict[str, int]]):
    for split, class_counts in stats.items():
        total_images = sum(class_counts.values())
        print(f"\n{split.upper()} Split:")
        print(f"Total images: {total_images}")
        print("Class distribution:")
        
        for class_name, count in class_counts.items():
            percentage = (count / total_images) * 100
            print(f"- {class_name}: {count} images ({percentage:.2f}%)")

if __name__ == "__main__":
    dataset_path = "data"
    stats = analyze_dataset_balance(dataset_path)
    print_dataset_analysis(stats)