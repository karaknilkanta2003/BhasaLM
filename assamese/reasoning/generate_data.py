"""
Assamese Synthetic Reasoning Dataset Generator (Phase 3)
========================================================
Generates programmatic comparative and basic reasoning tasks with:
1. Transitive Ordering (A > B and B > C => Extrema)
2. Direct Quantitative Comparisons (Ages, Heights, Quantities, Prices)
3. Multi-Hop Inequality Verification (Yes / No)

Enforces 100% Zero Train-Test Leakage via strictly disjoint entity name sets.
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple

# ==============================================================================
# Disjoint Name Pools (Prevents Train-Test Leakage)
# ==============================================================================

TRAIN_NAMES = [
    "ৰাহুল", "ৰমেন", "বিকাশ", "প্ৰণৱ", "ভাস্কৰ", "মানস", "দীপাংকৰ", "মৃদুল", "অনুপম", "হেমন্ত",
    "কল্যাণ", "পংকজ", "ধ্ৰুৱ", "নয়ন", "জয়ন্ত", "হিৰণ্য", "উৎপল", "অৰূপ", "বিপুল", "সঞ্জীৱ",
    "ৰীতা", "গীতা", "ৰশ্মি", "জোনালী", "ৰূপালী", "মিতালী", "দীপালী", "বৰ্ণালী", "পল্লৱী", "মামনি",
    "পূবালী", "অঞ্জলি", "কবিতা", "লখিমী", "ময়ূৰী", "প্ৰীতি", "মৃদুস্মিতা", "হিমানী", "চন্দনা", "সংগীতা"
]

VAL_NAMES = [
    "দেৱজিত", "অভিজিত", "পাৰ্থ", "সিদ্ধাৰ্থ", "কৌশিক", "প্ৰিয়ংকা", "টুনী", "ৰিমঝিম", "শিখা", "মৌচুমী"
]

TEST_NAMES = [
    "অনিন্দ্য", "তন্ময়", "সৌৰভ", "হিমাংশু", "নৱজিত", "ঋতুপৰ্ণ", "প্ৰীতম", "চিৰঞ্জিত", "ভাস্বতী", "দেৱযানী",
    "মধুস্মিতা", "পৰিস্মিতা", "কাকলি", "জ্যোতি", "নিবেদিতা"
]

# Verify zero leakage mathematically
assert len(set(TRAIN_NAMES) & set(VAL_NAMES)) == 0, "Leakage between Train and Val!"
assert len(set(TRAIN_NAMES) & set(TEST_NAMES)) == 0, "Leakage between Train and Test!"
assert len(set(VAL_NAMES) & set(TEST_NAMES)) == 0, "Leakage between Val and Test!"

# ==============================================================================
# Attribute Dimensions & Vocabulary
# ==============================================================================

COMPARATIVE_DIMENSIONS = [
    {
        "type": "height",
        "comp_pos": "ওখ", "comp_neg": "চাপৰ",
        "sup_pos": "ওখ", "sup_neg": "চাপৰ"
    },
    {
        "type": "age",
        "comp_pos": "ডাঙৰ", "comp_neg": "সৰু",
        "sup_pos": "ডাঙৰ", "sup_neg": "সৰু"
    },
    {
        "type": "weight",
        "comp_pos": "গধুৰ", "comp_neg": "পাতল",
        "sup_pos": "গধুৰ", "sup_neg": "পাতল"
    },
    {
        "type": "speed",
        "comp_pos": "বেগবান", "comp_neg": "ধীৰ",
        "sup_pos": "বেগবান", "sup_neg": "ধীৰ"
    }
]

QUANTITY_ITEMS = [
    {"name": "আপেল", "unit": "টা", "category": "fruit"},
    {"name": "আম", "unit": "টা", "category": "fruit"},
    {"name": "কিতাপ", "unit": "খন", "category": "book"},
    {"name": "কলম", "unit": "টা", "category": "stationery"},
    {"name": "টকা", "unit": "টকা", "category": "currency"},
    {"name": "নম্বৰ", "unit": "নম্বৰ", "category": "score"}
]

# ==============================================================================
# Task 1: Transitive Ordering (A > B > C)
# ==============================================================================

def generate_transitive_task(names_pool: List[str], rng: random.Random) -> Dict[str, Any]:
    a, b, c = rng.sample(names_pool, 3)
    dim = rng.choice(COMPARATIVE_DIMENSIONS)
    
    direction_desc = rng.choice([True, False])
    
    if direction_desc:
        pos_word = dim["comp_pos"]
        sup_max = dim["sup_pos"]
        sup_min = dim["sup_neg"]
        # A > B and B > C
        templates_premise = [
            f"{a} {b}তকৈ {pos_word}। {b} {c}তকৈ {pos_word}।",
            f"যদি {a} {b}তকৈ {pos_word} আৰু {b} {c}তকৈ {pos_word} হয়,",
            f"{b}ৰ তুলনাত {a} অধিক {pos_word}। {c}ৰ তুলনাত {b} অধিক {pos_word}।"
        ]
        max_entity = a
        min_entity = c
    else:
        pos_word = dim["comp_neg"]
        sup_max = dim["sup_neg"]
        sup_min = dim["sup_pos"]
        # A < B and B < C
        templates_premise = [
            f"{a} {b}তকৈ {pos_word}। {b} {c}তকৈ {pos_word}।",
            f"যদি {a} {b}তকৈ {pos_word} আৰু {b} {c}তকৈ {pos_word} হয়,",
            f"{b}ৰ তুলনাত {a} অধিক {pos_word}। {c}ৰ তুলনাত {b} অধিক {pos_word}।"
        ]
        max_entity = c
        min_entity = a

    premise = rng.choice(templates_premise)

    # Ask for maximum or minimum
    ask_max = rng.choice([True, False])
    if ask_max:
        questions = [
            f"আটাইতকৈ {sup_max} কোন?",
            f"এওঁলোকৰ ভিতৰত আটাইতকৈ {sup_max} কোন?",
            f"কওকচোন কোন আটাইতকৈ {sup_max} ব্যক্তি?"
        ]
        answer = max_entity
    else:
        questions = [
            f"আটাইতকৈ {sup_min} কোন?",
            f"এওঁলোকৰ ভিতৰত আটাইতকৈ {sup_min} কোন?",
            f"কওকচোন কোন আটাইতকৈ {sup_min} ব্যক্তি?"
        ]
        answer = min_entity

    question = rng.choice(questions)
    prompt = f"{premise} {question} উত্তৰ:"
    full_text = f"{prompt} {answer}"

    return {
        "task_type": "transitive_ordering",
        "prompt": prompt,
        "completion": answer,
        "text": full_text,
        "entities": [a, b, c],
        "target": answer,
        "dimension": dim["type"]
    }

# ==============================================================================
# Task 2: Direct Quantitative Comparisons
# ==============================================================================

def generate_quantitative_task(names_pool: List[str], rng: random.Random) -> Dict[str, Any]:
    a, b = rng.sample(names_pool, 2)
    item = rng.choice(QUANTITY_ITEMS)
    
    val_a = rng.randint(3, 50)
    val_b = rng.randint(3, 50)
    while val_a == val_b:
        val_b = rng.randint(3, 50)

    ask_more = rng.choice([True, False])
    unit = item["unit"]
    item_name = item["name"]

    if item["category"] == "currency":
        premise = rng.choice([
            f"{a}ৰ ওচৰত {val_a} টকা আছে আৰু {b}ৰ ওচৰত {val_b} টকা আছে।",
            f"{a}ৰ ওচৰত {val_a} টকা আছে, আনহাতে {b}ৰ ওচৰত {val_b} টকা আছে।"
        ])
        if ask_more:
            q = "কাৰ ওচৰত বেছি ধন আছে?"
            ans = a if val_a > val_b else b
        else:
            q = "কাৰ ওচৰত কম ধন আছে?"
            ans = a if val_a < val_b else b
    elif item["category"] == "score":
        premise = f"পৰীক্ষাত {a}ই {val_a} নম্বৰ পালে আৰু {b}ই {val_b} নম্বৰ পালে।"
        if ask_more:
            q = "কোনে বেছি নম্বৰ পালে?"
            ans = a if val_a > val_b else b
        else:
            q = "কোনে কম নম্বৰ পালে?"
            ans = a if val_a < val_b else b
    else:
        premise = rng.choice([
            f"{a}ৰ ওচৰত {val_a} {unit} {item_name} আছে আৰু {b}ৰ ওচৰত {val_b} {unit} {item_name} আছে।",
            f"{a}ই {val_a} {unit} {item_name} কিনিলে আৰু {b}ই {val_b} {unit} {item_name} কিনিলে।"
        ])
        if ask_more:
            q = f"কাৰ ওচৰত বেছি {item_name} আছে?"
            ans = a if val_a > val_b else b
        else:
            q = f"কাৰ ওচৰত কম {item_name} আছে?"
            ans = a if val_a < val_b else b

    prompt = f"{premise} {q} উত্তৰ:"
    full_text = f"{prompt} {ans}"

    return {
        "task_type": "quantitative_comparison",
        "prompt": prompt,
        "completion": ans,
        "text": full_text,
        "entities": [a, b],
        "values": {a: val_a, b: val_b},
        "target": ans,
        "item": item_name
    }

# ==============================================================================
# Task 3: Multi-Hop Inequality Verification (Yes / No)
# ==============================================================================

def generate_multihop_verification(names_pool: List[str], rng: random.Random) -> Dict[str, Any]:
    a, b, c = rng.sample(names_pool, 3)
    dim = rng.choice(COMPARATIVE_DIMENSIONS)
    
    pos = dim["comp_pos"]
    neg = dim["comp_neg"]

    premise = f"{a} {b}তকৈ {pos} আৰু {b} {c}তকৈ {pos}।"

    ask_true = rng.choice([True, False])
    if ask_true:
        # e.g., A Cতকৈ ওখ নে? => হয়
        q = f"{a} {c}তকৈ {pos} নে?"
        ans = "হয়"
    else:
        # e.g., C Aতকৈ ওখ নে? => নহয়
        flip = rng.choice(["invert_entities", "invert_attribute"])
        if flip == "invert_entities":
            q = f"{c} {a}তকৈ {pos} নে?"
        else:
            q = f"{a} {c}তকৈ {neg} নে?"
        ans = "নহয়"

    prompt = f"{premise} {q} উত্তৰ:"
    full_text = f"{prompt} {ans}"

    return {
        "task_type": "multihop_verification",
        "prompt": prompt,
        "completion": ans,
        "text": full_text,
        "entities": [a, b, c],
        "target": ans,
        "dimension": dim["type"]
    }

# ==============================================================================
# Master Split Generator
# ==============================================================================

def build_dataset_split(
    names_pool: List[str],
    num_samples: int,
    seed: int,
    split_name: str
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples = []
    
    t1_count = int(num_samples * 0.40)
    t2_count = int(num_samples * 0.35)
    t3_count = num_samples - t1_count - t2_count

    for i in range(t1_count):
        item = generate_transitive_task(names_pool, rng)
        item["id"] = f"as_reason_{split_name}_{len(samples) + 1}"
        samples.append(item)

    for i in range(t2_count):
        item = generate_quantitative_task(names_pool, rng)
        item["id"] = f"as_reason_{split_name}_{len(samples) + 1}"
        samples.append(item)

    for i in range(t3_count):
        item = generate_multihop_verification(names_pool, rng)
        item["id"] = f"as_reason_{split_name}_{len(samples) + 1}"
        samples.append(item)

    rng.shuffle(samples)
    return samples

def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data):,} examples -> {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate Assamese Synthetic Reasoning Dataset")
    parser.add_argument("--output_dir", type=str, default="assamese/reasoning/data", help="Output directory")
    parser.add_argument("--train_samples", type=int, default=5000, help="Number of training examples")
    parser.add_argument("--val_samples", type=int, default=500, help="Number of validation examples")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of test examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("Generating Assamese Synthetic Reasoning Dataset (Phase 3)")
    print(f"Train Names: {len(TRAIN_NAMES)} | Val Names: {len(VAL_NAMES)} | Test Names: {len(TEST_NAMES)} (Held-out)")
    print("=" * 70)

    train_data = build_dataset_split(TRAIN_NAMES, args.train_samples, args.seed + 10, "train")
    val_data = build_dataset_split(VAL_NAMES, args.val_samples, args.seed + 20, "val")
    test_data = build_dataset_split(TEST_NAMES, args.test_samples, args.seed + 30, "test")

    out_dir = args.output_dir
    save_jsonl(train_data, os.path.join(out_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(out_dir, "val.jsonl"))
    save_jsonl(test_data, os.path.join(out_dir, "test.jsonl"))

    train_entities = set()
    for d in train_data:
        train_entities.update(d["entities"])
    
    test_entities = set()
    for d in test_data:
        test_entities.update(d["entities"])

    leakage_overlap = train_entities & test_entities
    print(f"\n[Verification] Train Entity Count: {len(train_entities)}")
    print(f"[Verification] Held-out Test Entity Count: {len(test_entities)}")
    print(f"[Verification] Overlap (Leakage): {len(leakage_overlap)} entities (MUST BE ZERO)")
    assert len(leakage_overlap) == 0, "Train-test leakage detected!"

    stats = {
        "language": "Assamese (as)",
        "script": "Bengali-Assamese (Eastern Nagari)",
        "total_samples": len(train_data) + len(val_data) + len(test_data),
        "splits": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        },
        "tasks": {
            "transitive_ordering": 0.40,
            "quantitative_comparison": 0.35,
            "multihop_verification": 0.25
        },
        "leakage_prevention": {
            "train_names_count": len(TRAIN_NAMES),
            "val_names_count": len(VAL_NAMES),
            "test_names_count": len(TEST_NAMES),
            "leakage_overlap_count": len(leakage_overlap)
        }
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Dataset stats saved -> {os.path.join(out_dir, 'stats.json')}")

if __name__ == "__main__":
    main()
