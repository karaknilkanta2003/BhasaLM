"""
Hindi Synthetic Reasoning Dataset Generator (Phase 3)
=====================================================
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
    "अमित", "राहुल", "विकास", "रोहन", "सोहन", "अजय", "संजय", "मनीष", "दीपक", "नीरज",
    "प्रमोद", "विवेक", "सुमित", "गौरव", "आकाश", "कुणाल", "वरुण", "नितिन", "राजेश", "अशोक",
    "पूजा", "नेहा", "प्रिया", "अंजलि", "स्नेहा", "कविता", "रीता", "सुनीता", "मीना", "आरती",
    "प्रीति", "दीपा", "सीमा", "रश्मि", "भावना", "किरण", "श्वेता", "पूनम", "मोनिका", "अंशुल"
]

VAL_NAMES = [
    "हर्ष", "कार्तिक", "मयंक", "तरुण", "पार्थ", "श्रेया", "तनु", "दिव्या", "रिया", "ईशा"
]

TEST_NAMES = [
    "अभिनव", "ऋषभ", "चेतन", "तुषार", "निखिल", "मयूर", "लोकेश", "वैभव", "सिद्धार्थ", "हेमंत",
    "अदिति", "कृतिका", "तान्या", "पारुल", "मेघा"
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
        "comp_pos": "लंबा", "comp_neg": "छोटा",
        "sup_pos": "लंबा", "sup_neg": "छोटा",
        "nouns": ["कद", "ऊंचाई"]
    },
    {
        "type": "age",
        "comp_pos": "बड़ा", "comp_neg": "छोटा",
        "sup_pos": "बड़ा", "sup_neg": "छोटा",
        "nouns": ["आयु", "उम्र"]
    },
    {
        "type": "weight",
        "comp_pos": "भारी", "comp_neg": "हल्का",
        "sup_pos": "भारी", "sup_neg": "हल्का",
        "nouns": ["वज़न", "भार"]
    },
    {
        "type": "speed",
        "comp_pos": "तेज़", "comp_neg": "धीमा",
        "sup_pos": "तेज़", "sup_neg": "धीमा",
        "nouns": ["गति", "रफ़्तार"]
    }
]

QUANTITY_ITEMS = [
    {"name": "सेब", "unit": "किलो", "category": "fruit"},
    {"name": "आम", "unit": "टोकरी", "category": "fruit"},
    {"name": "किताबें", "unit": "पुस्तकें", "category": "book"},
    {"name": "कलम", "unit": "पेन", "category": "stationery"},
    {"name": "रुपये", "unit": "रुपये", "category": "currency"},
    {"name": "अंक", "unit": "मार्क्स", "category": "score"}
]

# ==============================================================================
# Task 1: Transitive Ordering (A > B > C)
# ==============================================================================

def generate_transitive_task(names_pool: List[str], rng: random.Random) -> Dict[str, Any]:
    a, b, c = rng.sample(names_pool, 3)
    dim = rng.choice(COMPARATIVE_DIMENSIONS)
    
    # Decide direction: True => A > B > C; False => A < B < C
    direction_desc = rng.choice([True, False])
    
    if direction_desc:
        pos_word = dim["comp_pos"]
        neg_word = dim["comp_neg"]
        sup_max = dim["sup_pos"]
        sup_min = dim["sup_neg"]
        # A > B and B > C
        # Phrasing templates
        templates_premise = [
            f"{a}, {b} से {pos_word} है। {b}, {c} से {pos_word} है।",
            f"यदि {a}, {b} से {pos_word} है और {b}, {c} से {pos_word} है।",
            f"{b} की तुलना में {a} {pos_word} है। {c} की तुलना में {b} {pos_word} है।"
        ]
        max_entity = a
        min_entity = c
    else:
        pos_word = dim["comp_neg"]
        neg_word = dim["comp_pos"]
        sup_max = dim["sup_neg"]
        sup_min = dim["sup_pos"]
        # A < B and B < C
        templates_premise = [
            f"{a}, {b} से {pos_word} है। {b}, {c} से {pos_word} है।",
            f"यदि {a}, {b} से {pos_word} है और {b}, {c} से {pos_word} है।",
            f"{b} की तुलना में {a} {pos_word} है। {c} की तुलना में {b} {pos_word} है।"
        ]
        max_entity = c
        min_entity = a

    premise = rng.choice(templates_premise)

    # Ask for maximum or minimum
    ask_max = rng.choice([True, False])
    if ask_max:
        questions = [
            f"इनमें से सबसे {sup_max} कौन है?",
            f"बताइए कि सबसे {sup_max} कौन है?",
            f"तीनों में से सबसे {sup_max} व्यक्ति कौन है?"
        ]
        answer = max_entity
    else:
        questions = [
            f"इनमें से सबसे {sup_min} कौन है?",
            f"बताइए कि सबसे {sup_min} कौन है?",
            f"तीनों में से सबसे {sup_min} व्यक्ति कौन है?"
        ]
        answer = min_entity

    question = rng.choice(questions)
    prompt = f"{premise} {question} उत्तर:"
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
    
    if item["category"] == "currency":
        premise = rng.choice([
            f"{a} के पास {val_a} रुपये हैं और {b} के पास {val_b} रुपये हैं।",
            f"{a} के पास {val_a} रुपये हैं, जबकि {b} के पास {val_b} रुपये हैं।"
        ])
        if ask_more:
            q = "किसके पास अधिक धन है?"
            ans = a if val_a > val_b else b
        else:
            q = "किसके पास कम धन है?"
            ans = a if val_a < val_b else b
    elif item["category"] == "score":
        premise = f"परीक्षा में {a} को {val_a} अंक मिले और {b} को {val_b} अंक मिले।"
        if ask_more:
            q = "किसको अधिक अंक मिले?"
            ans = a if val_a > val_b else b
        else:
            q = "किसको कम अंक मिले?"
            ans = a if val_a < val_b else b
    else:
        premise = rng.choice([
            f"{a} के पास {val_a} {item['name']} हैं और {b} के पास {val_b} {item['name']} हैं।",
            f"{a} ने {val_a} {item['name']} खरीदे, जबकि {b} ने {val_b} {item['name']} खरीदे।"
        ])
        if ask_more:
            q = f"किसके पास अधिक {item['name']} हैं?"
            ans = a if val_a > val_b else b
        else:
            q = f"किसके पास कम {item['name']} हैं?"
            ans = a if val_a < val_b else b

    prompt = f"{premise} {q} उत्तर:"
    full_text = f"{prompt} {ans}"

    return {
        "task_type": "quantitative_comparison",
        "prompt": prompt,
        "completion": ans,
        "text": full_text,
        "entities": [a, b],
        "values": {a: val_a, b: val_b},
        "target": ans,
        "item": item["name"]
    }

# ==============================================================================
# Task 3: Multi-Hop Inequality Verification (Yes / No)
# ==============================================================================

def generate_multihop_verification(names_pool: List[str], rng: random.Random) -> Dict[str, Any]:
    a, b, c = rng.sample(names_pool, 3)
    dim = rng.choice(COMPARATIVE_DIMENSIONS)
    
    pos = dim["comp_pos"]
    neg = dim["comp_neg"]

    # True relation: A > B and B > C
    premise = f"{a}, {b} से {pos} है और {b}, {c} से {pos} है।"

    # Question can be True or False
    ask_true = rng.choice([True, False])
    if ask_true:
        # e.g., क्या A, C से लंबा है? => हाँ
        q = f"क्या {a}, {c} से {pos} है?"
        ans = "हाँ"
    else:
        # e.g., क्या C, A से लंबा है? => नहीं
        # Or क्या A, C से छोटा है? => नहीं
        flip = rng.choice(["invert_entities", "invert_attribute"])
        if flip == "invert_entities":
            q = f"क्या {c}, {a} से {pos} है?"
        else:
            q = f"क्या {a}, {c} से {neg} है?"
        ans = "नहीं"

    prompt = f"{premise} {q} उत्तर:"
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
    
    # Task distribution: 40% Transitive Ordering, 35% Quantitative Comparison, 25% Multi-hop Verification
    t1_count = int(num_samples * 0.40)
    t2_count = int(num_samples * 0.35)
    t3_count = num_samples - t1_count - t2_count

    for i in range(t1_count):
        item = generate_transitive_task(names_pool, rng)
        item["id"] = f"hi_reason_{split_name}_{len(samples) + 1}"
        samples.append(item)

    for i in range(t2_count):
        item = generate_quantitative_task(names_pool, rng)
        item["id"] = f"hi_reason_{split_name}_{len(samples) + 1}"
        samples.append(item)

    for i in range(t3_count):
        item = generate_multihop_verification(names_pool, rng)
        item["id"] = f"hi_reason_{split_name}_{len(samples) + 1}"
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
    parser = argparse.ArgumentParser(description="Generate Hindi Synthetic Reasoning Dataset")
    parser.add_argument("--output_dir", type=str, default="hindi/reasoning/data", help="Output directory")
    parser.add_argument("--train_samples", type=int, default=5000, help="Number of training examples")
    parser.add_argument("--val_samples", type=int, default=500, help="Number of validation examples")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of test examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("Generating Hindi Synthetic Reasoning Dataset (Phase 3)")
    print(f"Train Names: {len(TRAIN_NAMES)} | Val Names: {len(VAL_NAMES)} | Test Names: {len(TEST_NAMES)} (Held-out)")
    print("=" * 70)

    train_data = build_dataset_split(TRAIN_NAMES, args.train_samples, args.seed + 1, "train")
    val_data = build_dataset_split(VAL_NAMES, args.val_samples, args.seed + 2, "val")
    test_data = build_dataset_split(TEST_NAMES, args.test_samples, args.seed + 3, "test")

    out_dir = args.output_dir
    save_jsonl(train_data, os.path.join(out_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(out_dir, "val.jsonl"))
    save_jsonl(test_data, os.path.join(out_dir, "test.jsonl"))

    # Compute statistics
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
        "language": "Hindi (hi)",
        "script": "Devanagari",
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
