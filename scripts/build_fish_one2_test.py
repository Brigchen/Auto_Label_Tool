# -*- coding: utf-8 -*-
"""Build Fish_one2 test split from Fish_one (see --help)."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _image_names(root: Path, split: str) -> set[str]:
    d = root / "images" / split
    if not d.is_dir():
        return set()
    return {
        p.name.lower()
        for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXT
    }


def _collect_one_pairs(one: Path) -> dict[str, tuple[Path, Path, str]]:
    """Map lowercase filename -> (image, label, source_split)."""
    out: dict[str, tuple[Path, Path, str]] = {}
    for split in ("train", "val"):
        img_dir = one / "images" / split
        lbl_dir = one / "labels" / split
        if not img_dir.is_dir():
            continue
        for img in img_dir.iterdir():
            if not img.is_file() or img.suffix.lower() not in IMG_EXT:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.is_file():
                continue
            out[img.name.lower()] = (img, lbl, split)
    return out


def _remove_from_split(root: Path, split: str, names: set[str]) -> tuple[int, int]:
    removed_img = removed_lbl = 0
    stems = {Path(n).stem.lower() for n in names}
    for sub, match_fn in (
        ("images", lambda p: p.name.lower() in names),
        ("labels", lambda p: p.stem.lower() in stems),
    ):
        d = root / sub / split
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if not p.is_file():
                continue
            if sub == "labels" and p.suffix.lower() != ".txt":
                continue
            if sub == "images" and p.suffix.lower() not in IMG_EXT:
                continue
            if match_fn(p):
                p.unlink()
                if sub == "images":
                    removed_img += 1
                else:
                    removed_lbl += 1
    return removed_img, removed_lbl


def build_test(
    one: Path,
    one2: Path,
    count: int,
    seed: int,
    yaml_path: Path,
) -> dict:
    one_pairs = _collect_one_pairs(one)
    one2_train = _image_names(one2, "train")
    one2_val = _image_names(one2, "val")
    one2_test = _image_names(one2, "test")
    one2_all = one2_train | one2_val | one2_test

    tier1 = [k for k in one_pairs if k not in one2_all]
    tier2 = [k for k in one_pairs if k in one2_val and k not in one2_train and k not in one2_test]
    tier3 = [k for k in one_pairs if k in one2_val and k not in one2_test]

    selected: list[str] = []
    for tier in (tier1, tier2, tier3):
        pool = [k for k in tier if k not in selected]
        random.Random(seed).shuffle(pool)
        need = count - len(selected)
        if need <= 0:
            break
        selected.extend(pool[:need])

    if len(selected) < count:
        raise SystemExit(
            f"可用样本不足: 需要 {count}，仅找到 {len(selected)} "
            f"(tier1={len(tier1)}, tier2={len(tier2)}, tier3={len(tier3)})"
        )

    img_out = one2 / "images" / "test"
    lbl_out = one2 / "labels" / "test"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in selected:
        img, lbl, split = one_pairs[name]
        dst_img = img_out / img.name
        dst_lbl = lbl_out / lbl.name
        shutil.copy2(img, dst_img)
        shutil.copy2(lbl, dst_lbl)
        copied.append(
            {
                "file": img.name,
                "fish_one_split": split,
                "was_in_one2_train": name in one2_train,
                "was_in_one2_val": name in one2_val,
            }
        )

    removed = {"train": {"images": 0, "labels": 0}, "val": {"images": 0, "labels": 0}}
    name_set = set(selected)
    for split in ("train", "val"):
        ri, rl = _remove_from_split(one2, split, name_set)
        removed[split]["images"] = ri
        removed[split]["labels"] = rl

    _update_yaml(yaml_path, one2)

    manifest = {
        "count": len(selected),
        "seed": seed,
        "tiers": {"tier1_new_names": len(tier1), "tier2_val_only": len(tier2), "tier3_from_val": len(tier3)},
        "removed_from_one2": removed,
        "files": copied,
    }
    manifest_path = one2 / "test_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _update_yaml(yaml_path: Path, one2: Path) -> None:
    text = yaml_path.read_text(encoding="utf-8")
    test_line = "test: images/test"
    if "test: images/test" in text and "#test: images/test" not in text:
        pass
    elif "#test: images/test" in text:
        text = text.replace("#test: images/test", "test: images/test", 1)
    else:
        anchor = "val: images/val"
        if anchor in text:
            text = text.replace(anchor, f"{anchor}\ntest: images/test")
        else:
            text += f"\ntest: images/test\n"
    yaml_path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Fish_one2 test set from Fish_one")
    ap.add_argument("--one", default=r"H:/Python/datasets/Fish_one")
    ap.add_argument("--one2", default=r"H:/Python/datasets/Fish_one2")
    ap.add_argument("--yaml", default=r"H:/Python/datasets/Fish_one2/fish_one.yaml")
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = build_test(
        Path(args.one),
        Path(args.one2),
        args.count,
        args.seed,
        Path(args.yaml),
    )
    print(json.dumps({"ok": True, "count": manifest["count"], "manifest": str(Path(args.one2) / "test_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
