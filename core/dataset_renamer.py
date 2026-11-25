# -*- coding: utf-8 -*-
"""
数据集图片与标签一键重命名模块

支持两种模式:
  1. 当前项目内重命名: 同步更新项目磁盘图片文件、SQLite images 表(filename/storage_path)、
     项目内已有 labels 文件及 AI 断点记录(ai_progress.json)，后续导出/训练自动生成同名 img_001.txt。
  2. 本地外部数据集目录重命名: 自动识别同目录混放或 images/ + labels/ 分目录(含 train/val/test 子目录)结构，
     配对重命名图片与同名标签文件(.txt / .xml / .json)，自动跳过 classes.txt / data.yaml 等元数据文件。

全部文件重命名均采用“两阶段临时名过渡”策略，即使原文件已命名为 img_002.jpg 需要重排为 img_001.jpg，
也绝不会发生文件名冲突或互相覆盖。
"""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from models.database import db

SUPPORTED_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'
}
SUPPORTED_LABEL_EXTENSIONS = ('.txt', '.xml', '.json')

# 严禁当作单图标签重命名的全局说明/配置/类别文件（小写比对）
RESERVED_METADATA_FILES = {
    'classes.txt',
    'predefined_classes.txt',
    'class_names.txt',
    'labels.txt',
    'obj.names',
    'readme.txt',
    'license.txt',
    'train.txt',
    'val.txt',
    'test.txt',
    'annotations.json',
    '_annotations.coco.json',
    'instances_train.json',
    'instances_val.json',
    'instances_test.json',
}

_ILLEGAL_PREFIX_CHARS = set('/\\:*?"<>|')


def natural_sort_key(name: str):
    """自然排序键: 使 img_2.jpg 排在 img_10.jpg 之前"""
    parts = re.split(r'(\d+)', str(name or ''))
    return [int(p) if p.isdigit() else p.casefold() for p in parts]


def validate_rename_params(prefix: str, start_index: int, min_digits: int) -> Tuple[str, int, int]:
    """校验并规范化重命名参数"""
    clean_prefix = (prefix or "").strip()
    if any(ch in _ILLEGAL_PREFIX_CHARS for ch in clean_prefix) or ".." in clean_prefix:
        raise ValueError('前缀不能包含路径分隔符或非法文件名字符（/ \\ : * ? " < > | ..）')
    if clean_prefix.lower().startswith("ai_"):
        raise ValueError("前缀请勿以 'ai_' 开头（'ai_' 为系统内部 AI 缓存文件保留前缀）")
    if start_index < 0:
        raise ValueError("起始编号不能小于 0")
    if min_digits < 1 or min_digits > 10:
        raise ValueError("编号位数需在 1 ~ 10 之间")
    return clean_prefix, int(start_index), int(min_digits)


def compute_effective_digits(count: int, start_index: int, min_digits: int = 3) -> int:
    """计算实际补零位数（若最大编号超过设定最少位数则自动扩容，避免截断）"""
    if count <= 0:
        return max(1, int(min_digits))
    max_num = int(start_index) + count - 1
    return max(int(min_digits), len(str(max_num)))


def format_target_stem(prefix: str, number: int, digits: int) -> str:
    """生成不含扩展名的目标主文件名，如 img_001"""
    return f"{prefix}{number:0{digits}d}"


def _execute_two_phase_file_moves(
    move_pairs: List[Tuple[Path, Path]],
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> None:
    """两阶段无冲突文件重命名（失败时自动回滚已移动文件）

    Args:
        move_pairs: [(src_path, dst_path), ...]，仅包含实际存在的源文件
    """
    active_pairs = [(Path(s), Path(d)) for s, d in move_pairs if Path(s) != Path(d)]
    if not active_pairs:
        return

    # 检查目标路径之间是否有重复，或是否会覆盖非本次重命名集合中的已有文件
    src_resolved_set = {p.resolve() for p, _ in active_pairs if p.exists()}
    dst_seen = set()
    for _, dst in active_pairs:
        dst_resolved = dst.resolve()
        if dst_resolved in dst_seen:
            raise RuntimeError(f"目标文件名冲突: {dst.name}")
        dst_seen.add(dst_resolved)
        if dst.exists() and dst_resolved not in src_resolved_set:
            raise RuntimeError(
                f"目标文件已存在且不在重命名队列中，为防覆盖已中止: {dst}"
            )

    token = uuid.uuid4().hex[:10]
    staged: List[Tuple[Path, Path, Path]] = []  # (orig_src, tmp_path, final_dst)
    completed_final: List[Tuple[Path, Path]] = []  # (orig_src, final_dst)

    try:
        total = len(active_pairs)
        # 阶段 1: 全部改名为唯一临时文件
        for idx, (src, dst) in enumerate(active_pairs):
            if not src.exists():
                continue
            tmp = src.parent / f".by_rename_tmp_{token}_{idx}{src.suffix}"
            os.replace(src, tmp)
            staged.append((src, tmp, dst))
            if progress_callback and total > 0 and idx % 10 == 0:
                pct = int((idx / (total * 2)) * 100)
                progress_callback(pct, f"准备重命名 ({idx + 1}/{total})...")

        # 阶段 2: 临时文件改名为最终目标文件
        for idx, (orig_src, tmp, dst) in enumerate(staged):
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, dst)
            completed_final.append((orig_src, dst))
            if progress_callback and len(staged) > 0 and idx % 10 == 0:
                pct = 50 + int(((idx + 1) / len(staged)) * 50)
                progress_callback(min(99, pct), f"写入新文件名 ({idx + 1}/{len(staged)})...")

    except Exception as exc:
        # 尽力回滚，保证磁盘不留半成品临时文件
        for orig_src, final_dst in reversed(completed_final):
            try:
                if final_dst.exists() and not orig_src.exists():
                    os.replace(final_dst, orig_src)
            except OSError:
                pass
        for orig_src, tmp, _ in reversed(staged):
            try:
                if tmp.exists() and not orig_src.exists():
                    os.replace(tmp, orig_src)
            except OSError:
                pass
        raise RuntimeError(f"文件重命名失败，已回滚原文件名: {exc}") from exc


# ==================== 模式 A: 当前项目内重命名 ====================

def _sort_project_images(images: List[Dict], sort_mode: str) -> List[Dict]:
    if sort_mode == "natural_name":
        return sorted(
            images,
            key=lambda img: (natural_sort_key(img.get('filename', '')), img.get('id', 0)),
        )
    # 默认按项目导入先后顺序 (get_project_images 已按 created_at 排序)
    return list(images)


def _find_project_sidecar_labels(project_storage: Path, image_record: Dict) -> List[Tuple[Path, str]]:
    """查找项目目录内可能已导出的同名标签文件: [(label_path, ext), ...]"""
    results = []
    labels_dir = project_storage / "labels"
    if not labels_dir.is_dir():
        return results

    old_stem = Path(image_record.get('filename') or '').stem
    if not old_stem:
        return results

    for ext in SUPPORTED_LABEL_EXTENSIONS:
        candidate = labels_dir / f"{old_stem}{ext}"
        if candidate.is_file() and candidate.name.lower() not in RESERVED_METADATA_FILES:
            results.append((candidate, ext))
    return results


def preview_project_rename(
    project_id: int,
    prefix: str = "img_",
    start_index: int = 1,
    min_digits: int = 3,
    sort_mode: str = "import_order",
) -> List[Dict]:
    """生成当前项目的重命名预览计划（不修改磁盘或数据库）"""
    clean_prefix, start_num, digits_cfg = validate_rename_params(prefix, start_index, min_digits)
    project = db.get_project(project_id)
    if not project:
        raise ValueError(f"项目不存在: {project_id}")

    project_storage = Path(project.get('storage_path') or '.')
    images = _sort_project_images(db.get_project_images(project_id), sort_mode)
    eff_digits = compute_effective_digits(len(images), start_num, digits_cfg)

    # 查询哪些图片在数据库中有标注或已标为负样本（导出时将生成对应 img_xxx.txt）
    annotated_ids = {
        img['id'] for img in images if img.get('status') == 'annotated'
    }

    plan = []
    for idx, img in enumerate(images):
        num = start_num + idx
        new_stem = format_target_stem(clean_prefix, num, eff_digits)
        old_storage = Path(img.get('storage_path') or '')
        ext = old_storage.suffix.lower() or Path(img.get('filename') or '').suffix.lower() or '.jpg'
        new_filename = f"{new_stem}{ext}"
        new_storage = old_storage.parent / new_filename if img.get('storage_path') else Path(new_filename)

        sidecars = _find_project_sidecar_labels(project_storage, img)
        sidecar_moves = [
            (lbl_path, lbl_path.parent / f"{new_stem}{lbl_ext}")
            for lbl_path, lbl_ext in sidecars
        ]

        plan.append({
            'id': img['id'],
            'old_filename': img.get('filename', ''),
            'new_filename': new_filename,
            'new_stem': new_stem,
            'old_storage': old_storage,
            'new_storage': new_storage,
            'has_db_label': img['id'] in annotated_ids,
            'sidecar_moves': sidecar_moves,
        })
    return plan


def rename_project_dataset(
    project_id: int,
    prefix: str = "img_",
    start_index: int = 1,
    min_digits: int = 3,
    sort_mode: str = "import_order",
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    """一键重命名当前项目内的全部图片及对应标注记录"""
    project = db.get_project(project_id)
    if not project:
        raise ValueError(f"项目不存在: {project_id}")

    plan = preview_project_rename(
        project_id=project_id,
        prefix=prefix,
        start_index=start_index,
        min_digits=min_digits,
        sort_mode=sort_mode,
    )
    if not plan:
        return {'renamed_images': 0, 'synced_db_labels': 0, 'renamed_label_files': 0, 'plan': []}

    file_moves: List[Tuple[Path, Path]] = []
    label_file_count = 0
    for item in plan:
        if item['old_storage'].is_file():
            file_moves.append((item['old_storage'], item['new_storage']))
        for src_lbl, dst_lbl in item['sidecar_moves']:
            file_moves.append((src_lbl, dst_lbl))
            label_file_count += 1

    # 1) 两阶段重命名磁盘文件
    _execute_two_phase_file_moves(file_moves, progress_callback=progress_callback)

    # 2) 批量更新 SQLite images 表
    db_updates = [
        (item['new_filename'], str(item['new_storage']), item['id'])
        for item in plan
    ]
    db.batch_update_image_filenames(db_updates)

    # 3) 同步更新 AI 断点记录 ai_progress.json 中的文件名，避免重命名后丢失已完成状态
    project_storage = Path(project.get('storage_path') or '')
    progress_file = project_storage / "ai_progress.json"
    if progress_file.is_file():
        try:
            old_to_new = {item['old_filename']: item['new_filename'] for item in plan}
            bp_data = json.loads(progress_file.read_text(encoding='utf-8')) or {}
            if isinstance(bp_data.get('completed'), list):
                bp_data['completed'] = [
                    old_to_new.get(fn, fn) for fn in bp_data['completed']
                ]
                progress_file.write_text(
                    json.dumps(bp_data, indent=4, ensure_ascii=False),
                    encoding='utf-8',
                )
        except Exception:
            pass

    if progress_callback:
        progress_callback(100, f"重命名完成: 共 {len(plan)} 张图片")

    synced_db_labels = sum(1 for item in plan if item['has_db_label'])
    return {
        'renamed_images': len(plan),
        'synced_db_labels': synced_db_labels,
        'renamed_label_files': label_file_count,
        'first_name': plan[0]['new_filename'],
        'last_name': plan[-1]['new_filename'],
        'plan': plan,
    }


# ==================== 模式 B: 本地外部数据集文件夹重命名 ====================

def _find_external_paired_labels(root_folder: Path, img_path: Path) -> List[Path]:
    """为外部数据集目录中的单张图片查找配对的同名标签文件(.txt / .xml / .json)

    自动兼容:
      - 同目录混放: <dir>/foo.jpg + <dir>/foo.txt
      - 标准分目录: <root>/images/foo.jpg + <root>/labels/foo.txt
      - 带划分分目录: <root>/images/train/foo.jpg + <root>/labels/train/foo.txt
      - 子集分目录: <root>/train/images/foo.jpg + <root>/train/labels/foo.txt
    """
    stem = img_path.stem
    candidate_dirs = [img_path.parent]

    # 查找把路径组件中的 'images' 替换为 'labels' 的镜像目录
    try:
        rel_parts = list(img_path.parent.relative_to(root_folder).parts)
        for idx in range(len(rel_parts) - 1, -1, -1):
            if rel_parts[idx].lower() == 'images':
                mirror_parts = list(rel_parts)
                mirror_parts[idx] = 'labels'
                mirror_dir = root_folder.joinpath(*mirror_parts)
                if mirror_dir not in candidate_dirs:
                    candidate_dirs.append(mirror_dir)
                break
    except ValueError:
        pass

    # 若用户直接选择了 images 目录本身，也顺便检查兄弟目录 ../labels
    if img_path.parent.name.lower() == 'images':
        sibling_labels = img_path.parent.parent / 'labels'
        if sibling_labels not in candidate_dirs:
            candidate_dirs.append(sibling_labels)

    matched: List[Path] = []
    seen = set()
    for cdir in candidate_dirs:
        if not cdir.is_dir():
            continue
        for ext in SUPPORTED_LABEL_EXTENSIONS:
            cand = cdir / f"{stem}{ext}"
            if not cand.is_file():
                continue
            if cand.name.lower() in RESERVED_METADATA_FILES:
                continue
            resolved = cand.resolve()
            if resolved not in seen:
                seen.add(resolved)
                matched.append(cand)
    return matched


def scan_external_dataset_images(folder_path: str, sort_mode: str = "natural_name") -> List[Path]:
    """扫描外部数据集目录下的全部有效图片并排序"""
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"无效的文件夹路径: {folder_path}")

    images = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.startswith("ai_"):
            continue
        if "ai_examples" in p.parts:
            continue
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(p)

    if sort_mode == "mtime":
        images.sort(key=lambda p: (p.stat().st_mtime, natural_sort_key(str(p.relative_to(root)))))
    else:
        images.sort(key=lambda p: natural_sort_key(str(p.relative_to(root))))
    return images


def preview_external_folder_rename(
    folder_path: str,
    prefix: str = "img_",
    start_index: int = 1,
    min_digits: int = 3,
    sort_mode: str = "natural_name",
) -> List[Dict]:
    """生成外部数据集目录重命名预览计划（不修改磁盘）"""
    clean_prefix, start_num, digits_cfg = validate_rename_params(prefix, start_index, min_digits)
    root = Path(folder_path)
    images = scan_external_dataset_images(folder_path, sort_mode=sort_mode)
    eff_digits = compute_effective_digits(len(images), start_num, digits_cfg)

    plan = []
    for idx, img_path in enumerate(images):
        num = start_num + idx
        new_stem = format_target_stem(clean_prefix, num, eff_digits)
        ext = img_path.suffix.lower()
        new_img_path = img_path.parent / f"{new_stem}{ext}"

        paired_labels = _find_external_paired_labels(root, img_path)
        label_moves = [
            (lbl, lbl.parent / f"{new_stem}{lbl.suffix.lower()}")
            for lbl in paired_labels
        ]

        plan.append({
            'old_filename': img_path.name,
            'new_filename': new_img_path.name,
            'new_stem': new_stem,
            'old_image_path': img_path,
            'new_image_path': new_img_path,
            'label_moves': label_moves,
        })
    return plan


def _sync_labelme_and_coco_references(root: Path, plan: List[Dict]) -> None:
    """若存在 LabelMe 单图 JSON 或 COCO annotations.json，同步其中的文件名引用"""
    # 1) LabelMe 单图 JSON 的 imagePath 字段
    for item in plan:
        for _, dst_lbl in item['label_moves']:
            if dst_lbl.suffix.lower() == '.json' and dst_lbl.is_file():
                try:
                    data = json.loads(dst_lbl.read_text(encoding='utf-8'))
                    if isinstance(data, dict) and 'imagePath' in data and 'shapes' in data:
                        old_ref = str(data.get('imagePath') or '')
                        if Path(old_ref).name == item['old_filename']:
                            parent_str = str(Path(old_ref).parent)
                            data['imagePath'] = (
                                item['new_filename']
                                if parent_str in ('', '.')
                                else f"{parent_str}/{item['new_filename']}"
                            )
                            dst_lbl.write_text(
                                json.dumps(data, indent=2, ensure_ascii=False),
                                encoding='utf-8',
                            )
                except Exception:
                    pass

    # 2) COCO JSON (annotations.json / _annotations.coco.json)
    name_map = {item['old_filename']: item['new_filename'] for item in plan}
    for coco_name in ('annotations.json', '_annotations.coco.json'):
        for coco_path in root.rglob(coco_name):
            if not coco_path.is_file():
                continue
            try:
                coco_data = json.loads(coco_path.read_text(encoding='utf-8'))
                if not isinstance(coco_data, dict) or not isinstance(coco_data.get('images'), list):
                    continue
                changed = False
                for img_entry in coco_data['images']:
                    if not isinstance(img_entry, dict):
                        continue
                    fn = str(img_entry.get('file_name') or '')
                    base_fn = Path(fn).name
                    if base_fn in name_map:
                        parent_str = str(Path(fn).parent)
                        img_entry['file_name'] = (
                            name_map[base_fn]
                            if parent_str in ('', '.')
                            else f"{parent_str}/{name_map[base_fn]}"
                        )
                        changed = True
                if changed:
                    coco_path.write_text(
                        json.dumps(coco_data, indent=2, ensure_ascii=False),
                        encoding='utf-8',
                    )
            except Exception:
                pass


def rename_external_folder_dataset(
    folder_path: str,
    prefix: str = "img_",
    start_index: int = 1,
    min_digits: int = 3,
    sort_mode: str = "natural_name",
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    """一键配对重命名本地外部数据集目录中的图片与同名标签文件"""
    root = Path(folder_path)
    plan = preview_external_folder_rename(
        folder_path=folder_path,
        prefix=prefix,
        start_index=start_index,
        min_digits=min_digits,
        sort_mode=sort_mode,
    )
    if not plan:
        return {'renamed_images': 0, 'renamed_label_files': 0, 'plan': []}

    move_pairs: List[Tuple[Path, Path]] = []
    label_count = 0
    for item in plan:
        move_pairs.append((item['old_image_path'], item['new_image_path']))
        for src_lbl, dst_lbl in item['label_moves']:
            move_pairs.append((src_lbl, dst_lbl))
            label_count += 1

    _execute_two_phase_file_moves(move_pairs, progress_callback=progress_callback)
    _sync_labelme_and_coco_references(root, plan)

    if progress_callback:
        progress_callback(100, f"重命名完成: {len(plan)} 张图片, {label_count} 个标签文件")

    return {
        'renamed_images': len(plan),
        'renamed_label_files': label_count,
        'first_name': plan[0]['new_filename'],
        'last_name': plan[-1]['new_filename'],
        'plan': plan,
    }
