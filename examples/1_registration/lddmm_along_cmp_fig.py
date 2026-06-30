

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import demeter.metamorphosis as mt
import demeter.utils.torchbox as tb
import sys
sys.path.insert(0, str(Path("examples/1_registration").resolve()))
from lddmm_along_utils import open_rainbow_minifish
from demeter import ROOT_DIRECTORY

ALLOWED_METHODS = {
    "affine",
    "rT_sT_tT",
    "rT_sF_tT",
    "rT_sT_tF",
}

ALLOWED_MODES = {
    "only",
    "along",
    "successive",
}

ALLOWED_TARGETS = {
    "full_affine",
    "rotation_translation_scaling",
    "rotation_translation",
    "rotation_scaling",
}

METHOD_ORDER = [
    "affine",
    "rT_sT_tT",
    "rT_sF_tT",
    "rT_sT_tF",
]

TARGET_ORDER = [
    "full_affine",
    "rotation_translation_scaling",
    "rotation_translation",
    "rotation_scaling",
]

METHOD_LABELS = {
    "affine": "Full affine",
    "rT_sT_tT": "(R+S+T)",
    "rT_sF_tT": "(R+T)",
    "rT_sT_tF": "(R+S)",
}

TARGET_LABELS = {
    "full_affine": "Target: Full affine",
    "rotation_translation_scaling": "Target: Rotation + Translation + Scaling",
    "rotation_translation": "Target: Rotation + Translation",
    "rotation_scaling": "Target: Rotation + Scaling",
}


@dataclass(frozen=True)
class ParsedRunFile:
    """
    Structured representation of one experiment filename.

    Parameters
    ----------
    file_name : str
        Original filename.
    method : str
        Registration method, e.g. 'affine', 'rT_sT_tT'.
    mode : str
        Registration mode, either 'along' or 'successive'.
    target : str
        Target transform family, e.g. 'full_affine'.
    toy_example : str
        Toy example name, e.g. 'fishes'.
    dimension : Optional[str]
        Optional dimension prefix, e.g. '2D'.
    date : Optional[str]
        Optional date string, e.g. '20260410'.
    sample_id : Optional[str]
        Optional trailing sample id, e.g. 'turtlefox_000'.
    part : Optional[str]
        For two-part successive registrations: 'part1' or 'part2'. None otherwise.
    extension : Optional[str]
        File extension without the leading dot, e.g. 'pk1'.
    """
    file_name: str
    method: str
    mode: str
    target: str
    toy_example: str
    dimension: Optional[str] = None
    date: Optional[str] = None
    sample_id: Optional[str] = None
    part: Optional[str] = None
    extension: Optional[str] = None


def parse_experiment_filename(
    file_name: str,
    *,
    validate: bool = True,
) -> ParsedRunFile:
    """
    Parse an experiment filename of the form

        2D_20260410_fishes_method_affine-along_target_rotation_scaling_turtlefox_000.pk1

    into structured metadata.

    Parameters
    ----------
    file_name : str
        Filename or path.
    validate : bool, default=True
        If True, checks method/mode/target against the allowed sets.

    Returns
    -------
    ParsedRunFile
        Parsed structured record.

    Raises
    ------
    ValueError
        If the filename does not match the expected structure.
    """
    basename = Path(file_name).name
    stem = Path(basename).stem
    extension = Path(basename).suffix[1:] if Path(basename).suffix else None

    method_anchor = "_method_"
    target_anchor = "_target_"

    if method_anchor not in stem:
        raise ValueError(f"Missing '{method_anchor}' in filename: {basename}")
    if target_anchor not in stem:
        raise ValueError(f"Missing '{target_anchor}' in filename: {basename}")

    prefix, rest = stem.split(method_anchor, 1)
    method_mode_block, suffix_block = rest.split(target_anchor, 1)

    # ---- prefix: typically "2D_20260410_fishes"
    prefix_parts = prefix.split("_")
    if len(prefix_parts) < 3:
        raise ValueError(
            "Prefix should contain at least dimension, date, and toy example: "
            f"{basename}"
        )

    dimension = prefix_parts[0]
    date = prefix_parts[1]
    toy_example = "_".join(prefix_parts[2:])

    # ---- method+mode: e.g. "affine-along", "rT_sT_tT-successive", "affine-successive-part1"
    # Check the longer suffixes before the shorter ones to avoid misparsing.
    if method_mode_block.endswith("-along"):
        method = method_mode_block[: -len("-along")]
        mode, part = "along", None
    elif method_mode_block.endswith("-successive-part1"):
        method = method_mode_block[: -len("-successive-part1")]
        mode, part = "successive", "part1"
    elif method_mode_block.endswith("-successive-part2"):
        method = method_mode_block[: -len("-successive-part2")]
        mode, part = "successive", "part2"
    elif method_mode_block.endswith("-successive"):
        method = method_mode_block[: -len("-successive")]
        mode, part = "successive", None
    elif method_mode_block.endswith("-only"):
        method = method_mode_block[: -len("-only")]
        mode, part = "only", None
    else:
        raise ValueError(
            "Could not parse mode from method block "
            f"'{method_mode_block}' in filename: {basename}"
        )

    # ---- target + trailing sample id
    # We know target belongs to a finite allowed set, so detect the longest matching prefix.
    target = _extract_target_prefix(suffix_block)
    if target is None:
        raise ValueError(
            "Could not parse target from suffix block "
            f"'{suffix_block}' in filename: {basename}"
        )

    remainder = suffix_block[len(target):]
    if remainder.startswith("_"):
        remainder = remainder[1:]

    sample_id = remainder if remainder else None

    if validate:
        if method not in ALLOWED_METHODS:
            raise ValueError(
                f"Unknown method '{method}' in filename: {basename}. "
                f"Allowed: {sorted(ALLOWED_METHODS)}"
            )
        if mode not in ALLOWED_MODES:
            raise ValueError(
                f"Unknown mode '{mode}' in filename: {basename}. "
                f"Allowed: {sorted(ALLOWED_MODES)}"
            )
        if target not in ALLOWED_TARGETS:
            raise ValueError(
                f"Unknown target '{target}' in filename: {basename}. "
                f"Allowed: {sorted(ALLOWED_TARGETS)}"
            )

    return ParsedRunFile(
        file_name=basename,
        method=method,
        mode=mode,
        target=target,
        toy_example=toy_example,
        dimension=dimension,
        date=date,
        sample_id=sample_id,
        part=part,
        extension=extension,
    )


def _extract_target_prefix(text: str) -> Optional[str]:
    """
    Return the longest allowed target that is a prefix of `text`.
    """
    matches = [t for t in ALLOWED_TARGETS if text == t or text.startswith(t + "_")]
    if not matches:
        return None
    return max(matches, key=len)


def parse_many(file_names: list[str], *, validate: bool = True) -> list[ParsedRunFile]:
    """
    Parse a list of filenames.
    """
    return [parse_experiment_filename(fn, validate=validate) for fn in file_names]


####################################
# Plotting



def plot_one_affine(file_p, ax, rainbow_img, load_path, *, show_titles=True):
    #
    mr = mt.load_optimize_geodesicShooting(
        file_p.file_name,
        path=load_path
    )

    affine = mr.mp.get_affine_deformator().detach().cpu()
    deform = mr.mp.get_deformation().to(affine.device)
    # deform = mr.mp.get_affine_deformation(deform).detach().cpu()
    deformator = mr.mp.get_deformator()

    img_rot = tb.imgDeform(mr.mp.image.detach().cpu(),affine,dx_convention='2square').cpu()
    source_rt = tb.imgDeform(mr.source.detach().cpu(),affine,dx_convention='2square').cpu()

    rainbow_d = tb.imgDeform(rainbow_img, deformator,dx_convention='2square').cpu()
    rainbow_d =  rainbow_d.transpose(1,3).transpose(1,2)[0]
    srt = tb.imCmp(mr.target, source_rt,method = 'seg')
    irt = tb.imCmp(mr.target, img_rot,method = 'seg')
    kwargs = {"origin": "upper", 'cmap': "gray"}
    ax[0,0].imshow(img_rot[0,0], **kwargs)
    if show_titles:
        ax[0,0].set_title("linear + diffeo")

    ax[0,1].imshow(source_rt[0,0], **kwargs)
    if show_titles:
        ax[0,1].set_title("only linear")

    ax[0,2].imshow(mr.mp.image[0,0].detach(), **kwargs)
    if show_titles:
        ax[0,2].set_title("only diffeo")

    ax[1,1].imshow(srt[0], **kwargs)
    if show_titles:
        ax[1,1].set_title("linear vs target")

    ax[1,0].imshow(irt[0], **kwargs)
    if show_titles:
        ax[1,0].set_title("diff + linear vs target")

    ax[1,2].imshow(rainbow_d, **kwargs)
    # tb.gridDef_plot_2d(affine, step = 40, ax = ax[1,2], color = None, alpha = .4)
    # tb.gridDef_plot_2d(deform.cpu(), step = 30, ax = ax[1,2])

    for axi in ax.flat:
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_xlabel("")
        axi.set_ylabel("")

def plot_one_successive(file_p, ax, rainbow_img, load_path, *, show_titles=True):
    #
    mr = mt.load_optimize_geodesicShooting(
        file_p.file_name,
        path=load_path
    )

    deform = mr.mp.get_deformation()
    # deform = mr.mp.get_affine_deformation(deform).detach().cpu()
    deformator = mr.mp.get_deformator()

    img = mr.mp.image.detach().cpu()
    source = mr.source.detach().cpu()

    rainbow_d = tb.imgDeform(rainbow_img, deformator,dx_convention='2square').cpu()
    rainbow_d =  rainbow_d.transpose(1,3).transpose(1,2)[0]
    srt = tb.imCmp(source,mr.target,method = 'seg')
    irt = tb.imCmp(img,mr.target,method = 'seg')
    kwargs = {"origin": "upper", 'cmap': "gray"}
    ax[0,0].imshow(img[0,0], **kwargs)
    ax[0,1].imshow(source[0,0], **kwargs)
    # ax[0,2].imshow(mr.mp.image[0,0].detach(), **kwargs)
    tb.gridDef_plot_2d(deform.cpu(), step = 30, ax = ax[0,2], origin="upper")

    ax[1,1].imshow(srt[0], **kwargs)
    ax[1,0].imshow(irt[0], **kwargs)
    ax[1,2].imshow(rainbow_d, **kwargs)
    # tb.gridDef_plot_2d(affine, step = 40, ax = ax[1,2], color = None, alpha = .4)

    for axi in ax.flat:
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_xlabel("")
        axi.set_ylabel("")

def plot_one_successive_v2(file_affine, file_diffeo, ax, rainbow_img, load_path, *, show_titles=True):
    #
    mr_a = mt.load_optimize_geodesicShooting(
        file_affine.file_name,
        path=load_path
    )
    mr_d = mt.load_optimize_geodesicShooting(
        file_diffeo.file_name,
        path=load_path
    )

    deform = mr_d.mp.get_deformation()
    affine = mr_a.mp.get_affine_deformator().detach().cpu()
    # deform = mr_d.mp.get_affine_deformation(deform).detach().cpu()
    deformator = mr_d.mp.get_deformator()
    img = mr_d.mp.image.detach().cpu()
    source = mr_d.source.detach().cpu()
    # ic(deformator.shape)
    # ic(deformator.min(), deformator.max())


    ic(file_affine.file_name, file_diffeo.file_name,
       mr_a.dx_convention, mr_d.dx_convention)
    rainbow_d = tb.imgDeform(rainbow_img, affine,dx_convention='2square').cpu()
    # ic(rainbow_d.shape)
    rainbow_d = tb.imgDeform(rainbow_d, deformator,dx_convention='pixel').cpu()
    rainbow_d =  rainbow_d.transpose(1,3).transpose(1,2)[0]
    # ic(rainbow_d.shape)
    srt = tb.imCmp(mr_d.target,source,method = 'seg')
    irt = tb.imCmp(mr_d.target,img,method = 'seg')
    kwargs = {"origin": "upper", 'cmap': "gray"}
    ax[0,0].imshow(img[0,0], **kwargs)
    ax[0,1].imshow(source[0,0], **kwargs)
    # ax[0,2].imshow(mr.mp.image[0,0].detach(), **kwargs)
    tb.gridDef_plot_2d(deform.cpu(), step = 30, ax = ax[0,2], origin="upper")

    ax[1,1].imshow(srt[0], **kwargs)
    ax[1,0].imshow(irt[0], **kwargs)
    ax[1,2].imshow(rainbow_d, **kwargs)
    # tb.gridDef_plot_2d(affine, step = 40, ax = ax[1,2], color = None, alpha = .4)

    for axi in ax.flat:
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_xlabel("")
        axi.set_ylabel("")

def _make_mode_grid(
    mode_files: list[ParsedRunFile],
    rainbow_img: torch.Tensor,
    load_path: str,
    plot_func: Callable,
    suptitle: str,
    prebuilt_lookup: dict | None = None,
):
    if prebuilt_lookup is not None:
        mode_lookup = prebuilt_lookup
    else:
        mode_lookup: dict[tuple[str, str], ParsedRunFile] = {}
        for p in mode_files:
            key = (p.method, p.target)
            prev = mode_lookup.get(key)
            if prev is None or (p.date or "") >= (prev.date or ""):
                mode_lookup[key] = p

    # Keep each 2x3 block close to a 3:2 aspect so inner images stay square.
    fig = plt.figure(figsize=(24, 16), constrained_layout=False)
    outer = fig.add_gridspec(
        len(METHOD_ORDER),
        len(TARGET_ORDER),
        wspace=0.03,
        hspace=0.04,
    )
    label_anchor_axes = {}
    block_axes = []

    for i, method in enumerate(METHOD_ORDER):
        for j, target in enumerate(TARGET_ORDER):
            # Leave a very thin but visible border between images in a block.
            inner = outer[i, j].subgridspec(2, 3, wspace=0.02, hspace=0.02)
            ax_block = np.empty((2, 3), dtype=object)
            for r in range(2):
                for c in range(3):
                    ax_block[r, c] = fig.add_subplot(inner[r, c])
                    ax_block[r, c].set_aspect("equal")
            block_axes.append(ax_block)

            if i == 0:
                label_anchor_axes[("col", j)] = ax_block[0, 1]
            if j == 0:
                label_anchor_axes[("row", i)] = ax_block[0, 0]

            run = mode_lookup.get((method, target))
            if run is None:
                for axi in ax_block.flat:
                    axi.axis("off")
                ax_block[0, 1].axis("on")
                ax_block[0, 1].text(
                    0.5,
                    0.5,
                    "missing",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="0.4",
                )
                ax_block[0, 1].set_xticks([])
                ax_block[0, 1].set_yticks([])
            else:
                plot_func(
                    run,
                    ax_block,
                    rainbow_img,
                    load_path,
                    show_titles=False,
                )

    fig.suptitle(suptitle, fontsize=17, y=0.998)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.03)

    for j, target in enumerate(TARGET_ORDER):
        ax = label_anchor_axes[("col", j)]
        pos = ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.01,
            TARGET_LABELS[target],
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
        )

    for i, method in enumerate(METHOD_ORDER):
        ax = label_anchor_axes[("row", i)]
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.014,
            (pos.y0 + pos.y1) / 2,
            METHOD_LABELS[method],
            ha="right",
            va="center",
            rotation=90,
            fontsize=11.5,
            fontweight="bold",
        )

    # Draw block borders after all layout adjustments so they align correctly.
    for ax_block in block_axes:
        pos_tl = ax_block[0, 0].get_position()
        pos_br = ax_block[1, 2].get_position()
        rect = Rectangle(
            (pos_tl.x0, pos_br.y0),
            pos_br.x1 - pos_tl.x0,
            pos_tl.y1 - pos_br.y0,
            transform=fig.transFigure,
            fill=False,
            lw=0.8,
            ec="0.6",
        )
        fig.add_artist(rect)

    return fig


def make_along_grid(
    along_files: list[ParsedRunFile],
    rainbow_img: torch.Tensor,
    load_path: str,
):
    return _make_mode_grid(
        mode_files=along_files,
        rainbow_img=rainbow_img,
        load_path=load_path,
        plot_func=plot_one_affine,
        suptitle="Along-Mode Registration: Method-by-Target Comparison",
    )


def make_successive_grid(
    successive_files: list[ParsedRunFile],
    rainbow_img: torch.Tensor,
    load_path: str,
):
    return _make_mode_grid(
        mode_files=successive_files,
        rainbow_img=rainbow_img,
        load_path=load_path,
        plot_func=plot_one_successive,
        suptitle="Successive-Mode Registration: Method-by-Target Comparison",
    )


def make_successive_grid_v2(
    successive_pairs: list[tuple[ParsedRunFile, ParsedRunFile]],
    rainbow_img: torch.Tensor,
    load_path: str,
):
    pairs_lookup = {(p1.method, p1.target): (p1, p2) for p1, p2 in successive_pairs}

    def _plot_pair(pair, ax_block, img, lp, *, show_titles=False):
        plot_one_successive_v2(*pair, ax_block, img, lp, show_titles=show_titles)

    return _make_mode_grid(
        mode_files=[],
        rainbow_img=rainbow_img,
        load_path=load_path,
        plot_func=_plot_pair,
        suptitle="Successive-Mode (Two-Part) Registration: Method-by-Target Comparison",
        prebuilt_lookup=pairs_lookup,
    )

def deduplicate_keep_latest(parsed: list[ParsedRunFile]) -> list[ParsedRunFile]:
    """Keep the most recent file per (method, mode, target, part) combination."""
    best: dict[tuple[str, str, str, Optional[str]], ParsedRunFile] = {}
    for p in parsed:
        key = (p.method, p.mode, p.target, p.part)
        prev = best.get(key)
        if prev is None or (p.date or "") >= (prev.date or ""):
            best[key] = p
    return list(best.values())


def warn_missing(
    parsed: list[ParsedRunFile],
    *,
    flag_successive_have_two_parts: bool = False,
) -> list[tuple[ParsedRunFile, ParsedRunFile]] | None:
    """Warn about missing (method, target) combinations for each mode present in parsed.
    Calls pair_successive_parts() internally and returns the resulting pairs when
    flag_successive_have_two_parts=True, so the caller does not need to call it again.

    For successive mode:
    - flag_successive_have_two_parts=False: checks only old-style files (part is None).
    - flag_successive_have_two_parts=True: builds pairs via pair_successive_parts()
      (which warns about orphans/date mismatches), then checks 16-combo completeness.
    """
    successive_pairs = None
    for mode in sorted({p.mode for p in parsed}):
        if mode == "successive":
            if flag_successive_have_two_parts:
                successive_pairs = pair_successive_parts(parsed)
                present = {(p1.method, p1.target) for p1, _ in successive_pairs}
            else:
                present = {(p.method, p.target) for p in parsed
                           if p.mode == "successive" and p.part is None}
        else:
            present = {(p.method, p.target) for p in parsed if p.mode == mode}

        missing = [
            (method, target)
            for method in METHOD_ORDER
            for target in TARGET_ORDER
            if (method, target) not in present
        ]
        if missing:
            lines = ", ".join(f"({m}, {t})" for m, t in missing)
            label = "Successive-parts" if (mode == "successive" and flag_successive_have_two_parts) else f"Mode '{mode}'"
            print(
                f"WARNING — {label}: {len(missing)}/16 combinations missing — {lines}",
                file=sys.stderr,
            )
    return successive_pairs


def load_parsed_files_from_path(load_path: str) -> list[ParsedRunFile]:
    """
    Scan *load_path* for files, attempt to parse each name, and return
    the successfully parsed ones.  Files whose names do not match the
    expected convention are silently skipped.
    """
    parsed = []
    for p in sorted(Path(load_path).iterdir()):
        if not p.is_file():
            continue
        try:
            parsed.append(parse_experiment_filename(p.name))
        except ValueError:
            pass
    return parsed


def pair_successive_parts(
    parsed: list[ParsedRunFile],
) -> list[tuple[ParsedRunFile, ParsedRunFile]]:
    """
    From successive files with parts, form (part1, part2) pairs by (method, target).
    Both parts must share the same date. Warns about:
      - orphaned files (only one part present for a given method/target)
      - date mismatches between part1 and part2
    Returns only complete, same-date pairs.
    Missing-combination warnings are handled by warn_missing().
    """
    part_files = [p for p in parsed if p.mode == "successive" and p.part is not None]
    by_key: dict[tuple[str, str], dict[str, ParsedRunFile]] = {}
    for p in part_files:
        by_key.setdefault((p.method, p.target), {})[p.part] = p

    pairs = []
    for (method, target), parts in by_key.items():
        has1, has2 = "part1" in parts, "part2" in parts
        if not (has1 and has2):
            missing_part = "part2" if has1 else "part1"
            print(
                f"WARNING — Successive-parts: method={method}, target={target} — "
                f"missing {missing_part}, skipping.",
                file=sys.stderr,
            )
            continue
        p1, p2 = parts["part1"], parts["part2"]
        if (p1.date or "") != (p2.date or ""):
            print(
                f"WARNING — Successive-parts: method={method}, target={target} — "
                f"date mismatch (part1={p1.date}, part2={p2.date}), skipping.",
                file=sys.stderr,
            )
            continue
        pairs.append((p1, p2))

    return pairs


if __name__ == '__main__':
    LOAD_PATH = "/home/turtlefox/Documents/11_metamorphoses/data/rigid_along_lddmm/"
    flag_successive_have_two_parts = True
    parsed = load_parsed_files_from_path(LOAD_PATH)
    parsed = deduplicate_keep_latest(parsed)

    successive_pairs = warn_missing(parsed, flag_successive_have_two_parts=flag_successive_have_two_parts)

    rainbow_img = open_rainbow_minifish()

    along_files = [p for p in parsed if p.mode == "along"]
    #%%
    # fig, ax = plt.subplots(2, 3)
    # plot_one_affine(along_files[0], ax, rainbow_img, LOAD_PATH)
    # plt.show()
    fig_along = make_along_grid(along_files, rainbow_img, LOAD_PATH)
    #
#     for i, (sp1, sp2) in enumerate(successive_pairs):
#         ic(i, sp1.file_name, sp2.file_name)
#     # raise ValueError("foknf,z")
#     if flag_successive_have_two_parts:
#         # fig, ax = plt.subplots(2, 3)
#         # i = 14
#         # plot_one_successive_v2(successive_pairs[i][0], successive_pairs[i][1], ax, rainbow_img, LOAD_PATH)
#         fig_successive = make_successive_grid_v2(successive_pairs, rainbow_img, LOAD_PATH)
#     else:
#         successive_files = [p for p in parsed if p.mode == "successive" and p.part is None]
# #         fig, ax = plt.subplots(2, 3)
# #         plot_one_successive(successive_files[0], ax, rainbow_img, LOAD_PATH)
#         # fig_successive = make_successive_grid(successive_files, rainbow_img, LOAD_PATH)


    plt.show()
