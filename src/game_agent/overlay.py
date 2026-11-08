"""Coordinate grid overlay: draw a CSS-pixel grid onto a screenshot.

The grid lets a vision model read coordinates directly off a scaled capture
and use them as CSS pixels in browser space. Every grid line carries a label
— x-axis labels along the bottom edge, y-axis labels along the left edge —
and the legend sits in the top-right corner on an opaque backing, so labels
never collide with it and grid lines never reduce its legibility. The source
image is never modified.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_LINE_COLOR = (255, 0, 255)
_STROKE_COLOR = (0, 0, 0)
_LEGEND_BG = (0, 0, 0)
_PAD = 3  # keep text this many px inside the image edges


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: load_default() takes no size
        return ImageFont.load_default()


def grid_lines(width: int, height: int, step: int, scale: float) -> tuple[list[int], list[int]]:
    """Pixel positions of the vertical (``xs``) and horizontal (``ys``) grid lines.

    Lines sit at ``0, pixel_step, 2*pixel_step, ...`` image px with
    ``pixel_step = max(1, round(step * scale))`` (never below 1), inclusive of
    the right/bottom edges when ``width`` / ``height`` are multiples of it.
    """
    pixel_step = max(1, round(step * scale))
    return (
        list(range(0, width + 1, pixel_step)),
        list(range(0, height + 1, pixel_step)),
    )


def format_label(pixel: int, scale: float) -> str:
    """CSS-px label for a line at ``pixel`` image px: int when whole, else at most 2 decimals."""
    css = round(pixel / scale, 2)
    return str(int(css)) if css.is_integer() else str(css)


def add_coordinate_grid(
    image_path: str | Path,
    *,
    step: int = 100,
    scale: float = 1.0,
    out_path: str | Path | None = None,
) -> str:
    """Write a copy of ``image_path`` with a coordinate grid drawn onto it.

    ``step`` (grid spacing) and the drawn labels are in CSS pixels; ``scale``
    converts CSS px to image px, so lines are spaced ``max(1, round(step *
    scale))`` image px apart and a line at image position ``i`` is labelled
    ``i / scale``. Every line gets a label: x-axis labels run along the bottom
    edge and y-axis labels along the left edge, each draw origin clamped so the
    text stays inside the image. The legend naming the units is drawn in the
    top-right corner on an opaque backing. The source file is never written
    to; the result goes to ``out_path`` or ``<stem>.grid.png`` next to the
    source. Returns the output path as ``str``.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    src = Path(image_path)
    dest = Path(out_path) if out_path is not None else src.with_name(f"{src.stem}.grid.png")
    if dest.resolve() == src.resolve():
        raise ValueError(f"out_path must differ from the source image: {src}")

    font_size = max(11, round(11 * scale))
    font = _load_font(font_size)
    stroke_width = max(1, font_size // 8)

    with Image.open(src) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        width, height = im.size
        draw = ImageDraw.Draw(im)

        def draw_text(xy: tuple[int, int], text: str) -> None:
            draw.text(
                xy, text, fill=_LINE_COLOR, font=font,
                stroke_width=stroke_width, stroke_fill=_STROKE_COLOR,
            )

        xs, ys = grid_lines(width, height, step, scale)
        for x in xs:
            draw.line([(x, 0), (x, height)], fill=_LINE_COLOR, width=1)
        for y in ys:
            draw.line([(0, y), (width, y)], fill=_LINE_COLOR, width=1)

        # X-axis labels along the bottom edge, y-axis labels along the left
        # edge: the label bands stay clear of the top-right legend, so no
        # label is ever skipped. Draw origins are clamped inside the image.
        for x in xs:
            label = format_label(x, scale)
            box = draw.textbbox((0, 0), label, font=font, stroke_width=stroke_width)
            tx = max(min(x + _PAD, width - _PAD - box[2]), 0)  # clamp at the right edge
            ty = max(height - _PAD - box[3], 0)                # hug the bottom edge
            draw_text((tx, ty), label)
        for y in ys:
            label = format_label(y, scale)
            box = draw.textbbox((0, 0), label, font=font, stroke_width=stroke_width)
            ty = max(min(y + _PAD, height - _PAD - box[3]), 0)  # clamp at the bottom edge
            draw_text((_PAD, ty), label)

        # Legend top-right on an opaque backing so grid lines can't bleed through.
        legend = f"grid {step}px · coords = CSS px"
        box = draw.textbbox((0, 0), legend, font=font, stroke_width=stroke_width)
        lx = max(width - _PAD - box[2], 0)  # right-align the ink in the corner
        ly = _PAD
        draw.rectangle(
            [
                max(lx + box[0] - 2, 0),
                max(ly + box[1] - 2, 0),
                min(lx + box[2] + 2, width),
                min(ly + box[3] + 2, height),
            ],
            fill=_LEGEND_BG,
        )
        draw_text((lx, ly), legend)

        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest)

    return str(dest)
