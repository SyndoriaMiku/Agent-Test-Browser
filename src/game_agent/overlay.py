"""Coordinate grid overlay: draw a CSS-pixel grid onto a screenshot.

The grid lets a vision model read coordinates directly off a scaled capture
and use them as CSS pixels in browser space. Labels and the legend state the
unit; the source image is never modified.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_LINE_COLOR = (255, 0, 255)
_STROKE_COLOR = (0, 0, 0)


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: load_default() takes no size
        return ImageFont.load_default()


def _format_css(value: float) -> str:
    """Render a CSS coordinate, as int when whole."""
    return str(int(value)) if value.is_integer() else f"{value:g}"


def add_coordinate_grid(
    image_path: str | Path,
    *,
    step: int = 100,
    scale: float = 1.0,
    out_path: str | Path | None = None,
) -> str:
    """Write a copy of ``image_path`` with a coordinate grid drawn onto it.

    ``step`` (grid spacing) and the drawn labels are in CSS pixels; ``scale``
    converts CSS px to image px, so lines are spaced ``round(step * scale)``
    image px apart and a line at image position ``i`` is labelled ``i / scale``.
    A legend naming the units is drawn at the top-left. The source file is
    never written to; the result goes to ``out_path`` or ``<stem>.grid.png``
    next to the source. Returns the output path as ``str``.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    src = Path(image_path)
    dest = Path(out_path) if out_path is not None else src.with_name(f"{src.stem}.grid.png")
    if dest.resolve() == src.resolve():
        raise ValueError(f"out_path must differ from the source image: {src}")

    pixel_step = round(step * scale)
    if pixel_step < 1:
        raise ValueError(
            f"step * scale = {step * scale} rounds to {pixel_step} image px; grid spacing must be >= 1 px"
        )

    font_size = max(11, round(11 * scale))
    font = _load_font(font_size)
    stroke_width = max(1, font_size // 8)

    with Image.open(src) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        width, height = im.size
        draw = ImageDraw.Draw(im)

        legend = f"grid {step}px · coords = CSS px"
        legend_box = draw.textbbox((3, 3), legend, font=font, stroke_width=stroke_width)
        legend_right, legend_bottom = legend_box[2], legend_box[3]

        for x in range(0, width, pixel_step):
            draw.line([(x, 0), (x, height)], fill=_LINE_COLOR, width=1)
            if x and x + 3 >= legend_right:  # skip 0 and labels colliding with the legend
                label = _format_css(x / scale)
                draw.text(
                    (x + 3, 3), label, fill=_LINE_COLOR, font=font,
                    stroke_width=stroke_width, stroke_fill=_STROKE_COLOR,
                )
        for y in range(0, height, pixel_step):
            draw.line([(0, y), (width, y)], fill=_LINE_COLOR, width=1)
            if y and y + 3 >= legend_bottom:  # skip 0 and labels colliding with the legend
                label = _format_css(y / scale)
                draw.text(
                    (3, y + 3), label, fill=_LINE_COLOR, font=font,
                    stroke_width=stroke_width, stroke_fill=_STROKE_COLOR,
                )

        draw.text(
            (3, 3), legend, fill=_LINE_COLOR, font=font,
            stroke_width=stroke_width, stroke_fill=_STROKE_COLOR,
        )

        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest)

    return str(dest)
