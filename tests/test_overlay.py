import pytest
from PIL import Image, ImageDraw
from game_agent.overlay import add_coordinate_grid, format_label, grid_lines

def make_png(tmp_path, w=200, h=120):
    p = tmp_path / "shot.png"
    Image.new("RGB", (w, h), (255, 255, 255)).save(p)
    return p

def test_grid_writes_new_file_same_size(tmp_path):
    src = make_png(tmp_path)
    before = src.read_bytes()
    out = add_coordinate_grid(src, step=50)
    assert out != str(src)
    from pathlib import Path
    assert Path(out).exists()
    assert Image.open(out).size == (200, 120)
    assert src.read_bytes() == before              # original untouched
    assert Path(out).read_bytes() != before        # grid actually drawn

def test_grid_scale_keeps_labels_in_css_px(tmp_path):
    src = make_png(tmp_path, w=400, h=200)
    out = add_coordinate_grid(src, step=100, scale=2.0)   # 400px image == 200 CSS px
    assert Image.open(out).size == (400, 200)             # size unchanged

def test_grid_rejects_bad_params(tmp_path):
    src = make_png(tmp_path)
    with pytest.raises(ValueError):
        add_coordinate_grid(src, step=0)
    with pytest.raises(ValueError):
        add_coordinate_grid(src, scale=0)

def test_grid_out_path_and_never_source(tmp_path):
    src = make_png(tmp_path)
    dest = tmp_path / "sub" / "gridded.png"
    out = add_coordinate_grid(src, step=50, out_path=dest)
    assert out == str(dest)
    assert dest.exists()
    with pytest.raises(ValueError):   # refusing to write the source itself
        add_coordinate_grid(src, out_path=src)

def test_grid_lines_are_multiples_of_pixel_step():
    xs, ys = grid_lines(400, 200, 100, 2.0)
    assert xs == [0, 200, 400]        # multiples of round(100 * 2.0) = 200, right edge included
    assert ys == [0, 200]
    xs, ys = grid_lines(200, 120, 50, 1.0)
    assert xs == [0, 50, 100, 150, 200]
    assert ys == [0, 50, 100]

def test_grid_lines_tiny_step_clamps_to_one_pixel(tmp_path):
    xs, ys = grid_lines(20, 20, 1, 0.5)   # round(1 * 0.5) = 0 -> clamped to 1, not an error
    assert xs == list(range(21))
    assert ys == list(range(21))
    src = make_png(tmp_path, w=20, h=20)
    out = add_coordinate_grid(src, step=1, scale=0.5)
    assert Image.open(out).size == (20, 20)

def test_format_label_is_css_px():
    assert format_label(200, 2.0) == "100"
    assert format_label(100, 1.0) == "100"
    assert format_label(0, 3.0) == "0"
    assert format_label(100, 3.0) == "33.33"   # non-whole: rounded to at most 2 decimals

def test_every_grid_line_gets_a_label(tmp_path, monkeypatch):
    captured = []
    real_text = ImageDraw.ImageDraw.text

    def spy(draw, xy, text, *args, **kwargs):
        captured.append(str(text))
        return real_text(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    src = make_png(tmp_path, w=400, h=200)
    add_coordinate_grid(src, step=100, scale=2.0)

    xs, ys = grid_lines(400, 200, 100, 2.0)
    expected = [format_label(x, 2.0) for x in xs]
    expected += [format_label(y, 2.0) for y in ys]
    expected.append("grid 100px · coords = CSS px")
    # exact multiset match: every line labelled exactly once, nothing skipped
    assert sorted(captured) == sorted(expected)

def test_labels_clamped_inside_image_and_legend_top_right(tmp_path):
    src = make_png(tmp_path, w=400, h=200)
    out = Image.open(add_coordinate_grid(src, step=100, scale=2.0)).convert("RGB")
    px = out.load()

    def region_has(x0, y0, x1, y1, match):
        return any(match(px[x, y]) for y in range(y0, y1) for x in range(x0, x1))

    # the x=400 edge label ("200") is clamped left, so its ink is inside the
    # bottom-right corner instead of being clipped off the right edge
    assert region_has(350, 170, 400, 200, lambda p: p[0] > 200 and p[2] > 200 and p[1] < 100)
    # the legend sits top-right on an opaque black backing
    assert region_has(350, 0, 400, 30, lambda p: p == (0, 0, 0))
