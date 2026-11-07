import pytest
from PIL import Image
from game_agent.overlay import add_coordinate_grid

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
