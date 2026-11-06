# Design — M1: Real Browser Controller (browser-only PoC)

> Ngày: 2026-11-06 · Trạng thái: chờ duyệt
> Tiền đề: skeleton đã merge vào `main`. Spec gốc: `2026-11-05-browser-game-agent-design.md`.
> Milestone: **M1** trong `docs/roadmap.md`.

---

## 1. Intent và lý do

Skeleton đã khoá các invariant (schema/validator/state/log) nhưng `controller.py` vẫn là stub.
M1 biến nó thành thật: agent mở được một browser **riêng**, chụp screenshot ở viewport đã biết, và
thực thi chuột/bàn phím **an toàn**. Đây là mốc "browser mở đúng + tọa độ click trúng" trong thứ tự
ưu tiên của plan gốc — chưa cần VLM.

## 2. Goals

- `BrowserController` hiện thực thật bằng **Playwright async**:
  `open()`, `goto(url)`, `screenshot() -> Observation`, `execute(Action)`, `release_all()`, `close()`.
- Browser **riêng biệt**: bundled Chromium; headed mặc định; viewport `1280×720`; `deviceScaleFactor=1`;
  `user-data-dir` riêng (không đụng Chrome/Edge cá nhân của người dùng).
- Mô hình tọa độ: screenshot và tọa độ chuột cùng hệ **CSS pixel**.
- **Ledger giữ phím/nút** + `release_all()` bảo đảm không để input "treo" sau lỗi (contract từ Task 9).
- Trang demo local `examples/demo.html` với target tọa độ đã biết + đổi trạng thái khi click (verify được).
- Test browser **opt-in** (marker `browser`); suite `pytest` mặc định vẫn offline và xanh.
- Cập nhật README mô tả M1.

## 3. Non-goals

- Không VLM, không vòng lặp `observe→decide→act→verify`, không state-runner (M2/M3).
- Không đọc DOM/accessibility tree để quyết định, không chiến thuật game (M4).
- Không Playwright MCP, không đa tab, không GUI.

## 4. Anti-goals (làm xong vẫn tính thất bại nếu…)

- `pytest` mặc định mở browser (phá tính offline/deterministic của suite).
- Để lại một tiến trình browser, một phím, hoặc một nút chuột bị giữ sau khi lỗi/đóng.
- Chạm vào profile Chrome/Edge cá nhân của người dùng.
- Click dựa trên giả định viewport khác với ảnh screenshot (lệch hệ tọa độ).

## 5. Constraints

- Chỉ dùng dependency đã có: `playwright`. Không thêm thư viện nặng.
- Tái sử dụng `Observation` (Task 4) và `Action` (Task 3); controller **chỉ nhận action đã qua validator**,
  nhưng vẫn tự kiểm tra phòng thủ (defense in depth).
- Python 3.14.7; Windows/PowerShell; Chromium tải bằng `playwright install chromium` (người dùng đã đồng ý).
- Không ghi secret/cookie/token vào log.

## 6. Approaches

- **A. Controller thật + trang demo local (đã chọn):** đúng kiến trúc đích, chạy lại được, verify được.
- **B. Spike vứt đi:** loại — không tiến kiến trúc.
- **C. Playwright MCP qua OpenCode:** loại — plan chọn Python controller; CLI MCP có bug `--`.

## 7. Thiết kế

### 7.1 Nguồn sự thật & constructor

`Settings` (đã có) là **nguồn duy nhất** cho viewport, `device_scale_factor`, `allowed_keys`, `allowed_buttons`.
`BrowserOptions` **không** lặp lại viewport — nó chỉ giữ hạ tầng:

```python
@dataclass(frozen=True)
class BrowserOptions:
    headless: bool = False
    user_data_dir: str = ".browser-profile"
    screenshot_dir: str = "screenshots"
    browser_channel: str | None = None   # None = bundled Chromium; "chrome"/"msedge" = hệ thống

class BrowserController:
    def __init__(self, settings: Settings, options: BrowserOptions | None = None,
                 clock: Clock | None = None): ...
```

Nhờ vậy không thể có chuyện validator cho `x ≤ 1920` trong khi browser rộng 1280. Context viewport cố
định từ `settings`; controller **không bao giờ** đổi viewport giữa phiên.

### 7.2 Vòng đời & điều hướng

- `open()`: `playwright.async_api` → `chromium.launch_persistent_context(user_data_dir, headless,
  viewport=..., device_scale_factor=..., channel=...)`. Giữ context + page. **Không** điều hướng.
- `goto(url)`: `await page.goto(url, wait_until="load")`. Cách gọi tới trang là tường minh, không nhồi vào `open()`.
- `page` (property): trả `Page` của Playwright — escape hatch cho test/observation layer (`page.evaluate`).
- `close()`: `release_all()` rồi đóng context và Playwright, luôn qua `try/finally`.

### 7.3 `screenshot() -> Observation`

`page.screenshot()` (viewport, không full-page) → lưu `screenshots/<session_id>/<observation_id>.png`.
`Observation(observation_id, screenshot_path, viewport_width, viewport_height, device_scale_factor,
timestamp=clock.now())`; viewport lấy từ `settings`, `observation_id` sinh trong phiên.

### 7.4 `execute(action)` — dispatch

| type | hành vi |
|---|---|
| `click` | `mouse.click(x, y, button, click_count)` |
| `move` | `mouse.move(x, y)`; cập nhật vị trí con trỏ |
| `press` | `keyboard.press(_key(k))` |
| `hold` | `down` → `asyncio.sleep` → `up`, **try/finally** |
| `release` | `up` phím/nút; **idempotent** (giải phóng thứ không giữ = no-op) |
| `drag` | `move(start)` → `down` → `move(end, steps=…)` → `up`; **nội suy theo `duration_ms`**, đi qua ledger |
| `wheel` | `move` tới vị trí con trỏ đã lưu (mặc định tâm viewport) rồi `mouse.wheel(dx, dy)` |
| `wait` | `asyncio.sleep(duration_ms/1000)` |
| `done` / `uncertain` | **không thực thi** — raise `ActionNotAllowed` (đây là tín hiệu của loop, không phải UI action) |

**Con trỏ:** controller theo dõi vị trí con trỏ cuối (khởi tạo tâm viewport) để `wheel` không cuộn ở `(0,0)`.

**Ledger:** tập hợp phím đang giữ + nút chuột đang giữ. `down` ghi vào ledger *trước* khi await; `up`
thành công mới xoá. `hold` dùng `try/finally` để up; nếu bản thân `up` lỗi (kể cả cancel), entry vẫn ở
ledger để `release_all()` thử lại. **`hold` là nguyên tử** (không có latch qua nhiều `execute`); ledger là
lưới an toàn khi lỗi, không phải cơ chế giữ phím lâu dài.

### 7.5 Defense in depth & exception

Controller tự kiểm (không thay validator, không lặp budget/rate vốn thuộc `SessionState`/validator):
allowlist key/button, bounds tọa độ theo viewport của chính nó, trần `duration_ms`. Vi phạm → raise.

```python
class ControllerError(Exception): ...
class ActionNotAllowed(ControllerError): ...   # vi phạm policy/allowlist
class ActionExecutionError(ControllerError): ...  # lỗi khi tương tác browser
```

### 7.6 Key vocabulary (bug thật phát hiện ở builder-check)

`Settings.allowed_keys` dùng tên thân thiện (`up`, `down`, `space`, `enter`, `escape`, `shift`, `tab`) —
Playwright yêu cầu tên chuẩn (`ArrowUp`, `Space`, `Enter`, `Escape`, `Shift`, `Tab`). Controller sở hữu
một **bảng dịch** `_KEY_MAP` (agent-name → Playwright-name); ký tự đơn đi thẳng. Nhờ vậy giữ nguyên
vocabulary cho VLM/M2 và không đổi `config.py`. Key không map được và không phải ký tự đơn → để Playwright báo lỗi.

### 7.7 Trang demo `examples/demo.html`

Tĩnh, không mạng. Target có tọa độ cố định: nút `#start`; một phần tử đổi text/`data-*` khi click; một
`window.__demoState` cập nhật để test đọc bằng `page.evaluate`. Verify: click trúng → trạng thái đổi.

### 7.8 Test

- Marker `browser` đăng ký trong `pyproject.toml`; `addopts` mặc định `-m "not browser"`.
- `tests/test_browser_controller.py` (marker `browser`): mở browser (headless cho nhanh), screenshot đúng
  kích thước, `goto(demo)` + click target → `__demoState` đổi, `press` một key đã map, `hold` rồi
  `release_all`, và ca lỗi giữa `hold` (monkeypatch `keyboard.up` lỗi một lần) → `release_all` để ledger sạch.
- Nếu browser chưa cài: `pytest.skip` với thông báo rõ (không hard-fail), để repo fresh vẫn không đỏ.

## 8. Quyết định đã chốt (từ builder-check)

1. **Ai sở hữu viewport/dsf/allowlist:** `Settings`, không phải `BrowserOptions`; controller nhận `settings`.
2. **Vocabulary phím:** bảng dịch trong controller (§7.6); `config.py` giữ nguyên.
3. **Điều hướng:** thêm `goto(url)` + property `page`; `open()` không điều hướng.
4. **`hold`:** nguyên tử; ledger là lưới an toàn; `release` idempotent; test unwinding bằng monkeypatch.
5. **Defense-in-depth:** allowlist + bounds + trần duration; ra `ControllerError`/`ActionNotAllowed`/`ActionExecutionError`.
6. **Chromium:** cài `playwright install chromium`; test thiếu browser → skip có thông báo; hỗ trợ `browser_channel`.
7. **`drag`:** nội suy theo `duration_ms` qua `mouse.move(steps=…)`; đi qua ledger.
8. **`wheel`:** move tới vị trí con trỏ đã lưu (mặc định tâm viewport) trước khi cuộn.

## 9. Điều còn để builder quyết

- Tọa độ/target cụ thể của `examples/demo.html`.
- Cách sinh `session_id` (timestamp/monotonic) — miễn ổn định trong một phiên.
- `steps` cho `drag` (mặc định hợp lý, tinh chỉnh sau).
