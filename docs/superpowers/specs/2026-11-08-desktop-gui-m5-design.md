# Design — M5: Desktop GUI (chat + observe the agent)

> Ngày: 2026-11-08 · Trạng thái: chờ duyệt
> Tiền đề: M1–M3 đã merge vào `main`. Milestone: **M5** trong `docs/roadmap.md`.

---

## 1. Intent và lý do

Hiện agent chỉ chạy qua CLI/script. M5 cho người dùng một **cửa sổ desktop** để: nhập mục tiêu (goal),
Start/Pause/Resume/Stop, **xem trực tiếp** screenshot browser + **log tương tác** (decision → action →
verify), trạng thái và **chi phí model**. Đây là lớp quan sát/điều khiển, không phải logic game.

## 2. Goals

- Cửa sổ Tkinter: ô **URL**, ô **goal**, **viewport (w×h)**, nút **Start / Pause / Resume / Stop**.
- **Preview screenshot real-time** (ảnh có lưới) của browser.
- **Log trực tiếp**: model nghĩ gì (reason), action (kể cả khi lỗi), verify, sự kiện runner.
- **Status**: state machine, chu kỳ, budget, màn hình hiện tại (`state`).
- **Chi phí model**: token in/out cộng dồn + ước tính $ nếu cấu hình giá; nếu không, hiện token.
- GUI **tự mở** Chromium riêng (headed) và `goto(URL)`.
- **Pause/Resume** giữa các chu kỳ (không cắt ngang một action).
- Toàn bộ logic **không-GUI** test offline; view Tk mỏng.

## 3. Non-goals

- Không chiến thuật game, **không** automation né chống-bot (từ chối tường minh).
- **Không** "chat" hướng dẫn giữa chừng (goal cố định trong một lần chạy); **không** approve/continue một
  `NEEDS_USER` (đó là trạng thái kết thúc — muốn tiếp thì sửa goal rồi Start lại).
- Không nhiều session/tab, không đóng gói .exe, không streaming log qua mạng.

## 4. Anti-goals

- UI thread bị block khi agent chạy.
- Lộ secret (API key) lên UI/log.
- Stop không đóng browser / để input treo.
- Pause cắt ngang giữa một action.
- Bắt buộc phải có GUI thì core mới chạy.

## 5. Constraints

- GUI dùng **Tkinter** (stdlib) + **Pillow** (đã có). Không thêm dependency nặng.
- Thay đổi core **nhỏ, additive**: `Session.pause()/resume()` + accessor read-only; `VLMClient` thêm
  `model`/`last_usage`; `SessionLogger.log_model`; `Session` gọi `log_action` cả khi thất bại.
- Không phá vỡ API hiện có.
- Offline-mặc-định: test phần không-GUI chạy `pytest`; view Tk smoke manual.

## 6. Approaches

- **A. Tkinter + runner thread + queue (đã chọn):** 0 dependency mới; tách phần test được.
- **B. PySide6/Qt:** loại (≈100MB; chưa chắc wheel cp314).
- **C. Web UI:** loại (thêm dependency + browser).

## 7. Thiết kế

### 7.1 Module

```
src/game_agent/
  gui_model.py     # logic hiển thị thuần (test offline)
  gui_runner.py    # runner thread + hàng đợi sự kiện (test với fakes)
  gui.py           # view Tkinter + entrypoint `python -m game_agent.gui`
```

### 7.2 Thay đổi core (additive)

- `Session.pause()/resume()`: cờ `self._paused`; đầu mỗi chu kỳ **trước** kiểm `_stopped`:
  `while self._paused and not self._stopped: await self._sleep(0.1)`. Dùng **`self._sleep`** (tiêm được)
  để test tất định; pause chỉ có hiệu lực giữa các chu kỳ.
- Accessor read-only: `Session.state` (state machine), `Session.cycles`, `Session.remaining_actions`.
- `VLMClient`: property `model` (tên model đã resolve); `last_usage` = **tổng** `usage` của (1–2) HTTP
  call trong một `decide` (không đếm thiếu lượt repair); reset đầu mỗi `decide`; `None` nếu không có usage.
- `SessionLogger.log_model(observation, model: str, usage: dict | None) -> None` → dòng `event="model"` với
  `prompt_tokens/completion_tokens/total_tokens`.
- `Session`: sau `decide`, nếu `getattr(vlm,"last_usage",None)` → `log_model(...)` (dùng `getattr` cho logger
  để giữ duck-typing). Và gọi `log_action(obs, action, result)` **cả khi `execute` lỗi** (log cố gắng + accepted).

### 7.3 Vòng đời (runner)

- `start(url, goal, viewport, grid_step)`: nếu còn controller cũ → đóng; mở controller mới (headed),
  `goto(url)`, tạo `Session` (+`_QueueLogger`), chạy `run(goal)` trong thread nền.
- **Run kết thúc (mọi lý do) → để browser MỞ** cho người dùng xem.
- **Stop**: `session.stop()` (không cắt ngang action/model call đang chạy; hiệu lực ở chu kỳ kế), rồi khi
  `run` trả về → **đóng browser** (release_all + close).
- **Đóng cửa sổ**: stop + đóng browser.
- Start lần sau: đóng controller cũ (nếu còn) rồi mở mới theo URL/viewport hiện tại.

### 7.4 Giao thức sự kiện runner→view (queue)

- Sự kiện logger (đúng dict JSONL): `observation`, `decision`, `action`, `verify`, `model`.
- Sự kiện runner: `{"event":"status","status":"running|paused|stopped|finished","detail":…}`,
  `{"event":"session_end","state":…,"cycles":…,"reason":…}`, `{"event":"error","message":…}`.
- `format_event(dict) -> str` xử lý **mọi** loại (kể cả lạ) một cách an toàn, không crash.
- Status bar lấy từ accessor của `Session` (state/cycles/remaining) + `UsageTracker`.

### 7.5 Model (`gui_model.py`) — thuần, test offline

- `format_event(event) -> str`; `preview_path(screenshot_path) -> str` (trả `<stem>.grid.png` nếu tồn tại,
  ngược lại ảnh gốc — xử lý cả `grid_step=None`/ảnh thiếu).
- `UsageTracker`: cộng dồn token; `cost_usd()` khi có `NVIDIA_PRICE_INPUT_PER_1M`/`NVIDIA_PRICE_OUTPUT_PER_1M`,
  nếu không → chỉ token.
- `state_label(...)`, `summarize_decision(...)`.

### 7.6 View (`gui.py`)

- Bố cục: hàng trên (URL, goal, w/h, Start/Pause/Resume/Stop); giữa: **preview ảnh** + **log**; dưới: **status**.
- `after(150, poll)`: rút queue → log + status; `observation` → preview `preview_path(...)` (Pillow→`ImageTk`).
- Không hiển thị API key. Entrypoint `python -m game_agent.gui` (`if __name__ == "__main__"`).

### 7.7 Cấu hình

`max_cycles` GUI mặc định 50; `grid_step` 100; poll 150ms. Không lưu goal/URL vào file.

### 7.8 Test

- **Offline (mặc định):**
  - `test_gui_model.py`: `format_event` cho từng loại (kể cả lạ); `preview_path` (có/không ảnh lưới, ảnh thiếu);
    `UsageTracker` cộng dồn + $ khi có giá, chỉ-token khi không.
  - `test_gui_runner.py` (fake controller + fake VLM): `start` chạy hết, queue nhận đủ
    `observation/decision/action/verify/model`, có `session_end`; `stop` dừng + đóng; `pause`/`resume` quan sát được.
  - `test_session_pause.py`: pause chặn giữa chu kỳ (fake sleep đảo `_paused`), resume tiếp; **stop thắng pause**.
  - `test_vlm_usage.py`: `last_usage` = tổng 2 call khi repair; `None` khi không có usage.
- **Marker `gui`** cho test cần display (loại khỏi mặc định qua `addopts`).
- **Manual:** `python -m game_agent.gui` chạy trên `examples/demo.html` (VLM thật, opt-in).

## 8. Quyết định đã chốt (từ builder-check)

1. **`NEEDS_USER` là trạng thái kết thúc**; "chat" = ô goal + log, **không** hướng dẫn giữa chừng.
2. **Vòng đời browser:** runner sở hữu open/close; run xong để mở; Stop đóng khi `run` trả về (không cắt
   ngang action); đóng cửa sổ = stop + close.
3. **Giao thức sự kiện:** logger events + `status`/`session_end`/`error`; `format_event` an toàn với loại lạ.
4. **Status lấy từ accessor Session** (`state`/`cycles`/`remaining_actions`) — thêm read-only, không suy diễn mơ hồ.
5. **Kênh lỗi:** `Session` gọi `log_action` cả khi lỗi; runner phát `error` (mở browser/URL); lý do cuối ở `session_end`.
6. **Model/usage:** `VLMClient.model` + `last_usage` = **tổng** các call; `None` → không có dòng `model`.
7. **Pause gate dùng `self._sleep(0.1)`** (tiêm được); stop kiểm trước trong điều kiện → stop thắng pause.
8. **Preview dùng ảnh lưới** qua `preview_path` (fallback ảnh gốc; xử lý `grid_step=None`).

## 9. Điều còn để builder quyết

- Kích thước/khung bố cục, màu sắc, font. `max_cycles`/poll interval. Không lưu cấu hình phiên.
