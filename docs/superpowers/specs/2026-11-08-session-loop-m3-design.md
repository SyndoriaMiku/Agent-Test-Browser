# Design — M3: closed-loop Session (observe → decide → act → verify)

> Ngày: 2026-11-08 · Trạng thái: chờ duyệt
> Tiền đề: M1 (controller) + M2 (VLM client) đã merge vào `main`.
> Milestone: **M3** trong `docs/roadmap.md`.

---

## 1. Intent và lý do

M1 cho "mắt + tay", M2 cho "bộ não". M3 ghép thành **vòng lặp khép kín**: quan sát → quyết định →
validate → thực thi → **xác minh**, có ngân sách, phanh an toàn và log. Đây là lần đầu hệ thống tự chạy
nhiều chu kỳ; mọi cơ chế chống loop vô hạn phải thật.

Bài học từ live test M2: model trả `Decision` hợp lệ nhưng **tọa độ ước lượng lệch** (click trượt).
Nên M3 cần (a) bước verify phát hiện trượt và (b) **phao tọa độ** (overlay lưới) để model đọc tọa độ chính xác hơn.

## 2. Goals

- `Session` điều phối vòng lặp, dùng đúng `StateMachine`, `SessionState`, `validate`, `SessionLogger`,
  `BrowserController`, `VLMClient`.
- **Phao tọa độ:** `add_coordinate_grid()` vẽ lưới + nhãn CSS-pixel lên screenshot (Pillow); không sửa ảnh gốc,
  **không đổi hệ tọa độ**.
- **Verify:** so ảnh trước/sau để biết action có hiệu lực; model tự báo `done`.
- **An toàn:** ngân sách action, trần retry liên tiếp → `NEEDS_USER`, trần chu kỳ, phát hiện action lặp,
  `release_all` khi **mọi** lối thoát, **`stop()` khẩn cấp**.
- **Log:** JSONL ghi observation/decision/action/verify, gắn `observation_id`.
- Test **offline** (fake controller + fake VLM + fake clock/sleep); 1 test **live** opt-in trên demo.

## 3. Non-goals

- Không chạy nền 24/7, không GUI, không đa tab, không nhiều tài khoản.
- Không học/fine-tune; không chiến thuật game cụ thể (M4).
- Không OCR/đọc DOM để quyết định; verify không cần ground truth.
- Không mở/đóng browser trong `Session` (caller sở hữu vòng đời browser).
- **Không** đổi chữ ký `VLMClient.decide`/`build_messages` (phao tọa độ tự mô tả trong ảnh).

## 4. Anti-goals

- Vòng lặp chạy vô hạn ở bất kỳ nhánh nào.
- Để input treo sau lỗi/dừng (không `release_all`).
- Grid overlay làm lệch tọa độ hoặc ghi đè ảnh gốc.
- Suite mặc định gọi mạng/API hoặc mở browser.
- Lộ secret trong log.

## 5. Constraints

- Thêm **một** dependency: `pillow` (đã xác nhận wheel cp314).
- `Session` nhận controller **đã mở**; không tự open/close.
- Tái dùng `Observation`, `Decision`, `Settings`, `SessionState`, `StateMachine`, `validate`, `SessionLogger`.
- Offline-mặc-định: marker `browser`/`live`; `addopts` loại cả hai.

## 6. Approaches (verify)

- **Ảnh-diff + model tự báo cáo (đã chọn).**
- Verifier cắm được: hoãn (YAGNI; M4).
- Hỏi VLM xác nhận mỗi chu kỳ: loại (tốn API).

## 7. Thiết kế

### 7.1 `overlay.py`

```python
def add_coordinate_grid(image_path, *, step: int = 100, scale: float = 1.0,
                        out_path=None) -> str: ...
```
- `step`/nhãn tính theo **CSS pixel**; bước pixel thật = `round(step * scale)` (để đúng khi `device_scale_factor != 1`).
- Vẽ lưới + nhãn tọa độ CSS, **kèm legend** (vd góc trên-trái: `"grid 100px · coords = CSS px"`) — để model
  tự hiểu ảnh có lưới mà **không cần đổi prompt** của `VLMClient`.
- Lưu ra file mới (mặc định `<stem>.grid.png`), trả đường dẫn; **không** sửa ảnh gốc; kích thước giữ nguyên.
- `step <= 0` hoặc `scale <= 0` → `ValueError`.

### 7.2 `session.py`

```python
@dataclass(frozen=True)
class SessionResult:
    state: State
    cycles: int
    reason: str
    last_decision: Decision | None

class Session:
    def __init__(self, controller, vlm, settings: Settings, *,
                 logger: SessionLogger | None = None,
                 max_consecutive_failures: int = 3,
                 repeat_limit: int = 3,
                 grid_step: int | None = 100,
                 clock: Clock | None = None,
                 sleep: Callable[[float], Awaitable[None]] | None = None): ...
    async def run(self, goal: str, max_cycles: int = 50) -> SessionResult: ...
    def stop(self) -> None: ...
```

`run` — bọc toàn bộ trong `try/finally`; `finally` **luôn** `await controller.release_all()` (best-effort).
Mỗi chu kỳ, kiểm `self._stopped` ở đầu → `STOPPED`:

1. **Nhịp:** nếu `last_action_at` cách `now` < `min_action_interval_ms`, `await self._sleep(remainder)`
   (fake sleep trong test đẩy `FakeClock`). Nhờ vậy `validate` không bị `RATE_LIMITED`.
2. `OBSERVING`: `obs = await controller.screenshot()`; nếu `grid_step` thì tạo ảnh lưới
   (`scale = obs.device_scale_factor`) và `vlm_obs = replace(obs, screenshot_path=gridded)`.
3. `DECIDING`: `decision = await vlm.decide(vlm_obs, goal, remaining)`; log decision.
4. `VALIDATING`: `validate(decision, snapshot, self._clock.now(), settings)`; reject → `NEEDS_USER`.
5. `action.type == "done"` → `DONE`. `"uncertain"` → `NEEDS_USER`.
6. `ACTING`: `await controller.execute(action)`; `session_state.record(action, now)`; log action.
7. **Lối thoát lỗi:** `VLMError` → `NEEDS_USER`. Lỗi từ `controller`/`overlay` (kể cả raw Playwright)
   → `record_failure()`; nếu ≥ `max_consecutive_failures` → `NEEDS_USER`; ngược lại ghi nhận, vòng tiếp
   (release_all do `finally`/lỗi thực thi đảm bảo).
8. `VERIFYING`: `after = await controller.screenshot()`; `changed = _image_changed(obs, after)`.
   - **Exempt:** `wait`/`move`/`release` không thể đổi pixel ⇒ **bỏ qua** luật failure-khi-không-đổi
     (coi là thành công/cập nhật nhịp). Các action còn lại (`click/drag/wheel/press/hold`): không đổi
     → `record_failure()`; ≥ trần → `NEEDS_USER`.
   - đổi → `record_success()`, reset đếm lặp.
9. **Lặp:** key `(action.model_dump_json(), hash(after))`; đếm lặp liên tiếp cùng key; đạt `repeat_limit`
   → `NEEDS_USER` (dùng bộ đếm riêng, không phụ thuộc trần failure).
10. Hết `max_cycles` hoặc `remaining_action_budget() <= 0` → `STOPPED`. Ngược lại vòng lại bước 1.

`stop()` set `self._stopped = True` (đồng bộ); chu kỳ kế phát hiện → `STOPPED` (vẫn `release_all`).

### 7.3 So ảnh

So **bytes** của hai PNG (Playwright tất định cùng nội dung) → `changed: bool`. Không cần Pillow.

### 7.4 Logger

Thêm `SessionLogger.log_verify(observation, changed: bool, note: str = "") -> None` (dòng JSON `event="verify"`).
Không đổi API cũ. Ngoài ra M3 dùng `log_observation/log_decision/log_action` sẵn có.

### 7.5 Cấu hình

`max_consecutive_failures`, `repeat_limit`, `grid_step`, `max_cycles`, `clock`, `sleep` là tham số **của phiên**
(`Session`/`run`), không nhồi vào `Settings`.

### 7.6 Test

- **Offline (mặc định):** `FakeController` (screenshot trỏ file PNG test tạo, đổi bytes theo script; `execute`
  ghi lại/raise được; `release_all` đếm) + `FakeVLM` (Decision theo script) + `FakeClock`/fake sleep. Kiểm:
  - happy path: click → ảnh đổi → done ⇒ `DONE`, có log verify.
  - ảnh không đổi liên tiếp (action loại ảnh-hưởng) ⇒ trần ⇒ `NEEDS_USER`.
  - `wait`/`move` không-đổi ⇒ **không** tính failure.
  - action lặp cùng key `repeat_limit` lần ⇒ `NEEDS_USER` (đặt `max_consecutive_failures` cao để cô lập).
  - `execute`/`screenshot` raise ⇒ `release_all` được gọi; cap ⇒ `NEEDS_USER`.
  - `vlm.decide` raise `VLMError` ⇒ `NEEDS_USER`.
  - hết budget ⇒ `STOPPED`; `stop()` giữa vòng ⇒ dừng chu kỳ kế.
  - `release_all` được gọi ở **mọi** lối thoát (DONE/NEEDS_USER/STOPPED/exception).
  - `grid_step` ⇒ `vlm.decide` nhận `screenshot_path` là ảnh lưới (khác gốc); `observation_id` không đổi.
  - logger đủ `observation/decision/action/verify` gắn `observation_id`.
- **`overlay` (offline):** PNG nhỏ → ra file mới tồn tại, cùng kích thước, khác bytes; `step<=0`/`scale<=0` → `ValueError`.
- **Live (opt-in, `live`+`browser`):** browser thật + VLM thật, `run(goal="Click the Start button, then press the up arrow")`
  trên demo; assert kết thúc `DONE` (hoặc `NEEDS_USER` nếu hết `max_cycles`); skip nếu thiếu key/model/browser.

## 8. Quyết định đã chốt (từ builder-check)

1. **Nhịp & clock:** tiêm `clock` + `sleep`; **chờ trước validate** đủ `min_action_interval_ms`; fake sleep đẩy `FakeClock`.
   `RATE_LIMITED` không còn là đường chết thông thường.
2. **Lỗi không thuộc `execute`:** `VLMError` → `NEEDS_USER`; lỗi controller/overlay → tính vào failure (cap → `NEEDS_USER`).
   `run()` có `finally` **luôn** `release_all()`.
3. **Overlay & dsf:** thêm `scale`; bước pixel = `step*scale`, nhãn theo CSS px; `Session` truyền `device_scale_factor`.
4. **Không đổi prompt VLM:** legend nằm trong ảnh lưới ⇒ giữ M3 trong `overlay.py`/`session.py`/logger.
5. **`release_all` ở mọi lối thoát** (qua `finally`).
6. **Luật lặp:** bộ đếm riêng theo cặp `(action, hash(ảnh))`, tham số `repeat_limit`; test cô lập bằng cách nâng trần failure.
7. **Miễn verify-failure cho `wait`/`move`/`release`** (không thể đổi pixel).

## 9. Điều còn để builder quyết

- Câu "nudge" khi ảnh không đổi (M3 tối thiểu: ghi chú, không đổi goal).
- `step` lưới mặc định (100), cỡ/độ đậm nhãn và legend.
- `max_cycles` mặc định (50).
