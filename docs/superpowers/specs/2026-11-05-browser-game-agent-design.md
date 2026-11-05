# Design — Visual Browser Game AI Agent (Skeleton, approach C)

> Ngày: 2026-11-05 · Trạng thái: chờ duyệt
> Nguồn gốc: `browser_game_ai_agent_plan.md` (research brief) + kết quả Milestone 0.
> Phạm vi tài liệu này: **bộ khung (skeleton) + kế hoạch triển khai**, không phải toàn bộ agent hoàn chỉnh.

---

## 1. Intent và lý do

Xây nền móng cho một agent quan sát browser game qua screenshot, quyết định bằng VLM, rồi thực thi
chuột/bàn phím và xác minh kết quả — vòng lặp `observe → decide → act → verify`.

Lý do làm skeleton trước: các **invariant cốt lõi** (schema action, validator, state machine, log)
quyết định độ an toàn và khả năng debug của cả hệ thống. Chúng **kiểm thử được mà không cần browser,
không cần API, không tốn tiền**. Dựng và khóa chúng trước khi chạm tới Playwright/VLM giúp các
milestone sau không phải viết lại nền.

## 2. Goals

- Có cấu trúc package rõ ràng phản ánh kiến trúc đích (Observation / Agent / Validator / Controller / State).
- Action schema + validator + state machine + logger: **đã hiện thực thật, có unit test xanh**.
- Controller (Playwright) và VLM client: **stub có interface rõ**, chờ M1/M2, không code giả.
- Có `pyproject.toml`, `.gitignore`, `.env.example`, README, và roadmap milestone.
- Chạy `pytest` xanh offline, không cần mạng.

## 3. Non-goals

- Không mở browser thật, không gọi VLM API trong phạm vi skeleton (đó là M1/M2).
- Không viết logic game cụ thể, không hard-code tọa độ/nút/chiến thuật.
- Không GUI, không long-running mode, không đa tab, không Rust.
- Không cấu hình OpenCode/Playwright MCP thay đổi máy người dùng trong bước này.

## 4. Anti-goals (làm xong vẫn tính là thất bại nếu…)

- Skeleton chứa module rỗng/suy đoán cho tính năng chưa tới lúc làm.
- Test phải cần browser/API/mạng mới chạy.
- Validator cho phép action ngoài viewport/ngoài allowlist lọt qua.
- Có state mà máy không thể rời (treo không có đường thoát).

## 5. Constraints (từ Milestone 0 — đã xác minh)

| Điều | Giá trị | Ảnh hưởng |
|---|---|---|
| OpenCode | v2.2.0 → **schema V2** | server config phải ở `mcp.servers.<name>`, dùng `disabled` |
| CLI `mcp add` | **bug: không tôn trọng `--`** (tái hiện được) | cấu hình MCP phải **sửa file tay**, không dùng CLI cho server có tham số gạch đầu dòng |
| Python | 3.14.7 | chọn dependency có wheel cho cp314; tránh thư viện lỗi thời |
| `uv` / `pipx` | chưa cài | dùng `python -m venv` + `pip` mặc định (không bắt cài thêm) |
| Provider | config máy hiện tại **không có OpenRouter**, đi qua proxy nội bộ | VLM client phải trừu tượng, không hard-code provider |
| Playwright MCP `@latest` | có `--caps vision`, `--isolated`, `--viewport-size`, `--image-responses`, `--snapshot-boxes` | đủ cho browser riêng + tọa độ, dùng ở M1/M4 |

## 6. Approaches đã cân nhắc

- **A — Skeleton đầy đủ mọi module stub:** loại. Tạo code rỗng suy đoán, trái nguyên tắc.
- **B — Vertical slice M1 trước (browser PoC):** loại làm hướng chính. Chạy được sớm nhưng chưa khóa
  được invariant cốt lõi; dễ phải viết lại khi thêm schema/validator.
- **C — Khung kiến trúc lõi + test offline (đã chọn):** hiện thực thật schema/validator/state/log,
  stub rõ cho controller/VLM. Cân bằng giữa đúng-kiến-trúc và không-đầu-cơ.

## 7. Kiến trúc trong phạm vi skeleton

```
Decision (VLM)  ──▶  ActionValidator  ──▶  StateMachine  ──▶  Controller(Playwright)  [stub]
      ▲                     │  ▲                 │
      │                     │  └─ SessionState ──┘   (counters + injected clock)
      └── VLMClient [stub]  └── ActionSchema
                                    ▲
                          Observation (id + screenshot path + viewport)  [value type]

Không có driver vòng lặp trong skeleton — module được test độc lập. `Session` điều phối cả vòng
observe→decide→act→verify thuộc M3.
```

Ranh giới trách nhiệm (giữ đúng plan):

- **VLMClient (stub, `async`):** nhận observation + goal → trả `Decision` có cấu trúc. Không chạy code tùy ý.
- **ActionValidator:** **hàm thuần**, nhận `(decision, snapshot)` trong đó `snapshot` là ảnh chụp bất biến
  của `SessionState`; từ chối tọa độ ngoài viewport, type/param không hợp lệ, vượt rate/budget, confidence thấp.
- **SessionState:** **sở hữu** mọi bộ đếm liên-action (thời điểm action trước, tổng action, thời gian `hold`
  cộng dồn, số retry) và một **đồng hồ tiêm vào (injected clock)**; validator không tự đọc giờ hệ thống.
- **StateMachine:** chỉ cho phép chuyển trạng thái hợp lệ; luôn có đường thoát khỏi mọi state.
- **Controller (stub, `async`):** thực thi allowlist action; theo dõi key/button đang giữ; chịu trách nhiệm
  release an toàn sau `hold`.
- **SessionLogger:** JSONL liên kết observation ↔ decision ↔ action ↔ kết quả; **nhận** `observation_id`
  và đường dẫn screenshot từ `Observation`, không tự sinh.

## 8. Cấu trúc package

```
Auto-Browser/
  pyproject.toml
  README.md
  .gitignore
  .env.example
  docs/
    roadmap.md
    superpowers/specs/2026-11-05-browser-game-agent-design.md
  src/game_agent/
    __init__.py
    config.py                 # Settings: viewport, budgets, rate, model
    schema.py                 # Decision + Action (Pydantic v2, discriminated union)
    observation.py            # Observation value type (id, screenshot path, viewport)
    session_state.py          # SessionState: counters/budget + injected clock
    validation.py             # ActionValidator (bounds/allowlist/rate/budget/confidence)
    state.py                  # StateMachine
    logging.py                # SessionLogger (JSONL)
    controller.py             # BrowserController stub (async interface)
    vlm.py                    # VLMClient stub (async interface)
  tests/
    test_schema.py
    test_validation.py
    test_state.py
    test_logging.py
    test_config.py
```

## 9. Action schema (lõi, dùng Pydantic v2)

`Decision`: `state: str`, `reason: str`, `action: Action`, `expected_result: str | None`,
`confidence: float` (0..1), `observation_id: str | None`.

`Action` = discriminated union theo `type`:

| type | tham số | quy tắc |
|---|---|---|
| `click` | x, y, button∈{left,right,middle}, click_count 1..3 | x,y trong viewport |
| `move` | x, y | trong viewport |
| `press` | key (không rỗng) | key nằm allowlist |
| `hold` | key/button, duration_ms ≤ max | cần release an toàn |
| `release` | key/button | hợp lệ độc lập |
| `drag` | start_x/y, end_x/y, duration_ms | cả hai điểm trong viewport |
| `wheel` | delta_x, delta_y | giới hạn biên độ |
| `wait` | duration_ms ≤ max | |
| `done` | summary | không thực thi browser |
| `uncertain` | reason | không thực thi browser |

Schema **từ chối** payload lạ (extra field) và giá trị ngoài biên ngay ở tầng parse.

**Chốt `hold` vs `release`** (theo plan gốc): `hold` là **một action nguyên tử** down → chờ `duration_ms` → up
(tự kết thúc, không để lại trạng thái giữ). `release` là lệnh **giải phóng an toàn** một key/button (dùng khi
recovery, hoặc khi `hold` bị exception giữa đường) — hợp lệ độc lập, kể cả khi không có `hold` trước đó.

**`Observation`** là value type nhỏ: `observation_id`, `screenshot_path`, `viewport` (w/h), `device_scale_factor`,
`timestamp`. Nó là **seam** giữa Observation layer và logger; skeleton chỉ định nghĩa shape, không chụp ảnh.

## 10. Quy tắc Validator

1. **Bounds:** mọi tọa độ trong `[0, viewport_w/h]` (CSS px).
2. **Allowlist:** type ∈ schema; button/key ∈ allowlist cấu hình.
3. **Rate limit:** khoảng cách tối thiểu giữa hai action.
4. **Budget:** tổng số action/phiên ≤ ngưỡng; `hold` cộng dồn thời gian giữ.
5. **Confidence:** dưới ngưỡng → chuyển thành `uncertain` (dừng hỏi người), không thực thi.
6. Signature: `validate(decision, snapshot, now) -> accepted | rejected(reason)`. **Thuần hàm**: bộ đếm
   nằm trong `SessionState`; validator chỉ đọc `snapshot` bất biến + `now` truyền vào → test tất định, không side effect.
   Sau khi accept, `SessionState.record(action, now)` cập nhật bộ đếm (nơi duy nhất thay đổi trạng thái liên-action).

## 11. State machine

States: `idle → observing → deciding → validating → acting → verifying → (observing | done | recovering | needs_user | stopped)`.
Quy tắc: chỉ chuyển theo bảng cho phép; mọi state có đường thoát (`stopped`/`needs_user`); đếm retry
có trần; `verifying` thất bại quá N lần → `needs_user`.

**`recovering`**: vào từ `verifying` thất bại, controller ném lỗi, hoặc mất focus/kết nối tạm thời.
Trong `recovering` hệ thống phát lệnh `release` an toàn cho mọi key/button đang giữ, rồi thử lại có trần.
Thoát về `observing` khi phục hồi được, hoặc → `needs_user` khi hết budget retry (mất kết nối thì → `needs_user`
ngay, không tự đoán).

## 12. Logging

JSONL mỗi dòng một sự kiện, `observation_id` xuyên suốt; lưu đường dẫn screenshot trước/sau; ghi
decision, action, kết quả validate, thời lượng. **Không log secret/token.**

## 13. Config & secrets

`Settings` đọc từ file + env: viewport (mặc định 1280×720), budget, rate, ngưỡng confidence, key
allowlist, thông số model. API key qua `.env` (có `.env.example`); `.env` và thư mục profile/screenshot
nằm trong `.gitignore`.

## 14. Dependencies

Runtime tối thiểu: `playwright`, `pydantic>=2`, `python-dotenv`. Dev: `pytest`, `pytest-asyncio`.
VLM SDK thêm ở M2 khi biết provider thật. Tạo venv bằng `python -m venv .venv`.

## 15. Test strategy

- **Offline, không browser, không API, không driver vòng lặp.** Chạy `pytest`. Test dùng `SessionState`
  với **đồng hồ giả** (fake clock) để tất định — không `sleep`.
- `test_session_state`: đếm budget, cộng dồn `hold`, rate theo clock giả, reset/trần retry.
- `test_schema`: parse hợp lệ; reject extra field, type lạ, giá trị ngoài biên.
- `test_validation`: bounds, allowlist, rate, budget, confidence→uncertain.
- `test_state`: chuyển hợp lệ; chặn chuyển sai; mọi state có đường thoát; retry cap.
- `test_logging`: ghi/đọc JSONL, liên kết observation_id.
- `test_config`: mặc định + override từ env.
- Test browser/VLM thật để dành M1/M2 (đánh dấu riêng, không chạy mặc định).

## 16. Roadmap triển khai (thứ tự thực hiện)

1. Scaffold: `pyproject.toml`, `.gitignore`, `.env.example`, package rỗng, venv.
2. `schema.py` + `test_schema.py`.
3. `observation.py` (value type).
4. `session_state.py` (counters + injected clock) + `test_session_state.py`.
5. `validation.py` + `test_validation.py`.
6. `state.py` + `test_state.py`.
7. `logging.py` + `test_logging.py`.
8. `config.py` + `test_config.py`.
9. Stub `controller.py`, `vlm.py` (`async` interface + `NotImplementedError` rõ ràng).
10. `README.md` + `docs/roadmap.md`; chạy toàn bộ `pytest` xanh.

Sau skeleton (ngoài phạm vi tài liệu này): M1 browser PoC → M2 VLM PoC → M3 closed-loop → M4 game.

## 17. Quyết định đã tự chốt (từ builder-check)

- **State liên-action** nằm ở `SessionState` (bộ đếm + clock tiêm vào); validator thuần, nhận snapshot + `now`.
- **`hold`** nguyên tử down→wait→up; **`release`** là lệnh giải phóng an toàn độc lập (không phải cặp latch).
- **Async**: `controller.py` và `vlm.py` dùng `async def`; `schema/validation/state/session_state/logging/config` là sync.
- **Đồng hồ**: tiêm vào `SessionState`; không đọc giờ hệ thống trong validator/state → test tất định.
- **`observation_id` + đường dẫn screenshot** do `Observation` sở hữu; logger chỉ nhận, không tự sinh.
- **Không có `Session`/driver vòng lặp trong skeleton** — module test độc lập; orchestrator thuộc M3.
- **`recovering`**: vào khi verify fail / controller lỗi / mất focus tạm thời; phát `release` an toàn rồi retry có trần.

## 18. Điều còn để builder quyết

- Tên cấu hình cụ thể của key/button allowlist (chốt ở M1 khi biết game).
- Ngưỡng budget/rate/confidence chính xác (giá trị mặc định hợp lý, tinh chỉnh ở M3).
- Có dùng `pydantic-settings` hay tự đọc env (mặc định: tự đọc, ít dependency).
