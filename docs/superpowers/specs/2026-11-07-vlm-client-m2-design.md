# Design — M2: VLM client (vision model via NVIDIA API)

> Ngày: 2026-11-07 · Trạng thái: chờ duyệt
> Tiền đề: M1 đã merge (browser controller thật). Milestone: **M2** trong `docs/roadmap.md`.

---

## 1. Intent và lý do

M1 cho agent "con mắt và tay" (browser + screenshot + input) nhưng chưa có "bộ não". M2 hiện thực
`VLMClient` — nhận một `Observation` (ảnh) + mục tiêu, gọi một **model vision** qua **NVIDIA API**
(OpenAI-compatible), và trả về một `Decision` **có cấu trúc, đã validate**. Đây là mốc M2: model
nhận ảnh thật và trả action hợp lệ.

## 2. Goals

- `VLMClient.decide(observation, goal, remaining_actions) -> Decision` là thật (không còn stub).
- Kết nối **NVIDIA API** (`https://integrate.api.nvidia.com/v1/chat/completions`), key + model đọc từ env:
  `NVIDIA_API_KEY`, `NVIDIA_MODEL`, tùy chọn `NVIDIA_BASE_URL`/timeout/max_tokens/temperature.
- **Không hard-code** key hay model; nạp `.env` để người dùng để key trong `.env` (gitignored).
- Ảnh screenshot gửi dạng base64 data URL trong content của message.
- Output **bắt buộc** parse thành `Decision`; nếu model trả JSON hỏng → thử sửa **1 lần**; vẫn hỏng → trả `uncertain`.
- Lỗi vận chuyển/HTTP/thiếu key → raise `VLMError` (rõ ràng, không lộ secret).
- Test **offline** bằng transport giả; thêm 1 test **live** opt-in (marker `live`).

## 3. Non-goals

- Không vòng lặp orchestrate (M3).
- Không đa model/routing, không streaming, không fine-tune, không đếm chi phí.
- Không tiền xử lý ảnh (crop/resize) — M2 gửi nguyên PNG viewport.
- Không kiểm bounds viewport trong client (validator M3 sở hữu).

## 4. Anti-goals (làm xong vẫn tính thất bại nếu…)

- Key hoặc ảnh bị ghi vào log.
- Suite `pytest` mặc định gọi mạng/API (phá offline).
- Model trả code tùy ý được thực thi (M2 chỉ trả `Decision`).
- Hard-code model/key vào code.

## 5. Constraints

- **Không thêm dependency HTTP**: venv không có `httpx`/`requests`/`openai`. Dùng stdlib `urllib.request`
  chạy trong thread (`asyncio.to_thread`). `python-dotenv` đã là dependency.
- Tái dùng `Observation`, `Decision`/action schema.
- Ảnh rời máy tới provider → chỉ gửi khi người dùng chủ động chạy; ghi chú quyền riêng tư.
- Không ghi secret; key chỉ đọc từ env.

## 6. Approaches

- **A. stdlib `urllib` + Pydantic + 1 repair (đã chọn):** không thêm dependency; kiểm soát payload.
- **B. thêm `httpx`:** hoãn.
- **C. SDK `openai`:** loại.

## 7. Thiết kế

### 7.1 Cấu hình

```python
@dataclass(frozen=True)
class VLMConfig:
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str | None = None       # từ NVIDIA_MODEL; bắt buộc khi gọi
    api_key: str | None = None     # từ NVIDIA_API_KEY; bắt buộc khi gọi
    timeout_s: float = 60.0
    max_tokens: int = 512
    temperature: float = 0.0

def load_vlm_config(env: Mapping[str, str] | None = None) -> VLMConfig: ...
```

**Contract của `load_vlm_config`** (tất định cho test):
- `env` được truyền (kể cả `{}`) → **chỉ** dùng `env`, KHÔNG nạp `.env`, KHÔNG đọc `os.environ`.
  (Giống `load_settings`: `env={}` = "không override".)
- `env is None` → đọc `.env` bằng `dotenv_values()` (KHÔNG mutate `os.environ`), rồi overlay `os.environ`
  (process env thắng), đọc từ đó.
- Không set `model`/`api_key` mặc định.

### 7.2 Client

```python
class VLMError(Exception): ...

class VLMClient:
    def __init__(self, config: VLMConfig | None = None,
                 transport: Callable[[str, dict, dict], dict] | None = None): ...
    async def decide(self, observation: Observation, goal: str,
                     remaining_actions: int) -> Decision: ...
```

- `transport(url, headers, payload) -> dict` là điểm tiêm cho test; mặc định là POST thật (urllib),
  gọi qua `asyncio.to_thread`.
- `decide`:
  1. Thiếu `api_key`/`model` → `VLMError`.
  2. Đọc PNG tại `observation.screenshot_path` → data URL `data:image/png;base64,…` (thiếu ảnh → `VLMError`).
  3. `build_messages(...)` → payload; gọi transport.
  4. Trích `choices[0].message.content`. **Không trích được** (không `choices`, content `null`, body không phải JSON)
     → `VLMError` (lỗi giao thức). Có content nhưng JSON hỏng → đường parse.
  5. **Chuẩn hoá tối thiểu:** strip code-fence ```` ```json ```` và khoảng trắng; **không** coerce kiểu,
     **không** bỏ key lạ. Rồi `Decision.model_validate_json`.
  6. Hỏng → gọi lại 1 lần (repair, xem §7.5); vẫn hỏng → `Decision` fallback `uncertain`.
  7. `observation_id` do client set = `observation.observation_id` (ghi đè giá trị model trả), qua `model_copy`.

### 7.3 Prompt

- System: vai trò; **schema JSON chính xác** của `Decision` cùng danh sách action; quy tắc: tọa độ CSS px
  theo viewport đã cho; `hold` cần `duration_ms`; `state` là nhãn màn hình ngắn; dùng `uncertain` khi không chắc;
  dùng `done` khi đạt mục tiêu; **chỉ trả một object JSON, không kèm văn bản/fence**.
- User: `goal`, `remaining_actions`, kích thước viewport, và ảnh.

### 7.4 Fallback `uncertain` (giá trị chính xác)

Khi parse hỏng cả 2 lần, trả:
```python
Decision(state="uncertain", reason="model output was not valid JSON after one repair",
         action=UncertainAction(type="uncertain", reason="unparseable model output"),
         expected_result=None, confidence=1.0, observation_id=observation.observation_id)
```
`confidence=1.0` là **có chủ đích**: `validate()` kiểm confidence TRƯỚC khi bypass terminal action, nên
confidence thấp sẽ bị `LOW_CONFIDENCE` và M3 không bao giờ thấy "không đọc được màn hình". `uncertain` là
báo cáo terminal hợp lệ, không phải quyết định tồi.

### 7.5 Repair

Repair = **tiếp nối hội thoại**: giữ nguyên `messages` (ảnh vẫn ở lượt user cũ, không gửi lại ảnh lần hai),
thêm lượt `assistant` chứa output hỏng, thêm lượt `user`: "Previous response was invalid: `<error>`.
Reply with ONLY a single valid JSON object matching the schema." Gọi transport lần 2.

### 7.6 Phân loại lỗi

| Tình huống | Hành vi |
|---|---|
| Thiếu key/model | raise `VLMError` |
| Thiếu file ảnh | raise `VLMError` |
| HTTP != 200 / timeout / transport ném lỗi | raise `VLMError` |
| Body 200 nhưng không trích được content | raise `VLMError` |
| Content có nhưng JSON hỏng (kể cả bị cắt vì `max_tokens`) | repair 1 lần → vẫn lỗi → fallback `uncertain` |
| Model trả `uncertain` | giữ nguyên (action hợp lệ) |

### 7.7 Test

- **Offline (mặc định):** transport giả. Kiểm: headers có `Authorization: Bearer …` (test không in key/message),
  `model`, ảnh là data URL đúng prefix, prompt chứa goal/viewport; parse JSON hợp lệ; **repair retry**
  (rác lần 1 → JSON lần 2, đúng 2 lần gọi, có lượt assistant chứa output hỏng); fallback `uncertain` (rác cả 2 lần)
  với đúng field §7.4; `VLMError` khi thiếu key/model/ảnh; transport ném lỗi → `VLMError`; `observation_id` được set.
- **Live (opt-in):** marker `live` **và** `browser` — chụp `examples/demo.html` bằng controller thật rồi gọi API thật;
  skip nếu thiếu `NVIDIA_API_KEY`/`NVIDIA_MODEL`. Chạy: `pytest -m "live and browser"`.
- Marker `live` đăng ký; `addopts` mặc định `-m "not browser and not live"`.

## 8. Quyết định đã chốt (từ builder-check)

1. **Fallback `uncertain`:** `state="uncertain"`, `confidence=1.0` (báo cáo terminal hợp lệ, không bị LOW_CONFIDENCE).
2. **`load_vlm_config`:** `env` truyền vào là nguồn DUY NHẤT (không đọc `.env`/`os.environ`); `env=None` mới đọc
   `.env` + `os.environ` (process thắng), dùng `dotenv_values()` không mutate global.
3. **Body 200 hỏng giao thức** (không content) → `VLMError`; **content có nhưng JSON hỏng/cắt** → repair → `uncertain`.
4. **Parse:** strict `Decision.model_validate_json` sau khi strip code-fence; KHÔNG coerce kiểu, KHÔNG bỏ key lạ
   (không tạo tầng schema thứ hai).
5. **Bounds viewport:** client KHÔNG kiểm; validator M3 sở hữu (một nguồn sự thật).
6. **`observation_id`** do client ghi đè = của `Observation`; `state`/`reason`/`expected_result` do model.
7. **Repair** là tiếp nối hội thoại (ảnh chỉ gửi một lần).
8. **Live test** dùng controller thật chụp `examples/demo.html` → marker `live and browser`; model ví dụ trong
   `.env.example`/README (KHÔNG mặc định trong code).

## 9. Điều còn để builder quyết

- Câu chữ repair chính xác.
- `max_tokens`/`temperature` mặc định (chỉnh sau khi thử model).
- Có log token usage hay không (M2: bỏ qua).
