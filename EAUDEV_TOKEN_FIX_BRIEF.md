# EauDev Token Count Fix Brief
**For:** Qwen Code  
**File:** `eaudev/commands/run/command.py`  
**Change:** 1 targeted fix — use response total_tokens instead of /slots n_past  
**Risk:** LOW

---

## Background

EauDev runs on mlx_lm.server which does NOT implement the llama.cpp `/slots` endpoint.
`_get_n_past()` queries `/slots` and always returns 0. This is why the token bar
always shows `0/32.8K`.

`total_tokens` IS correctly returned from mlx_lm in the standard OpenAI response
format — it's captured in `_chat_complete()` (line 508) and `_chat_stream()` 
(lines 604, 633) and returned as the second element of their tuples.

The fix: use `total_tokens` from the response tuple instead of `n_past` from `/slots`.

---

## Step 1: Find the agent loop where tokens are set

Locate this block in `_run_agent()` (around line 980–1010):

```python
response, _, tc_data = _chat_complete(session.message_history, config, tools=all_tools)
```

or the streaming version:

```python
response, _, tc_data = _chat_stream(...)
```

**Note the `_` — that's `total_tokens` being discarded.**

---

## Step 2: Capture total_tokens from the response

Find every call to `_chat_complete()` and `_chat_stream()` inside `_run_agent()`
and change `_` to `resp_tokens`:

```python
# Change:
response, _, tc_data = _chat_complete(session.message_history, config, tools=all_tools)

# To:
response, resp_tokens, tc_data = _chat_complete(session.message_history, config, tools=all_tools)
```

Do the same for any `_chat_stream()` calls that use `_` for the token position.

---

## Step 3: Replace n_past with resp_tokens

Find this block (around line 1002–1006):

```python
# Get real KV cache usage from /slots — always update, even on final response
n_past = _get_n_past(config.agent.inference.endpoint)
if n_past > 0:
    last_tokens = n_past
```

Replace with:

```python
# Use token count from response (mlx_lm returns this via standard OpenAI usage field)
if resp_tokens > 0:
    last_tokens = resp_tokens
```

---

## Step 4: Initialise resp_tokens safely

At the top of `_run_agent()` where `last_tokens = 0` is set (line ~958), add:

```python
last_tokens = 0
resp_tokens = 0
```

---

## What NOT to change

- Do NOT delete `_get_n_past()` — it may be used elsewhere or useful for llama.cpp
- Do NOT modify `_get_context_limit()` — context limit detection is working correctly
- Do NOT touch `_chat_complete()` or `_chat_stream()` — they already return total_tokens correctly
- Do NOT modify `token_display.py` or `user_input_panel.py`
- Commit message: `Fix: Token count from response usage (mlx_lm has no /slots endpoint)`

---

## Verification

1. Launch EauDev: `cd ~/Desktop/Code/Cluster && eaudev`
2. Type `hello` and press Enter
3. On the next prompt, token bar should show a non-zero count e.g. `Session context: ▮▮▮▮▮▮▮▮▮▮ 1.2K/32.8K`
4. Send a few more messages — count should increase each turn
5. Confirm bar never shows `0/32.8K` after the first response
