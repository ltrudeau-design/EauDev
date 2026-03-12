# EauDev UI Fix Brief
**For:** Qwen Code  
**File:** `eaudev/commands/run/command.py`  
**Changes:** 2 targeted fixes — response panel + token bar threshold  
**Risk:** LOW

---

## Fix 1: Response Panel (command.py ~line 981)

### Problem
The response is printed after a bare `Rule("[green]Response[/green]")` horizontal
line. This gives no visual separation — the response text runs directly into
surrounding UI elements with no border or padding.

### Current Code (lines ~981–1039)
```python
console.print(Rule("[green]Response[/green]", style="green", align="left"))

# ... inference call ...

# Non-streaming: response was not printed during generation — print it now.
if not streamed:
    response_clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if config and config.agent.markdown:
        console.print(Markdown(response_clean))
    else:
        print(response_clean)
```

### Fix
**Step 1:** Remove the Rule print entirely:
```python
# DELETE this line:
console.print(Rule("[green]Response[/green]", style="green", align="left"))
```

**Step 2:** Wrap the non-streaming response in a Panel:
```python
# Replace:
if not streamed:
    response_clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if config and config.agent.markdown:
        console.print(Markdown(response_clean))
    else:
        print(response_clean)

# With:
if not streamed:
    response_clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if config and config.agent.markdown:
        console.print(Panel(
            Markdown(response_clean),
            title="[green]Response[/green]",
            border_style="green",
            padding=(1, 2),
        ))
    else:
        console.print(Panel(
            response_clean,
            title="[green]Response[/green]",
            border_style="green",
            padding=(1, 2),
        ))
```

**Note:** `Panel` is already imported at line 25. No new imports needed.

---

## Fix 2: Token Bar Always Visible (command.py ~line 2319)

### Problem
The token bar is suppressed on the first exchange because `show_tokens` requires
`last_tokens > 0` OR `session.num_messages > 2`. On the first turn, `last_tokens`
starts at 0 and there is only 1 message — so the bar never appears on turn 1.

`last_tokens` is only set after `_run_agent()` returns (line 1006: `last_tokens = n_past`).
By the time `prompt_async` is called again, `last_tokens` is still 0 for the first turn.

### Current Code (line ~2319)
```python
show_tokens = (last_tokens > 0) or (session.num_messages > 2)
```

### Fix
Always show the token bar after the first message has been sent:
```python
show_tokens = session.num_messages >= 1
```

This means the bar appears from the second prompt onward (after the first response),
which is the correct behaviour — there is nothing to show before any messages exist.

---

## Verification

After applying both fixes:

1. Launch EauDev: `cd ~/Desktop/Code/Cluster && eaudev`
2. Type `hello` and press Enter
3. **Confirm:** Response appears inside a green-bordered panel with padding
4. **Confirm:** Token bar appears below the response panel on the next prompt
5. **Confirm:** Token bar shows correct `▮` graph and `Xk/32k` counts
6. **Confirm:** No visual regression — slash commands, memory stats, tool output unaffected

---

## What NOT To Touch

- Do NOT modify `token_display.py` — the bar rendering is correct
- Do NOT modify `user_input_panel.py` — the panel is stable at this checkpoint
- Do NOT modify streaming response path — only fix the `if not streamed:` block
- Do NOT batch additional changes into this commit
- Commit with message: `Fix: Response panel + token bar threshold`
