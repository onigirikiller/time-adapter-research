# Realtime Time Adapter WebUI

A local research UI for microphone streaming, VAD ticks, Time Adapter action scoring, and optional response generation with a compatible Qwen2.5-Omni control-response LoRA.

## Requirements

The source-only public release does not include checkpoints. The server expects the English control-response LoRA and shared Time Adapter at the paths declared near the top of `scripts/run_control_response_lora_webui_v1.py`.

Start without speech synthesis while validating model loading and action scoring:

```powershell
python scripts/run_control_response_lora_webui_v1.py --host 127.0.0.1 --port 7865 --autoload --tts-backend none
```

Then open `http://127.0.0.1:7865`, grant microphone access, and select **Start mic**.

## Runtime path

```text
browser PCM16 microphone chunks
-> WebSocket
-> server-side VAD and audio-prefix snapshots
-> Time Adapter action-token logits
-> WAIT / BACKCHANNEL / SUPPORT telemetry
-> optional text and speech generation on response events
```

The UI stores local session logs and generated responses under ignored `artifacts/omni3b_realtime_webui_v1/` paths.

## Security

The app has no authentication or authorization layer. Keep the default loopback binding. Do not use `--host 0.0.0.0` on an untrusted network. Session logs may contain microphone audio and generated text; review them before sharing.

This WebUI is an experiment harness, not a hardened service.
