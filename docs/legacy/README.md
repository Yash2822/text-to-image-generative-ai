# Legacy reference

The original provided files, kept for reference only — not imported by
anything in `src/` or `app/`. The refactored, tested version lives in
`src/inference/diffusion_pipeline.py` (generator logic) and
`app/gradio_app.py` (UI).

One behavioral difference from the original: `set_scheduler()` used to
skip re-applying the scheduler if the requested name matched
`current_scheduler`, including on the very first call from `__init__`.
That meant the pipeline's actual scheduler was never explicitly set to
the configured default on initialization — it silently kept whatever
scheduler the pretrained model shipped with (typically PNDM, not
Euler Ancestral) until the user switched away and back. The refactor
always applies the scheduler on init, which is a intentional bug fix,
not an accidental behavior change.
