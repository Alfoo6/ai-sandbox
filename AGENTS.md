- This is a learning and experimentation project.
- Prefer simple, understandable implementations.
- Explain important design decisions.
- Do not introduce dependencies without justification.
- Treat README.md as the source of truth for v0.1 scope and mechanics; keep
  advanced systems listed as future work out of v0.1.
- Keep simulation logic independent from rendering.
- Use one shared simulation loop for visual and headless execution. Rendering
  must never influence simulation state, timing, or randomness.
- Keep controllers independent from the world implementation.
- Pass observations and actions across the controller boundary, not world internals.
- Avoid hard-coding intelligent behavior into the environment.
- Use explicit, separate RNGs for the world and controller. Evaluate controllers
  across the same seeds for reproducible comparisons.
- Run relevant checks after changes.
- Never create Git commits; the user commits manually.
