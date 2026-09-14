# Working on this repo

## The photoreal face is the product

The GPU face (`musetalk`, or `liveavatar`) is a **required** part of any demo,
not an optional upgrade. The `procedural` renderer is a CPU scaffold for tests
and local plumbing — it is never what a client sees.

So whenever you list what's outstanding, what's left to deploy, or what the
demo needs, **the GPU pod and the photoreal avatar are always on that list**.
Never describe them as optional, nice-to-have, or "only if you want a real
face". If the pod isn't up, the demo isn't ready.

Face quality is the highest bar in this project. `qav-face selftest` output
gets watched before anything is presented.

## Branding

Avatars are **QAV**. Two logos ship with the demo: Lux Sanans on the left,
Quantanite on the right, matched by cap height (not CSS height — equal heights
look wrong because the letterforms fill their canvases differently).

## Design

Apple's current design language — Liquid Glass: translucency, specular edges,
concentric radii, depth for hierarchy. Apple system fonts. The call is front
and centre; settings stay tucked away. It must hold up at phone width.

## Deployment shape

Vercel (`apps/web`, Root Directory `apps/web`) → Railway (`qav-api` +
`qav-engine`) → LiveKit Cloud (media) → RunPod (the face worker, per demo).
`infra/DEPLOY.md` and `infra/gpu/RUNBOOK.md` are the step-by-steps.
