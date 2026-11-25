# Releasing GPU memory while a player is hidden

Branch `arglax-gpu-release`. Opt-in API: nothing changes unless an embedder calls it.

## Problem

An embedder that hides a player but must keep its scripts running (MVS army clients: each AQW
client is its own Android WebView, forced "visible" so Chromium does not suspend it) still holds
every GPU object Ruffle created. On a Poco X6 Pro a hidden client keeps ~569 MB of "Graphics"
(EGL/GL mtrack) out of 1132 MB PSS even with rendering skipped and the canvas at 0.25x.

Where the WebGL backend's GPU memory lives (`render/webgl/src/lib.rs`):

| What | Owner | CPU copy? |
| --- | --- | --- |
| Bitmap textures from `register_bitmap` (SWF bitmaps, BitmapData) | `RegistryData` behind `BitmapHandle` | yes: `BitmapCharacter::compressed`, `BitmapRawData::pixels` |
| Textures the GPU wrote (`create_empty_texture` for cacheAsBitmap, `render_offscreen` targets for `BitmapData.draw`) | `RegistryData` | no (or stale until a sync) |
| Offscreen render-target pool (color + stencil renderbuffers, blend layers, resolve textures) | `target_pool` | not needed: scratch |
| Stage MSAA buffers (multisampled color + stencil, resolve texture) | `msaa_buffers` | not needed: scratch |
| Shape vertex/index buffers, gradient uniforms, programs, VAOs | `Mesh`, `ShaderProgram` | no (tessellation is not kept) |

## Design

`RenderBackend` gets four methods, all with no-op defaults (wgpu and canvas keep current
behavior):

- `release_gpu_resources() -> usize` frees what can be recreated and returns an estimate of
  the bytes freed.
- `restore_gpu_resources()` rebuilds the backend's own scratch buffers (MSAA).
- `texture_needs_upload(&BitmapHandle) -> bool` says a texture was emptied.
- `keep_texture_resident(&BitmapHandle)` excludes a texture whose owner cannot upload it again.

WebGL implementation:

- Bitmap textures are **shrunk to 1x1, not deleted**. Every `BitmapHandle`, and every shape draw
  that holds one, keeps naming a valid texture object, so nothing dangles and a stray draw shows
  transparent pixels instead of raising GL errors. `update_texture` refills the same object.
- Textures the GPU wrote are marked `resident` (in `create_empty_texture` and on entry to
  `render_offscreen`) and never emptied: their only up-to-date copy is on the GPU. This covers
  `BitmapData.draw`, filters into BitmapData and cacheAsBitmap surfaces.
- The offscreen target pool and the MSAA buffers are deleted outright; `acquire_target` and
  `restore_gpu_resources` recreate them.
- The backend tracks `register_bitmap` textures with `Weak` handles (pruned amortized), so it
  holds no extra reference and dropped bitmaps still free immediately.

Core re-uploads:

- `BitmapRawData::try_bitmap_handle` (every path that gets a BitmapData handle: Bitmap display
  objects, `copyPixels`, `draw` sources/targets, filters) re-uploads the CPU pixels when the
  texture was emptied and the CPU copy is current (not `GpuModified`).
- `Library::restore_gpu_textures` decodes SWF bitmap characters again (shape bitmap fills hold
  their handle, so they are restored eagerly rather than lazily).
- Drawing-API bitmap fills (`Graphics.beginBitmapFill`, `drawTriangles`, AVM1
  `MovieClip.beginBitmapFill`) store a handle no one refreshes: those textures are marked
  `keep_texture_resident`.
- `Player::release_gpu_resources` sets a flag; `Player::render` calls
  `restore_gpu_resources` first when it is set, so correctness never depends on the embedder
  calling restore. Calling it early only moves the upload out of the first frame.

Web API (`PlayerV1`, `player.ruffle()`): `releaseGpuResources(): number` and
`restoreGpuResources(): void`.

## Measured (Poco X6 Pro, Mali-G615, Android 16, MVS mvs-1.0.6, CI build of 5377a0b0)

Hidden AQW army client, `dumpsys meminfo`, in MB:

| Step | Graphics | Total PSS |
| --- | --- | --- |
| Before (visible-sized canvas) | 568 | 1119 |
| Page ratio 0.25 + `releaseGpuResources` (reported ~146 MB of textures/targets/MSAA, plus shape tessellations) | 464 | 998 |
| Player `visibility:hidden` | 464 | 998 |
| WebGL canvas `width = height = 1` | 145 | 669 |
| Chromium critical memory pressure (`Application.onTrimMemory`) | 145 | 663 |

Afterwards GL mtrack was 128 MB. Restoring (canvas size back, ratio back) redrew correctly with
no errors.

So the largest holder was not Ruffle's own objects but the WebGL drawing buffer and Chromium's
copies of it (shared-image / mailbox backings), which stay allocated at the canvas size even when
nothing is drawn. Shrinking the canvas is an embedder-side step (no Ruffle change needed; Ruffle
must just not be asked to render meanwhile). Ruffle's release adds ~100 MB on top. Hiding the
element and memory pressure freed nothing measurable.

A possible upstream follow-up: have the release API (or a `gpuReleaseWhenHidden` option) also
set the canvas to 1x1 and let the next resize restore it.

## Expected savings (before measuring)

Estimates for a hidden AQW client (to be measured with `dumpsys meminfo` "Graphics"):

- Shape tessellations (added in 5377a0b0): `Library::release_shape_meshes` drops every
  `Graphic` character's tessellation cache, so the vertex buffers go; shapes tessellate again when
  next drawn.
- Bitmap textures: SWF bitmaps of the town map, UI, loaded gear/pets — the bulk; estimated
  150-300 MB on a busy map (each 1024x1024 RGBA texture is 4 MB, plus driver padding).
- Offscreen target pool: up to 24 pooled targets with MSAA renderbuffers and blend layers;
  estimated 20-80 MB.
- Stage MSAA: at 0.25x pixel ratio already small (a few MB); at full ratio ~4x the canvas.

Not freed: GPU-written textures (cacheAsBitmap, BitmapData.draw), drawing-API fill textures,
shape meshes, programs. A rough total of 200-350 MB of the 569 MB is the target.

## Risks

- Video frames (video backend owns its handles and does not re-upload): a paused video shows
  nothing after a release until its next frame. AQW does not use video.
- One slow frame on return (texture upload; SWF JPEG/PNG decode for library bitmaps).
- A texture emptied while a frame is drawn without going through `Player::render` (none
  known) would show transparent until its owner next touches it.

## WebGL context loss (not in this branch)

Recovering a lost context needs everything above plus programs, VAOs, quad meshes and every
shape mesh (core would re-register shapes, since tessellations are not kept). This branch is the
first half of that: textures can be refilled from CPU data. MVS should not call
`WEBGL_lose_context().loseContext()` with this build; Ruffle still cannot recover from it.

## Upstream PR plan

Small commits, in order:

1. `render: add release_gpu_resources / restore_gpu_resources / texture_needs_upload /
   keep_texture_resident to RenderBackend` (default no-ops, docs).
2. `core: re-upload BitmapData and bitmap characters whose texture was released`
   (`try_bitmap_handle`, `BitmapCharacter::restore_texture`, `Library::restore_gpu_textures`,
   `keep_texture_resident` at the drawing-API call sites).
3. `core: Player::release_gpu_resources, auto-restore in render`.
4. `webgl: implement GPU release` (1x1 shrink, resident marking, target pool, MSAA). Upstream's
   webgl has no target pool; this commit would drop that part and keep textures + MSAA.
5. `web: PlayerV1.releaseGpuResources / restoreGpuResources` (+ changelog, d.ts).
6. Optional follow-up: `wgpu` implementation (drop `texture`, keep a re-creatable descriptor) and
   a `gpuReleaseWhenHidden` config option that calls it on `visibilitychange`.
7. Follow-up PR: WebGL context loss/restore on top of this.

Tests: upstream render tests run on wgpu (desktop), which this does not change; a unit test for
the core re-upload path would need a mock backend that reports `texture_needs_upload` — worth
adding in commit 2 with the existing `NullRenderer` extended by a flag.
