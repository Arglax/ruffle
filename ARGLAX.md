# Ruffle, Arglax fork

This is a fork of [Ruffle](https://github.com/ruffle-rs/ruffle), the Flash Player emulator, used by
the Arglax apps DreamQuest (DragonFable, AdventureQuest, MechQuest) and MoVibeSkua (AQWorlds). It
tracks upstream Ruffle. Fixes that would help everyone are meant to go back upstream as pull requests.

## Branches

- `arglax`: the fork, as upstream release + our commits on top.
- `upstream-nightly-*`: the upstream release each `arglax` is based on.

## What it adds

- MoVibeSkua's patches (see the first commit on `arglax`):
  - `BitmapData.draw` on WebGL;
  - more blend modes;
  - culling of off-screen and masked-out shapes;
  - fewer WebGL calls (uniform and texture-parameter caching);
  - memory fixes (weak `Dictionary` keys, `useWeakReference`, freeing unused movie libraries);
  - in-betweening (`window.__mvsInterp`);
  - even frame pacing;
  - frame and memory counters for the apps.
- Fixes for AVM1 games (DragonFable etc.) where the above was only proven on AVM2.

## Builds

Every push to `arglax` builds the self-hosted web package in GitHub Actions
(`.github/workflows/arglax-web.yml`, artifact `ruffle-arglax-selfhosted-<sha>`). Tags `arglax-v*`
publish it as a release.

Locally, the same as upstream (`web/README.md`), from `web/`:
`CARGO_FEATURES=lzma npm run build --workspace=ruffle-core && npm run build --workspace=ruffle-selfhosted`.

## License

Same as Ruffle: MIT or Apache-2.0, at your option (`LICENSE.md`). The vendored `gc-arena` keeps its
own license files.
