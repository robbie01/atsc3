# atsc3 — PlutoSDR ATSC 3.0 receiver and gateway

A real-time ATSC 3.0 (A/321, A/322, A/330, A/331) receiver for a PlutoSDR running the pimpmypluto
firmware, and an HTTP gateway in front of it: HDHomeRun-style lineup, MPEG-TS and Matroska streams
with the AC-4 audio decoded to Opus and the IMSC1 captions kept, HLS of the untouched broadcast
objects, trampolines for the internet-delivered channels and the broadcaster apps, the ESG as a
guide, a live signal meter and an outage log.  Transport only: nothing is decrypted.

    ./run.sh --gain 34 --port 8023 --bind 0.0.0.0        # then open http://<host>:8023/

## Layout

| path | what |
|---|---|
| `a3rx/` | Rust receiver: acquisition, OFDM demodulation, channel estimation, demapping, LDPC/BCH; emits baseband packets and per-frame stats on stdout. Table-driven: `a3rx/tables/` holds this station's configuration (RF 23 Columbus: 16K FFT, GI 512, SP24_4, PLP0 16QAM 9/15, PLP1 256QAM 9/15). |
| `gateway/live.py` | the HTTP gateway (Python, numpy).  `route.py`: ALP → IP/UDP → ROUTE/LCT object reassembly.  `cpsnr.py`: fast SNR from the guard interval.  `tsprobe.py`: measures how far a live stream runs ahead of the clock. |
| `tools/tables/` | the validated Python decoder the receiver's tables are exported from (`export_tables.py OUTDIR`), for a different station or configuration.  Reads A/322 constants from the gr-atsc3 sources. |
| `third_party/ac3forge` | AC-4 decoder (C++23, GPL-3), the only open one that handles the immersive presentations.  Build with Homebrew LLVM: `cmake -B build/llvm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos.llvm.toolchain.cmake -DAC3FORGE_BUILD_GUI=OFF -DAC3FORGE_BUILD_TESTS=OFF -DAC3FORGE_BUILD_HEARTH=OFF && cmake --build build/llvm --target ac3cli` |
| `third_party/gr-atsc3` | drmpeg/gr-atsc3 checkout, read for its constant tables only (not built). |
| `state/` | runtime: `signal.log` (outages and signalling events, JSON lines), `services_last.json`, logs. |

Requirements: Rust, uv, Homebrew `ffmpeg` (libopus).  The receiver's `pmpstream` crate is a git dependency on
[pimpmypluto](https://github.com/robbie01/pimpmypluto); to build against a local checkout instead, copy
`a3rx/.cargo/config.toml.example` to `a3rx/.cargo/config.toml` and set the path.

## Endpoints

`/` lineup · `/guide` · `/meter` · `/events.json` · `/lineup.json` `/lineup.m3u` `/discover.json` (HDHomeRun) ·
`/auto/v6.1` MPEG-TS (HEVC copy, AC-4 → Opus; `?audio=1` second track, `?ch=2|8`, `?lead=N`) ·
`/auto/v6.1.mkv` the same plus captions as a positioned ASS track ·
`/live/<serviceId>/master.m3u8` HLS of the broadcast objects · `/app/<serviceId>` broadcaster app ·
`/tables/` low-level signalling · `/status.json` `/snr.json` `/signal.txt` `/services.json` `/sls/<serviceId>`.

Environment overrides: `A3RX`, `AC3CLI`, `A3_STATE`, `A3RX_TABLES`, `GR_ATSC3`, `GW_DEBUG=1`.

## Licence

AGPL-3.0-or-later (see `LICENSE`).  `third_party/ac3forge` is GPL-3.0 (its own licence applies); `third_party/gr-atsc3`
(GPL-3.0) is read for its constant tables and is not part of this repository.  The `pmpstream` crate the receiver uses
comes from [pimpmypluto](https://github.com/robbie01/pimpmypluto) (AGPL-3.0-or-later).
