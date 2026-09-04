# Next-phase test matrix

This matrix is intentionally a plan, not a claim that these tests were run during UI implementation.

| Area | Environment / test | Expected evidence |
| --- | --- | --- |
| Python unit | Existing unit test suite | Existing contracts and helpers remain green. |
| CLI | `doctor`, `models`, `analyze` validation paths | Safe errors and documented exit codes. |
| API | Endpoint/auth/validation tests | Loopback default, LAN token requirement, CORS allow-list, no path leakage. |
| Job queue | Two submitted GPU jobs | Only one runs; the second remains queued. |
| Cancellation | Queued and running jobs | Safe-boundary cancellation, no completed marker for partial work. |
| Models | Authorised three-model smoke test | Health, instrument and pose models resolve and run once. |
| Privacy | Skeleton-only artifact audit | No source RGB/audio/face crop leakage; audit evidence retained. |
| Desktop dev | Windows Tauri development session | Managed/external API, picker, polling, cancel and artifacts work. |
| Desktop release | Windows Tauri release build | Signed-off executable with sidecar strategy and no embedded model weights. |
| Installer | Fresh Windows machine | Installer, local API lifecycle and model path resolution work. |
| Android emulator | Emulator using configurable `10.0.2.2` address | Secure connection, picker, upload, polling and skeleton playback work. |
| Physical Android | Trusted LAN, explicit token | TLS warning, SecureStore token, upload and no local inference. |
| iOS simulator | Configurable loopback address | Connection and artifact behavior work. |
| Mobile LAN | Wi-Fi / VPN boundary cases | LAN token is required; no public exposure or port forwarding. |
| Large upload | Representative large authorised video | Streaming upload, progress/cancel, max-size limit and `.part` cleanup. |
| Network recovery | Disconnect/reconnect during upload and polling | Safe error, resumed polling after foreground, no duplicate job. |
| Backgrounding | Android and iOS app background/restore | Current job resumes polling without creating a second job. |
| Artifact downloads | All allowed types | Job-root confinement, traversal rejection and correct MIME/name. |
| Sensitive blur | Blur/both outputs | No auto-download/share; explicit confirmation required. |
| Video playback | Skeleton and blur (with consent) | Correct local playback and a clear privacy label. |
| Lottie | Reduced motion and invalid animation fallback | Static indicator remains visible. |
| Storage | Long authorised video, low free disk | Preflight warning, no accidental source deletion. |
| GPU | CUDA OOM and CPU fallback policy | Safe error card; no traceback or fake completion. |
| Cleanup | Cancel during writing/finalising | Partial media is not exposed as completed; temporary artifacts are managed. |
