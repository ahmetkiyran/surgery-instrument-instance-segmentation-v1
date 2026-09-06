# Local Lottie assets

`apps/desktop/assets/lottie/` contains seven small, original JSON animations: `idle`, `checking_models`, `uploading`, `analyzing`, `finalizing`, `completed`, and `error`.

They are hand-authored geometric ring animations made for this repository. No CDN, remote URL, third-party art, or external license is used. The Expo client imports the same checked-in JSON files from the desktop asset directory, so both clients run fully locally. The Gradio client embeds small animated WebP previews rendered from these repository-owned status definitions. All clients retain visible status text, and desktop/mobile use a static status dot when the platform reports reduced motion.
