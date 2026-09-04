# Local Lottie assets

`apps/desktop/assets/lottie/` contains six small, original JSON animations: `idle`, `checking_models`, `uploading`, `analyzing`, `completed`, and `error`.

They are hand-authored geometric ring animations made for this repository. No CDN, remote URL, third-party art, or external license is used. The Expo client imports the same checked-in JSON files from the desktop asset directory, so both clients run fully locally. Both clients use a static status dot when the platform reports reduced motion; a static UI status remains visible if Lottie cannot render.
