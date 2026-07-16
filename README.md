# MetalSlide
A slideshow app for macOS using Metal 4 and MetalFX for high quality image viewing.
![MetalSlide User Interface](demo.png)
## Features
- Displays photos from a folder (shuffled or sorted)
- JPEG, PNG, HEIC/HEIF, TIFF, WebP, AVIF, GIF, BMP — with EXIF orientation
- HDR photos (gain map / PQ / HLG) rendered with EDR headroom
- Metal 4 pipeline with native rendering (no OS scaling)
- MetalFX upscaling
- Auto-advance timer
- Settings (scaling, shuffle, auto-advance) persist between runs
## Keybinds
| Key | Action |
|-----|--------|
| Right / Down / Space | Next slide |
| Left / Up | Previous slide |
| Delete | Trash current image |
| f | Toggle Fullscreen |
| i | Toggle info overlay |
| 0-9 | Set auto-advance interval (0 = off) |
| Esc | Quit |
## Info Overlay
Press `i` to show the info overlay, which displays image details and provides toggles for:
- **Scaling** - Enable/disable all scalings
- **Shuffle** - Switch between shuffled and sorted order
## Requirements
- macOS 26+
- Apple Silicon