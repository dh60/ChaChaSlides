# Clean build
rm -rf MetalSlide.app

# Create the app bundle structure first
mkdir -p MetalSlide.app/Contents/MacOS

# Compile the Swift code
swiftc MetalSlide.swift -o MetalSlide.app/Contents/MacOS/MetalSlide -framework SwiftUI -framework Metal -framework MetalKit -framework MetalFX -parse-as-library

# Create Info.plist
cat << EOF > MetalSlide.app/Contents/Info.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>MetalSlide</string>
    <key>CFBundleIdentifier</key>
    <string>com.dh60.MetalSlide</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>LSMinimumSystemVersion</key>
    <string>26.0</string>
</dict>
</plist>
EOF

echo "Build complete!"