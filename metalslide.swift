import SwiftUI
import Observation
import Metal
import MetalKit
import MetalFX
import ImageIO

@main
struct MetalSlide: App {
    var body: some Scene {
        Window("MetalSlide", id: "main") {
            MetalView()
        }
        .windowStyle(.hiddenTitleBar)
    }
}

struct MetalView: View {
    @State private var renderer = Renderer()
    @State private var isImporting = true

    var body: some View {
        @Bindable var renderer = renderer
        ZStack(alignment: .topLeading) {
            MetalViewRepresentable(renderer: renderer)
                .ignoresSafeArea()

            if renderer.showInfo {
                VStack(alignment: .leading) {
                    Text(renderer.info)

                    Toggle("Scaling", isOn: $renderer.scalingEnabled)
                        .onChange(of: renderer.scalingEnabled) { renderer.view?.needsDisplay = true }

                    Toggle("Shuffle", isOn: $renderer.shuffleEnabled)
                        .onChange(of: renderer.shuffleEnabled) { renderer.toggleShuffle() }
                }
                .padding(15)
                .glassEffect(in: .rect(cornerRadius: 30))
                .padding(.leading, 8)
            }
        }

        .focusable()
        .focusEffectDisabled()
        .onMoveCommand { direction in
            switch direction {
            case .left, .up:
                renderer.goToSlide(renderer.currentIndex - 1)
            case .right, .down:
                renderer.goToSlide(renderer.currentIndex + 1)
            default:
                break
            }
        }
        .onKeyPress(.space) { renderer.goToSlide(renderer.currentIndex + 1); return .handled }
        .onDeleteCommand { renderer.deleteSlide() }
        .onExitCommand { NSApplication.shared.terminate(nil) }
        .onKeyPress("i") { renderer.showInfo.toggle(); return .handled }
        .onKeyPress("f") { NSApplication.shared.keyWindow?.toggleFullScreen(nil); return .handled }
        .onKeyPress("0") { renderer.autoadvanceInterval = 0; return .handled }
        .onKeyPress("1") { renderer.autoadvanceInterval = 1; return .handled }
        .onKeyPress("2") { renderer.autoadvanceInterval = 2; return .handled }
        .onKeyPress("3") { renderer.autoadvanceInterval = 3; return .handled }
        .onKeyPress("4") { renderer.autoadvanceInterval = 4; return .handled }
        .onKeyPress("5") { renderer.autoadvanceInterval = 5; return .handled }
        .onKeyPress("6") { renderer.autoadvanceInterval = 6; return .handled }
        .onKeyPress("7") { renderer.autoadvanceInterval = 7; return .handled }
        .onKeyPress("8") { renderer.autoadvanceInterval = 8; return .handled }
        .onKeyPress("9") { renderer.autoadvanceInterval = 9; return .handled }

        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: [.directory],
            allowsMultipleSelection: false
        ) { result in
            guard case .success(let urls) = result, let url = urls.first,
                  let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil) else { return }
            let exts: Set = ["jpg", "jpeg", "png", "heic", "heif", "tiff", "tif", "webp", "avif", "gif", "bmp"]
            renderer.imagePaths = files.allObjects
                .compactMap { $0 as? URL }
                .filter { exts.contains($0.pathExtension.lowercased()) }
                .sorted { $0.path < $1.path }
            if renderer.shuffleEnabled { renderer.imagePaths.shuffle() }
            renderer.goToSlide(0)
        }
    }
}

struct MetalViewRepresentable: NSViewRepresentable {
    let renderer: Renderer

    func makeCoordinator() -> Renderer { renderer }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        // Everything after decode lives in linear extended-range light: images are
        // drawn into a half-float extended-linear-sRGB bitmap, so the drawable is
        // rgba16Float and the layer is tagged to match. The OS color-manages from
        // there (and drives EDR when wantsExtendedDynamicRangeContent is set).
        view.colorPixelFormat = .rgba16Float
        (view.layer as? CAMetalLayer)?.colorspace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)
        view.delegate = context.coordinator
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        context.coordinator.view = view
        context.coordinator.initializeMetal(view)
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}

@Observable
class Renderer: NSObject, MTKViewDelegate {
    var imagePaths: [URL] = []
    var currentIndex = 0
    // Generation counter so a slow decode can't overwrite a newer slide.
    var loadGeneration = 0

    var device: MTLDevice!
    var queue: MTL4CommandQueue!
    var pipeline: MTLRenderPipelineState?
    var pipelineEWA: MTLRenderPipelineState?
    var kernelLUT: MTLTexture!
    var allocator: MTL4CommandAllocator!
    var argumentTable: MTL4ArgumentTable!
    var residencySet: MTLResidencySet!
    var texture: MTLTexture?
    var textureIsHDR = false
    weak var view: MTKView?

    var scaleBuffer: MTLBuffer!
    var compiler: MTL4Compiler!
    var scaler: (MTL4FXSpatialScaler, MTLTexture)?
    var scalerFence: MTLFence!
    var commandBuffer: MTL4CommandBuffer!
    // The command allocator's memory is read by the GPU during execution, so it can
    // only be reset (or recorded into again) once the previous submission finished.
    // The queue signals this event after each frame; draw() waits on it before reuse.
    var frameEvent: MTLSharedEvent!
    var pendingFrameValue: UInt64 = 0

    var info = ""
    var showInfo = false
    // Persisted settings — restored from UserDefaults at launch, written on change.
    var scalingEnabled = UserDefaults.standard.object(forKey: "scalingEnabled") as? Bool ?? true {
        didSet { UserDefaults.standard.set(scalingEnabled, forKey: "scalingEnabled") }
    }
    var shuffleEnabled = UserDefaults.standard.bool(forKey: "shuffleEnabled") {
        didSet { UserDefaults.standard.set(shuffleEnabled, forKey: "shuffleEnabled") }
    }
    var autoadvanceInterval = UserDefaults.standard.integer(forKey: "autoadvanceInterval") {
        didSet { UserDefaults.standard.set(autoadvanceInterval, forKey: "autoadvanceInterval") }
    }
    var slideChangedTime = Date()
    var autoadvanceTimer: Timer?

    deinit { autoadvanceTimer?.invalidate() }

    func initializeMetal(_ view: MTKView) {
        device = view.device
        queue = device.makeMTL4CommandQueue()
        compiler = try! device.makeCompiler(descriptor: MTL4CompilerDescriptor())

        let tableDesc = MTL4ArgumentTableDescriptor()
        tableDesc.maxTextureBindCount = 2
        tableDesc.maxBufferBindCount = 1
        argumentTable = try! device.makeArgumentTable(descriptor: tableDesc)

        // Uniforms: float2 quadScale + float ratio (+ pad).
        scaleBuffer = device.makeBuffer(length: 16, options: .storageModeShared)
        residencySet = try! device.makeResidencySet(descriptor: .init())
        allocator = device.makeCommandAllocator()
        scalerFence = device.makeFence()
        commandBuffer = device.makeCommandBuffer()
        frameEvent = device.makeSharedEvent()

        autoadvanceTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self, autoadvanceInterval > 0, !imagePaths.isEmpty,
                  Date().timeIntervalSince(slideChangedTime) >= Double(autoadvanceInterval) else { return }
            goToSlide(currentIndex + 1)
        }

        // ewa_lanczossharp kernel LUT (libplacebo, same as MetalFrame): jinc *
        // jinc-window with a 0.98125 blur factor, 3rd jinc zero as support,
        // precomputed into a 1D LUT indexed by r / maxR ∈ [0, 1].
        let lutSize = 512
        let kernelRadius = 3.2383154841662362
        let blur = 0.98125058372237073
        let maxR = kernelRadius * blur
        func jinc(_ x: Double) -> Double {
            if abs(x) < 1e-8 { return 1.0 }
            return 2.0 * j1(.pi * x) / (.pi * x)
        }
        var lutData = [Float](repeating: 0, count: lutSize)
        for i in 0..<lutSize {
            let r = (Double(i) / Double(lutSize - 1)) * maxR
            let rPrime = r / blur
            lutData[i] = rPrime >= kernelRadius ? 0 : Float(jinc(rPrime) * jinc(rPrime / kernelRadius))
        }
        let lutDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: lutSize, height: 1, mipmapped: false)
        lutDesc.usage = .shaderRead
        lutDesc.storageMode = .shared
        kernelLUT = device.makeTexture(descriptor: lutDesc)
        lutData.withUnsafeBytes { bytes in
            kernelLUT.replace(region: MTLRegionMake2D(0, 0, lutSize, 1), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: lutSize * MemoryLayout<Float>.size)
        }

        let shaderSource = """
            #include <metal_stdlib>
            using namespace metal;

            struct VertexOut {
                float4 position [[position]];
                float2 texCoord;
            };

            struct Uniforms { float2 scale; float ratio; };

            vertex VertexOut vertexShader(uint vid [[vertex_id]], constant Uniforms &u [[buffer(0)]]) {
                float2 positions[6] = {
                    float2(-1,-1), float2(1,-1), float2(-1,1),
                    float2(-1,1), float2(1,-1), float2(1,1)
                };
                float2 texCoords[6] = {
                    float2(0,1), float2(1,1), float2(0,0),
                    float2(0,0), float2(1,1), float2(1,0)
                };
                return { float4(positions[vid] * u.scale, 0, 1), texCoords[vid] };
            }

            fragment half4 fragmentShader(VertexOut in [[stage_in]], texture2d<half> tex [[texture(0)]]) {
                constexpr sampler s(coord::normalized,
                                   address::clamp_to_edge,
                                   filter::linear);
                return tex.sample(s, in.texCoord);
            }

            // ewa_lanczossharp (libplacebo), same kernel as MetalFrame. Input is
            // already linear light and the layer is linear-tagged, so unlike
            // MetalFrame no transfer-function round trip is needed. At ratio == 1
            // the kernel collapses to its natural support; downscale ratios widen
            // it to low-pass at the output Nyquist.
            fragment half4 fragmentShaderEWA(VertexOut in [[stage_in]],
                                              texture2d<half> tex [[texture(0)]],
                                              texture2d<float> kernelLUT [[texture(1)]],
                                              constant Uniforms &u [[buffer(0)]]) {
                constexpr sampler texSampler(coord::pixel, address::clamp_to_edge, filter::nearest);
                constexpr sampler lutSampler(coord::normalized, address::clamp_to_edge, filter::linear);
                constexpr float kernelRadius = 3.2383154841662362;
                constexpr float blur = 0.98125058372237073;
                constexpr float maxR = kernelRadius * blur;

                float ratio = max(u.ratio, 1.0);
                int taps = min(int(ceil(maxR * ratio)), 20);

                float texW = float(tex.get_width());
                float texH = float(tex.get_height());
                float cx = in.texCoord.x * texW;
                float cy = in.texCoord.y * texH;
                int baseX = int(floor(cx - 0.5));
                int baseY = int(floor(cy - 0.5));

                float4 sum = float4(0);
                float wSum = 0;
                for (int j = -taps + 1; j <= taps; ++j) {
                    float sy = float(baseY + j) + 0.5;
                    float dy = (sy - cy) / ratio;
                    for (int i = -taps + 1; i <= taps; ++i) {
                        float sx = float(baseX + i) + 0.5;
                        float dx = (sx - cx) / ratio;
                        float r = sqrt(dx * dx + dy * dy);
                        if (r >= maxR) continue;
                        float w = kernelLUT.sample(lutSampler, float2(r / maxR, 0.5)).x;
                        sum += w * float4(tex.sample(texSampler, float2(sx, sy)));
                        wSum += w;
                    }
                }
                return half4(sum / max(wSum, 1e-6));
            }
            """

        let pixelFormat = view.colorPixelFormat
        Task { [weak self] in
            guard let self else { return }
            let libDesc = MTL4LibraryDescriptor()
            libDesc.source = shaderSource
            do {
                let library = try await compiler.makeLibrary(descriptor: libDesc)
                let desc = MTL4RenderPipelineDescriptor()
                desc.vertexFunctionDescriptor = {
                    let d = MTL4LibraryFunctionDescriptor()
                    d.name = "vertexShader"
                    d.library = library
                    return d
                }()
                desc.fragmentFunctionDescriptor = {
                    let d = MTL4LibraryFunctionDescriptor()
                    d.name = "fragmentShader"
                    d.library = library
                    return d
                }()
                desc.colorAttachments[0].pixelFormat = pixelFormat
                pipeline = try await compiler.makeRenderPipelineState(descriptor: desc)
                desc.fragmentFunctionDescriptor = {
                    let d = MTL4LibraryFunctionDescriptor()
                    d.name = "fragmentShaderEWA"
                    d.library = library
                    return d
                }()
                pipelineEWA = try await compiler.makeRenderPipelineState(descriptor: desc)
                await MainActor.run { self.view?.needsDisplay = true }
            } catch {
                print("Shader compilation error: \(error.localizedDescription)")
            }
        }
    }

    // EXIF orientation → transform into an upright canvas (CG bottom-left coords,
    // image drawn visually upright in the given rect). Cases 5–8 swap the canvas.
    static func uprightTransform(orientation o: Int, w: Int, h: Int) -> (CGAffineTransform, Int, Int) {
        let (fw, fh) = (CGFloat(w), CGFloat(h))
        switch o {
        case 2: return (CGAffineTransform(translationX: fw, y: 0).scaledBy(x: -1, y: 1), w, h)
        case 3: return (CGAffineTransform(translationX: fw, y: fh).rotated(by: .pi), w, h)
        case 4: return (CGAffineTransform(translationX: 0, y: fh).scaledBy(x: 1, y: -1), w, h)
        case 5: return (CGAffineTransform(translationX: fh, y: fw).rotated(by: .pi / 2).scaledBy(x: -1, y: 1), h, w)
        case 6: return (CGAffineTransform(translationX: 0, y: fw).rotated(by: -.pi / 2), h, w)
        case 7: return (CGAffineTransform(rotationAngle: -.pi / 2).scaledBy(x: -1, y: 1), h, w)
        case 8: return (CGAffineTransform(translationX: fh, y: 0).rotated(by: .pi / 2), h, w)
        default: return (.identity, w, h)
        }
    }

    // Decode off the main thread into linear extended-range half-float RGBA:
    // DecodeToHDR expands gain-map / PQ / HLG sources to their full range, and the
    // extended-linear-sRGB context makes CG do the (colorspace-aware) conversion.
    func loadSlide(_ url: URL) {
        loadGeneration += 1
        let generation = loadGeneration
        guard let device else { return }
        Task.detached(priority: .userInitiated) { [weak self] in
            let decoded: (texture: MTLTexture, isHDR: Bool)? = {
                guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceDecodeRequest: kCGImageSourceDecodeToHDR] as CFDictionary)
                else { return nil }
                let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
                let orientation = props?[kCGImagePropertyOrientation] as? Int ?? 1
                let (transform, w, h) = Self.uprightTransform(orientation: orientation, w: image.width, h: image.height)
                guard let colorspace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB),
                      let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 16, bytesPerRow: w * 8,
                                          space: colorspace,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue |
                                                      CGBitmapInfo.floatComponents.rawValue |
                                                      CGBitmapInfo.byteOrder16Little.rawValue)
                else { return nil }
                ctx.concatenate(transform)
                ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
                let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
                desc.usage = .shaderRead
                desc.storageMode = .shared
                guard let tex = device.makeTexture(descriptor: desc), let data = ctx.data else { return nil }
                tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: data, bytesPerRow: ctx.bytesPerRow)
                return (tex, image.contentHeadroom > 1.0)
            }()
            await MainActor.run { [weak self] in
                guard let self, generation == self.loadGeneration else { return }
                texture = decoded?.texture
                textureIsHDR = decoded?.isHDR ?? false
                (view?.layer as? CAMetalLayer)?.wantsExtendedDynamicRangeContent = textureIsHDR
                view?.needsDisplay = true
            }
        }
    }

    func draw(in view: MTKView) {
        guard let pipeline, let pipelineEWA, let drawable = view.currentDrawable,
              let passDescriptor = view.currentMTL4RenderPassDescriptor else { return }

        // Reusing the single command buffer / allocator is only safe once the GPU
        // finished the previous submission — wait before touching any shared state.
        if pendingFrameValue > 0 {
            _ = frameEvent.wait(untilSignaledValue: pendingFrameValue, timeoutMS: 1000)
        }
        allocator.reset()

        guard let inputTexture = texture else {
            // No image (empty folder, last slide deleted): clear to black.
            info = "No images"
            commandBuffer.beginCommandBuffer(allocator: allocator)
            if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor, options: MTL4RenderEncoderOptions()) {
                encoder.endEncoding()
            }
            commandBuffer.endCommandBuffer()
            queue.waitForDrawable(drawable)
            queue.commit([commandBuffer])
            pendingFrameValue += 1
            queue.signalEvent(frameEvent, value: pendingFrameValue)
            queue.signalDrawable(drawable)
            drawable.present()
            return
        }

        let viewportSize = CGSize(width: drawable.texture.width,
                                 height: drawable.texture.height)

        let aspectImage = CGFloat(inputTexture.width) / CGFloat(inputTexture.height)
        let aspectView  = viewportSize.width / viewportSize.height

        let fitSize: CGSize = aspectImage > aspectView
            ? CGSize(width: viewportSize.width,
                    height: viewportSize.width / aspectImage)
            : CGSize(width: viewportSize.height * aspectImage,
                    height: viewportSize.height)

        let displaySize = scalingEnabled ? fitSize : CGSize(width: inputTexture.width,
                                                          height: inputTexture.height)

        let needsUpscale = Int(fitSize.width) > inputTexture.width ||
                          Int(fitSize.height) > inputTexture.height
        // Content is linear light after decode; .hdr additionally tells MetalFX to
        // expect values beyond [0, 1].
        let colorMode: MTLFXSpatialScalerColorProcessingMode = textureIsHDR ? .hdr : .linear

        var scalerRebuilt = false

        // Scalers are expensive to build (pipeline compilation), so keep the current
        // one unless the geometry or color mode changed. Replacing here is safe: the
        // event wait above guarantees the old scaler is no longer executing.
        if !scalingEnabled || !needsUpscale || !MTLFXSpatialScalerDescriptor.supportsMetal4FX(device) {
            scaler = nil
        } else if scaler == nil
            || scaler!.0.inputWidth != inputTexture.width
            || scaler!.0.inputHeight != inputTexture.height
            || scaler!.0.outputWidth != Int(fitSize.width)
            || scaler!.0.outputHeight != Int(fitSize.height)
            || scaler!.0.colorProcessingMode != colorMode {
            scaler = nil
            let desc = MTLFXSpatialScalerDescriptor()
            desc.inputWidth = inputTexture.width
            desc.inputHeight = inputTexture.height
            desc.outputWidth = Int(fitSize.width)
            desc.outputHeight = Int(fitSize.height)
            desc.colorTextureFormat = .rgba16Float
            desc.outputTextureFormat = .rgba16Float
            desc.colorProcessingMode = colorMode

            if let s = desc.makeSpatialScaler(device: device, compiler: compiler) {
                s.fence = scalerFence
                let outDesc = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .rgba16Float,
                    width: Int(fitSize.width),
                    height: Int(fitSize.height),
                    mipmapped: false)
                outDesc.usage = s.outputTextureUsage
                // MTLFXSpatialScaler.h: the output texture must use private
                // storage. (The macOS default is managed.)
                outDesc.storageMode = .private
                if let outTex = device.makeTexture(descriptor: outDesc) {
                    scaler = (s, outTex)
                    scalerRebuilt = true
                }
            }
        }

        // EWA runs whenever we sample the source ourselves (no MetalFX): the
        // downscale path proper, and the upscale fallback when Metal4FX is
        // unavailable. Scaling off is 1:1 in pixel space — bilinear suffices.
        let useEWA = scalingEnabled && scaler == nil
        let scalingMode = !scalingEnabled ? ""
            : scaler != nil ? "\nUpscaling: MetalFX"
            : needsUpscale ? "\nUpscaling: ewa_lanczossharp"
            : "\nDownscaling: ewa_lanczossharp"

        if let (s, output) = scaler {
            s.colorTexture = inputTexture
            s.outputTexture = output
            s.inputContentWidth = inputTexture.width
            s.inputContentHeight = inputTexture.height
        }

        info = """
            Slide: \(currentIndex + 1) of \(imagePaths.count)
            File: \(imagePaths.indices.contains(currentIndex) ? imagePaths[currentIndex].lastPathComponent : "—")
            Input: \(inputTexture.width)x\(inputTexture.height)\(textureIsHDR ? " HDR" : "")
            Output: \(Int(displaySize.width))x\(Int(displaySize.height))\(scalingMode)
            """

        let u = scaleBuffer.contents().assumingMemoryBound(to: Float.self)
        u[0] = Float(displaySize.width / viewportSize.width)
        u[1] = Float(displaySize.height / viewportSize.height)
        u[2] = useEWA ? Float(max(CGFloat(inputTexture.width) / max(fitSize.width, 1),
                                  CGFloat(inputTexture.height) / max(fitSize.height, 1))) : 1

        residencySet.removeAllAllocations()
        residencySet.addAllocation(inputTexture)
        residencySet.addAllocation(scaleBuffer)
        residencySet.addAllocation(kernelLUT)
        if let (_, output) = scaler { residencySet.addAllocation(output) }
        residencySet.commit()

        let finalTexture = scaler?.1 ?? inputTexture

        argumentTable.setTexture(finalTexture.gpuResourceID, index: 0)
        if useEWA { argumentTable.setTexture(kernelLUT.gpuResourceID, index: 1) }
        argumentTable.setAddress(scaleBuffer.gpuAddress, index: 0)

        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)
        scaler?.0.encode(commandBuffer: commandBuffer)
        // A freshly created scaler's first encode can emit NaN tiles (black
        // rectangles on screen): the Metal-4 MetalFX path reads uninitialized
        // internal scratch memory on first use (macOS 26.5). Re-encoding the same
        // work is reliably clean, so warm every new scaler up with a second pass;
        // its fence orders the two encodes and the presented result comes from
        // the clean one.
        if scalerRebuilt { scaler?.0.encode(commandBuffer: commandBuffer) }

        guard let encoder = commandBuffer.makeRenderCommandEncoder(
            descriptor: passDescriptor,
            options: MTL4RenderEncoderOptions()) else { return }

        encoder.waitForFence(scalerFence, beforeEncoderStages: .fragment)
        encoder.setRenderPipelineState(useEWA ? pipelineEWA : pipeline)
        encoder.setArgumentTable(argumentTable, stages: [.vertex, .fragment])
        encoder.drawPrimitives(primitiveType: .triangle, vertexStart: 0, vertexCount: 6)
        encoder.endEncoding()

        commandBuffer.endCommandBuffer()

        queue.waitForDrawable(drawable)
        queue.commit([commandBuffer])
        pendingFrameValue += 1
        queue.signalEvent(frameEvent, value: pendingFrameValue)
        queue.signalDrawable(drawable)
        drawable.present()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        view.needsDisplay = true
    }

    func deleteSlide() {
        guard !imagePaths.isEmpty else { return }
        try? FileManager.default.trashItem(at: imagePaths[currentIndex], resultingItemURL: nil)
        imagePaths.remove(at: currentIndex)
        goToSlide(currentIndex)
    }

    func goToSlide(_ index: Int) {
        slideChangedTime = Date()
        guard !imagePaths.isEmpty else {
            loadGeneration += 1  // cancel any in-flight decode
            texture = nil
            view?.needsDisplay = true
            return
        }
        currentIndex = (index % imagePaths.count + imagePaths.count) % imagePaths.count
        loadSlide(imagePaths[currentIndex])
    }

    func toggleShuffle() {
        guard !imagePaths.isEmpty else { return }
        let currentPath = imagePaths[currentIndex]
        imagePaths = shuffleEnabled ? imagePaths.shuffled() : imagePaths.sorted { $0.path < $1.path }
        currentIndex = imagePaths.firstIndex(of: currentPath) ?? 0
    }
}
