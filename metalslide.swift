import SwiftUI
import Metal
import MetalKit
import MetalFX

@main
struct MetalSlide: App {
    var body: some Scene {
        Window("MetalSlide", id: "main") {
            MetalView()
        }
    }
}

struct MetalView: NSViewRepresentable {
    func makeCoordinator() -> Renderer { Renderer() }

    func makeNSView(context: Context) -> MTKView {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.runModal()
        let url = panel.url!
        context.coordinator.imagePaths = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil)!.allObjects
            .map { $0 as! URL }
            .filter { ["jpg", "jpeg", "png"].contains($0.pathExtension) }
            .shuffled()
        context.coordinator.lastIndex = -1
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.delegate = context.coordinator
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) {
            context.coordinator.handleKey($0)
            return nil
        }
        context.coordinator.view = view
        view.needsDisplay = true
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
    }
}

class Renderer: NSObject, MTKViewDelegate {
    var imagePaths: [URL] = []
    var currentIndex = 0
    var lastIndex = -1
    var device: MTLDevice!
    var queue: MTL4CommandQueue!
    var pipeline: MTLRenderPipelineState?
    var allocator: MTL4CommandAllocator!
    var argumentTable: MTL4ArgumentTable!
    var residencySet: MTLResidencySet!
    var texture: MTLTexture?
    weak var view: MTKView?
    var scaleBuffer: MTLBuffer!
    var compiler: MTL4Compiler!
    var spatialScaler: MTL4FXSpatialScaler?
    var scalerOutput: MTLTexture?
    var lastViewportSize = CGSize.zero
    var scalerFence: MTLFence!
    var commandBuffer: MTL4CommandBuffer!
    var preloadedTextures: [Int: MTLTexture] = [:]
    var preloadedScalers: [Int: (MTL4FXSpatialScaler, MTLTexture)] = [:]
    var preloadQueue = DispatchQueue(label: "preload", qos: .userInitiated)
    var pipelineReady = false

    func initializeMetal(_ view: MTKView) {
        device = view.device
        queue = device.makeMTL4CommandQueue()
        compiler = try! device.makeCompiler(descriptor: MTL4CompilerDescriptor())

        let tableDesc = MTL4ArgumentTableDescriptor()
        tableDesc.maxTextureBindCount = 1
        tableDesc.maxBufferBindCount = 1
        argumentTable = try! device.makeArgumentTable(descriptor: tableDesc)

        scaleBuffer = device.makeBuffer(length: 8, options: .storageModeShared)
        residencySet = try! device.makeResidencySet(descriptor: MTLResidencySetDescriptor())
        allocator = device.makeCommandAllocator()
        scalerFence = device.makeFence()
        commandBuffer = device.makeCommandBuffer()

        let shaderSource = """
            #include <metal_stdlib>
            using namespace metal;
            struct VertexOut { float4 position [[position]]; float2 texCoord; };
            vertex VertexOut vertexShader(uint vid [[vertex_id]], constant float2 &scale [[buffer(0)]]) {
                float2 positions[6] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(-1,1), float2(1,-1), float2(1,1) };
                float2 texCoords[6] = { float2(0,1), float2(1,1), float2(0,0), float2(0,0), float2(1,1), float2(1,0) };
                return { float4(positions[vid] * scale, 0, 1), texCoords[vid] };
            }

            float lanczos(float x, float a) {
                if (x == 0.0) return 1.0;
                if (abs(x) >= a) return 0.0;
                float pi_x = M_PI_F * x;
                return (a * sin(pi_x) * sin(pi_x / a)) / (pi_x * pi_x);
            }

            fragment half4 fragmentShader(VertexOut in [[stage_in]], texture2d<half> tex [[texture(0)]]) {
                float2 texSize = float2(tex.get_width(), tex.get_height());
                float2 texelPos = in.texCoord * texSize;

                half4 color = half4(0.0);
                float totalWeight = 0.0;

                const int radius = 3;
                for (int y = -radius; y <= radius; y++) {
                    for (int x = -radius; x <= radius; x++) {
                        float2 offset = float2(x, y);
                        float2 centerPos = floor(texelPos) + offset + 0.5;
                        float2 samplePos = centerPos / texSize;
                        float2 delta = texelPos - centerPos;
                        float weight = lanczos(delta.x, 3.0) * lanczos(delta.y, 3.0);

                        constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);
                        color += tex.sample(s, samplePos) * weight;
                        totalWeight += weight;
                    }
                }

                return color / totalWeight;
            }
            """
        let options = MTLCompileOptions()
        options.mathMode = .fast

        device.makeLibrary(source: shaderSource, options: options) { [weak self] library, error in
            guard let self = self, let library = library else { return }

            let vertDesc = MTL4LibraryFunctionDescriptor()
            vertDesc.name = "vertexShader"
            vertDesc.library = library

            let fragDesc = MTL4LibraryFunctionDescriptor()
            fragDesc.name = "fragmentShader"
            fragDesc.library = library

            let pipelineDesc = MTL4RenderPipelineDescriptor()
            pipelineDesc.vertexFunctionDescriptor = vertDesc
            pipelineDesc.fragmentFunctionDescriptor = fragDesc
            pipelineDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat

            Task {
                do {
                    self.pipeline = try await self.compiler.makeRenderPipelineState(descriptor: pipelineDesc)
                    self.pipelineReady = true
                    await MainActor.run { self.view?.needsDisplay = true }
                } catch {}
            }
        }
    }

    func loadTexture(index: Int) -> MTLTexture? {
        guard let src = CGImageSourceCreateWithURL(imagePaths[index] as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldAllowFloat: true] as CFDictionary) else { return nil }

        let (width, height) = (cgImage.width, cgImage.height)
        var data = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &data, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.interpolationQuality = .none
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        desc.usage = .shaderRead
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }
        tex.replace(region: MTLRegion(origin: MTLOrigin(), size: MTLSize(width: width, height: height, depth: 1)),
                   mipmapLevel: 0, withBytes: data, bytesPerRow: width * 4)
        return tex
    }

    func preloadAdjacentImages() {
        guard let view = view else { return }
        let viewportSize = CGSize(width: view.drawableSize.width, height: view.drawableSize.height)
        let next = (currentIndex + 1) % imagePaths.count
        let prev = (currentIndex - 1 + imagePaths.count) % imagePaths.count

        preloadQueue.async { [weak self] in
            guard let self = self else { return }

            for index in [next, prev] {
                if self.preloadedTextures[index] == nil, let tex = self.loadTexture(index: index) {
                    self.preloadedTextures[index] = tex

                    let imageAspect = CGFloat(tex.width) / CGFloat(tex.height)
                    let viewportAspect = viewportSize.width / viewportSize.height
                    let fitSize = imageAspect > viewportAspect
                        ? CGSize(width: viewportSize.width, height: viewportSize.width / imageAspect)
                        : CGSize(width: viewportSize.height * imageAspect, height: viewportSize.height)
                    let (outputWidth, outputHeight) = (Int(fitSize.width), Int(fitSize.height))

                    if outputWidth > tex.width || outputHeight > tex.height,
                       MTLFXSpatialScalerDescriptor.supportsDevice(self.device) {
                        let desc = MTLFXSpatialScalerDescriptor()
                        desc.inputWidth = tex.width
                        desc.inputHeight = tex.height
                        desc.outputWidth = outputWidth
                        desc.outputHeight = outputHeight
                        desc.colorTextureFormat = .rgba8Unorm
                        desc.outputTextureFormat = .rgba8Unorm
                        desc.colorProcessingMode = .perceptual

                        if let scaler = desc.makeSpatialScaler(device: self.device, compiler: self.compiler) {
                            scaler.fence = self.scalerFence
                            let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: outputWidth, height: outputHeight, mipmapped: false)
                            outDesc.usage = scaler.outputTextureUsage
                            if let output = self.device.makeTexture(descriptor: outDesc) {
                                self.preloadedScalers[index] = (scaler, output)
                            }
                        }
                    }
                }
            }
        }
    }

    func updateTexture() {
        guard device != nil, lastIndex != currentIndex else { return }
        lastIndex = currentIndex

        if let preloaded = preloadedTextures[currentIndex] {
            texture = preloaded
            preloadedTextures.removeValue(forKey: currentIndex)

            if let (scaler, output) = preloadedScalers[currentIndex] {
                spatialScaler = scaler
                scalerOutput = output
                preloadedScalers.removeValue(forKey: currentIndex)
            } else {
                spatialScaler = nil
            }
        } else {
            texture = loadTexture(index: currentIndex)
            spatialScaler = nil
        }

        preloadedTextures.removeAll()
        preloadedScalers.removeAll()
        preloadAdjacentImages()
    }

    func draw(in view: MTKView) {
        guard !imagePaths.isEmpty else { return }

        if !pipelineReady {
            if pipeline == nil { initializeMetal(view) }
            return
        }

        updateTexture()

        guard let drawable = view.currentDrawable, let inputTexture = texture, let pipeline = pipeline else { return }

        let viewportSize = CGSize(width: drawable.texture.width, height: drawable.texture.height)
        let imageAspect = CGFloat(inputTexture.width) / CGFloat(inputTexture.height)
        let viewportAspect = viewportSize.width / viewportSize.height
        let fitSize = imageAspect > viewportAspect
            ? CGSize(width: viewportSize.width, height: viewportSize.width / imageAspect)
            : CGSize(width: viewportSize.height * imageAspect, height: viewportSize.height)

        if spatialScaler == nil || lastViewportSize != viewportSize {
            lastViewportSize = viewportSize
            let (outputWidth, outputHeight) = (Int(fitSize.width), Int(fitSize.height))

            if outputWidth > inputTexture.width || outputHeight > inputTexture.height,
               MTLFXSpatialScalerDescriptor.supportsDevice(device) {
                let desc = MTLFXSpatialScalerDescriptor()
                desc.inputWidth = inputTexture.width
                desc.inputHeight = inputTexture.height
                desc.outputWidth = outputWidth
                desc.outputHeight = outputHeight
                desc.colorTextureFormat = .rgba8Unorm
                desc.outputTextureFormat = .rgba8Unorm
                desc.colorProcessingMode = .perceptual

                if let scaler = desc.makeSpatialScaler(device: device, compiler: compiler) {
                    scaler.fence = scalerFence
                    spatialScaler = scaler
                    let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: outputWidth, height: outputHeight, mipmapped: false)
                    outDesc.usage = scaler.outputTextureUsage
                    scalerOutput = device.makeTexture(descriptor: outDesc)
                }
            } else {
                spatialScaler = nil
                scalerOutput = nil
            }
        }

        let renderTexture: MTLTexture
        if let scaler = spatialScaler, let output = scalerOutput {
            scaler.colorTexture = inputTexture
            scaler.outputTexture = output
            scaler.inputContentWidth = inputTexture.width
            scaler.inputContentHeight = inputTexture.height
            renderTexture = output
        } else {
            renderTexture = inputTexture
        }

        let scale = scaleBuffer.contents().assumingMemoryBound(to: Float.self)
        (scale[0], scale[1]) = (Float(fitSize.width / viewportSize.width), Float(fitSize.height / viewportSize.height))

        residencySet.removeAllAllocations()
        residencySet.addAllocation(inputTexture)
        residencySet.addAllocation(scaleBuffer)
        scalerOutput.map { residencySet.addAllocation($0) }
        residencySet.commit()

        argumentTable.setTexture(renderTexture.gpuResourceID, index: 0)
        argumentTable.setAddress(scaleBuffer.gpuAddress, index: 0)

        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)

        spatialScaler?.encode(commandBuffer: commandBuffer)

        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: view.currentMTL4RenderPassDescriptor!, options: MTL4RenderEncoderOptions())!
        if spatialScaler != nil { encoder.waitForFence(scalerFence, beforeEncoderStages: .fragment) }
        encoder.setRenderPipelineState(pipeline)
        encoder.setArgumentTable(argumentTable, stages: [.vertex, .fragment])
        encoder.drawPrimitives(primitiveType: .triangle, vertexStart: 0, vertexCount: 6)
        encoder.endEncoding()
        commandBuffer.endCommandBuffer()
        queue.waitForDrawable(drawable)
        queue.commit([commandBuffer], options: nil)
        queue.signalDrawable(drawable)
        drawable.present()
        allocator.reset()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        view.needsDisplay = true
    }

    func handleKey(_ event: NSEvent) {
        switch event.keyCode {
        case 53: NSApp.terminate(nil)
        case 123:
            currentIndex = (currentIndex - 1 + imagePaths.count) % imagePaths.count
            view?.needsDisplay = true
        case 124, 49:
            currentIndex = (currentIndex + 1) % imagePaths.count
            view?.needsDisplay = true
        case 51:
            try? FileManager.default.trashItem(at: imagePaths[currentIndex], resultingItemURL: nil)
            imagePaths.remove(at: currentIndex)
            if currentIndex >= imagePaths.count {
                currentIndex = 0
            }
            lastIndex = -1
            view?.needsDisplay = true
        default: break
        }
    }
}