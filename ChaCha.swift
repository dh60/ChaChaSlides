import SwiftUI
import Metal
import MetalKit
import MetalFX

@main
struct ChaCha: App {
    var body: some Scene {
        Window("ChaCha", id: "main") {
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
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.delegate = context.coordinator
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) {
            context.coordinator.handleKey($0)
            return nil
        }
        context.coordinator.view = view
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
    var pipeline: MTLRenderPipelineState!
    var allocator: MTL4CommandAllocator!
    var argumentTable: MTL4ArgumentTable!
    var residencySet: MTLResidencySet!
    var texture: MTLTexture?
    weak var view: MTKView?
    var scaleBuffer: MTLBuffer!
    var imageSize = CGSize.zero
    var compiler: MTL4Compiler!
    var spatialScaler: MTL4FXSpatialScaler?
    var scalerOutput: MTLTexture?
    var lastViewportSize = CGSize.zero
    var scalerFence: MTLFence!

    func initializeMetal(_ view: MTKView) {
        device = view.device
        queue = device.makeMTL4CommandQueue()
        compiler = try! device.makeCompiler(descriptor: MTL4CompilerDescriptor())

        let library = try! device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            struct VertexOut { float4 position [[position]]; float2 texCoord; };
            vertex VertexOut vertexShader(uint vid [[vertex_id]], constant float2 &scale [[buffer(0)]]) {
                float2 positions[6] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(-1,1), float2(1,-1), float2(1,1) };
                float2 texCoords[6] = { float2(0,1), float2(1,1), float2(0,0), float2(0,0), float2(1,1), float2(1,0) };
                return { float4(positions[vid] * scale, 0, 1), texCoords[vid] };
            }
            fragment half4 fragmentShader(VertexOut in [[stage_in]], texture2d<half> tex [[texture(0)]]) {
                constexpr sampler s(mag_filter::linear, min_filter::linear);
                return tex.sample(s, in.texCoord);
            }
            """, options: nil)

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
        pipeline = try! compiler.makeRenderPipelineState(descriptor: pipelineDesc, dynamicLinkingDescriptor: nil, compilerTaskOptions: nil)

        let tableDesc = MTL4ArgumentTableDescriptor()
        tableDesc.maxTextureBindCount = 1
        tableDesc.maxBufferBindCount = 1
        argumentTable = try! device.makeArgumentTable(descriptor: tableDesc)

        scaleBuffer = device.makeBuffer(length: 8, options: .storageModeShared)
        residencySet = try! device.makeResidencySet(descriptor: MTLResidencySetDescriptor())
        allocator = device.makeCommandAllocator()
        scalerFence = device.makeFence()
    }

    func updateTexture() {
        guard device != nil,
              lastIndex != currentIndex,
              let src = CGImageSourceCreateWithURL(imagePaths[currentIndex] as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return }

        lastIndex = currentIndex
        let (width, height) = (cgImage.width, cgImage.height)
        imageSize = CGSize(width: width, height: height)

        var data = [UInt8](repeating: 0, count: width * height * 4)
        CGContext(data: &data, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            .draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        if texture?.width != width || texture?.height != height {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
            desc.usage = .shaderRead
            texture = device.makeTexture(descriptor: desc)
            spatialScaler = nil
        }

        texture?.replace(region: MTLRegion(origin: MTLOrigin(), size: MTLSize(width: width, height: height, depth: 1)),
                        mipmapLevel: 0, withBytes: data, bytesPerRow: width * 4)
    }

    func setupScaler(viewportSize: CGSize) {
        guard let inputTexture = texture else { return }

        let imageAspect = imageSize.width / imageSize.height
        let viewportAspect = viewportSize.width / viewportSize.height
        let (outputWidth, outputHeight) = imageAspect > viewportAspect
            ? (Int(viewportSize.width), Int(viewportSize.width / imageAspect))
            : (Int(viewportSize.height * imageAspect), Int(viewportSize.height))

        guard outputWidth > inputTexture.width || outputHeight > inputTexture.height else {
            spatialScaler = nil
            scalerOutput = nil
            return
        }

        let desc = MTLFXSpatialScalerDescriptor()
        desc.inputWidth = inputTexture.width
        desc.inputHeight = inputTexture.height
        desc.outputWidth = outputWidth
        desc.outputHeight = outputHeight
        desc.colorTextureFormat = .rgba8Unorm
        desc.outputTextureFormat = .rgba8Unorm
        desc.colorProcessingMode = .perceptual

        guard MTLFXSpatialScalerDescriptor.supportsDevice(device),
              let scaler = desc.makeSpatialScaler(device: device, compiler: compiler) else { return }

        scaler.fence = scalerFence
        spatialScaler = scaler

        let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: outputWidth, height: outputHeight, mipmapped: false)
        outDesc.usage = scaler.outputTextureUsage
        scalerOutput = device.makeTexture(descriptor: outDesc)
    }

    func draw(in view: MTKView) {
        guard !imagePaths.isEmpty else { return }

        if pipeline == nil {
            initializeMetal(view)
        }

        updateTexture()

        guard let drawable = view.currentDrawable, let inputTexture = texture else { return }

        let viewportSize = CGSize(width: drawable.texture.width, height: drawable.texture.height)

        if spatialScaler == nil || lastViewportSize != viewportSize {
            lastViewportSize = viewportSize
            setupScaler(viewportSize: viewportSize)
        }

        let (renderTexture, renderSize): (MTLTexture, CGSize)
        if let scaler = spatialScaler, let output = scalerOutput {
            scaler.colorTexture = inputTexture
            scaler.outputTexture = output
            scaler.inputContentWidth = inputTexture.width
            scaler.inputContentHeight = inputTexture.height
            (renderTexture, renderSize) = (output, CGSize(width: output.width, height: output.height))
        } else {
            (renderTexture, renderSize) = (inputTexture, imageSize)
        }

        let scalePtr = scaleBuffer.contents().assumingMemoryBound(to: Float.self)
        (scalePtr[0], scalePtr[1]) = (Float(renderSize.width / viewportSize.width), Float(renderSize.height / viewportSize.height))

        residencySet.removeAllAllocations()
        residencySet.addAllocation(inputTexture)
        scalerOutput.map { residencySet.addAllocation($0) }
        residencySet.addAllocation(scaleBuffer)
        residencySet.commit()

        argumentTable.setTexture(renderTexture.gpuResourceID, index: 0)
        argumentTable.setAddress(scaleBuffer.gpuAddress, index: 0)

        allocator.reset()
        let commandBuffer = device.makeCommandBuffer()!
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
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func handleKey(_ event: NSEvent) {
        switch event.keyCode {
        case 53: NSApp.terminate(nil)
        case 123:
            currentIndex = (currentIndex - 1 + imagePaths.count) % imagePaths.count
        case 124, 49:
            currentIndex = (currentIndex + 1) % imagePaths.count
        case 51:
            try? FileManager.default.trashItem(at: imagePaths[currentIndex], resultingItemURL: nil)
        default: break
        }
    }
}