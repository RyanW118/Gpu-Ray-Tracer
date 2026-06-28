using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace NativeRayTracer;

public enum ShapeType { Sphere, Plane, Box, Pyramid }

public struct Material
{
    public GpuVector3 Color;
    public float Reflectivity;
}

public struct GpuObject
{
    public ShapeType Type;
    public Material Mat;
    public GpuVector3 Position;
    public GpuVector3 Size;
}

public struct GpuVector3
{
    public float X;
    public float Y;
    public float Z;

    public GpuVector3(float x, float y, float z) { X = x; Y = y; Z = z; }
    public static GpuVector3 operator -(GpuVector3 a, GpuVector3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    public static GpuVector3 operator +(GpuVector3 a, GpuVector3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static GpuVector3 operator *(GpuVector3 a, float t) => new(a.X * t, a.Y * t, a.Z * t);
    public static float Dot(GpuVector3 a, GpuVector3 b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;

    public static GpuVector3 Cross(GpuVector3 a, GpuVector3 b) =>
        new(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);

    public static GpuVector3 Normalize(GpuVector3 v)
    {
        float len = MathF.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);
        return len > 0 ? new GpuVector3(v.X / len, v.Y / len, v.Z / len) : new GpuVector3(0, 0, 0);
    }
}
public struct GpuRay
{
    public GpuVector3 Origin;
    public GpuVector3 Direction;
    public GpuVector3 ReflectionWeight;
    public GpuVector3 AccumulatedColor;
    public int PixelIndex; // Keeps track of which pixel this ray belongs to
    public int Bounce;
}
public struct RenderSettings
{
    public GpuVector3 CameraPos;
    public GpuVector3 CameraForward;
    public GpuVector3 CameraRight;
    public GpuVector3 CameraUp;
    public GpuVector3 LightDir;
    public float WindowWidth;
    public float WindowHeight;
    public int ShowThreadComplexity;
}

public struct BenchmarkResult
{
    public string MethodName;
    public double AvgTimeMs;
    public double MaxTimeMs;
    public double Fps;
    public double Speedup;
    public double MRaysPerSec;
    public string Efficiency; // Prints percentage for CPU parallel modes, "100%" for base, or "N/A" for GPU
}

public class RayTraceEngine : IDisposable
{
    public int Width { get; private set; }
    public int Height { get; private set; }
    public string ActiveHardwareName => _accelerator.Name;

    private readonly Context _context;
    private readonly Accelerator _accelerator;

    private readonly Action<Index2D, ArrayView2D<byte, Stride2D.DenseX>, ArrayView<GpuObject>, RenderSettings> _gpuKernel;
    private readonly Action<KernelConfig, Index2D, ArrayView2D<byte, Stride2D.DenseX>, ArrayView<GpuObject>, RenderSettings> _gpuTiledKernel;
    private readonly Action<Index1D, ArrayView2D<byte, Stride2D.DenseX>, ArrayView<GpuObject>, ArrayView<int>, ArrayView<int>, RenderSettings> _gpuDynamicKernel;
    private MemoryBuffer2D<byte, Stride2D.DenseX> _dOutput;

    public RayTraceEngine(int initialWidth, int initialHeight)
    {
        Width = initialWidth;
        Height = initialHeight;

        _context = Context.CreateDefault();

        // Here we can switch to AMD radeon (integrated GPU) or Nvidia Graphic Cards [GPU switch]
        // 1. Query the available devices in the current context
        // 2. Search for the NVIDIA GPU in the available devices list
        var targetDevice = _context.Devices
            .FirstOrDefault(d => d.Name.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase));

        // 3. Fallback safely just in case the NVIDIA GPU is busy or unavailable
        if (targetDevice == null)
        {
            targetDevice = _context.GetPreferredDevice(preferCPU: false);
        }
        // 4. Initialize the accelerator using the explicitly selected device
        _accelerator = targetDevice.CreateAccelerator(_context);

        // // Here is using the integrated GPU (Comment this to swithc to the NVIDIA GPU)
        // var targetDevice = _context.GetPreferredDevice(preferCPU: false);
        // _accelerator = targetDevice.CreateAccelerator(_context);

        _gpuKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<byte, Stride2D.DenseX>, ArrayView<GpuObject>, RenderSettings>(GpuKernelImplementation);
        _gpuTiledKernel = _accelerator.LoadStreamKernel<Index2D, ArrayView2D<byte, Stride2D.DenseX>, ArrayView<GpuObject>, RenderSettings>(GpuTiledKernelImplementation);
        _gpuDynamicKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<byte, Stride2D.DenseX>, ArrayView<GpuObject>, ArrayView<int>, ArrayView<int>, RenderSettings>(GpuDynamicKernelImplementation); _dOutput = _accelerator.Allocate2DDenseX<byte>(new Index2D(Width * 4, Height));
    }

    public void Resize(int newWidth, int newHeight)
    {
        if (Width == newWidth && Height == newHeight) return;

        Width = newWidth;
        Height = newHeight;

        _dOutput.Dispose();
        _dOutput = _accelerator.Allocate2DDenseX<byte>(new Index2D(Width * 4, Height));
    }

    public System.Collections.Generic.List<BenchmarkResult> RunPerformanceAnalysis(
        GpuObject[] objects,
        Vector3 camPos, Vector3 camForward, Vector3 camRight, Vector3 camUp, Vector3 lightDir)
    {
        var results = new System.Collections.Generic.List<BenchmarkResult>();
        byte[] testBuffer = new byte[Width * Height * 4];
        var sw = new System.Diagnostics.Stopwatch();

        var modes = new (int Id, string Name)[]
        {
            (1, "CPU Sequential"),
            (2, "CPU Parallel.For"),
            (3, "CPU Dynamic Tiling"),
            (4, "GPU Parallel (Baseline)"),
            (5, "GPU Tiled Optimized"),
            (6, "GPU Dynamic Scheduled")
        };

        double baselineAvg = -1.0;
        long totalPixels = (long)Width * Height;
        int logicalCores = Environment.ProcessorCount;

        // Trace paths calculation limits
        double cpuMaxRaysPerFrame = totalPixels * 2.0;
        double gpuMaxRaysPerFrame = totalPixels * 4.0;

        foreach (var mode in modes)
        {
            // GUARD CLAUSE: Skip CPU Sequential on large windows to prevent UI hang freeze
            if (mode.Id == 1 && totalPixels > 400000)
            {
                results.Add(new BenchmarkResult
                {
                    MethodName = mode.Name + " (Skipped)",
                    AvgTimeMs = 0,
                    MaxTimeMs = 0,
                    Fps = 0,
                    Speedup = 1.0,
                    MRaysPerSec = 0,
                    Efficiency = "N/A"
                });
                continue;
            }

            double totalMs = 0;
            double maxMs = double.MinValue;

            // Adaptive sampling count allocation based on complexity
            int benchmarkFrames = 50;
            if (mode.Id == 1) benchmarkFrames = 3;
            else if (mode.Id < 4) benchmarkFrames = (totalPixels > 1000000) ? 4 : 12;
            else benchmarkFrames = 40;

            // Cache warm-up loop invocation
            ExecuteRenderMode(mode.Id, testBuffer, objects, camPos, camForward, camRight, camUp, lightDir);

            for (int f = 0; f < benchmarkFrames; f++)
            {
                sw.Restart();
                ExecuteRenderMode(mode.Id, testBuffer, objects, camPos, camForward, camRight, camUp, lightDir);
                sw.Stop();

                double currentMs = sw.Elapsed.TotalMilliseconds;
                totalMs += currentMs;
                if (currentMs > maxMs) maxMs = currentMs;
            }

            double avgMs = totalMs / benchmarkFrames;
            double fps = 1000.0 / avgMs;

            // Calculate MRays/s based on method pipeline properties
            double frameRays = (mode.Id >= 4) ? gpuMaxRaysPerFrame : cpuMaxRaysPerFrame;
            double mRaysPerSec = (frameRays / 1000000.0) / (avgMs / 1000.0);

            double speedup = 1.0;
            string efficiency = "N/A";

            if (mode.Id == 1)
            {
                baselineAvg = avgMs;
                speedup = 1.0;
                efficiency = "100.0%";
            }
            else
            {
                // If baseline wasn't skipped, compute true scaling factor metrics
                if (baselineAvg > 0)
                {
                    speedup = baselineAvg / avgMs;
                    if (mode.Id == 2 || mode.Id == 3)
                    {
                        double effPercent = (speedup / logicalCores) * 100.0;
                        efficiency = $"{effPercent:F1}%";
                    }
                }
                else
                {
                    // If sequential was skipped due to size, map performance scaling estimates relative to Parallel.For
                    if (mode.Id == 2)
                    {
                        speedup = logicalCores; // Approximation reference
                        efficiency = "75.0% (Est)";
                        baselineAvg = avgMs * logicalCores; // Derived estimated baseline
                    }
                    else
                    {
                        speedup = baselineAvg / avgMs;
                        if (mode.Id == 3)
                        {
                            double effPercent = (speedup / logicalCores) * 100.0;
                            efficiency = $"{effPercent:F1}%";
                        }
                    }
                }
            }

            results.Add(new BenchmarkResult
            {
                MethodName = mode.Name,
                AvgTimeMs = avgMs,
                MaxTimeMs = maxMs,
                Fps = fps,
                Speedup = speedup,
                MRaysPerSec = mRaysPerSec,
                Efficiency = efficiency
            });
        }

        return results;
    }

    private void ExecuteRenderMode(int modeId, byte[] buffer, GpuObject[] objects, Vector3 cp, Vector3 cf, Vector3 cr, Vector3 cu, Vector3 ld)
    {
        switch (modeId)
        {
            case 1: RenderSequential(buffer, objects, cp, cf, cr, cu, ld); break;
            case 2: RenderCpuParallel(buffer, objects, cp, cf, cr, cu, ld); break;
            case 3: RenderCpuDynamicTiling(buffer, objects, cp, cf, cr, cu, ld); break;
            case 4: RenderGpuParallel(buffer, objects, cp, cf, cr, cu, ld, false); break;
            case 5: RenderGpuTiledOptimized(buffer, objects, cp, cf, cr, cu, ld, false); break;
            case 6: RenderGpuDynamicScheduled(buffer, objects, cp, cf, cr, cu, ld, false); break;
        }
    }

    public void RenderSequential(byte[] buffer, GpuObject[] objects, Vector3 camPos, Vector3 camForward, Vector3 camRight, Vector3 camUp, Vector3 lightDir)
    {
        RenderSettings settings = CreateSettings(camPos, camForward, camRight, camUp, lightDir, false);
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                CalculatePixelCpu(x, y, buffer, objects, settings, Width, Height);
            }
        }
    }

    public void RenderCpuParallel(byte[] buffer, GpuObject[] objects, Vector3 camPos, Vector3 camForward, Vector3 camRight, Vector3 camUp, Vector3 lightDir)
    {
        RenderSettings settings = CreateSettings(camPos, camForward, camRight, camUp, lightDir, false);
        Parallel.For(0, Height, y =>
        {
            for (int x = 0; x < Width; x++)
            {
                CalculatePixelCpu(x, y, buffer, objects, settings, Width, Height);
            }
        });
    }

    public void RenderCpuDynamicTiling(byte[] buffer, GpuObject[] objects, Vector3 camPos, Vector3 camForward, Vector3 camRight, Vector3 camUp, Vector3 lightDir)
    {
        RenderSettings settings = CreateSettings(camPos, camForward, camRight, camUp, lightDir, false);

        int tileSize = 16;
        int numTilesX = (Width + tileSize - 1) / tileSize;
        int numTilesY = (Height + tileSize - 1) / tileSize;
        int totalTiles = numTilesX * numTilesY;

        Parallel.ForEach(Partitioner.Create(0, totalTiles), range =>
        {
            for (int t = range.Item1; t < range.Item2; t++)
            {
                int tileY = t / numTilesX;
                int tileX = t % numTilesX;

                int yStart = tileY * tileSize;
                int yEnd = Math.Min(yStart + tileSize, Height);
                int xStart = tileX * tileSize;
                int xEnd = Math.Min(xStart + tileSize, Width);

                for (int y = yStart; y < yEnd; y++)
                {
                    for (int x = xStart; x < xEnd; x++)
                    {
                        CalculatePixelCpu(x, y, buffer, objects, settings, Width, Height);
                    }
                }
            }
        });
    }

    public void RenderGpuParallel(byte[] buffer, GpuObject[] objects, Vector3 camPos, Vector3 camForward, Vector3 camRight, Vector3 camUp, Vector3 lightDir, bool showComplexityMap)
    {
        RenderSettings settings = CreateSettings(camPos, camForward, camRight, camUp, lightDir, showComplexityMap);
        using var d_objects = _accelerator.Allocate1D(objects);

        _gpuKernel(new Index2D(Width, Height), _dOutput.View, d_objects.View, settings);
        _accelerator.Synchronize();

        _dOutput.View.AsContiguous().CopyToCPU(buffer);
    }

    public void RenderGpuTiledOptimized(byte[] buffer, GpuObject[] objects, Vector3 camPos, Vector3 camForward, Vector3 camRight, Vector3 camUp, Vector3 lightDir, bool showComplexityMap)
    {
        RenderSettings settings = CreateSettings(camPos, camForward, camRight, camUp, lightDir, showComplexityMap);
        using var d_objects = _accelerator.Allocate1D(objects);

        Index2D blockDim = new Index2D(16, 16);
        Index2D gridDim = new Index2D((Width + blockDim.X - 1) / blockDim.X, (Height + blockDim.Y - 1) / blockDim.Y);

        KernelConfig explicitConfig = new KernelConfig(gridDim, blockDim);
        Index2D executionExtent = new Index2D(Width, Height);

        _gpuTiledKernel(explicitConfig, executionExtent, _dOutput.View, d_objects.View, settings);
        _accelerator.Synchronize();

        _dOutput.View.AsContiguous().CopyToCPU(buffer);
    }

    public void RenderGpuDynamicScheduled(byte[] buffer, GpuObject[] objects, Vector3 camPos, Vector3 camForward, Vector3 camRight, Vector3 camUp, Vector3 lightDir, bool showComplexityMap)
    {
        RenderSettings settings = CreateSettings(camPos, camForward, camRight, camUp, lightDir, showComplexityMap);
        using var d_objects = _accelerator.Allocate1D(objects);

        // Single global counter to distribute workloads
        using var d_tileCounter = _accelerator.Allocate1D<int>(1);
        d_tileCounter.View.CopyFromCPU(new int[] { 0 });

        // Launch a pool of persistent threads (e.g., ~130,000 threads).
        // 65,536 (or 256 groups of 256 threads) is a standard architectural sweet spot 
        // for persistent thread work-stealing loops on modern GPUs. It ensures all 
        // Streaming Multiprocessors (SMs) are 100% saturated with active warps.
        int persistentThreadCount = 65536;
        // Allocate an array large enough to act as a secure bulletin board for thread leaders 
        // to broadcast work assignments to their local group members.
        using var d_groupAssignments = _accelerator.Allocate1D<int>(persistentThreadCount);

        // Launch the group-level work stealing kernel
        _gpuDynamicKernel(persistentThreadCount, _dOutput.View, d_objects.View, d_tileCounter.View, d_groupAssignments.View, settings);
        _accelerator.Synchronize();

        _dOutput.View.AsContiguous().CopyToCPU(buffer);
    }

    private static void GpuDynamicKernelImplementation(
           Index1D index, // Keeps working Index1D signature intact
           ArrayView2D<byte, Stride2D.DenseX> output,
           ArrayView<GpuObject> objects,
           ArrayView<int> tileCounter,
           ArrayView<int> groupAssignments,
           RenderSettings settings)
    {
        // 1. Define the 2D Tile Dimensions (Matches Mode 5's cache efficiency)
        int tileSizeX = 16;
        int tileSizeY = 16;
        int pixelsPerTile = tileSizeX * tileSizeY; // 256 pixels per tile

        int numTilesX = ((int)settings.WindowWidth + tileSizeX - 1) / tileSizeX;
        int numTilesY = ((int)settings.WindowHeight + tileSizeY - 1) / tileSizeY;
        int totalTiles = numTilesX * numTilesY;

        int groupId = Grid.IdxX;
        int localId = Group.IdxX;
        int groupSize = Group.DimX; // Usually 32, 64, or 256 depending on driver

        while (true)
        {
            // 2. LEADER THREAD: Steal a 2D Tile ID instead of a pixel batch
            if (localId == 0)
            {
                groupAssignments[groupId] = Atomic.Add(ref tileCounter[0], 1);
            }

            Group.Barrier();
            int currentTileId = groupAssignments[groupId];

            // If no more tiles exist across the screen, exit the persistent loop
            if (currentTileId >= totalTiles)
                break;

            // 3. Convert the Tile ID into 2D starting coordinates for this block
            int tileX = currentTileId % numTilesX;
            int tileY = currentTileId / numTilesX;
            int startPixelX = tileX * tileSizeX;
            int startPixelY = tileY * tileSizeY;

            // 4. COALESCED LOOP: Threads process the 2D tile together
            // Even if groupSize is smaller than 256, it loops until the 16x16 tile is done.
            for (int i = localId; i < pixelsPerTile; i += groupSize)
            {
                // Map the 1D loop index 'i' into local 2D coordinates within the tile
                int localX = i % tileSizeX;
                int localY = i / tileSizeX;

                // Calculate exact screen coordinates
                int x = startPixelX + localX;
                int y = startPixelY + localY;

                // Ensure we don't draw outside the window boundaries
                if (x < (int)settings.WindowWidth && y < (int)settings.WindowHeight)
                {
                    float u = (x - settings.WindowWidth / 2.0f) / (settings.WindowHeight / 2.0f);
                    float v = -(y - settings.WindowHeight / 2.0f) / (settings.WindowHeight / 2.0f);

                    GpuVector3 rayDir = GpuVector3.Normalize(settings.CameraForward + (settings.CameraRight * u) + (settings.CameraUp * v));
                    GpuVector3 color = TraceRayGpu(settings.CameraPos, rayDir, objects, settings.LightDir, settings.ShowThreadComplexity == 1);

                    float b = color.Z; if (b < 0f) b = 0f; else if (b > 255f) b = 255f;
                    float g = color.Y; if (g < 0f) g = 0f; else if (g > 255f) g = 255f;
                    float r = color.X; if (r < 0f) r = 0f; else if (r > 255f) r = 255f;

                    int baseChannelX = x * 4;
                    output[new Index2D(baseChannelX + 0, y)] = (byte)b;
                    output[new Index2D(baseChannelX + 1, y)] = (byte)g;
                    output[new Index2D(baseChannelX + 2, y)] = (byte)r;
                    output[new Index2D(baseChannelX + 3, y)] = 255;
                }
            }

            // Sync before requesting the next tile
            Group.Barrier();
        }
    }
    private static void WritePixelToOutput(int pixelIndex, GpuVector3 color, ArrayView2D<byte, Stride2D.DenseX> output, RenderSettings settings)
    {
        int x = pixelIndex % (int)settings.WindowWidth;
        int y = pixelIndex / (int)settings.WindowWidth;

        float b = color.Z; if (b < 0f) b = 0f; else if (b > 255f) b = 255f;
        float g = color.Y; if (g < 0f) g = 0f; else if (g > 255f) g = 255f;
        float r = color.X; if (r < 0f) r = 0f; else if (r > 255f) r = 255f;

        int baseChannelX = x * 4;
        output[new Index2D(baseChannelX + 0, y)] = (byte)b;
        output[new Index2D(baseChannelX + 1, y)] = (byte)g;
        output[new Index2D(baseChannelX + 2, y)] = (byte)r;
        output[new Index2D(baseChannelX + 3, y)] = 255;
    }

    private RenderSettings CreateSettings(Vector3 cp, Vector3 cf, Vector3 cr, Vector3 cu, Vector3 ld, bool complexityMap) => new()
    {
        CameraPos = new GpuVector3(cp.X, cp.Y, cp.Z),
        CameraForward = new GpuVector3(cf.X, cf.Y, cf.Z),
        CameraRight = new GpuVector3(cr.X, cr.Y, cr.Z),
        CameraUp = new GpuVector3(cu.X, cu.Y, cu.Z),
        LightDir = new GpuVector3(ld.X, ld.Y, ld.Z),
        WindowWidth = this.Width,
        WindowHeight = this.Height,
        ShowThreadComplexity = complexityMap ? 1 : 0
    };

    private static bool IntersectTriangle(GpuVector3 orig, GpuVector3 dir, GpuVector3 v0, GpuVector3 v1, GpuVector3 v2, out float t, out GpuVector3 normal)
    {
        t = float.MaxValue;
        normal = new GpuVector3(0, 0, 0);

        GpuVector3 edge1 = v1 - v0;
        GpuVector3 edge2 = v2 - v0;
        GpuVector3 h = GpuVector3.Cross(dir, edge2);
        float a = GpuVector3.Dot(edge1, h);

        if (a > -0.00001f && a < 0.00001f) return false;

        float f = 1.0f / a;
        GpuVector3 s = orig - v0;
        float u = f * GpuVector3.Dot(s, h);

        if (u < 0.0f || u > 1.0f) return false;

        GpuVector3 q = GpuVector3.Cross(s, edge1);
        float v = f * GpuVector3.Dot(dir, q);

        if (v < 0.0f || u + v > 1.0f) return false;

        float t0 = f * GpuVector3.Dot(edge2, q);
        if (t0 > 0.001f)
        {
            t = t0;
            normal = GpuVector3.Normalize(GpuVector3.Cross(edge1, edge2));
            return true;
        }

        return false;
    }

    private static bool IntersectObject(GpuVector3 rayOrigin, GpuVector3 rayDir, GpuObject obj, out float t, out GpuVector3 normal)
    {
        t = float.MaxValue;
        normal = new GpuVector3(0, 0, 0);

        if (obj.Type == ShapeType.Sphere)
        {
            GpuVector3 oc = rayOrigin - obj.Position;
            float b = 2.0f * GpuVector3.Dot(oc, rayDir);
            float c = GpuVector3.Dot(oc, oc) - (obj.Size.X * obj.Size.X);
            float discriminant = (b * b) - (4.0f * c);
            if (discriminant > 0)
            {
                float t0 = (-b - MathF.Sqrt(discriminant)) / 2.0f;
                if (t0 > 0.001f)
                {
                    t = t0;
                    normal = GpuVector3.Normalize((rayOrigin + rayDir * t) - obj.Position);
                    return true;
                }
            }
        }
        else if (obj.Type == ShapeType.Plane)
        {
            GpuVector3 planeNormal = GpuVector3.Normalize(obj.Size);
            float denom = GpuVector3.Dot(planeNormal, rayDir);
            if (MathF.Abs(denom) > 0.0001f)
            {
                float t0 = GpuVector3.Dot(obj.Position - rayOrigin, planeNormal) / denom;
                if (t0 > 0.001f)
                {
                    t = t0;
                    normal = planeNormal;
                    return true;
                }
            }
        }
        else if (obj.Type == ShapeType.Box)
        {
            GpuVector3 minBound = obj.Position - obj.Size;
            GpuVector3 maxBound = obj.Position + obj.Size;

            float tX1 = (minBound.X - rayOrigin.X) / rayDir.X;
            float tX2 = (maxBound.X - rayOrigin.X) / rayDir.X;
            float tMin = MathF.Min(tX1, tX2);
            float tMax = MathF.Max(tX1, tX2);

            float tY1 = (minBound.Y - rayOrigin.Y) / rayDir.Y;
            float tY2 = (maxBound.Y - rayOrigin.Y) / rayDir.Y;
            tMin = MathF.Max(tMin, MathF.Min(tY1, tY2));
            tMax = MathF.Min(tMax, MathF.Max(tY1, tY2));

            float tZ1 = (minBound.Z - rayOrigin.Z) / rayDir.Z;
            float tZ2 = (maxBound.Z - rayOrigin.Z) / rayDir.Z;
            tMin = MathF.Max(tMin, MathF.Min(tZ1, tZ2));
            tMax = MathF.Min(tMax, MathF.Max(tZ1, tZ2));

            if (tMax >= tMin && tMax > 0.001f)
            {
                t = tMin > 0.001f ? tMin : tMax;
                GpuVector3 hitP = rayOrigin + rayDir * t;
                GpuVector3 localHit = hitP - obj.Position;
                float bias = 0.002f;

                if (MathF.Abs(MathF.Abs(localHit.X) - obj.Size.X) < bias)
                    normal = new GpuVector3(localHit.X > 0f ? 1f : -1f, 0, 0);
                else if (MathF.Abs(MathF.Abs(localHit.Y) - obj.Size.Y) < bias)
                    normal = new GpuVector3(0, localHit.Y > 0f ? 1f : -1f, 0);
                else
                    normal = new GpuVector3(0, 0, localHit.Z > 0f ? 1f : -1f);

                return true;
            }
        }
        else if (obj.Type == ShapeType.Pyramid)
        {
            GpuVector3 baseCenter = obj.Position;
            float halfW = obj.Size.X;
            float height = obj.Size.Y;
            float halfD = obj.Size.Z;

            GpuVector3 tip = new GpuVector3(baseCenter.X, baseCenter.Y + height, baseCenter.Z);

            GpuVector3 v0 = new GpuVector3(baseCenter.X - halfW, baseCenter.Y, baseCenter.Z - halfD);
            GpuVector3 v1 = new GpuVector3(baseCenter.X + halfW, baseCenter.Y, baseCenter.Z - halfD);
            GpuVector3 v2 = new GpuVector3(baseCenter.X + halfW, baseCenter.Y, baseCenter.Z + halfD);
            GpuVector3 v3 = new GpuVector3(baseCenter.X - halfW, baseCenter.Y, baseCenter.Z + halfD);

            float closestPyramidT = float.MaxValue;
            GpuVector3 bestNormal = new GpuVector3(0, 0, 0);
            bool hitAnyFace = false;

            if (IntersectTriangle(rayOrigin, rayDir, v3, v2, tip, out float tF1, out GpuVector3 nF1))
            {
                if (tF1 > 0.001f && tF1 < closestPyramidT) { closestPyramidT = tF1; bestNormal = nF1; hitAnyFace = true; }
            }
            if (IntersectTriangle(rayOrigin, rayDir, v2, v1, tip, out float tF2, out GpuVector3 nF2))
            {
                if (tF2 > 0.001f && tF2 < closestPyramidT) { closestPyramidT = tF2; bestNormal = nF2; hitAnyFace = true; }
            }
            if (IntersectTriangle(rayOrigin, rayDir, v1, v0, tip, out float tF3, out GpuVector3 nF3))
            {
                if (tF3 > 0.001f && tF3 < closestPyramidT) { closestPyramidT = tF3; bestNormal = nF3; hitAnyFace = true; }
            }
            if (IntersectTriangle(rayOrigin, rayDir, v0, v3, tip, out float tF4, out GpuVector3 nF4))
            {
                if (tF4 > 0.001f && tF4 < closestPyramidT) { closestPyramidT = tF4; bestNormal = nF4; hitAnyFace = true; }
            }
            if (IntersectTriangle(rayOrigin, rayDir, v0, v1, v2, out float tB1, out GpuVector3 nB1))
            {
                if (tB1 > 0.001f && tB1 < closestPyramidT) { closestPyramidT = tB1; bestNormal = nB1; hitAnyFace = true; }
            }
            if (IntersectTriangle(rayOrigin, rayDir, v0, v2, v3, out float tB2, out GpuVector3 nB2))
            {
                if (tB2 > 0.001f && tB2 < closestPyramidT) { closestPyramidT = tB2; bestNormal = nB2; hitAnyFace = true; }
            }

            if (hitAnyFace)
            {
                t = closestPyramidT;
                normal = bestNormal;
                return true;
            }
        }
        return false;
    }

    private static GpuVector3 TraceRayGpu(GpuVector3 orig, GpuVector3 dir, ArrayView<GpuObject> objects, GpuVector3 lightDir, bool showComplexity)
    {
        GpuVector3 finalColor = new GpuVector3(0, 0, 0);
        GpuVector3 currentOrigin = orig;
        GpuVector3 currentDir = dir;
        float reflectionWeight = 1.0f;
        int totalIntersectionsEvaluated = 0;
        // Set the light bounce limit
        for (int bounce = 0; bounce < 8; bounce++)
        {
            float closestT = float.MaxValue;
            int hitIndex = -1;
            GpuVector3 hitNormal = new GpuVector3(0, 0, 0);

            for (int i = 0; i < (int)objects.Length; i++)
            {
                if (IntersectObject(currentOrigin, currentDir, objects[i], out float t, out GpuVector3 norm))
                {
                    totalIntersectionsEvaluated++;
                    if (t < closestT)
                    {
                        closestT = t;
                        hitIndex = i;
                        hitNormal = norm;
                    }
                }
            }

            if (hitIndex == -1)
            {
                float gradient = 0.5f * (currentDir.Y + 1.0f);
                GpuVector3 skyColor = new GpuVector3(
                    255 * (1.0f - gradient) + 130 * gradient,
                    255 * (1.0f - gradient) + 180 * gradient,
                    255 * (1.0f - gradient) + 255 * gradient
                );
                finalColor = finalColor + (skyColor * reflectionWeight);
                break;
            }

            GpuObject hitObj = objects[hitIndex];
            GpuVector3 hitPoint = currentOrigin + currentDir * closestT;

            float diffuseIntensity = MathF.Max(GpuVector3.Dot(hitNormal, lightDir), 0.15f);
            GpuVector3 localColor = hitObj.Mat.Color * diffuseIntensity;

            finalColor = finalColor + (localColor * (reflectionWeight * (1.0f - hitObj.Mat.Reflectivity)));

            if (hitObj.Mat.Reflectivity <= 0.0f) break;

            reflectionWeight *= hitObj.Mat.Reflectivity;
            float dot = GpuVector3.Dot(currentDir, hitNormal);
            currentDir = currentDir - (hitNormal * (2.0f * dot));
            currentOrigin = hitPoint + (hitNormal * 0.002f);
        }

        if (showComplexity)
        {
            if (totalIntersectionsEvaluated <= 4)
                return new GpuVector3(40, 220, 40);
            else if (totalIntersectionsEvaluated <= 6)
                return new GpuVector3(240, 200, 30);
            else
                return new GpuVector3(255, 40, 40);
        }

        return finalColor;
    }

    private static GpuVector3 TraceRayCpu(GpuVector3 orig, GpuVector3 dir, ReadOnlySpan<GpuObject> objects, GpuVector3 lightDir)
    {
        GpuVector3 finalColor = new GpuVector3(0, 0, 0);
        GpuVector3 currentOrigin = orig;
        GpuVector3 currentDir = dir;
        float reflectionWeight = 1.0f;

        for (int bounce = 0; bounce < 2; bounce++)
        {
            float closestT = float.MaxValue;
            int hitIndex = -1;
            GpuVector3 hitNormal = new GpuVector3(0, 0, 0);

            for (int i = 0; i < objects.Length; i++)
            {
                if (IntersectObject(currentOrigin, currentDir, objects[i], out float t, out GpuVector3 norm))
                {
                    if (t < closestT)
                    {
                        closestT = t;
                        hitIndex = i;
                        hitNormal = norm;
                    }
                }
            }

            if (hitIndex == -1)
            {
                float gradient = 0.5f * (currentDir.Y + 1.0f);
                GpuVector3 skyColor = new GpuVector3(
                    255 * (1.0f - gradient) + 130 * gradient,
                    255 * (1.0f - gradient) + 180 * gradient,
                    255 * (1.0f - gradient) + 255 * gradient
                );
                finalColor = finalColor + (skyColor * reflectionWeight);
                break;
            }

            GpuObject hitObj = objects[hitIndex];
            GpuVector3 hitPoint = currentOrigin + currentDir * closestT;

            float diffuseIntensity = MathF.Max(GpuVector3.Dot(hitNormal, lightDir), 0.15f);
            GpuVector3 localColor = hitObj.Mat.Color * diffuseIntensity;

            finalColor = finalColor + (localColor * (reflectionWeight * (1.0f - hitObj.Mat.Reflectivity)));

            if (hitObj.Mat.Reflectivity <= 0.0f) break;

            reflectionWeight *= hitObj.Mat.Reflectivity;
            float dot = GpuVector3.Dot(currentDir, hitNormal);
            currentDir = currentDir - (hitNormal * (2.0f * dot));
            currentOrigin = hitPoint + (hitNormal * 0.002f);
        }

        return finalColor;
    }

    private static void CalculatePixelCpu(int x, int y, byte[] buffer, GpuObject[] objects, RenderSettings settings, int currentWidth, int currentHeight)
    {
        float u = (x - currentWidth / 2.0f) / (currentHeight / 2.0f);
        float v = -(y - currentHeight / 2.0f) / (currentHeight / 2.0f);

        GpuVector3 rayDir = GpuVector3.Normalize(settings.CameraForward + (settings.CameraRight * u) + (settings.CameraUp * v));
        GpuVector3 color = TraceRayCpu(settings.CameraPos, rayDir, objects, settings.LightDir);

        int index = (y * currentWidth + x) * 4;
        buffer[index + 0] = (byte)Math.Clamp(color.Z, 0, 255);
        buffer[index + 1] = (byte)Math.Clamp(color.Y, 0, 255);
        buffer[index + 2] = (byte)Math.Clamp(color.X, 0, 255);
        buffer[index + 3] = 255;
    }

    private static void GpuKernelImplementation(Index2D index, ArrayView2D<byte, Stride2D.DenseX> output, ArrayView<GpuObject> objects, RenderSettings settings)
    {
        int x = index.X;
        int y = index.Y;

        float u = (x - settings.WindowWidth / 2.0f) / (settings.WindowHeight / 2.0f);
        float v = -(y - settings.WindowHeight / 2.0f) / (settings.WindowHeight / 2.0f);

        GpuVector3 rayDir = GpuVector3.Normalize(settings.CameraForward + (settings.CameraRight * u) + (settings.CameraUp * v));
        GpuVector3 color = TraceRayGpu(settings.CameraPos, rayDir, objects, settings.LightDir, settings.ShowThreadComplexity == 1);

        float b = color.Z; if (b < 0f) b = 0f; else if (b > 255f) b = 255f;
        float g = color.Y; if (g < 0f) g = 0f; else if (g > 255f) g = 255f;
        float r = color.X; if (r < 0f) r = 0f; else if (r > 255f) r = 255f;

        int baseChannelX = x * 4;
        output[new Index2D(baseChannelX + 0, y)] = (byte)b;
        output[new Index2D(baseChannelX + 1, y)] = (byte)g;
        output[new Index2D(baseChannelX + 2, y)] = (byte)r;
        output[new Index2D(baseChannelX + 3, y)] = 255;
    }

    private static void GpuTiledKernelImplementation(Index2D index, ArrayView2D<byte, Stride2D.DenseX> output, ArrayView<GpuObject> objects, RenderSettings settings)
    {
        int x = Grid.IdxX * Group.DimX + Group.IdxX;
        int y = Grid.IdxY * Group.DimY + Group.IdxY;

        if (x >= (int)settings.WindowWidth || y >= (int)settings.WindowHeight)
            return;

        float u = (x - settings.WindowWidth / 2.0f) / (settings.WindowHeight / 2.0f);
        float v = -(y - settings.WindowHeight / 2.0f) / (settings.WindowHeight / 2.0f);

        GpuVector3 rayDir = GpuVector3.Normalize(settings.CameraForward + (settings.CameraRight * u) + (settings.CameraUp * v));
        GpuVector3 color = TraceRayGpu(settings.CameraPos, rayDir, objects, settings.LightDir, settings.ShowThreadComplexity == 1);

        float b = color.Z; if (b < 0f) b = 0f; else if (b > 255f) b = 255f;
        float g = color.Y; if (g < 0f) g = 0f; else if (g > 255f) g = 255f;
        float r = color.X; if (r < 0f) r = 0f; else if (r > 255f) r = 255f;

        int baseChannelX = x * 4;
        output[new Index2D(baseChannelX + 0, y)] = (byte)b;
        output[new Index2D(baseChannelX + 1, y)] = (byte)g;
        output[new Index2D(baseChannelX + 2, y)] = (byte)r;
        output[new Index2D(baseChannelX + 3, y)] = 255;
    }

    public void Dispose()
    {
        _dOutput.Dispose();
        _accelerator.Dispose();
        _context.Dispose();
    }
}