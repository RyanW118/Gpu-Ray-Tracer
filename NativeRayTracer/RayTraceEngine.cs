using System;
using System.Numerics;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;

namespace NativeRayTracer;

public struct Sphere
{
    public Vector3 Center;
    public float Radius;
    public Vector3 Color;
}

// FIXED: Plain data structure completely free of hidden CPU .NET 10 hardware SIMD intrinsics
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
    public static GpuVector3 Normalize(GpuVector3 v)
    {
        float len = MathF.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);
        return len > 0 ? new GpuVector3(v.X / len, v.Y / len, v.Z / len) : new GpuVector3(0, 0, 0);
    }
}

public struct GpuSphere
{
    public GpuVector3 Center;
    public float Radius;
    public GpuVector3 Color;
}

public class RayTraceEngine : IDisposable
{
    public const int Width = 640;
    public const int Height = 480;

    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly Action<Index2D, ArrayView2D<byte, Stride2D.DenseX>, GpuSphere> _gpuKernel;

    public RayTraceEngine()
    {
        _context = Context.CreateDefault();
        _accelerator = _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);
        _gpuKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<byte, Stride2D.DenseX>, GpuSphere>(GpuKernelImplementation);
    }

    public void RenderSequential(byte[] buffer, Sphere sphere, Vector3 lightDir)
    {
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                CalculatePixel(x, y, buffer, sphere, lightDir);
            }
        }
    }

    public void RenderCpuParallel(byte[] buffer, Sphere sphere, Vector3 lightDir)
    {
        Parallel.For(0, Height, y =>
        {
            for (int x = 0; x < Width; x++)
            {
                CalculatePixel(x, y, buffer, sphere, lightDir);
            }
        });
    }

    public void RenderGpuParallel(byte[] buffer, Sphere sphere)
    {
        // Safely map values into our GPU-safe structures before invocation
        GpuSphere gpuSphere = new GpuSphere
        {
            Center = new GpuVector3(sphere.Center.X, sphere.Center.Y, sphere.Center.Z),
            Radius = sphere.Radius,
            Color = new GpuVector3(sphere.Color.X, sphere.Color.Y, sphere.Color.Z)
        };

        using var d_output = _accelerator.Allocate2DDenseX<byte>(new Index2D(Width * 4, Height));
        _gpuKernel(new Index2D(Width, Height), d_output.View, gpuSphere);
        _accelerator.Synchronize();
        d_output.View.AsContiguous().CopyToCPU(buffer);
    }

    private static void CalculatePixel(int x, int y, byte[] buffer, Sphere sphere, Vector3 lightDir)
    {
        Vector3 cameraPos = Vector3.Zero;
        float u = (x - Width / 2.0f) / (Height / 2.0f);
        float v = -(y - Height / 2.0f) / (Height / 2.0f);

        Vector3 rayDir = Vector3.Normalize(new Vector3(u, v, -1.0f));
        Vector3 oc = cameraPos - sphere.Center;

        float a = Vector3.Dot(rayDir, rayDir);
        float b = 2.0f * Vector3.Dot(oc, rayDir);
        float c = Vector3.Dot(oc, oc) - (sphere.Radius * sphere.Radius);
        float discriminant = (b * b) - (4 * a * c);

        int index = (y * Width + x) * 4;

        if (discriminant > 0)
        {
            float t = (-b - MathF.Sqrt(discriminant)) / (2.0f * a);
            Vector3 hitPoint = cameraPos + rayDir * t;
            Vector3 normal = Vector3.Normalize(hitPoint - sphere.Center);
            float intensity = MathF.Max(Vector3.Dot(normal, lightDir), 0.1f);

            buffer[index + 0] = (byte)(sphere.Color.Z * intensity); // B
            buffer[index + 1] = (byte)(sphere.Color.Y * intensity); // G
            buffer[index + 2] = (byte)(sphere.Color.X * intensity); // R
            buffer[index + 3] = 255;                                // A
        }
        else
        {
            float gradient = 0.5f * (rayDir.Y + 1.0f);
            buffer[index + 0] = (byte)(255 * (1.0f - gradient) + 255 * gradient);
            buffer[index + 1] = (byte)(255 * (1.0f - gradient) + 200 * gradient);
            buffer[index + 2] = (byte)(255 * (1.0f - gradient) + 150 * gradient);
            buffer[index + 3] = 255;
        }
    }

    // FIXED: Uses GpuVector3 structures to compile directly down to GPU assemblies without .NET 10 vector crashes
    private static void GpuKernelImplementation(Index2D index, ArrayView2D<byte, Stride2D.DenseX> output, GpuSphere sphere)
    {
        int x = index.X;
        int y = index.Y;
        GpuVector3 cameraPos = new GpuVector3(0, 0, 0);
        GpuVector3 lightDir = GpuVector3.Normalize(new GpuVector3(1.0f, 1.0f, -1.0f));

        float u = (x - 640 / 2.0f) / (480 / 2.0f);
        float v = -(y - 480 / 2.0f) / (480 / 2.0f);

        GpuVector3 rayDir = GpuVector3.Normalize(new GpuVector3(u, v, -1.0f));
        GpuVector3 oc = cameraPos - sphere.Center;

        float a = GpuVector3.Dot(rayDir, rayDir);
        float b = 2.0f * GpuVector3.Dot(oc, rayDir);
        float c = GpuVector3.Dot(oc, oc) - (sphere.Radius * sphere.Radius);
        float discriminant = (b * b) - (4 * a * c);

        int baseChannelX = x * 4;

        if (discriminant > 0)
        {
            float t = (-b - MathF.Sqrt(discriminant)) / (2.0f * a);
            GpuVector3 hitPoint = cameraPos + rayDir * t;
            GpuVector3 normal = GpuVector3.Normalize(hitPoint - sphere.Center);
            float intensity = MathF.Max(GpuVector3.Dot(normal, lightDir), 0.1f);

            output[new Index2D(baseChannelX + 0, y)] = (byte)(sphere.Color.Z * intensity);
            output[new Index2D(baseChannelX + 1, y)] = (byte)(sphere.Color.Y * intensity);
            output[new Index2D(baseChannelX + 2, y)] = (byte)(sphere.Color.X * intensity);
            output[new Index2D(baseChannelX + 3, y)] = 255;
        }
        else
        {
            float gradient = 0.5f * (rayDir.Y + 1.0f);
            output[new Index2D(baseChannelX + 0, y)] = (byte)(255 * (1.0f - gradient) + 255 * gradient);
            output[new Index2D(baseChannelX + 1, y)] = (byte)(255 * (1.0f - gradient) + 200 * gradient);
            output[new Index2D(baseChannelX + 2, y)] = (byte)(255 * (1.0f - gradient) + 150 * gradient);
            output[new Index2D(baseChannelX + 3, y)] = 255;
        }
    }

    public void Dispose()
    {
        _accelerator.Dispose();
        _context.Dispose();
    }
}