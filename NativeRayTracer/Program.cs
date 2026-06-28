using System;
using System.Diagnostics;
using System.Numerics;
using Silk.NET.Input;
using Silk.NET.OpenGL;
using Silk.NET.Windowing;

namespace NativeRayTracer;

/// Main orchestration and application entry point for the cross-platform Interactive Ray Tracer.
/// Handles window life cycle, OpenGL initialization, user interactive inputs, and runtime performance rendering switching.

class Program
{
    //The UI interactive window context wrapper handled by Silk.NET.
    private static IWindow? _window;

    //The OpenGL API connection handle instance for texture compilation and display buffering.
    private static GL? _gl;

    //The active hardware acceleration execution logic module for Ray Tracing computing patterns.
    private static RayTraceEngine? _engine;

    //The unmanaged flat array stream memory buffer containing target screen frame colors.
    private static byte[]? _pixelBuffer;

    //The cross-compiled active standard reference handle to the screen canvas pipeline texture mapping context.
    private static uint _textureId;

    //The vertex array object configuration reference identifier binding data descriptors.
    private static uint _vao;

    //The vertex allocation data buffer segment identifier array pointer map.
    private static uint _vbo;

    //The linked multi-stage glsl shader translation application system program layout pointer mapping block.
    private static uint _shaderProgram;

    //The index representing current thread configuration mode mapping schemes (1 to 5).
    private static int _currentMode = 1;

    //The descriptive string matching the running algorithmic strategy layout representation.
    private static string _modeName = "1. CPU Sequential";

    //A toggle flag forcing execution engines to map localized execution density metrics via a visualization heatmap view.
    private static bool _showComplexityMap = false;

    //Flag utilized to instruct systems to instantly compile multi-tier architecture evaluation sweeps via F5.
    private static bool _triggerBenchmarkSnapshot = false;

    //The localized 3-space positional representation vector of the active viewport viewer.
    private static Vector3 _cameraPos = new Vector3(0, 0, 0);

    //The horizontal angular direction modifier mapping side-to-side rotation constraints.
    private static float _yaw = -90.0f;

    //The vertical angular direction tracking constraint variable mapping horizon pitch bounds.
    private static float _pitch = 0.0f;

    //The relative state coordinate snapshot mapping mouse vector positioning on previous updates.
    private static Vector2 _lastMousePos;

    //A state tracker checking if a left click drag interaction matrix update loop is executing.
    private static bool _isMouseDragging = false;

    //The pointer targeting active peripheral components transmitting keyboard button flags.
    private static IKeyboard? _activeKeyboard;

    // A toggle flag to spawn a highly reflective sphere to stress test GPU dynamic scheduling.
    private static bool _showReflectiveSphere = false;

    // Program execution launcher. Configures environment variables, configures the execution environment size, and runs loops.
    static void Main(string[] args)
    {
        var options = WindowOptions.Default;
        options.Size = new Silk.NET.Maths.Vector2D<int>(640, 480);
        options.Title = "Interactive Ray Tracer Setup";
        options.VSync = false;

        _window = Window.Create(options);
        _window.Load += OnLoad;
        _window.Update += OnUpdate;
        _window.Render += OnRender;
        _window.Closing += OnClosing;

        _window.Run();
    }

    // Allocates critical device dependencies, establishes inputs, and builds required layout buffers.
    private static unsafe void OnLoad()
    {
        if (_window == null) return;

        _gl = _window.CreateOpenGL();
        _engine = new RayTraceEngine(_window.Size.X, _window.Size.Y);
        _pixelBuffer = new byte[_engine.Width * _engine.Height * 4];

        _window.Resize += OnWindowResize;

        var inputContext = _window.CreateInput();

        if (inputContext.Keyboards.Count > 0)
        {
            _activeKeyboard = inputContext.Keyboards[0];
            _activeKeyboard.KeyDown += OnKeyDown;
        }
        if (inputContext.Mice.Count > 0)
        {
            var mouse = inputContext.Mice[0];
            mouse.MouseDown += OnMouseDown;
            mouse.MouseUp += OnMouseUp;
            mouse.MouseMove += OnMouseMove;
        }

        string vertexShaderSource = @"#version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }";

        string fragmentShaderSource = @"#version 330 core
        out vec4 FragColor;
        in vec2 TexCoord;
        uniform sampler2D texture1;
        void main() {
            FragColor = texture2D(texture1, TexCoord);
        }";

        uint vertexShader = _gl.CreateShader(ShaderType.VertexShader);
        _gl.ShaderSource(vertexShader, vertexShaderSource);
        _gl.CompileShader(vertexShader);

        uint fragmentShader = _gl.CreateShader(ShaderType.FragmentShader);
        _gl.ShaderSource(fragmentShader, fragmentShaderSource);
        _gl.CompileShader(fragmentShader);

        _shaderProgram = _gl.CreateProgram();
        _gl.AttachShader(_shaderProgram, vertexShader);
        _gl.AttachShader(_shaderProgram, fragmentShader);
        _gl.LinkProgram(_shaderProgram);

        _gl.DeleteShader(vertexShader);
        _gl.DeleteShader(fragmentShader);

        float[] vertices = {
            -1.0f,  1.0f,  0.0f, 0.0f,
            -1.0f, -1.0f,  0.0f, 1.0f,
             1.0f, -1.0f,  1.0f, 1.0f,

            -1.0f,  1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 1.0f,
             1.0f,  1.0f,  1.0f, 0.0f
        };

        _vao = _gl.GenVertexArray();
        _vbo = _gl.GenBuffer();

        _gl.BindVertexArray(_vao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        fixed (float* v = vertices)
        {
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertices.Length * sizeof(float)), v, BufferUsageARB.StaticDraw);
        }

        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, 4 * sizeof(float), (void*)0);
        _gl.EnableVertexAttribArray(1);
        _gl.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, 4 * sizeof(float), (void*)(2 * sizeof(float)));

        _textureId = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _textureId);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
    }

    // Reacts to OS canvas resize instructions to recalculate viewport sizing parameters gracefully.
    private static void OnWindowResize(Silk.NET.Maths.Vector2D<int> newSize)
    {
        if (_engine == null || _gl == null || newSize.X == 0 || newSize.Y == 0) return;

        _engine.Resize(newSize.X, newSize.Y);
        _pixelBuffer = new byte[_engine.Width * _engine.Height * 4];
        _gl.Viewport(0, 0, (uint)newSize.X, (uint)newSize.Y);
    }

    // Listens for peripheral keystroke activities to manipulate thread paradigms or toggle diagnostic views.
    private static void OnKeyDown(IKeyboard keyboard, Key key, int keyCode)
    {
        if (key == Key.Number1) { _currentMode = 1; _modeName = "1. CPU Sequential"; _showComplexityMap = false; }
        if (key == Key.Number2) { _currentMode = 2; _modeName = "2. CPU Parallel For"; _showComplexityMap = false; }
        if (key == Key.Number3) { _currentMode = 3; _modeName = "3. CPU Dynamic Tiling"; _showComplexityMap = false; }
        if (key == Key.Number4) { _currentMode = 4; _modeName = "4. GPU Baseline SIMT"; }
        if (key == Key.Number5) { _currentMode = 5; _modeName = "5. GPU Tiled Optimization"; }
        if (key == Key.Number6) { _currentMode = 6; _modeName = "6. GPU Dynamic Thread Scheduling"; }

        if (key == Key.H)
        {
            if (_currentMode == 4 || _currentMode == 5 || _currentMode == 6)
            {
                _showComplexityMap = !_showComplexityMap;
                string baseGpuName = _currentMode == 4 ? "4. GPU Baseline SIMT" :
                              _currentMode == 5 ? "5. GPU Tiled Optimization" :
                                                  "6. GPU Dynamic Thread Scheduling";
                _modeName = _showComplexityMap ? $"{baseGpuName} (Heatmap Active)" : baseGpuName;
            }
        }
        if (key == Key.R)
        {
            _showReflectiveSphere = !_showReflectiveSphere;
        }

        // Trigger snapshot evaluation run via F5
        if (key == Key.F5)
        {
            _triggerBenchmarkSnapshot = true;
        }
    }

    private static void OnMouseDown(IMouse mouse, MouseButton button)
    {
        if (button == MouseButton.Left)
        {
            _isMouseDragging = true;
            _lastMousePos = mouse.Position;
        }
    }

    private static void OnMouseUp(IMouse mouse, MouseButton button)
    {
        if (button == MouseButton.Left) _isMouseDragging = false;
    }

    private static void OnMouseMove(IMouse mouse, Vector2 position)
    {
        if (!_isMouseDragging) return;

        float deltaX = position.X - _lastMousePos.X;
        float deltaY = position.Y - _lastMousePos.Y;
        _lastMousePos = position;

        const float sensitivity = 0.15f;
        _yaw += deltaX * sensitivity;
        _pitch -= deltaY * sensitivity;

        _pitch = Math.Clamp(_pitch, -89.0f, 89.0f);
    }

    private static void OnUpdate(double deltaTime)
    {
        if (_activeKeyboard == null) return;

        float radYaw = _yaw * (MathF.PI / 180.0f);
        float radPitch = _pitch * (MathF.PI / 180.0f);

        Vector3 forward;
        forward.X = MathF.Cos(radYaw) * MathF.Cos(radPitch);
        forward.Y = MathF.Sin(radPitch);
        forward.Z = MathF.Sin(radYaw) * MathF.Cos(radPitch);
        forward = Vector3.Normalize(forward);

        Vector3 right = Vector3.Normalize(Vector3.Cross(forward, Vector3.UnitY));
        Vector3 up = Vector3.UnitY;

        float moveSpeed = 4.0f * (float)deltaTime;

        if (_activeKeyboard.IsKeyPressed(Key.W)) _cameraPos += forward * moveSpeed;
        if (_activeKeyboard.IsKeyPressed(Key.S)) _cameraPos -= forward * moveSpeed;
        if (_activeKeyboard.IsKeyPressed(Key.A)) _cameraPos -= right * moveSpeed;
        if (_activeKeyboard.IsKeyPressed(Key.D)) _cameraPos += right * moveSpeed;
        if (_activeKeyboard.IsKeyPressed(Key.Space)) _cameraPos += up * moveSpeed;
        if (_activeKeyboard.IsKeyPressed(Key.ShiftLeft)) _cameraPos -= up * moveSpeed;
    }

    // RENDER OBJECTS
    private static unsafe void OnRender(double deltaTime)
    {
        _gl!.Clear((uint)ClearBufferMask.ColorBufferBit);

        GpuObject[] sceneObjects = new GpuObject[30];

        sceneObjects[0] = new GpuObject
        {
            Type = ShapeType.Sphere,
            Position = new GpuVector3(-1.6f, 0.0f, -5.0f),
            Size = new GpuVector3(1.4f, 0.0f, 0.0f),
            Mat = new Material { Color = new GpuVector3(240, 40, 40), Reflectivity = 0.1f }
        };

        sceneObjects[1] = new GpuObject
        {
            Type = ShapeType.Plane,
            Position = new GpuVector3(0.0f, -1.4f, 0.0f),
            Size = new GpuVector3(0.0f, 1.0f, 0.0f),
            Mat = new Material { Color = new GpuVector3(50, 180, 50), Reflectivity = 0.15f }
        };

        sceneObjects[2] = new GpuObject
        {
            Type = ShapeType.Box,
            Position = new GpuVector3(2.3f, -0.2f, -4.5f),
            Size = new GpuVector3(0.8f, 1.2f, 0.8f),
            Mat = new Material { Color = new GpuVector3(153, 51, 255), Reflectivity = 0.50f }
        };

        sceneObjects[3] = new GpuObject
        {
            Type = ShapeType.Pyramid,
            Position = new GpuVector3(0.3f, -1.4f, -4.7f),
            Size = new GpuVector3(0.7f, 1.5f, 0.7f),
            Mat = new Material { Color = new GpuVector3(240, 190, 30), Reflectivity = 0.45f }
        };

        sceneObjects[4] = new GpuObject
        {
            Type = ShapeType.Sphere,
            Position = new GpuVector3(0.0f, 1.2f, -3.5f),
            Size = new GpuVector3(0.6f, 0.0f, 0.0f),
            Mat = new Material { Color = new GpuVector3(255, 255, 255), Reflectivity = 0.95f }
        };

        sceneObjects[5] = new GpuObject
        {
            Type = ShapeType.Sphere,
            Position = new GpuVector3(0.0f, 1.2f, -3.5f),
            Size = new GpuVector3(0.7f, 0.0f, 0.0f),
            Mat = new Material { Color = new GpuVector3(200, 200, 255), Reflectivity = 0.85f }
        };

        for (int i = 0; i < 4; i++)
        {
            float angle = i * (MathF.PI / 2.0f);
            sceneObjects[6 + i] = new GpuObject
            {
                Type = ShapeType.Sphere,
                Position = new GpuVector3(0.0f + MathF.Cos(angle) * 1.1f, 1.2f + MathF.Sin(angle) * 0.5f, -3.5f),
                Size = new GpuVector3(0.25f, 0.0f, 0.0f),
                Mat = new Material { Color = new GpuVector3(255, 100, 250), Reflectivity = 0.70f }
            };
        }
        // Generates 5 non-overlapping highly reflective stress-test mirror spheres
        for (int i = 0; i < 5; i++)
        {
            // (Spacing of 1.5f ensures no overlap since the sphere diameter is 1.4f)
            float xOffset = -2.0f + (i * 1.5f);

            sceneObjects[10 + i] = new GpuObject
            {
                Type = ShapeType.Sphere,
                // If toggle is ON, space them along the X-axis. If OFF, hide them underground.
                Position = _showReflectiveSphere
                    ? new GpuVector3(xOffset, 1.2f, -5.5f)
                    : new GpuVector3(0.0f, -1000.0f, 0.0f),
                Size = new GpuVector3(0.7f, 0.0f, 0.0f),
                Mat = new Material { Color = new GpuVector3(255, 255, 255), Reflectivity = 0.95f }
            };
        }
        // Another 5 non-overlapping highly reflective stress-test mirror spheres (different side)
        for (int i = 0; i < 5; i++)
        {
            // (Spacing of 1.5f ensures no overlap since the sphere diameter is 1.4f)
            float xOffset = -2.0f + (i * 1.5f);

            sceneObjects[15 + i] = new GpuObject
            {
                Type = ShapeType.Sphere,
                // If toggle is ON, space them along the X-axis. If OFF, hide them underground.
                Position = _showReflectiveSphere
                    ? new GpuVector3(xOffset, 1.2f, -2.6f)
                    : new GpuVector3(0.0f, -1000.0f, 0.0f),
                Size = new GpuVector3(0.7f, 0.0f, 0.0f),
                Mat = new Material { Color = new GpuVector3(255, 255, 255), Reflectivity = 0.95f }
            };
        }
        // Another 5 non-overlapping highly reflective stress-test mirror spheres (different side)
        for (int i = 0; i < 5; i++)
        {
            // (Spacing of 1.5f ensures no overlap since the sphere diameter is 1.4f)
            float zOffset = -5.5f + (i * 1.5f);

            sceneObjects[20 + i] = new GpuObject
            {
                Type = ShapeType.Sphere,
                // If toggle is ON, space them along the X-axis. If OFF, hide them underground.
                Position = _showReflectiveSphere
                    ? new GpuVector3(-3.0f, 1.2f, zOffset)
                    : new GpuVector3(0.0f, -1000.0f, 0.0f),
                Size = new GpuVector3(0.7f, 0.0f, 0.0f),
                Mat = new Material { Color = new GpuVector3(255, 255, 255), Reflectivity = 0.95f }
            };
        }
        // Another 5 non-overlapping highly reflective stress-test mirror spheres (different side)
        for (int i = 0; i < 5; i++)
        {
            // (Spacing of 1.5f ensures no overlap since the sphere diameter is 1.4f)
            float zOffset = -5.5f + (i * 1.5f);

            sceneObjects[25 + i] = new GpuObject
            {
                Type = ShapeType.Sphere,
                // If toggle is ON, space them along the X-axis. If OFF, hide them underground.
                Position = _showReflectiveSphere
                    ? new GpuVector3(4.0f, 1.2f, zOffset)
                    : new GpuVector3(0.0f, -1000.0f, 0.0f),
                Size = new GpuVector3(0.7f, 0.0f, 0.0f),
                Mat = new Material { Color = new GpuVector3(255, 255, 255), Reflectivity = 0.95f }
            };
        }

        Vector3 lightDirection = Vector3.Normalize(new Vector3(0.5f, 1.0f, 0.3f));

        float radYaw = _yaw * (MathF.PI / 180.0f);
        float radPitch = _pitch * (MathF.PI / 180.0f);

        Vector3 forward;
        forward.X = MathF.Cos(radYaw) * MathF.Cos(radPitch);
        forward.Y = MathF.Sin(radPitch);
        forward.Z = MathF.Sin(radYaw) * MathF.Cos(radPitch);
        forward = Vector3.Normalize(forward);

        Vector3 right = Vector3.Normalize(Vector3.Cross(forward, Vector3.UnitY));
        Vector3 up = Vector3.Normalize(Vector3.Cross(right, forward));

        // EXECUTE PERFORMANCE BENCHMARK SNAPSHOT IF REQUESTED
        // OUTPUT PERFORMANCE ANALYSIS REPORT
        if (_triggerBenchmarkSnapshot)
        {
            _triggerBenchmarkSnapshot = false; // Reset immediately
            // Console.Clear();
            Console.WriteLine("=====================================================================================================");
            Console.WriteLine($" EVALUATION ANALYZER SNAPSHOT ({_engine!.Width}x{_engine!.Height} | 10 (30) scene objects)");
            Console.WriteLine($" Camera Position: ({_cameraPos.X:F2}, {_cameraPos.Y:F2}, {_cameraPos.Z:F2})");
            Console.WriteLine($"[Hardware Detected]: {_engine!.ActiveHardwareName}");
            Console.WriteLine("=====================================================================================================");
            Console.WriteLine(string.Format("| {0,-25} | {1,8} | {2,8} | {3,7} | {4,9} | {5,9} | {6,11} |",
                "Algorithm Engine", "Avg (ms)", "Max (ms)", "FPS", "Speedup", "MRays/s", "Efficiency"));
            Console.WriteLine("-----------------------------------------------------------------------------------------------------");

            var report = _engine.RunPerformanceAnalysis(sceneObjects, _cameraPos, forward, right, up, lightDirection);

            foreach (var metric in report)
            {
                // Format text cleanly depending on whether a mode was skipped or processed
                if (metric.AvgTimeMs == 0)
                {
                    Console.WriteLine(string.Format("| {0,-25} | {1,8} | {2,8} | {3,7} | {4,9} | {5,9} | {6,11} |",
                        metric.MethodName, "N/A", "N/A", "N/A", "1.00x", "0.00", "N/A"));
                }
                else
                {
                    Console.WriteLine(string.Format("| {0,-25} | {1,8:F2} | {2,8:F2} | {3,7:F1} | {4,8:F2}x | {5,9:F2} | {6,11} |",
                        metric.MethodName, metric.AvgTimeMs, metric.MaxTimeMs, metric.Fps, metric.Speedup, metric.MRaysPerSec, metric.Efficiency));
                }
            }
            Console.WriteLine("=====================================================================================================\n");
        }

        Stopwatch sw = Stopwatch.StartNew();

        switch (_currentMode)
        {
            case 1:
                _engine!.RenderSequential(_pixelBuffer!, sceneObjects, _cameraPos, forward, right, up, lightDirection);
                break;
            case 2:
                _engine!.RenderCpuParallel(_pixelBuffer!, sceneObjects, _cameraPos, forward, right, up, lightDirection);
                break;
            case 3:
                _engine!.RenderCpuDynamicTiling(_pixelBuffer!, sceneObjects, _cameraPos, forward, right, up, lightDirection);
                break;
            case 4:
                _engine!.RenderGpuParallel(_pixelBuffer!, sceneObjects, _cameraPos, forward, right, up, lightDirection, _showComplexityMap);
                break;
            case 5:
                _engine!.RenderGpuTiledOptimized(_pixelBuffer!, sceneObjects, _cameraPos, forward, right, up, lightDirection, _showComplexityMap);
                break;
            case 6:
                _engine!.RenderGpuDynamicScheduled(_pixelBuffer!, sceneObjects, _cameraPos, forward, right, up, lightDirection, _showComplexityMap);
                break;
        }

        sw.Stop();

        _window!.Title = $"[{_modeName}] Latency: {sw.Elapsed.TotalMilliseconds:F2} ms | Cam: ({_cameraPos.X:F1}, {_cameraPos.Y:F1}, {_cameraPos.Z:F1}) | [F5] Analyze";

        _gl.BindTexture(TextureTarget.Texture2D, _textureId);
        fixed (byte* p = _pixelBuffer)
        {
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                           (uint)_engine!.Width, (uint)_engine!.Height,
                           0, PixelFormat.Bgra, PixelType.UnsignedByte, p);
        }

        _gl.UseProgram(_shaderProgram);
        _gl.BindVertexArray(_vao);
        _gl.DrawArrays(PrimitiveType.Triangles, 0, 6);
    }

    private static void OnClosing()
    {
        _gl?.DeleteVertexArray(_vao);
        _gl?.DeleteBuffer(_vbo);
        _gl?.DeleteProgram(_shaderProgram);
        _gl?.DeleteTexture(_textureId);
        _engine?.Dispose();
    }
}