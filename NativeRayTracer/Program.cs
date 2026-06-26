using System;
using System.Diagnostics;
using System.Numerics;
using Silk.NET.Input;
using Silk.NET.OpenGL;
using Silk.NET.Windowing;

namespace NativeRayTracer;

class Program
{
    private static IWindow? _window;
    private static GL? _gl;
    private static RayTraceEngine? _engine;
    private static byte[]? _pixelBuffer;

    private static uint _textureId;
    private static uint _vao;
    private static uint _vbo;
    private static uint _shaderProgram;

    private static int _currentMode = 1; // 1 = Sequential, 2 = CPU Parallel, 3 = GPU Parallel
    private static string _modeName = "CPU Sequential";

    static void Main(string[] args)
    {
        var options = WindowOptions.Default;
        options.Size = new Silk.NET.Maths.Vector2D<int>(RayTraceEngine.Width, RayTraceEngine.Height);
        options.Title = "Native Ray Tracer Benchmark (Keys 1, 2, 3 to swap)";
        options.VSync = false; // Disable VSync to observe true performance unthrottled

        _window = Window.Create(options);
        _window.Load += OnLoad;
        _window.Render += OnRender;
        _window.Closing += OnClosing;

        _window.Run();
    }

    private static unsafe void OnLoad()
    {
        _gl = _window!.CreateOpenGL();
        _engine = new RayTraceEngine();
        _pixelBuffer = new byte[RayTraceEngine.Width * RayTraceEngine.Height * 4];

        // Wire up input handlers
        var inputContext = _window!.CreateInput();
        foreach (var keyboard in inputContext.Keyboards)
        {
            keyboard.KeyDown += OnKeyDown;
        }

        // 1. Create fully compliant Modern OpenGL Shaders to display our 2D texture array
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

        // 2. Setup full-screen viewport coordinates mapping layout properties
        float[] vertices = {
            // Positions   // TexCoords
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

        // 3. Instantiate standard hardware texture memory
        _textureId = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _textureId);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
    }

    private static void OnKeyDown(IKeyboard keyboard, Key key, int keyCode)
    {
        if (key == Key.Number1) { _currentMode = 1; _modeName = "CPU Sequential"; }
        if (key == Key.Number2) { _currentMode = 2; _modeName = "CPU Parallel"; }
        if (key == Key.Number3) { _currentMode = 3; _modeName = "GPU Parallel (ILGPU)"; }
    }

    private static unsafe void OnRender(double deltaTime)
    {
        _gl!.Clear((uint)ClearBufferMask.ColorBufferBit);

        Sphere demoSphere = new Sphere
        {
            Center = new Vector3(0, 0, -5.0f),
            Radius = 1.6f,
            Color = new Vector3(240, 60, 60)
        };
        Vector3 lightDirection = Vector3.Normalize(new Vector3(1.0f, 1.0f, -1.0f));

        Stopwatch sw = Stopwatch.StartNew();

        if (_currentMode == 1) _engine!.RenderSequential(_pixelBuffer!, demoSphere, lightDirection);
        else if (_currentMode == 2) _engine!.RenderCpuParallel(_pixelBuffer!, demoSphere, lightDirection);
        else if (_currentMode == 3) _engine!.RenderGpuParallel(_pixelBuffer!, demoSphere);

        sw.Stop();

        _window!.Title = $"[{_modeName}] Execution Time: {sw.Elapsed.TotalMilliseconds:F2} ms";

        // Upload our flat byte array to GPU video memory texture slots
        _gl.BindTexture(TextureTarget.Texture2D, _textureId);
        fixed (byte* p = _pixelBuffer)
        {
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                           (uint)RayTraceEngine.Width, (uint)RayTraceEngine.Height,
                           0, PixelFormat.Bgra, PixelType.UnsignedByte, p);
        }

        // Use modern shader array call routines to render our frame quad layout bounds
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