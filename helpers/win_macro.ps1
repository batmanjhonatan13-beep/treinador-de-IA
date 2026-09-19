# Gravacao / replay de teclado e mouse no desktop do Windows.
# So roda quando o usuario dispara. Sem persistencia, sem rede.
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("ping", "record", "replay")]
    [string]$Mode,

    [string]$OutFile = "",
    [string]$InFile = "",
    [string]$Skill = "",
    [int]$MoveMinPx = 4
)

$ErrorActionPreference = "Stop"

Add-Type @"
using System;
using System.Runtime.InteropServices;

public class WinIo {
    public const int INPUT_MOUSE = 0;
    public const int INPUT_KEYBOARD = 1;
    public const uint KEYEVENTF_KEYUP = 0x0002;
    public const uint MOUSEEVENTF_MOVE = 0x0001;
    public const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
    public const uint MOUSEEVENTF_LEFTUP = 0x0004;
    public const uint MOUSEEVENTF_RIGHTDOWN = 0x0008;
    public const uint MOUSEEVENTF_RIGHTUP = 0x0010;
    public const uint MOUSEEVENTF_MIDDLEDOWN = 0x0020;
    public const uint MOUSEEVENTF_MIDDLEUP = 0x0040;
    public const uint MOUSEEVENTF_WHEEL = 0x0800;
    public const uint MOUSEEVENTF_ABSOLUTE = 0x8000;
    public const int SM_CXSCREEN = 0;
    public const int SM_CYSCREEN = 1;

    [StructLayout(LayoutKind.Sequential)]
    public struct POINT { public int X; public int Y; }

    [StructLayout(LayoutKind.Sequential)]
    public struct INPUT {
        public int type;
        public INPUTUNION u;
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct INPUTUNION {
        [FieldOffset(0)] public MOUSEINPUT mi;
        [FieldOffset(0)] public KEYBDINPUT ki;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MOUSEINPUT {
        public int dx;
        public int dy;
        public uint mouseData;
        public uint dwFlags;
        public uint time;
        public IntPtr dwExtraInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct KEYBDINPUT {
        public ushort wVk;
        public ushort wScan;
        public uint dwFlags;
        public uint time;
        public IntPtr dwExtraInfo;
    }

    [DllImport("user32.dll")] public static extern short GetAsyncKeyState(int vKey);
    [DllImport("user32.dll")] public static extern bool GetCursorPos(out POINT lpPoint);
    [DllImport("user32.dll")] public static extern int GetSystemMetrics(int nIndex);
    [DllImport("user32.dll", SetLastError = true)]
    public static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);
}
"@

function Get-ScreenSize {
    return @{
        w = [WinIo]::GetSystemMetrics(0)
        h = [WinIo]::GetSystemMetrics(1)
    }
}

function Test-Down([int]$vk) {
    return ([WinIo]::GetAsyncKeyState($vk) -band 0x8000) -ne 0
}

function Get-Cursor {
    $p = New-Object WinIo+POINT
    [void][WinIo]::GetCursorPos([ref]$p)
    return @{ x = $p.X; y = $p.Y }
}

function Get-VkName([int]$vk) {
    $map = @{
        1 = "mouse_left"; 2 = "mouse_right"; 4 = "mouse_middle"
        8 = "backspace"; 9 = "tab"; 13 = "enter"; 16 = "shift"; 17 = "ctrl"
        18 = "alt"; 19 = "pause"; 20 = "caps"; 27 = "esc"; 32 = "space"
        33 = "pageup"; 34 = "pagedown"; 35 = "end"; 36 = "home"
        37 = "left"; 38 = "up"; 39 = "right"; 40 = "down"
        45 = "insert"; 46 = "delete"; 91 = "win"; 92 = "rwin"
        112 = "F1"; 113 = "F2"; 114 = "F3"; 115 = "F4"; 116 = "F5"
        117 = "F6"; 118 = "F7"; 119 = "F8"; 120 = "F9"; 121 = "F10"
        122 = "F11"; 123 = "F12"; 160 = "lshift"; 161 = "rshift"
        162 = "lctrl"; 163 = "rctrl"; 164 = "lalt"; 165 = "ralt"
    }
    if ($map.ContainsKey($vk)) { return $map[$vk] }
    if ($vk -ge 48 -and $vk -le 90) { return [char]$vk }
    return "vk$vk"
}

if ($Mode -eq "ping") {
    $s = Get-ScreenSize
    Write-Output ("win-macro-ok {0}x{1}" -f $s.w, $s.h)
    exit 0
}

if ($Mode -eq "record") {
    if (-not $OutFile) { throw "OutFile obrigatorio no record" }
    $screen = Get-ScreenSize
    Write-Host ""
    Write-Host "GRAVANDO skill='$Skill'  $($screen.w)x$($screen.h)"
    Write-Host "F8 = parar e salvar   F9 = pausa (digite senha aqui)   F10 = cancelar"
    Write-Host "NAO digite senha, cartao ou token enquanto estiver GRAVANDO."
    Write-Host ""

    $watch = @(
        1, 2, 4, 8, 9, 13, 16, 17, 18, 20, 27, 32, 33, 34, 35, 36, 37, 38, 39, 40, 45, 46, 91
    )
    48..90 | ForEach-Object { $watch += $_ }
    112..123 | ForEach-Object { $watch += $_ }
    160..165 | ForEach-Object { $watch += $_ }
    $watch += 186..192
    $watch += 219..222
    $watch = $watch | Select-Object -Unique

    $prev = @{}
    foreach ($vk in $watch) { $prev[$vk] = $false }

    $events = New-Object System.Collections.Generic.List[object]
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $paused = $false
    $lastPos = Get-Cursor
    $f9Was = $false
    $exitCode = 1

    while ($true) {
        if (Test-Down 119) { $exitCode = 0; break }       # F8
        if (Test-Down 121) { $exitCode = 2; break }       # F10
        $f9Now = Test-Down 120
        if ($f9Now -and -not $f9Was) {
            $paused = -not $paused
            if ($paused) { Write-Host "PAUSA - F9 de novo para voltar a gravar" }
            else { Write-Host "GRAVANDO de novo" }
        }
        $f9Was = $f9Now

        if (-not $paused) {
            $t = [math]::Round($sw.Elapsed.TotalSeconds, 3)
            $pos = Get-Cursor
            $dx = [math]::Abs($pos.x - $lastPos.x)
            $dy = [math]::Abs($pos.y - $lastPos.y)
            if ($dx -ge $MoveMinPx -or $dy -ge $MoveMinPx) {
                $events.Add([pscustomobject]@{
                        t = $t; type = "mouse_move"; x = $pos.x; y = $pos.y
                    })
                $lastPos = $pos
            }
            foreach ($vk in $watch) {
                if ($vk -eq 119 -or $vk -eq 120 -or $vk -eq 121) { continue }
                $now = Test-Down $vk
                if ($now -eq $prev[$vk]) { continue }
                $kind = if ($now) { "down" } else { "up" }
                if ($vk -in 1, 2, 4) {
                    $btn = @{ 1 = "left"; 2 = "right"; 4 = "middle" }[$vk]
                    $events.Add([pscustomobject]@{
                            t = $t; type = ("mouse_" + $kind); button = $btn; x = $pos.x; y = $pos.y
                        })
                }
                else {
                    $events.Add([pscustomobject]@{
                            t      = $t
                            type   = ("key_" + $kind)
                            vk     = $vk
                            name   = Get-VkName $vk
                        })
                }
                $prev[$vk] = $now
            }
        }
        Start-Sleep -Milliseconds 10
    }

    if ($exitCode -eq 2) {
        Write-Host "cancelado. nada salvo."
        exit 2
    }

    $payload = [pscustomobject]@{
        skill   = $Skill
        backend = "windows"
        screen  = $screen
        events  = $events
    }
    $dir = Split-Path -Parent $OutFile
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -Path $OutFile -Encoding UTF8
    Write-Host ("salvo {0} eventos em {1}" -f $events.Count, $OutFile)
    exit 0
}

if ($Mode -eq "replay") {
    if (-not $InFile) { throw "InFile obrigatorio no replay" }
    if (-not (Test-Path $InFile)) { throw "arquivo nao existe: $InFile" }
    $macro = Get-Content -Path $InFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $screen = Get-ScreenSize
    $srcW = [double]$macro.screen.w
    $srcH = [double]$macro.screen.h
    if ($srcW -le 0) { $srcW = $screen.w }
    if ($srcH -le 0) { $srcH = $screen.h }

    function Send-MouseAbs([int]$x, [int]$y, [uint32]$flags, [uint32]$data = 0) {
        $nx = [int](($x / $srcW) * 65535)
        $ny = [int](($y / $srcH) * 65535)
        $inp = New-Object WinIo+INPUT
        $inp.type = [WinIo]::INPUT_MOUSE
        $inp.u.mi.dx = $nx
        $inp.u.mi.dy = $ny
        $inp.u.mi.mouseData = $data
        $inp.u.mi.dwFlags = $flags -bor [WinIo]::MOUSEEVENTF_ABSOLUTE
        [void][WinIo]::SendInput(1, @($inp), [System.Runtime.InteropServices.Marshal]::SizeOf([type][WinIo+INPUT]))
    }

    function Send-Key([uint16]$vk, [bool]$up) {
        $inp = New-Object WinIo+INPUT
        $inp.type = [WinIo]::INPUT_KEYBOARD
        $inp.u.ki.wVk = $vk
        $inp.u.ki.dwFlags = $(if ($up) { [WinIo]::KEYEVENTF_KEYUP } else { [uint32]0 })
        [void][WinIo]::SendInput(1, @($inp), [System.Runtime.InteropServices.Marshal]::SizeOf([type][WinIo+INPUT]))
    }

    Write-Host ("replay $($macro.skill)  $($macro.events.Count) eventos  F10 cancela")
    $lastT = 0.0
    foreach ($ev in $macro.events) {
        if (Test-Down 121) { Write-Host "replay cancelado"; exit 2 }
        $wait = [double]$ev.t - $lastT
        if ($wait -gt 0) { Start-Sleep -Milliseconds ([int]($wait * 1000)) }
        $lastT = [double]$ev.t
        switch ($ev.type) {
            "mouse_move" { Send-MouseAbs ([int]$ev.x) ([int]$ev.y) ([WinIo]::MOUSEEVENTF_MOVE) }
            "mouse_down" {
                $f = @{ left = [WinIo]::MOUSEEVENTF_LEFTDOWN; right = [WinIo]::MOUSEEVENTF_RIGHTDOWN; middle = [WinIo]::MOUSEEVENTF_MIDDLEDOWN }[$ev.button]
                Send-MouseAbs ([int]$ev.x) ([int]$ev.y) ($f)
            }
            "mouse_up" {
                $f = @{ left = [WinIo]::MOUSEEVENTF_LEFTUP; right = [WinIo]::MOUSEEVENTF_RIGHTUP; middle = [WinIo]::MOUSEEVENTF_MIDDLEUP }[$ev.button]
                Send-MouseAbs ([int]$ev.x) ([int]$ev.y) ($f)
            }
            "key_down" { Send-Key ([uint16]$ev.vk) $false }
            "key_up" { Send-Key ([uint16]$ev.vk) $true }
        }
    }
    Write-Host "replay ok"
    exit 0
}
