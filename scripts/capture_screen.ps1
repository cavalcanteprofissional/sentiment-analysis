param(
    [Parameter(Mandatory = $true)]
    [string]$OutputFile
)

<#
Captura a tela completa (todos os monitores) e salva em PNG.
Uso:
    powershell -ExecutionPolicy Bypass -File scripts/capture_screen.ps1 -OutputFile docs/prints/00_mlflow_ui.png
Sem dependencias externas (System.Windows.Forms + System.Drawing do .NET).
#>

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)

$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$graphics.Dispose()

$parent = Split-Path -Parent $OutputFile
if ($parent -and -not (Test-Path $parent)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
}

$bitmap.Save($OutputFile, [System.Drawing.Imaging.ImageFormat]::Png)
$bitmap.Dispose()

Write-Host "Captura salva: $OutputFile"