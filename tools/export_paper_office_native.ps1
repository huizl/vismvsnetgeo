param(
    [ValidateSet('visio','word')][string]$Mode = 'visio'
)
$ErrorActionPreference = 'Stop'
$paperRoot = Split-Path $PSScriptRoot -Parent
$paperOut = Join-Path $paperRoot 'docs\paper_word_visio'
$paperPreview = Join-Path $paperOut 'visio_previews'
$paperQa = Join-Path $paperOut 'qa'
New-Item -ItemType Directory -Path $paperPreview,$paperQa -Force | Out-Null
$paperCulture = [System.Globalization.CultureInfo]::InvariantCulture

function Number-U([double]$Value) { return $Value.ToString('0.########', $paperCulture) }
function Color-U([string]$Hex) {
    $r = [Convert]::ToInt32($Hex.Substring(1,2),16)
    $g = [Convert]::ToInt32($Hex.Substring(3,2),16)
    $b = [Convert]::ToInt32($Hex.Substring(5,2),16)
    return "RGB($r,$g,$b)"
}

if ($Mode -eq 'visio') {
    $spec = Get-Content -LiteralPath (Join-Path $paperOut 'visio_diagrams_spec.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $app = New-Object -ComObject Visio.InvisibleApp
    $doc = $null
    $restoreRaster = $false
    try {
        $app.AlertResponse = 7
        $doc = $app.Documents.Add('')
        try {
            [int]$oldResolution=0; [double]$oldWidth=96; [double]$oldHeight=96; [int]$oldUnits=0
            $app.Settings.GetRasterExportResolution([ref]$oldResolution,[ref]$oldWidth,[ref]$oldHeight,[ref]$oldUnits)
            $restoreRaster=$true
            $app.Settings.SetRasterExportResolution(3,180,180,0)
        } catch { Write-Output 'Using native Visio default PNG export resolution.' }
        $report = @()
        $pageIndex=0
        foreach ($pageSpec in $spec.pages) {
            if ($pageIndex -eq 0) { $page = $doc.Pages.Item(1) } else { $page = $doc.Pages.Add() }
            $pageIndex++
            $page.Name = $pageSpec.name
            $page.PageSheet.CellsU('PageWidth').FormulaU = (Number-U $pageSpec.width)+' in'
            $page.PageSheet.CellsU('PageHeight').FormulaU = (Number-U $pageSpec.height)+' in'
            $shapeMap = @{}
            foreach ($item in $pageSpec.shapes) {
                $shape=$page.DrawRectangle([double]$item.x,[double]$item.y,[double]($item.x+$item.w),[double]($item.y+$item.h))
                $shape.NameU = $item.id
                $shape.Text = $item.text
                $shape.CellsU('Char.Font').FormulaU = 'FONT("Microsoft YaHei")'
                $shape.CellsU('Char.Size').FormulaU = (Number-U ([double]$item.font))+' pt'
                $shape.CellsU('Char.Style').FormulaU = $(if ($item.bold) {'1'} else {'0'})
                $shape.CellsU('Char.Color').FormulaU = 'RGB(0,0,0)'
                $shape.CellsU('Para.HorzAlign').FormulaU = '1'
                $shape.CellsU('VerticalAlign').FormulaU = '1'
                $shape.CellsU('LeftMargin').FormulaU = '0.075 in'
                $shape.CellsU('RightMargin').FormulaU = '0.075 in'
                $shape.CellsU('TopMargin').FormulaU = '0.04 in'
                $shape.CellsU('BottomMargin').FormulaU = '0.04 in'
                if ($item.fill -eq 'none') { $shape.CellsU('FillPattern').FormulaU='0' }
                else { $shape.CellsU('FillForegnd').FormulaU=Color-U $item.fill; $shape.CellsU('FillPattern').FormulaU='1' }
                if ($item.stroke -eq 'none') { $shape.CellsU('LinePattern').FormulaU='0' }
                else { $shape.CellsU('LineColor').FormulaU=Color-U $item.stroke; $shape.CellsU('LineWeight').FormulaU='1 pt' }
                $shapeMap[$item.id]=$shape
            }
            foreach ($item in $pageSpec.connectors) {
                $source=$shapeMap[$item.from]; $target=$shapeMap[$item.to]
                $ports=@{right=@(1.0,0.5);left=@(0.0,0.5);top=@(0.5,1.0);bottom=@(0.5,0.0);center=@(0.5,0.5)}
                $a=$ports[$item.begin]; $b=$ports[$item.end]
                $line=$page.DrawLine(0.0,0.0,1.0,1.0)
                $line.CellsU('BeginX').GlueToPos($source,[double]$a[0],[double]$a[1])
                $line.CellsU('EndX').GlueToPos($target,[double]$b[0],[double]$b[1])
                $line.CellsU('LineColor').FormulaU=Color-U $item.color
                $line.CellsU('LineWeight').FormulaU='1.2 pt'
                $line.CellsU('EndArrow').FormulaU=$(if ($item.head -eq $false) {'0'} else {'4'})
                if ($item.dashed) { $line.CellsU('LinePattern').FormulaU='2' }
            }
            $page.Export((Join-Path $paperPreview ($pageSpec.name+'.svg')))
            $page.Export((Join-Path $paperPreview ($pageSpec.name+'.png')))
            $report += [pscustomobject]@{page=$pageSpec.name;shapes=$page.Shapes.Count;connectors=$pageSpec.connectors.Count}
            Write-Output ('Visio page saved: '+$pageSpec.name)
        }
        $file=Join-Path $paperOut '论文图表_可编辑.vsdx'
        $doc.SaveAs($file)
        $doc.Close(); $doc=$null
        $check=$app.Documents.Open($file)
        if ($check.Pages.Count -ne 10) { throw 'Visio page count mismatch.' }
        $check.Close()
        $report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $paperOut 'visio_validation.json') -Encoding utf8
        Write-Output ('Native Visio file verified: '+$file)
    } finally {
        if ($null -ne $doc) { $doc.Saved=$true; $doc.Close() }
        if ($restoreRaster) { $app.Settings.SetRasterExportResolution($oldResolution,$oldWidth,$oldHeight,$oldUnits) }
        $app.Quit()
    }
} else {
    $app=New-Object -ComObject Word.Application
    $doc=$null
    try {
        $app.Visible=$false
        $app.DisplayAlerts=0
        $app.AutomationSecurity=3
        $file=Join-Path $paperOut '大视差与复杂遮挡深度估计论文.docx'
        $doc=$app.Documents.Open($file,$false,$false,$false)
        $doc.Fields.Update() | Out-Null
        $doc.Repaginate()
        $pages=$doc.ComputeStatistics(2)
        $math=$doc.OMaths.Count
        $doc.Save()
        $pdf=Join-Path $paperQa 'paper_native_word.pdf'
        $doc.ExportAsFixedFormat($pdf,17)
        [pscustomobject]@{pages=$pages;native_math_objects=$math;native_word_pdf=$pdf;word_version=$app.Version} |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $paperOut 'word_render_validation.json') -Encoding utf8
        Write-Output ('Native Word pages: '+$pages+'; math objects: '+$math)
        $doc.Close(0); $doc=$null
    } finally {
        if ($null -ne $doc) { $doc.Close(0) }
        $app.Quit()
    }
}
