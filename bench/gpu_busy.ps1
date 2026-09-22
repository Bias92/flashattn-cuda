# Prints "<percent> <process>" for the busiest GPU user that is not the WSL VM.
# The WSL VM (vmwp / vmmem) is where the benchmarks themselves run, so it is left out.
# Five one-second samples, median per process: a browser repaint is a blip, a game is not.
$sets = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 5 -ErrorAction SilentlyContinue)
$perPid = @{}
foreach ($set in $sets) {
    $sum = @{}
    foreach ($s in $set.CounterSamples) {
        if ($s.InstanceName -match 'pid_(\d+)') {
            $procId = [int]$Matches[1]
            $sum[$procId] = [double]$sum[$procId] + $s.CookedValue
        }
    }
    foreach ($procId in $sum.Keys) {
        if (-not $perPid.ContainsKey($procId)) { $perPid[$procId] = @() }
        $perPid[$procId] += $sum[$procId]
    }
}
$worst = 0.0
$who = 'none'
foreach ($procId in $perPid.Keys) {
    $name = try { (Get-Process -Id $procId -ErrorAction Stop).ProcessName } catch { 'unknown' }
    if ($name -in @('vmwp', 'vmmem', 'vmmemWSL')) { continue }
    $sorted = $perPid[$procId] | Sort-Object
    $median = $sorted[[int][math]::Floor($sorted.Count / 2)]
    if ($median -gt $worst) { $worst = $median; $who = $name }
}
'{0:N0} {1}' -f $worst, $who
